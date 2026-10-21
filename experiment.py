"""AG News: development-only selection, frozen test evaluation, and retrieval ablation.

Commands: python experiment.py prepare | develop | test | analyze | all
"""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import random
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'results'
LABELS = ['World', 'Sports', 'Business', 'Sci/Tech']
MODELS = {'qwen': 'Qwen/Qwen2.5-0.5B-Instruct',
          'smol': 'HuggingFaceTB/SmolLM2-360M-Instruct'}
MODEL_REVISIONS = {'qwen': '7ae557604adf67be50417f59c2c2f167def9a775',
                   'smol': 'a10cc1512eabd3dde888204e902eca88bddb4951'}
SEED = 42


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def prepare():
    from prepare_data import prepare_data
    if (RUN / 'data/manifest.json').exists():
        return
    splits = prepare_data()
    manifest = {}
    seen = set()
    for name, data in zip(['train', 'dev', 'test'], splits):
        rows = []
        for i, item in enumerate(data):
            digest = hashlib.sha256(item['text'].strip().encode()).hexdigest()
            if digest in seen:
                # Exact duplicates invalidate the working split.
                raise ValueError('Duplicate news across/within splits: ' + digest)
            seen.add(digest)
            rows.append(dict(id=f'{name}-{i}', text=item['text'], label=item['label'], sha256=digest))
        save(RUN / f'data/{name}.json', rows)
        manifest[name] = {'n': len(rows), 'labels': [sum(r['label'] == c for r in rows) for c in range(4)],
                          'sha256': hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()}
    save(RUN / 'data/manifest.json', manifest)


def prompt(text, examples, style='short'):
    instruction = ('Classify the news into one of these categories: World, Sports, Business, Sci/Tech. '
                   'Output only the category name.\n\n')
    if style == 'definitions':
        instruction += ('World: politics, diplomacy, conflicts and international affairs.\n'
                        'Sports: athletes, teams and sporting competitions.\n'
                        'Business: markets, companies, finance and the economy.\n'
                        'Sci/Tech: science, computing, technology and research.\n'
                        'Choose the main topic of the story.\n\n')
    demonstration_text = ''.join(f"News: {r['text']}\nCategory: {LABELS[r['label']]}\n\n" for r in examples)
    if style == 'explicit':
        return (instruction + 'Labelled examples:\n' + demonstration_text +
                f'Now classify this news item:\n{text}\n\n' +
                'Return exactly one label: World, Sports, Business, or Sci/Tech. Do not add an explanation or another news item.')
    return instruction + demonstration_text + f'News: {text}\nCategory:'


def messages_for(text, examples, style='short'):
    if style != 'chat':
        return [{'role': 'user', 'content': prompt(text, examples, style)}]
    messages = [{'role': 'system', 'content':
                 'Classify news as World, Sports, Business, or Sci/Tech. Reply with the exact label only.'}]
    for example in examples:
        messages += [{'role': 'user', 'content': example['text']},
                     {'role': 'assistant', 'content': LABELS[example['label']]}]
    messages.append({'role': 'user', 'content': text})
    return messages


def demos(train, k, seed=SEED):
    rng = random.Random(seed)
    pools = [[r for r in train if r['label'] == c] for c in range(4)]
    for pool in pools:
        rng.shuffle(pool)
    # Balanced 4/8-shot prompts; 1-shot uses a random class.
    order = list(range(4))
    rng.shuffle(order)
    return [pools[order[i % 4]][i // 4] for i in range(k)]


def balanced_subset(rows, n):
    rng = random.Random(SEED)
    chosen = []
    for c in range(4):
        pool = [r for r in rows if r['label'] == c]
        rng.shuffle(pool)
        chosen.extend(pool[:n//4])
    rng.shuffle(chosen)
    return chosen


class Retrieval:
    def __init__(self, train):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.train = train
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=30000, sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform([r['text'] for r in train])

    def get(self, text, k):
        import numpy as np
        scores = (self.matrix @ self.vectorizer.transform([text]).T).toarray().ravel()
        # Stable ordering; only training texts/labels enter retrieval.
        indices = np.argsort(-scores, kind='stable')
        return [self.train[i] for i in indices if self.train[i]['text'].strip() != text.strip()][:k]


def load_model(key, adapter=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = MODELS[key]
    cache = os.environ.get('CMU_MODEL_CACHE', str(ROOT / 'model_cache'))
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=MODEL_REVISIONS[key], cache_dir=cache)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=MODEL_REVISIONS[key], cache_dir=cache,
             torch_dtype=(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32),
             attn_implementation='sdpa')
    save(RUN / f'models/{key}.json', {'model_id': model_id,
         'revision': getattr(model.config, '_commit_hash', None),
         'layers': model.config.num_hidden_layers, 'hidden_size': model.config.hidden_size,
         'vocab_size': model.config.vocab_size,
         'base_parameters': sum(p.numel() for p in model.parameters())})
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter))
    # Neutralize sampling defaults from model cards for clean greedy generation.
    model.generation_config.temperature = 1.0
    model.generation_config.top_p = 1.0
    model.generation_config.top_k = 50
    return model.to('cuda' if torch.cuda.is_available() else 'cpu'), tokenizer


def metrics(rows):
    from sklearn.metrics import classification_report
    gold = [r['label'] for r in rows]
    pred = [r['prediction'] for r in rows]
    n = len(rows)
    if not n:
        raise ValueError('Empty evaluation set')
    correct = sum(a == b for a, b in zip(gold, pred))
    p = correct / n
    z = 1.96
    center = (p + z*z/(2*n)) / (1+z*z/n)
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1+z*z/n)
    report = classification_report(gold, pred, labels=list(range(4)), target_names=LABELS,
                                   output_dict=True, zero_division=0)
    return {'n': n, 'correct': correct, 'accuracy': p, 'wilson_95': [center-half, center+half],
            'macro_f1': report['macro avg']['f1-score'], 'invalid_rate': sum(x == -1 for x in pred)/n,
            'per_class': report,
            'confusion_matrix_4x5': [[sum(r['label'] == c and r['prediction'] == d for r in rows)
                                     for d in [0, 1, 2, 3, -1]] for c in range(4)]}


def evaluate(model, tokenizer, rows, train, config, path):
    import torch
    if path.exists():
        cached = read(path)
        if cached['config'] != config or [r['id'] for r in cached['predictions']] != [r['id'] for r in rows]:
            raise ValueError('Incompatible cached evaluation: ' + str(path))
        return cached
    model.eval()
    model.config.use_cache = True
    retrieval = Retrieval(train) if config.get('retrieval') else None
    examples = demos(train, config.get('shots', 0), config.get('seed', SEED))
    predictions = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    batch_size = 4 if model.device.type == 'cuda' else 1
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i+batch_size]
        sequences, batch_demos = [], []
        for item in batch:
            ex = retrieval.get(item['text'], config['shots']) if retrieval else examples
            text = item['text'][:800]
            ex = [dict(r, text=r['text'][:800]) for r in ex]
            messages = messages_for(text, ex, config.get('style', 'short'))
            sequences.append(tokenizer.apply_chat_template(messages,
                                                           tokenize=True, add_generation_prompt=True))
            batch_demos.append([r['id'] for r in ex])
        width = max(map(len, sequences))
        ids = torch.tensor([[tokenizer.pad_token_id]*(width-len(s))+s for s in sequences], device=model.device)
        masks = torch.tensor([[0]*(width-len(s))+[1]*len(s) for s in sequences], device=model.device)
        if width + 16 > model.config.max_position_embeddings:
            raise ValueError('Prompt exceeds context window; reduce shots before development')
        with torch.inference_mode():
            out = model.generate(input_ids=ids, attention_mask=masks,
                                 max_new_tokens=16, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        for j, item in enumerate(batch):
            raw = tokenizer.decode(out[j, width:], skip_special_tokens=True).strip()
            label = next((c for c, name in enumerate(LABELS) if raw.casefold() == name.casefold()), -1)
            predictions.append(dict(item, prediction=label, raw_output=raw, demo_ids=batch_demos[j]))
        if len(predictions) % 100 == 0:
            print(f'{path.stem}: {len(predictions)}/{len(rows)}', flush=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    result = {'config': config, 'metrics': metrics(predictions), 'seconds': time.perf_counter()-start,
              'model_revision': getattr(model.config, '_commit_hash', None),
              'inference_batch_size': batch_size,
              'peak_gpu_mb': torch.cuda.max_memory_allocated()/1024**2 if torch.cuda.is_available() else 0,
              'predictions': predictions}
    save(path, result)
    print(path.stem, result['metrics']['accuracy'], flush=True)
    return result


def release(model):
    import torch
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def answer_only_loss(model, inputs):
    """Exact masked causal-LM loss without computing unused prompt vocabulary logits.

    Both selected architectures expose .model and .lm_head. All input tokens still
    traverse the transformer; only supervised next-token positions enter the head.
    """
    import torch.nn.functional as F
    base = model.get_base_model()
    hidden = base.model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                        use_cache=False, return_dict=True).last_hidden_state
    targets = inputs['labels'][:, 1:]
    active = targets != -100
    logits = base.lm_head(hidden[:, :-1][active])
    return F.cross_entropy(logits.float(), targets[active])


def train_adapter(key, train, dev, lr, epochs, directory):
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import Trainer, TrainingArguments, set_seed
    if (directory / 'complete.json').exists():
        return
    set_seed(SEED)
    model, tokenizer = load_model(key)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16,
                           lora_dropout=0.05, target_modules=['q_proj', 'v_proj'], bias='none'))
    model.enable_input_require_grads()
    parameter_counts = {'total': sum(p.numel() for p in model.parameters()),
                        'trainable': sum(p.numel() for p in model.parameters() if p.requires_grad)}

    def encode(item):
        messages = [{'role': 'user', 'content': prompt(item['text'][:800], [])}]
        prefix = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        full = tokenizer.apply_chat_template(messages + [{'role': 'assistant', 'content': LABELS[item['label']]}], tokenize=True)
        if full[:len(prefix)] != prefix:
            raise ValueError('Chat template prefix mismatch')
        # Shorten news if tokenizer expands it beyond a 384-token training budget.
        text = item['text'][:800]
        while len(full) > 384:
            text = text[:int(len(text)*0.8)]
            if not text:
                raise ValueError('Instruction alone exceeds training budget')
            messages = [{'role': 'user', 'content': prompt(text, [])}]
            prefix = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            full = tokenizer.apply_chat_template(messages + [{'role': 'assistant', 'content': LABELS[item['label']]}], tokenize=True)
            if full[:len(prefix)] != prefix:
                raise ValueError('Chat template prefix mismatch')
        return {'input_ids': full, 'attention_mask': [1]*len(full),
                'labels': [-100]*len(prefix) + full[len(prefix):]}

    def collate(batch):
        width = max(len(x['input_ids']) for x in batch)
        return {k: torch.tensor([x[k] + [pad]*(width-len(x[k])) for x in batch])
                for k, pad in [('input_ids', tokenizer.pad_token_id), ('attention_mask', 0), ('labels', -100)]}

    class AnswerTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_bf16_supported()):
                loss = answer_only_loss(model, inputs)
            return (loss, {'loss': loss}) if return_outputs else loss

    args = TrainingArguments(output_dir=str(directory / 'checkpoints'), num_train_epochs=epochs,
           learning_rate=lr, per_device_train_batch_size=2, per_device_eval_batch_size=2,
           gradient_accumulation_steps=4, bf16=torch.cuda.is_bf16_supported(),
           gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
           eval_strategy='epoch', prediction_loss_only=True, save_strategy='no', logging_steps=25, report_to='none',
           optim='adamw_torch', seed=SEED, data_seed=SEED, dataloader_num_workers=0)
    trainer = AnswerTrainer(model=model, args=args, train_dataset=Dataset.from_list([encode(r) for r in train]),
                      eval_dataset=Dataset.from_list([encode(r) for r in dev]), data_collator=collate)
    trainer.model_accepts_loss_kwargs = False
    trainer.train()
    adapter_norms = {n: float(p.detach().float().norm()) for n, p in model.named_parameters() if 'lora_B' in n}
    if not all(math.isfinite(v) for v in adapter_norms.values()) or not any(v > 0 for v in adapter_norms.values()):
        raise RuntimeError('LoRA B matrices are zero or non-finite: adapter training failed')
    trainer.save_model(str(directory))
    tokenizer.save_pretrained(str(directory))
    training_record = {'lr': lr, 'epochs': epochs, 'parameters': parameter_counts,
                       'log_history': trainer.state.log_history, 'lora_B_norms': adapter_norms}
    save(directory / 'complete.json', training_record)
    save(RUN / f'training_logs/{key}_{lr}.json', training_record)
    del trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def develop(keys, smoke=False, quick=False):
    from transformers import set_seed
    set_seed(SEED)
    train, dev = [read(RUN / f'data/{s}.json') for s in ['train', 'dev']]
    if smoke:
        train, dev = train[:16], dev[:8]
    if quick:
        train, dev = balanced_subset(train, 128), balanced_subset(dev, 80)
        save(RUN / 'protocol.json', {'mode': 'simplified', 'seed': SEED, 'train_used': len(train),
             'dev_used': len(dev), 'test_used': 400, 'train_ids': [r['id'] for r in train],
             'dev_ids': [r['id'] for r in dev], 'learning_rates': [2e-4],
             'limitations': 'No LoRA hyperparameter search or extra retrieval experiment.'})
    selection = read(RUN / 'selected.json') if (RUN / 'selected.json').exists() else {}
    for key in keys:
        candidates = []
        model, tokenizer = load_model(key)
        prompt_candidates = [('short', 0), ('short', 4)] if quick else [('short', 0), ('short', 1), ('short', 4), ('definitions', 4), ('explicit', 4), ('chat', 4)]
        for style, shots in prompt_candidates:
            config = {'model': key, 'method': 'icl', 'style': style, 'shots': shots, 'seed': SEED}
            result = evaluate(model, tokenizer, dev, train, config, RUN / f'dev/{key}_icl_{style}_{shots}.json')
            if shots > 0:
                candidates.append((result['metrics']['accuracy'], config))
        best = max(candidates, key=lambda x: x[0])[1]
        # Extra experiment: retrieval vs balanced fixed examples, matched style and shot count.
        fixed = dict(best, shots=4)
        retrieval = dict(fixed, retrieval=True)
        if not quick:
            evaluate(model, tokenizer, dev, train, fixed, RUN / f'dev/{key}_extra_fixed.json')
            evaluate(model, tokenizer, dev, train, retrieval, RUN / f'dev/{key}_extra_retrieval.json')
        del model
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        ft = []
        for lr in ([2e-4] if quick else [1e-4, 2e-4]):
            directory = RUN / f'adapters/{key}_{lr}'
            train_adapter(key, train, dev, lr, 1, directory)
            model, tokenizer = load_model(key, directory)
            config = {'model': key, 'method': 'lora', 'shots': 0, 'style': 'short', 'lr': lr,
                      'adapter': directory.relative_to(ROOT).as_posix()}
            result = evaluate(model, tokenizer, dev, train, config, RUN / f'dev/{key}_lora_{lr}.json')
            ft.append((result['metrics']['accuracy'], config))
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        selection[key] = {'icl': best, 'lora': max(ft, key=lambda x: x[0])[1],
                          'extra_fixed': fixed, 'extra_retrieval': retrieval}
        if quick:
            selection[key].pop('extra_fixed')
            selection[key].pop('extra_retrieval')
        save(RUN / 'selected.json', selection)


def test(keys, quick=False):
    # Final configuration is written before any test labels are read for evaluation.
    selected = read(RUN / 'selected.json')
    signature = hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest()
    lock = RUN / 'test_lock.json'
    if lock.exists() and read(lock)['selection_sha256'] != signature:
        raise ValueError('Selection changed after test evaluation. Use a new experiment directory.')
    save(lock, {'selection_sha256': signature})
    train, test_rows = [read(RUN / f'data/{s}.json') for s in ['train', 'test']]
    if quick:
        train = balanced_subset(train, 128)
    for key in keys:
        for name, config in selected[key].items():
            control_path = RUN / f'test/{key}_{name}.json'
            main_path = RUN / f'test/{key}_icl.json'
            if name == 'extra_fixed' and config == selected[key]['icl'] and not control_path.exists():
                control = read(main_path)
                control['reused_from'] = main_path.name
                save(control_path, control)
                continue
            model, tokenizer = load_model(key, ROOT / config['adapter'] if 'adapter' in config else None)
            evaluate(model, tokenizer, test_rows, train, config, RUN / f'test/{key}_{name}.json')
            if name == 'lora':
                evaluate(model, tokenizer, train, train, config, RUN / f'train/{key}_lora.json')
            del model
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def analyze():
    import csv
    import itertools
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.plot_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    results = {p.stem: read(p) for p in sorted((RUN / 'test').glob('*.json'))}
    if not results:
        raise ValueError('No final test results. Run development and test first.')
    lines = ['# AG News 实验结果', '',
             '|系统|正确数/总数|Accuracy|95% Wilson 区间|Macro-F1|无效输出率|秒/条|',
             '|---|---|---|---|---|---|---|']
    with (RUN / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['system', 'n', 'correct', 'accuracy', 'macro_f1', 'invalid_rate', 'seconds'])
        for name, result in results.items():
            m = result['metrics']
            lo, hi = m['wilson_95']
            lines.append(f"|{name}|{m['correct']}/{m['n']}|{m['accuracy']:.2%}|{lo:.2%}–{hi:.2%}|{m['macro_f1']:.4f}|{m['invalid_rate']:.2%}|{result['seconds']/m['n']:.3f}|")
            writer.writerow([name, m['n'], m['correct'], m['accuracy'], m['macro_f1'], m['invalid_rate'], result['seconds']])
            fig, ax = plt.subplots(figsize=(7, 4))
            matrix = m['confusion_matrix_4x5']
            ax.imshow(matrix, cmap='Blues')
            ax.set(xticks=range(5), xticklabels=LABELS+['Invalid'], yticks=range(4), yticklabels=LABELS,
                   xlabel='Prediction', ylabel='Gold', title=name)
            for i in range(4):
                for j in range(5):
                    cell_color = 'white' if matrix[i][j] > max(max(row) for row in matrix)/2 else 'black'
                    ax.text(j, i, matrix[i][j], ha='center', va='center', color=cell_color)
            fig.tight_layout()
            fig.savefig(RUN / f'{name}_confusion.png', dpi=160)
            plt.close(fig)
    lines += ['', '## 同一模型中 ICL 与 LoRA 的差异']
    for key in MODELS:
        a, b = results.get(key+'_icl'), results.get(key+'_lora')
        if a and b:
            delta = b['metrics']['accuracy'] - a['metrics']['accuracy']
            lines.append(f"- {key}: LoRA − ICL = {delta*100:+.2f} 个百分点。")
            train_path = RUN / f'train/{key}_lora.json'
            if train_path.exists():
                tr = read(train_path)['metrics']['accuracy']
                lines.append(f"  训练准确率 {tr:.2%}，测试准确率 {b['metrics']['accuracy']:.2%}；差距 {100*(tr-b['metrics']['accuracy']):.2f} 个百分点。结合验证 loss 曲线判断过拟合，不能仅凭训练 loss 下降下结论。")
    lines += ['', '## 配对错误分析', '', '|系统A|系统B|A对B错|A错B对|两者都错|精确McNemar p|', '|---|---|---|---|---|---|']
    main = {k: v for k, v in results.items() if k.endswith('_icl') or k.endswith('_lora')}
    from scipy.stats import binomtest
    for (an, a), (bn, b) in itertools.combinations(main.items(), 2):
        ar, br = a['predictions'], b['predictions']
        if [r['id'] for r in ar] != [r['id'] for r in br]:
            raise ValueError('Unpaired samples')
        ab = ba = both = 0
        for x, y in zip(ar, br):
            ac, bc = x['prediction'] == x['label'], y['prediction'] == y['label']
            ab += ac and not bc
            ba += bc and not ac
            both += not ac and not bc
        p = binomtest(ab, ab+ba, 0.5).pvalue if ab+ba else 1.0
        lines.append(f'|{an}|{bn}|{ab}|{ba}|{both}|{p:.4g}|')
    lines += ['', '配对检验是探索性分析，未进行多重比较校正；单次训练不能反映训练随机性。', '', '## 错误示例']
    for name, result in main.items():
        lines += ['', f'### {name}']
        for r in [r for r in result['predictions'] if r['prediction'] != r['label']][:5]:
            lines += [f"- {r['id']}，真实：{LABELS[r['label']]}；生成：{r['raw_output']!r}。", f"  新闻：{r['text']}"]
    lines += ['', '## 部署选择', '',
              '优先比较测试准确率与每类召回率；再结合延迟、显存和训练成本决定。检索方法需计入检索时间；LoRA 合并后延迟可能变化，本实验未测量合并部署。',
              '额外实验用相同模型、相同提示说明、相同4个示例数量比较固定与TF-IDF检索；检索只能访问训练集。提升不能归因于模型大小，因为两个主模型同时在规模和训练方法上存在差异。']
    (RUN / 'analysis_zh.md').write_text('\n'.join(lines), encoding='utf-8')
    fig, ax = plt.subplots(figsize=(7.5, 4))
    names = list(main)
    values = [results[n]['metrics']['accuracy'] for n in names]
    errors = [[values[i]-results[n]['metrics']['wilson_95'][0] for i, n in enumerate(names)],
              [results[n]['metrics']['wilson_95'][1]-values[i] for i, n in enumerate(names)]]
    ax.bar(names, values, yerr=errors, capsize=3,
           color=['#bd683b' if name.endswith('_lora') else '#497aab' for name in names])
    ax.set(ylabel='Accuracy', ylim=(0, 1), title='AG News test accuracy (95% Wilson intervals)')
    plt.xticks(rotation=25, ha='right')
    fig.tight_layout()
    fig.savefig(RUN / 'accuracy.png', dpi=180)
    plt.close(fig)


def main():
    import torch
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=['prepare', 'develop', 'test', 'analyze', 'all'])
    parser.add_argument('--models', nargs='+', choices=list(MODELS), default=list(MODELS))
    parser.add_argument('--smoke', action='store_true', help='Separate development-only quick check; never use as final results')
    parser.add_argument('--quick', action='store_true', help='Simplified real experiment: 128 train, 80 dev, 400 test; no extra experiment')
    parser.add_argument('--offline', action='store_true', help='Use downloaded dataset/model files without remote checks')
    parser.add_argument('--results-dir', help='New run directory relative to the project root')
    args = parser.parse_args()
    if args.offline:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['HF_DATASETS_OFFLINE'] = '1'
    global RUN
    if args.results_dir:
        RUN = (ROOT / args.results_dir).resolve()
        if not RUN.is_relative_to(ROOT) or RUN == ROOT:
            parser.error('The results directory must be a subdirectory of the project.')
        if args.quick or args.smoke:
            parser.error('--results-dir cannot be combined with --quick or --smoke')
    if args.quick:
        RUN = ROOT / 'quick_results'
    if args.quick and args.smoke:
        parser.error('Use either --quick or --smoke')
    if args.smoke:
        RUN = ROOT / 'smoke_results_verified'
        if args.stage not in ['prepare', 'develop']:
            parser.error('Smoke runs may not evaluate final test data')
    if args.stage in ['develop', 'all'] and (RUN / 'test_lock.json').exists():
        raise ValueError('Test already evaluated; development is frozen.')
    RUN.mkdir(parents=True, exist_ok=True)
    environment = {p: importlib.metadata.version(p) for p in
                   ['torch', 'transformers', 'peft', 'datasets', 'accelerate', 'scikit-learn',
                    'numpy', 'scipy', 'matplotlib', 'tokenizers', 'huggingface-hub']}
    environment['device'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    environment['cuda_runtime'] = torch.version.cuda
    save(RUN / f'environments/{args.stage}.json', environment)
    if not (RUN / 'environment.json').exists():
        save(RUN / 'environment.json', environment)
    if args.stage in ['prepare', 'all', 'develop']:
        prepare()
    if args.stage in ['develop', 'all']:
        develop(args.models, args.smoke, args.quick)
    if args.stage in ['test', 'all']:
        test(args.models, args.quick)
    if args.stage in ['analyze', 'all']:
        analyze()


if __name__ == '__main__':
    main()
