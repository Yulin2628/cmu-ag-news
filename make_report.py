"""Build the report from frozen experiments; no model inference is performed."""

import itertools
import argparse
from experiment import ROOT, RUN, MODELS, LABELS, read, prompt, demos, messages_for


def pct(value):
    return f'{value:.2%}'


def table(headers, rows):
    lines = ['|' + '|'.join(headers) + '|', '|' + '|'.join(['---'] * len(headers)) + '|'] + [
        '|' + '|'.join(map(str, row)) + '|' for row in rows]
    return ['\n'.join(lines)]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--results-dir', default='results')
    args = parser.parse_args()
    global RUN
    RUN = (ROOT / args.results_dir).resolve()
    report_root = ROOT if RUN == ROOT / 'results' else RUN
    asset_prefix = RUN.relative_to(report_root).as_posix()
    selected = read(RUN / 'selected.json')
    tests = {f'{key}_{method}': read(RUN / f'test/{key}_{method}.json')
             for key in MODELS for method in ['icl', 'lora', 'extra_fixed', 'extra_retrieval']}
    training = read(RUN / 'data/train.json')
    primary = {name: result for name, result in tests.items() if name.endswith(('_icl', '_lora'))}
    best = max(primary, key=lambda name: primary[name]['metrics']['accuracy'])
    lines = [
        '# News classification with in-context examples and LoRA',
        'CMU 11-667 mini-project implementation',
        '## Abstract',
        'This study compares two compact instruction-tuned models on AG News. We select prompts and LoRA learning rates on a development set, then evaluate the frozen systems on a fresh, balanced test sample. A separate matched experiment compares fixed and retrieved examples. All scores below come from saved generated predictions, including outputs that fail to match a class label.',
        f'The strongest of the four primary systems is {best}, with {pct(primary[best]["metrics"]["accuracy"])} test accuracy. The class-level analysis and format-error rates qualify this aggregate result.',
        '## 1. Task and data',
        'The task assigns an English news item to World, Sports, Business or Sci/Tech. Topic routing is useful for news organisation and search. It requires identifying the main event rather than relying only on named entities: a technology company can appear in a finance story, and government decisions can affect economic reporting.',
        'AG News has 120,000 official training examples and 7,600 official test examples [1]. We remove exact duplicate text from the test pool, then remove duplicate training texts and any training text also present in the official test pool. Sampling is stratified by class. The working splits contain 2,000 train, 400 development and 400 test examples, with 500/100/100 per class. The training and development selections use seed 42.',
        'An earlier exploratory run used a 400-item test sample drawn with seed 42. Its results had already been inspected. This revision excludes those texts from the official test pool and draws a new balanced test sample with seed 43. No fresh-test predictions are used to select prompts, learning rates or retrieval settings. The old pilot results do not appear in the final comparison. Text hashes and split counts are saved in results/data/manifest.json.',
        f'Example training item ({training[0]["id"]}, label {LABELS[training[0]["label"]]}):',
        '> ' + training[0]['text'].replace('\n', ' '),
        '### Ethical considerations',
        'The source contains older English-language news, with the geographic and editorial biases of its publishers. A single label can obscure stories that span politics, finance and technology. Incorrect routing can hide relevant information from readers. Public reporting can still discuss sensitive events or identify people, so qualitative analysis quotes only the text needed to explain an error. We have not audited whether these news items appeared in either base model\'s pretraining data. The dataset card does not specify a clear dataset license; the source should be checked before downstream redistribution [1].',
        '## 2. Task formulation and evaluation',
        'We express classification as short text generation. The user message contains a task instruction, optional labelled demonstrations and the target news item. The assistant target is the correct category name followed by the model\'s chat end marker. For example:',
        '```text', prompt(training[0]['text'][:800], []),
        'Assistant target: ' + LABELS[training[0]['label']], '```',
        'Only assistant target tokens contribute to training cross-entropy. Prompt and padding labels are -100. Apart from duplicate filtering and truncation, the dataset text is retained as provided; escaped punctuation and HTML-like entities are not additionally cleaned. News is limited to 800 characters at inference. During training, if the full dialogue exceeds 384 tokens, the news text is shortened until the complete answer fits. Different tokenizers can therefore retain different amounts of the same training item.',
        'Inference uses the native chat template, greedy decoding and at most 16 new tokens. We score only newly generated text. After stripping surrounding whitespace and ignoring case, the entire string must equal one of the four class names. Explanations, multiple categories and alternative names such as Science/Tech are invalid and count as incorrect. This measures both topic classification and compliance with the requested output format. It avoids selecting a convenient class word from a longer answer after seeing the gold label.',
        'Accuracy is correct predictions divided by all predictions. Uniform random guessing has expected accuracy 25% on these balanced splits. We also report macro-F1, per-class precision/recall/F1 and invalid-output rate. Invalid predictions count as false negatives for their true class. Confusion matrices add an Invalid column. Wilson 95% intervals describe test-sample uncertainty. Pairwise exact McNemar tests compare correctness on the same news items, with no correction for multiple comparisons. These are exploratory comparisons, not evidence about training-seed variability.',
        '## 3. Models and development',
        'Qwen2.5-0.5B-Instruct and SmolLM2-360M-Instruct are compact instruction-tuned causal language models from different families [2,3]. Their model sizes, tokenizers and training histories differ. The comparison can show which complete system works better here, but cannot isolate the causal effect of parameter count or model family.',
        '### Prompt search',
        'Each model independently compares a short instruction with one demonstration, the same instruction with four demonstrations, and a four-example prompt with category definitions. A zero-example prompt is evaluated as a separate baseline. The primary ICL system must use at least one demonstration. Four fixed examples contain one training item per class. Demonstration selection is deterministic with seed 42. One-shot uses one class chosen by the same fixed random procedure, so class imbalance in that prompt is a limitation.',
        'The initial SmolLM2 one-shot validation trial scored only 3.25%, with 92% invalid outputs, often including a prefix such as Category: World rather than the bare label. We therefore add two development candidates for both models: an explicit final instruction forbidding extra news or explanations, and demonstrations as alternating user/assistant chat turns. This is iterative development on the validation set, not a fully preregistered prompt search. The fresh test remains unevaluated during this change.',
        'We select the ICL configuration with the highest strict generated-label development accuracy. Ties follow the candidate order: short 1-shot, short 4-shot, definitions 4-shot, explicit 4-shot, chat 4-shot. Definitions test explicit class boundaries, while the added formats address instruction following. We retain every completed trial, including failures. Four unchanged full-size Qwen prompt-validation trials from the preceding development attempt are reused; no previous final-test score enters selection.',
    ]
    rows = []
    model_rows = []
    for key in MODELS:
        meta = read(RUN / f'models/{key}.json')
        model_rows.append([key, f'{meta["base_parameters"]:,}', meta['layers'], meta['hidden_size'], meta['vocab_size']])
    # Insert quantitative architecture details before the validation trials.
    insert_at = lines.index('### Prompt search')
    lines[insert_at:insert_at] = table(['Model','Base parameters','Layers','Hidden size','Vocabulary'], model_rows)
    for path in sorted((RUN / 'dev').glob('*_icl_*.json')):
        r = read(path)
        rows.append([path.stem, pct(r['metrics']['accuracy']), pct(r['metrics']['invalid_rate'])])
    lines += table(['Validation prompt trial', 'Accuracy', 'Invalid'], rows)
    for key in MODELS:
        cfg = selected[key]['icl']
        chosen_trial = read(RUN / f'dev/{key}_icl_{cfg["style"]}_{cfg["shots"]}.json')
        baseline = read(RUN / f'dev/{key}_icl_short_0.json')['metrics']
        gain = 100*(chosen_trial['metrics']['accuracy']-baseline['accuracy'])
        lines += [f'For {key}, the selected {cfg["shots"]}-shot {cfg["style"]} prompt has development accuracy {pct(chosen_trial["metrics"]["accuracy"])} and invalid-output rate {pct(chosen_trial["metrics"]["invalid_rate"])}. Its development difference from zero-shot is {gain:+.2f} points. This selected-development comparison is not an independent estimate of the benefit of examples.']
        scores = sorted([read(path)['metrics']['accuracy'] for path in (RUN/'dev').glob(f'{key}_icl_*.json')
                         if read(path)['config']['shots'] > 0], reverse=True)
        if len(scores)>1 and scores[0]-scores[1] <= 0.01:
            margin = round(400*(scores[0]-scores[1]))
            lines += [f'The top two few-shot settings differ by only {margin} of 400 development items. The selected setting should not be interpreted as a robust prompt advantage.']
        failed = [r for r in chosen_trial['predictions'] if r['prediction'] == -1]
        if failed:
            lines += [f'An actual invalid development output for {key} is {failed[0]["raw_output"][:150]!r}. This illustrates the output-control issue visible in the trial table, rather than a change to the scoring rule.']
        if key == 'smol':
            old = read(RUN/'dev/smol_icl_short_4.json')['metrics']
            explicit = read(RUN/'dev/smol_icl_explicit_4.json')['metrics']
            lines += [f'Keeping four demonstrations, the explicit-final-instruction variant changes invalid-output rate from {pct(old["invalid_rate"])} to {pct(explicit["invalid_rate"])} and accuracy from {pct(old["accuracy"])} to {pct(explicit["accuracy"])}. Fewer format failures do not by themselves imply correct topic classification.']
        shown_examples = [dict(r, text=r['text'][:800]) for r in demos(training, cfg['shots'], cfg['seed'])]
        shown_messages = '\n\n'.join(f'[{message["role"]}]\n{message["content"]}'
                                     for message in messages_for('<target news item>', shown_examples, cfg['style']))
        lines += [f'### Selected {key} prompt', f'Style: {cfg["style"]}. Demonstrations: {cfg["shots"]}. These role/content messages, including the selected training examples, are passed to the model\'s native chat template.',
                  '```text', shown_messages, '```']
    lines += [
        '### LoRA training',
        'LoRA freezes the pretrained weights and trains low-rank updates [4]. Both models use rank 8, scaling alpha 16, dropout 0.05 and query/value projection targets. We compare learning rates 0.0001 and 0.0002 independently for each model. Holding the other settings fixed makes the learning-rate comparison interpretable under the available compute budget. It does not establish a globally optimal tuning setup.',
        'Each adapter trains for one epoch on all 2,000 training items using AdamW, physical batch size 2 and gradient accumulation 4, giving effective batch size 8. Training and data seeds are 42. Other Trainer defaults include a linear learning-rate schedule, no warmup and weight decay zero. The run uses gradient checkpointing and BF16 on an RTX 3050 Laptop GPU with 4 GB VRAM. Only supervised next-token positions are projected to vocabulary logits, which saves memory while preserving the masked causal-LM objective. A regression check compares both loss and all LoRA gradients against the standard objective on small Qwen2 and Llama models.',
        'Learning-rate selection uses generated-label development accuracy rather than training loss. Every saved adapter is checked for finite, nonzero LoRA B-matrix updates. Model-specific trainable parameter counts and observed losses follow.',
    ]
    rows = []
    for key in MODELS:
        for lr in [1e-4, 2e-4]:
            log = RUN / f'training_logs/{key}_{lr}.json'
            info = read(log if log.exists() else RUN / f'adapters/{key}_{lr}/complete.json')
            tr = next(x['train_loss'] for x in reversed(info['log_history']) if 'train_loss' in x)
            dv = next(x['eval_loss'] for x in reversed(info['log_history']) if 'eval_loss' in x)
            m = read(RUN / f'dev/{key}_lora_{lr}.json')['metrics']
            rows.append([key, lr, f'{tr:.4f}', f'{dv:.4f}', pct(m['accuracy']), f'{info["parameters"]["trainable"]:,}'])
    lines += table(['Model', 'LR', 'Train loss', 'Dev loss', 'Dev accuracy', 'Trainable parameters'], rows)
    lines += ['; '.join(f'{key} selects LR {selected[key]["lora"]["lr"]}' for key in MODELS) + '.',
              '## 4. Final test results',
              'All primary systems use the same 400 fresh test news items. One item changes accuracy by 0.25 percentage points. Configuration hashes are written before final prediction. The intervals below reflect test sampling, not repeated training.']
    rows = []
    for name, r in primary.items():
        m = r['metrics']
        rows.append([name, f'{m["correct"]}/{m["n"]}', pct(m['accuracy']),
                     f'{pct(m["wilson_95"][0])} to {pct(m["wilson_95"][1])}',
                     f'{m["macro_f1"]:.3f}', pct(m['invalid_rate']), f'{r["seconds"]/m["n"]:.3f}'])
    lines += table(['System', 'Correct', 'Accuracy', '95% interval', 'Macro-F1', 'Invalid', 's/item'], rows)
    for method in ['icl', 'lora']:
        names = [f'{key}_{method}' for key in MODELS]
        leader = max(names, key=lambda name: tests[name]['metrics']['accuracy'])
        other = next(name for name in names if name != leader)
        difference = 100*(tests[leader]['metrics']['accuracy']-tests[other]['metrics']['accuracy'])
        lines += [f'Across the two {method.upper()} systems, {leader} has the higher test accuracy, by {difference:.2f} percentage points over {other}. The per-class and paired comparisons below indicate where that aggregate difference comes from. Because model family, size and training history all differ, the result cannot identify which factor caused the difference.']
    lines += [f'![Test accuracy and Wilson intervals]({asset_prefix}/accuracy.png)',
              'Figure 1. The four primary systems, with 95% Wilson intervals. The matched retrieval experiment is reported separately.']
    for key in MODELS:
        a, b = tests[key+'_icl']['metrics'], tests[key+'_lora']['metrics']
        train_m = read(RUN / f'train/{key}_lora.json')['metrics']
        dev_m = read(RUN / f'dev/{key}_lora_{selected[key]["lora"]["lr"]}.json')['metrics']
        fixed = sum(x['prediction'] != x['label'] and y['prediction'] == y['label'] for x,y in zip(tests[key+'_icl']['predictions'], tests[key+'_lora']['predictions']))
        regressed = sum(x['prediction'] == x['label'] and y['prediction'] != y['label'] for x,y in zip(tests[key+'_icl']['predictions'], tests[key+'_lora']['predictions']))
        lines += [f'### {key}: ICL and fine-tuning',
                  f'ICL scores {pct(a["accuracy"])} and LoRA scores {pct(b["accuracy"])}. The difference is {100*(b["accuracy"]-a["accuracy"]):+.2f} percentage points. LoRA corrects {fixed} ICL errors but changes {regressed} previously correct predictions into errors. It is therefore important to inspect regressions as well as the net change.',
                  f'The selected adapter has training accuracy {pct(train_m["accuracy"])}, development accuracy {pct(dev_m["accuracy"])} and test accuracy {pct(b["accuracy"])}. The train-test gap is {100*(train_m["accuracy"]-b["accuracy"]):+.2f} points. One epoch gives only one end-of-epoch development loss, so this cannot establish a train/dev divergence over time. Development accuracy is also optimistic after configuration selection. These results give limited evidence about overfitting rather than proving its presence or absence.',
                  f'ICL invalid-output rate is {pct(a["invalid_rate"])} versus {pct(b["invalid_rate"])} for LoRA. Fixing every invalid ICL answer could add at most {100*a["invalid_rate"]:.2f} points without correcting any valid-but-wrong class predictions. This upper bound helps distinguish output-format gains from other changes.']
        drops = [label for label in LABELS if b['per_class'][label]['recall'] < a['per_class'][label]['recall']]
        if drops:
            lines += ['The aggregate gain is not uniform across classes. Observed recall decreases are: ' +
                      '; '.join(f'{label}, {pct(a["per_class"][label]["recall"])} to {pct(b["per_class"][label]["recall"])}' for label in drops) +
                      '. Each class has 100 test items, so a one-item change moves recall by one percentage point. These point estimates should be read alongside the confusion matrices.']
    lines += ['## 5. Error analysis']
    rows = []
    for name, r in primary.items():
        for label in LABELS:
            m = r['metrics']['per_class'][label]
            rows.append([name, label, f'{m["precision"]:.3f}', f'{m["recall"]:.3f}', f'{m["f1-score"]:.3f}'])
    lines += table(['System', 'Class', 'Precision', 'Recall', 'F1'], rows)
    note_path = ROOT / 'qualitative_notes.json'
    for name, r in primary.items():
        matrix = r['metrics']['confusion_matrix_4x5']
        biggest = max((matrix[i][j], LABELS[i], LABELS[j]) for i in range(4) for j in range(4) if i != j)
        lines += [f'### {name}', f'The largest valid-label confusion is {biggest[1]} classified as {biggest[2]} ({biggest[0]} items).',
                  f'![Confusion matrix for {name}]({asset_prefix}/{name}_confusion.png)',
                  'Rows are gold labels. Columns are predicted labels, with a separate column for invalid output.']
        totals = [sum(row[c] for row in matrix) for c in range(4)]
        dominant = max(range(4), key=lambda c: totals[c])
        if totals[dominant] > len(r['predictions'])*0.6:
            lines += [f'The system emits {LABELS[dominant]} for {totals[dominant]} of 400 items, although the gold set contains 100 items per class. This is a strong concentration in one output class, not random guessing. It helps explain the aggregate score without attributing an unobserved reasoning process to the model.']
        errors = [row for row in r['predictions'] if row['label'] != row['prediction']]
        for item in (errors[:1] if not note_path.exists() else []):
            lines += [f'Gold: {LABELS[item["label"]]}. Generated: {item["raw_output"]!r}.', '> ' + item['text'][:350].replace('\n',' ')]
    if note_path.exists():
        index = {row['id']:row for row in next(iter(primary.values()))['predictions']}
        for note in read(note_path):
            item = index[note['id']]
            if item['sha256'] != note['sha256']:
                raise ValueError('Qualitative note refers to a different test item.')
            lines += ['### ' + note['title'], '> ' + item['text'].replace('\n',' '),
                      'Gold label: ' + LABELS[item['label']] + '.']
            case_rows = []
            for name,result in primary.items():
                case = next(r for r in result['predictions'] if r['id']==note['id'])
                case_rows.append([name,case['raw_output'].replace('\n',' ').replace('|','/')])
            lines += table(['System','Actual generated output'],case_rows)
            lines += [note['analysis']]
    common = [i for i in range(400) if all(r['predictions'][i]['prediction'] != r['predictions'][i]['label'] for r in primary.values())]
    lines += [f'All four primary systems miss {len(common)} of the same test items. This count includes both format failures and wrong valid labels.']
    from scipy.stats import binomtest
    rows = []
    for (an,a),(bn,b) in itertools.combinations(primary.items(),2):
        ab = ba = both = 0
        for x,y in zip(a['predictions'],b['predictions']):
            ac,bc = x['prediction']==x['label'], y['prediction']==y['label']
            ab += ac and not bc
            ba += bc and not ac
            both += not ac and not bc
        p = binomtest(ab,ab+ba,0.5).pvalue if ab+ba else 1
        rows.append([an,bn,ab,ba,both,f'{p:.4g}'])
    lines += table(['A','B','A only correct','B only correct','Both wrong','Exact paired p'],rows)
    lines += ['The examples are observational evidence, not access to model reasoning. Cross-topic news, dataset labelling conventions, truncation and strict category naming can all contribute to errors. A factual explanation of each selected item requires reading its actual text rather than attributing every mistake to a general lack of knowledge.',
              '## 6. Additional experiment: retrieved demonstrations',
              'Hypothesis: examples with vocabulary related to the target news item will improve classification compared with a fixed balanced set. We fit a TF-IDF vectorizer only on the 2,000 training texts, with unigram/bigram features, sublinear term frequency and at most 30,000 features. Default L2 normalisation makes the sparse dot product a cosine similarity. Four highest-scoring training items supply both text and known labels. Retrieval ranks with the full raw target text; the language model still sees at most 800 target characters and 800 characters per demonstration. The target label and development/test examples never enter the index.',
              'The matched fixed and retrieved systems use the same base model, instruction style and four demonstrations. The instruction style comes from that model\'s selected ICL configuration. When the selected primary ICL configuration is already this exact four-shot control, its measured predictions are reused and marked in the result metadata; these are not additional independent replicates. Fixed examples are class-balanced; retrieved ones need not be. Thus the contrast tests the full selection strategy, including relevance and class mix, rather than similarity alone.']
    rows = []
    for key in MODELS:
        a,b = tests[key+'_extra_fixed']['metrics'],tests[key+'_extra_retrieval']['metrics']
        delta = 100*(b['accuracy']-a['accuracy'])
        rows.append([key,pct(a['accuracy']),pct(b['accuracy']),pct(tests[key+'_lora']['metrics']['accuracy']),f'{delta:+.2f}'])
        lines += [f'For {key}, retrieval changes test accuracy by {delta:+.2f} percentage points. This {"supports" if delta>0 else "does not support"} the hypothesis for this model and fixed experiment. A single split does not justify a general claim that retrieval always helps.']
        ab = ba = 0
        for x,y in zip(tests[key+'_extra_fixed']['predictions'],tests[key+'_extra_retrieval']['predictions']):
            ac,bc = x['prediction']==x['label'],y['prediction']==y['label']
            ab += ac and not bc
            ba += bc and not ac
        p = binomtest(ab,ab+ba,0.5).pvalue if ab+ba else 1
        lines += [f'Retrieval corrects {ba} fixed-prompt errors but introduces {ab} regressions (exact paired p={p:.4g}, uncorrected). Per-class recall changes from fixed to retrieved are: ' +
                  '; '.join(f'{label} {pct(a["per_class"][label]["recall"])} to {pct(b["per_class"][label]["recall"])}' for label in LABELS) + '.']
        retrieved_counts = [sum(row[c] for row in b['confusion_matrix_4x5']) for c in range(4)]
        dominant = max(range(4),key=lambda c: retrieved_counts[c])
        if retrieved_counts[dominant] > 240:
            fixed_count = sum(row[dominant] for row in a['confusion_matrix_4x5'])
            lines += [f'The retrieved system predicts {LABELS[dominant]} on {retrieved_counts[dominant]}/400 items, compared with {fixed_count}/400 for the fixed control. This output concentration characterises the observed result, but does not isolate whether relevance, example class balance or another prompt interaction produced it.']
    lines += table(['Model','Fixed 4-shot','Retrieved 4-shot','Selected LoRA','Retrieval difference (points)'],rows)
    lines += ['## 7. Deployment and limitations',
              f'For an accuracy-first deployment trial, {best} is the strongest primary candidate on this test set ({pct(primary[best]["metrics"]["accuracy"])}). The weakest class and invalid-output rate still matter: a news router that loses a topic systematically may be unsuitable even with a higher average score. Close candidates should also be compared using paired errors and realistic traffic, not only a ranking by rounded accuracy.',
              'Timing divides wall-clock inference time by the number of items with GPU batch size four. It includes prompt processing, decoding and retrieval where used, but excludes model loading. It measures throughput cost rather than single-request latency. Long host pauses occurred during training trials, so recorded training wall time is not a valid model-speed comparison. Inference timings also come from an uncontrolled laptop session and should be treated as rough observations. Fixed ICL repeats demonstrations in every prompt. Retrieval adds an index and lookup work. LoRA needs offline training and adapter version management. Adapter merging and production serving have not been benchmarked.',
              'The study uses one training seed and one fresh test sample, a modest development search and only one epoch per adapter. Exact deduplication misses paraphrases. The news is old, in English and potentially present in pretraining. Model size and family are confounded. The earlier pilot informed the choice to run the full protocol; the fresh holdout limits direct test reuse but does not erase this history. These boundaries should carry into any interpretation of the results.',
              '## 8. Reproducibility',
              'PROTOCOL.json defines the experiments. The selected configuration, test lock, sample hashes, software versions and per-item predictions are included under results/. Scripts reproduce prompt selection, training, scoring and figures. Development cannot proceed after a test lock has been created. Run a new results directory after changing code, because evaluation caching checks sample IDs and settings rather than source hashes.',
              '## References',
              '[1] Zhang et al. (2015). Character-level Convolutional Networks for Text Classification. AG News dataset: https://huggingface.co/datasets/fancyzhx/ag_news',
              '[2] Qwen team. Qwen2.5-0.5B-Instruct model card. https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct',
              '[3] Hugging Face. SmolLM2-360M-Instruct model card. https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct',
              '[4] Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. https://arxiv.org/abs/2106.09685']
    (report_root / 'report.md').write_text('\n\n'.join(lines) + '\n',encoding='utf-8')


if __name__ == '__main__':
    main()
