"""Check final prediction records, metrics and split identities without running models."""

import hashlib
import json
import math
from pathlib import Path
from experiment import ROOT, LABELS, MODELS, MODEL_REVISIONS, metrics, read


def main():
    run = ROOT / 'results'
    manifest = read(run / 'data/manifest.json')
    seen = set()
    data = {}
    for split, count in [('train', 2000), ('dev', 400), ('test', 400)]:
        rows = read(run / f'data/{split}.json')
        assert len(rows) == count, split
        assert [sum(r['label'] == label for r in rows) for label in range(4)] == [count//4]*4
        hashes = {hashlib.sha256(r['text'].strip().encode()).hexdigest() for r in rows}
        assert len(hashes) == count and not hashes & seen
        assert all(r['sha256'] == hashlib.sha256(r['text'].strip().encode()).hexdigest() for r in rows)
        assert manifest[split]['sha256'] == hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
        seen |= hashes
        data[split] = rows
    selected = read(run / 'selected.json')
    signature = hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest()
    assert read(run / 'test_lock.json')['selection_sha256'] == signature
    for key in MODELS:
        assert read(run / f'models/{key}.json')['revision'] == MODEL_REVISIONS[key]
        assert selected[key]['icl']['shots'] > 0
        assert '\\' not in selected[key]['lora']['adapter']
        for method in ['icl', 'lora', 'extra_fixed', 'extra_retrieval']:
            result = read(run / f'test/{key}_{method}.json')
            assert result['config'] == selected[key][method]
            predictions = result['predictions']
            if 'reused_from' in result:
                source = read(run / 'test' / result['reused_from'])
                assert result['config'] == source['config'] and predictions == source['predictions']
            assert [(r['id'], r['sha256'], r['label']) for r in predictions] == [
                (r['id'], r['sha256'], r['label']) for r in data['test']]
            assert [r['text'] for r in predictions] == [r['text'] for r in data['test']]
            for row in predictions:
                parsed = next((i for i, name in enumerate(LABELS)
                               if row['raw_output'].strip().casefold() == name.casefold()), -1)
                assert parsed == row['prediction']
            recalculated = metrics(predictions)
            for metric in ['accuracy', 'macro_f1', 'invalid_rate']:
                assert math.isclose(result['metrics'][metric], recalculated[metric], abs_tol=1e-12)
            assert result['metrics']['confusion_matrix_4x5'] == recalculated['confusion_matrix_4x5']
        for lr in [1e-4, 2e-4]:
            path = run / f'training_logs/{key}_{lr}.json'
            info = read(path if path.exists() else run / f'adapters/{key}_{lr}/complete.json')
            assert info['lr'] == lr and info['epochs'] == 1
            assert any(row.get('step') == 250 and 'train_loss' in row for row in info['log_history'])
            norms = list(info['lora_B_norms'].values())
            assert norms and all(math.isfinite(value) for value in norms) and any(value > 0 for value in norms)
        training = read(run / f'train/{key}_lora.json')
        assert training['config'] == selected[key]['lora']
        assert [(r['id'], r['sha256']) for r in training['predictions']] == [
            (r['id'], r['sha256']) for r in data['train']]
        assert [r['text'] for r in training['predictions']] == [r['text'] for r in data['train']]
        assert math.isclose(training['metrics']['accuracy'], metrics(training['predictions'])['accuracy'], abs_tol=1e-12)
    print('Verified disjoint balanced splits, frozen selection, eight paired test records and four trained adapters.')


if __name__ == '__main__':
    main()
