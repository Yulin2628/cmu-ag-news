"""Offline checks of scoring, demonstration selection and saved data isolation."""
import unittest
from experiment import LABELS, ROOT, demos, metrics, read


class EvaluationChecks(unittest.TestCase):
    def test_invalid_outputs_count_as_errors_and_false_negatives(self):
        rows = [dict(label=i, prediction=i) for i in range(4)]
        rows[0]['prediction'] = -1
        result = metrics(rows)
        self.assertEqual(result['correct'], 3)
        self.assertEqual(result['accuracy'], 0.75)
        self.assertEqual(result['invalid_rate'], 0.25)
        self.assertEqual(result['confusion_matrix_4x5'][0][4], 1)
        self.assertEqual(result['per_class']['World']['recall'], 0)
        self.assertEqual(result['macro_f1'], 0.75)
        self.assertLess(result['wilson_95'][0], 0.75)
        self.assertGreater(result['wilson_95'][1], 0.75)

    def test_balanced_deterministic_demonstrations(self):
        train = [dict(id=f'{c}-{i}', text=str(i), label=c) for c in range(4) for i in range(3)]
        chosen = demos(train, 4)
        self.assertEqual(sorted(r['label'] for r in chosen), list(range(4)))
        self.assertEqual(chosen, demos(train, 4))

    def test_saved_data_is_disjoint_and_balanced(self):
        directory = ROOT / 'results/data'
        if not (directory / 'manifest.json').exists():
            self.skipTest('Run prepare first')
        seen = set()
        for split, count in [('train', 2000), ('dev', 400), ('test', 400)]:
            rows = read(directory / f'{split}.json')
            self.assertEqual(len(rows), count)
            self.assertEqual([sum(r['label'] == c for r in rows) for c in range(4)], [count//4]*4)
            hashes = {r['sha256'] for r in rows}
            self.assertEqual(len(hashes), count)
            self.assertFalse(hashes & seen)
            seen |= hashes


if __name__ == '__main__':
    unittest.main()
