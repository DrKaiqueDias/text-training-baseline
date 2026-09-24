import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

from train import evaluate, fit, load_records, main, predict, run, tokens


TRAINING = [{"text": "red red blue", "label": "color"},
            {"text": "cat dog", "label": "animal"}]
TESTING = [{"text": "red blue blue", "label": "color"},
           {"text": "cat cat", "label": "animal"}]


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.train_path, self.test_path = self.root / "train.jsonl", self.root / "test.jsonl"
        self.write(self.train_path, TRAINING)
        self.write(self.test_path, TESTING)

    def write(self, path, records):
        path.write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")

    def test_laplace_probabilities_match_hand_calculation(self):
        model = fit(TRAINING)
        self.assertAlmostEqual(math.exp(model["log_likelihood"]["color"]["red"]), 3 / 7)
        self.assertAlmostEqual(math.exp(model["log_likelihood"]["animal"]["red"]), 1 / 6)
        self.assertAlmostEqual(math.exp(model["log_prior"]["color"]), 0.5)
        for label in model["labels"]:
            self.assertAlmostEqual(sum(math.exp(value) for value in model["log_likelihood"][label].values()), 1)

    def test_repeated_words_and_log_scores(self):
        model = fit(TRAINING)
        self.assertEqual(predict(model, "red " * 10000), "color")
        self.assertEqual(predict(model, "dog dog cat"), "animal")

    def test_tokenization_and_unknown_word_fallback(self):
        self.assertEqual(tokens("ＣＡＦÉ café! 42"), ["café", "café", "42"])
        self.assertEqual(predict(fit(TRAINING), "unseen"), "animal")
        self.assertEqual(predict(fit(TRAINING), ""), "animal")

    def test_test_vocabulary_cannot_change_training(self):
        model = fit(TRAINING)
        before = json.dumps(model, sort_keys=True)
        report = evaluate(model, [{"text": "unknownword", "label": "color"}])
        self.assertEqual(before, json.dumps(model, sort_keys=True))
        self.assertNotIn("unknownword", model["vocabulary"])
        self.assertEqual(report["unknown_token_fraction"], 1)

    def test_metrics_with_errors_and_missing_predictions(self):
        rows = [{"text": "red", "label": "color"},
                {"text": "red", "label": "animal"}]
        report = evaluate(fit(TRAINING), rows)
        self.assertEqual(report["accuracy"], 0.5)
        self.assertAlmostEqual(report["macro_f1"], 1 / 3)
        self.assertEqual(report["confusion_matrix"]["animal"]["color"], 1)
        self.assertEqual(report["per_class"]["animal"]["precision"], 0)

    def test_absent_test_class_has_zero_support(self):
        report = evaluate(fit(TRAINING), TESTING[:1])
        self.assertEqual(report["per_class"]["animal"]["support"], 0)
        self.assertEqual(report["macro_f1"], 0.5)

    def test_alpha_and_training_classes(self):
        for alpha in (0, -1, float("nan"), float("inf"), True, 1e-300, 1e300):
            with self.assertRaises(ValueError):
                fit(TRAINING, alpha)
        with self.assertRaises(ValueError):
            fit(TRAINING[:1])
        with self.assertRaises(ValueError):
            evaluate(fit(TRAINING), [{"text": "red", "label": "new"}])
        with self.assertRaises(ValueError):
            evaluate(fit(TRAINING), [])

    def test_duplicate_bags_and_conflicting_labels(self):
        for label in ("color", "animal"):
            self.write(self.train_path, [TRAINING[0], {"text": "BLUE red RED!", "label": label}])
            with self.assertRaises(ValueError):
                load_records(self.train_path)

    def test_leakage_rejected_before_output(self):
        self.write(self.test_path, [{"text": "BLUE RED red", "label": "color"}])
        with self.assertRaisesRegex(ValueError, "overlap"):
            run(self.train_path, self.test_path, self.root / "bad")
        self.assertFalse((self.root / "bad").exists())

    def test_invalid_records_and_json(self):
        for row in ({}, [], {"text": "!!!", "label": "x"},
                    {"text": True, "label": "x"}, {"text": "ok", "label": " x"},
                    {"text": "x", "label": "y", "extra": 1},
                    {"text": "x", "label": "\ud800"}):
            self.write(self.train_path, [row])
            with self.assertRaises(ValueError):
                load_records(self.train_path)
        for text in ('{"text":"x","text":"y","label":"z"}', '{bad', '',
                     '{"text":NaN,"label":"z"}'):
            self.train_path.write_text(text, encoding="utf-8")
            with self.assertRaises(ValueError):
                load_records(self.train_path)

    def test_bom_and_blank_lines(self):
        self.train_path.write_text("\ufeff\n" + json.dumps(TRAINING[0]) + "\n\n", encoding="utf-8")
        self.assertEqual(load_records(self.train_path), TRAINING[:1])

    def test_reproducible_saved_model_and_predictions(self):
        first = run(self.train_path, self.test_path, self.root / "first")
        second = run(self.train_path, self.test_path, self.root / "second")
        self.assertEqual(first, second)
        self.assertEqual(first["accuracy"], 1)
        self.assertEqual(first["majority_baseline_accuracy"], 0.5)
        for name in ("metrics.json", "model.json"):
            self.assertEqual((self.root / "first" / name).read_bytes(), (self.root / "second" / name).read_bytes())
        model = json.loads((self.root / "first" / "model.json").read_text(encoding="utf-8"))
        self.assertEqual(predict(model, "cat"), "animal")

    def test_does_not_overwrite_or_modify_inputs(self):
        original = self.train_path.read_bytes()
        output = self.root / "existing"
        output.mkdir()
        with self.assertRaises(FileExistsError):
            run(self.train_path, self.test_path, output)
        self.assertEqual(self.train_path.read_bytes(), original)
        self.assertEqual(list(output.iterdir()), [])

    def test_cli_success_and_error(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            code = main(["--train", str(self.train_path), "--test", str(self.test_path),
                         "--out", str(self.root / "run")])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.getvalue())["test_records"], 2)
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["--train", "missing", "--test", str(self.test_path),
                                   "--out", str(self.root / "bad")]), 2)


if __name__ == "__main__":
    unittest.main()
