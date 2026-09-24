"""Train and evaluate a small multinomial Naive Bayes text classifier."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import unicodedata


def tokens(text):
    return re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", text).casefold())


def signature(text):
    return tuple(sorted(Counter(tokens(text)).items()))


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError("non-standard JSON constant")


def load_records(path):
    records, seen = [], set()
    with Path(path).open(encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line, object_pairs_hook=strict_object,
                                 parse_constant=reject_constant)
                if not isinstance(row, dict) or set(row) != {"text", "label"}:
                    raise ValueError("expected exactly text and label")
                if any(not isinstance(row[field], str) or not row[field].strip()
                       for field in ("text", "label")):
                    raise ValueError("text and label must be non-empty strings")
                if any(unicodedata.category(char) == "Cs"
                       for field in ("text", "label") for char in row[field]):
                    raise ValueError("unpaired Unicode surrogate")
                if row["label"] != row["label"].strip():
                    raise ValueError("labels cannot have surrounding whitespace")
                key = signature(row["text"])
                if not key:
                    raise ValueError("text must contain letters or numbers")
                if key in seen:
                    raise ValueError("duplicate bag of words")
                seen.add(key)
                records.append(row)
            except (ValueError, TypeError, RecursionError) as error:
                raise ValueError(f"line {line_number}: invalid labeled text ({type(error).__name__})") from error
    if not records:
        raise ValueError("dataset is empty")
    return records


def fit(records, alpha=1.0):
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or not 1e-9 <= alpha <= 1e6:
        raise ValueError("alpha must be finite and between 1e-9 and 1e6")
    labels = sorted({row["label"] for row in records})
    if len(labels) < 2:
        raise ValueError("training needs at least two classes")
    documents = Counter(row["label"] for row in records)
    counts = {label: Counter() for label in labels}
    for row in records:
        counts[row["label"]].update(tokens(row["text"]))
    vocabulary = sorted({word for count in counts.values() for word in count})
    if not vocabulary:
        raise ValueError("training vocabulary is empty")
    denominators = {label: math.log(sum(counts[label].values()) + alpha * len(vocabulary))
                    for label in labels}
    return {
        "schema_version": 1, "algorithm": "multinomial-naive-bayes", "alpha": alpha,
        "tokenizer": "NFKC-casefold-unicode-alphanumeric-v1", "labels": labels,
        "vocabulary": vocabulary,
        "log_prior": {label: math.log(documents[label] / len(records)) for label in labels},
        "log_likelihood": {
            label: {word: math.log(counts[label][word] + alpha)
                    - denominators[label]
                    for word in vocabulary}
            for label in labels},
    }


def predict(model, text):
    counts = Counter(tokens(text))
    scores = {
        label: model["log_prior"][label] + sum(
            count * model["log_likelihood"][label][word]
            for word, count in counts.items() if word in model["log_likelihood"][label])
        for label in model["labels"]}
    return max(model["labels"], key=scores.get)


def evaluate(model, records):
    if not records:
        raise ValueError("test set is empty")
    labels = model["labels"]
    matrix = {actual: {predicted: 0 for predicted in labels} for actual in labels}
    for row in records:
        if row["label"] not in labels:
            raise ValueError("test contains a class absent from training")
        matrix[row["label"]][predict(model, row["text"])] += 1
    per_class = {}
    for label in labels:
        true_positive = matrix[label][label]
        predicted_count = sum(matrix[actual][label] for actual in labels)
        support = sum(matrix[label].values())
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    vocabulary = set(model["vocabulary"])
    all_tokens = [word for row in records for word in tokens(row["text"])]
    return {
        "test_records": len(records), "labels": labels, "confusion_matrix": matrix,
        "accuracy": sum(matrix[label][label] for label in labels) / len(records),
        "macro_f1": sum(item["f1"] for item in per_class.values()) / len(labels),
        "per_class": per_class,
        "unknown_token_fraction": sum(word not in vocabulary for word in all_tokens) / len(all_tokens)
        if all_tokens else 0.0,
    }


def run(train_path, test_path, output_dir, alpha=1.0):
    training, testing = load_records(train_path), load_records(test_path)
    train_keys = {signature(row["text"]) for row in training}
    if any(signature(row["text"]) in train_keys for row in testing):
        raise ValueError("train/test overlap: matching bags of words")
    model = fit(training, alpha)
    metrics = evaluate(model, testing)
    majority = max(model["labels"], key=model["log_prior"].get)
    metrics["majority_baseline_accuracy"] = sum(row["label"] == majority for row in testing) / len(testing)
    def fingerprint(records):
        data = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(data.encode("utf-8")).hexdigest()
    metrics["training_records"] = len(training)
    metrics["dataset_sha256"] = {"train": fingerprint(training), "test": fingerprint(testing)}
    serialized = {"model.json": json.dumps(model, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False),
                  "metrics.json": json.dumps(metrics, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, content in serialized.items():
        with (output_dir / name).open("x", encoding="utf-8", newline="\n") as target:
            target.write(content + "\n")
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args(argv)
    try:
        report = run(args.train, args.test, args.out, args.alpha)
    except (ValueError, OSError, UnicodeError, OverflowError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
