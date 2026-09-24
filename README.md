# Text Training Baseline

[![Python checks](https://github.com/DrKaiqueDias/text-training-baseline/actions/workflows/tests.yml/badge.svg)](https://github.com/DrKaiqueDias/text-training-baseline/actions)

A small text classifier with the whole training process visible: token counts, learned probabilities and evaluation on a separate test set.

My work in AI quality makes me interested in what sits behind a score. This project gives me a simple baseline to inspect before reaching for a larger model. It trains **multinomial Naive Bayes** from scratch in Python, using only the standard library.

## Train and evaluate

Requires Python 3.11+.

```sh
python train.py --train examples/train.jsonl --test examples/test.jsonl --out runs/demo
python -m unittest discover -v
```

The command writes `model.json` and `metrics.json` into a new folder. Existing output folders are never overwritten. Choose a new name to run it again.

Each input line contains exactly `text` and `label`:

```json
{"text": "The answer includes a claim without evidence.", "label": "factuality"}
```

The synthetic demo has 18 training examples and 6 test examples across three labels: `factuality`, `format` and `privacy`. These labels describe review notes, not a model's actual safety.

## What is learned

Text is normalized with Unicode NFKC and case folding, then split into sequences of letters and numbers. Word counts and the vocabulary come only from training data.

Class priors use document frequencies. Word likelihoods use additive smoothing (`--alpha 1.0` by default). Prediction adds log probabilities to avoid underflow. Unknown words are ignored; an all-unknown input falls back to the class prior. Ties use alphabetical label order.

The algorithm follows the [Stanford Introduction to Information Retrieval explanation of multinomial Naive Bayes](https://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html).

## Reading the results

The metrics include accuracy, macro F1, precision and recall per class, a confusion matrix, unknown-token fraction and a majority-class baseline. In the confusion matrix, rows are actual labels and columns are predicted labels. Macro F1 includes every training class; a class with no test examples gets F1 zero and support zero.

Dataset fingerprints record the parsed records in input order. The same inputs and settings reproduce the model and report. The saved model uses JSON, so it can be inspected without loading a pickle.

To use a model created by this script:

```python
import json
from pathlib import Path
from train import predict

model = json.loads(Path("runs/demo/model.json").read_text(encoding="utf-8"))
print(predict(model, "This response does not follow the requested format."))
```

## Checks and limits

The command rejects malformed rows, duplicate JSON keys, duplicate bags of words within either file, and matching bags of words across train and test. It also rejects test labels absent from training. Exit code **0** means the run completed; **2** means an input, configuration or file error.

These overlap checks do not catch paraphrases or related source documents. For real experiments, separate data by source and use a validation set for model choices before evaluating once on a held-out test set. Do not tune repeatedly on the demo test file.

Word order is discarded. This is a classical machine-learning baseline, not an LLM or a safety classifier for production use. A high score on six synthetic examples says very little about real performance. No employer data is included.

Training keeps the dataset and class/vocabulary tables in memory. An interrupted write can leave a partial output folder. Saved vocabulary can retain sensitive words from your input; review artifacts before publishing them.

## Changes

Issues and pull requests are welcome. A useful report includes a small synthetic example and the expected result. Keep tests with changes to the calculations. Code and bundled examples use the [MIT license](LICENSE).
