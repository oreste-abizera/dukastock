"""
Fine-tune XLM-R on the 200-message annotated commerce-domain Kinyarwanda
NER test set (proposal Research Question 2).

Expects a JSONL annotations file where each line is:
    {"text": "Nabagurishije isukari ibiro bitatu",
     "entities": [{"start": 14, "end": 21, "label": "PRODUCT"},
                  {"start": 22, "end": 32, "label": "UNIT"},
                  {"start": 33, "end": 40, "label": "QUANTITY"}]}

Reports precision/recall/F1 per entity type plus Cohen's Kappa
inter-annotator agreement when a second annotator's file is supplied,
exactly as specified in proposal Chapter 1.3.1 and Table 6.

Run on a GPU runtime (Kaggle/Colab) — see ml_experiments/notebooks/ for the
notebook version of this same pipeline with full visualisation of training
curves and the confusion matrix.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

LABEL_LIST = ["O", "B-PRODUCT", "I-PRODUCT", "B-QUANTITY", "I-QUANTITY", "B-UNIT", "I-UNIT"]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


def load_annotations(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def char_spans_to_bio_tags(text: str, entities: list[dict], tokenizer) -> tuple[list[str], list[str]]:
    """Convert character-offset entity spans into BIO tags aligned to the
    tokenizer's wordpiece tokens (a standard preprocessing step for
    transformer-based token classification)."""
    encoding = tokenizer(text, return_offsets_mapping=True, truncation=True)
    offsets = encoding["offset_mapping"]
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])

    tags = ["O"] * len(offsets)
    for ent in entities:
        started = False
        for i, (start, end) in enumerate(offsets):
            if start == end:  # special tokens
                continue
            if start >= ent["start"] and end <= ent["end"]:
                tags[i] = f"{'B' if not started else 'I'}-{ent['label']}"
                started = True
    return tokens, tags


def compute_cohens_kappa(annotator_a_path: str, annotator_b_path: str) -> float:
    """Deprecated in favor of ml_experiments/scripts/compute_kappa.py, which
    has the same core logic plus diagnostics (mismatched message sets,
    mismatched entity counts per message) and doesn't require importing
    this module's heavy transformers/torch dependencies just to check
    inter-annotator agreement. Kept here only as a thin wrapper so any
    existing external call sites don't break.

    Run `python ml_experiments/scripts/compute_kappa.py --annotator-a ...
    --annotator-b ...` directly instead — see docs/ANNOTATION_GUIDE.md."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from compute_kappa import compute_cohens_kappa as _compute
    return _compute(annotator_a_path, annotator_b_path)["cohens_kappa"]


def train(annotations_path: str, output_dir: str, epochs: int = 10):
    from datasets import Dataset
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
    )

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    model = AutoModelForTokenClassification.from_pretrained(
        "xlm-roberta-base", num_labels=len(LABEL_LIST), id2label=ID2LABEL, label2id=LABEL2ID
    )

    records = load_annotations(annotations_path)
    processed = []
    for r in records:
        tokens, tags = char_spans_to_bio_tags(r["text"], r["entities"], tokenizer)
        processed.append({
            "input_ids": tokenizer.convert_tokens_to_ids(tokens),
            "labels": [LABEL2ID[t] for t in tags],
        })

    split_idx = int(0.8 * len(processed))
    train_ds = Dataset.from_list(processed[:split_idx])
    eval_ds = Dataset.from_list(processed[split_idx:])

    collator = DataCollatorForTokenClassification(tokenizer)
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,  # each checkpoint includes optimizer state (~3x
        # the model size); unbounded accumulation over many epochs fills the
        # disk. load_best_model_at_end still keeps the best checkpoint even
        # if it ages out of this window.
        load_best_model_at_end=True,
        # Select the checkpoint with the highest token-level F1, not lowest
        # loss. Minimizing loss can diverge from maximising F1 on imbalanced
        # NER datasets (where 'O' tokens dominate) — F1 is the metric that
        # maps directly to Research Question 2's evaluation criterion.
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        seed=42,
        logging_steps=10,
        fp16=False,  # set to True manually if running on a GPU with tensor cores
    )

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)
        true_labels = [[ID2LABEL[l] for l in label if l != -100] for label in labels]
        true_preds = [
            [ID2LABEL[p] for p, l in zip(pred, label) if l != -100]
            for pred, label in zip(predictions, labels)
        ]
        return {
            "precision": precision_score(true_labels, true_preds),
            "recall": recall_score(true_labels, true_preds),
            "f1": f1_score(true_labels, true_preds),
        }

    trainer = Trainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=collator, compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics = trainer.evaluate()
    print(json.dumps(metrics, indent=2))
    Path(output_dir, "eval_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Model and tokenizer saved. Run on Kaggle GPU with requirements installed.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--annotations", default="ml_experiments/data/annotations.jsonl")
    p.add_argument("--output-dir", default="backend/app/nlp/xlmr_commerce_ner")
    p.add_argument("--epochs", type=int, default=10)
    args = p.parse_args()
    train(args.annotations, args.output_dir, args.epochs)
