"""
Compute Cohen's Kappa inter-annotator agreement between two independently
annotated copies of the same message set.

This is deliberately separate from train_xlmr_ner.py: that script needs the
full transformers/torch stack just to import, but checking annotator
agreement only needs scikit-learn. Run this right after both annotators
finish, BEFORE spending time fine-tuning anything — if Kappa comes back low,
the label scheme needs clarifying first (see docs/ANNOTATION_GUIDE.md,
"Resolving disagreements"), and there's no point training on data that two
humans can't agree on.

Usage:
    python ml_experiments/scripts/compute_kappa.py \\
        --annotator-a ml_experiments/data/annotations_annotator_a.jsonl \\
        --annotator-b ml_experiments/data/annotations_annotator_b.jsonl

Both files must be in DukaStock's schema (output of convert_doccano_export.py):
    {"text": "...", "entities": [{"start": int, "end": int, "label": str}, ...]}

Messages are matched by exact text match, so both annotators must have
annotated the identical message set (same text, same order not required).
"""
import argparse
import json
import math

from sklearn.metrics import cohen_kappa_score


def load_annotations(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_cohens_kappa(annotator_a_path: str, annotator_b_path: str) -> dict:
    a_records = {r["text"]: r for r in load_annotations(annotator_a_path)}
    b_records = {r["text"]: r for r in load_annotations(annotator_b_path)}

    shared_texts = set(a_records.keys()) & set(b_records.keys())
    only_in_a = set(a_records.keys()) - set(b_records.keys())
    only_in_b = set(b_records.keys()) - set(a_records.keys())

    a_labels, b_labels = [], []
    entity_count_mismatches = 0

    for text in shared_texts:
        a_ents = sorted(a_records[text]["entities"], key=lambda e: e["start"])
        b_ents = sorted(b_records[text]["entities"], key=lambda e: e["start"])
        if len(a_ents) != len(b_ents):
            entity_count_mismatches += 1
        n = min(len(a_ents), len(b_ents))
        for i in range(n):
            a_labels.append(a_ents[i]["label"])
            b_labels.append(b_ents[i]["label"])

    kappa = float(cohen_kappa_score(a_labels, b_labels)) if a_labels else float("nan")

    return {
        "cohens_kappa": kappa,
        "shared_messages": len(shared_texts),
        "only_in_annotator_a": len(only_in_a),
        "only_in_annotator_b": len(only_in_b),
        "messages_with_mismatched_entity_count": entity_count_mismatches,
        "compared_entity_pairs": len(a_labels),
    }


def interpret(kappa: float) -> str:
    # Landis & Koch (1977) benchmark scale
    if math.isnan(kappa):
        return "undefined (no overlapping entities to compare)"
    if kappa < 0.0:
        return "worse than chance — stop and revisit the label definitions"
    if kappa < 0.20:
        return "slight agreement — label scheme likely too ambiguous to use"
    if kappa < 0.40:
        return "fair agreement — clarify label definitions before proceeding"
    if kappa < 0.60:
        return "moderate agreement — usable, but expect noisy training data"
    if kappa < 0.80:
        return "substantial agreement — good, proceed"
    return "almost perfect agreement — excellent"


def main(annotator_a_path: str, annotator_b_path: str):
    result = compute_cohens_kappa(annotator_a_path, annotator_b_path)

    print(json.dumps(result, indent=2))
    print()
    print(f"Interpretation: {interpret(result['cohens_kappa'])}")

    if result["only_in_annotator_a"] or result["only_in_annotator_b"]:
        print()
        print(f"WARNING: {result['only_in_annotator_a']} message(s) only in annotator A's file, "
              f"{result['only_in_annotator_b']} only in annotator B's file.")
        print("Both annotators should be labeling the EXACT same message set. Check for:")
        print("  - Typos/whitespace differences introduced during annotation")
        print("  - One annotator skipping messages")

    if result["messages_with_mismatched_entity_count"] > 0:
        print()
        print(f"NOTE: {result['messages_with_mismatched_entity_count']} message(s) where the two "
              "annotators found a different NUMBER of entities (e.g. one annotator missed a "
              "QUANTITY). Kappa here only compares the labels of however many entities each found, "
              "pairwise in span order — it does not penalize missed entities directly. Review "
              "these messages manually; they're usually the most informative disagreements.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotator-a", required=True)
    parser.add_argument("--annotator-b", required=True)
    args = parser.parse_args()
    main(args.annotator_a, args.annotator_b)
