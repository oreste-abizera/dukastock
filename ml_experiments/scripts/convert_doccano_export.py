"""
Convert a Doccano sequence-labeling export into the schema
ml_experiments/scripts/train_xlmr_ner.py expects.

Doccano's export format has changed across versions and export-button
choices, so this script accepts the three shapes seen in practice:

  Shape A (older "JSONL" / format 0):
    {"text": "...", "labels": [[start, end, "LABEL"], ...]}

  Shape B (newer "JSONL(Text-Labels)"):
    {"id": 10, "data": "...", "label": [[start, end, "LABEL"], ...]}

  Shape C ("JSONL(Relation)" / entities export):
    {"text": "...", "entities": [{"id":0,"start_offset":0,"end_offset":6,"label":"LABEL"}, ...]}

DukaStock's training script wants one shape, used everywhere else in this
project (train_xlmr_ner.py, notebook 3, docs/SOURCES.md):

    {"text": "...", "entities": [{"start": 0, "end": 6, "label": "LABEL"}, ...]}

Usage:
    python ml_experiments/scripts/convert_doccano_export.py \\
        --input raw_export.jsonl \\
        --output ml_experiments/data/annotations.jsonl

Run once per annotator. For the inter-annotator agreement step
(see docs/ANNOTATION_GUIDE.md), convert each annotator's export
separately and pass both converted files to compute_kappa.py.
"""
import argparse
import json
import sys


def convert_record(raw: dict) -> dict:
    text = raw.get("text") or raw.get("data")
    if text is None:
        raise ValueError(f"Record has neither 'text' nor 'data' field: {raw}")

    entities = []

    if "entities" in raw:
        # Shape C
        for e in raw["entities"]:
            start = e.get("start_offset", e.get("start"))
            end = e.get("end_offset", e.get("end"))
            label = e["label"]
            entities.append({"start": start, "end": end, "label": label})
    elif "labels" in raw or "label" in raw:
        # Shape A or B: list of [start, end, label] triples
        triples = raw.get("labels") or raw.get("label")
        for triple in triples:
            start, end, label = triple
            entities.append({"start": start, "end": end, "label": label})
    else:
        raise ValueError(f"Record has no recognizable label field: {raw}")

    return {"text": text, "entities": entities}


def main(input_path: str, output_path: str):
    converted = []
    skipped = 0
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            try:
                converted.append(convert_record(raw))
            except (ValueError, KeyError) as exc:
                print(f"  Skipping line {line_num}: {exc}", file=sys.stderr)
                skipped += 1

    with open(output_path, "w", encoding="utf-8") as f:
        for record in converted:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Converted {len(converted)} records -> {output_path}")
    if skipped:
        print(f"Skipped {skipped} malformed record(s) — see warnings above.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Raw Doccano JSONL export file")
    parser.add_argument("--output", required=True, help="Converted output path (DukaStock schema)")
    args = parser.parse_args()
    main(args.input, args.output)
