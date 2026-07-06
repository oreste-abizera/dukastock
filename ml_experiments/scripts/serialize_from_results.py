"""
Serialize deployable model artifacts from an already-completed
ml_benchmark_results.csv, without re-running the full density/CV sweep.

run_experiment.py's expensive part is the walk-forward CV across all 6
density levels x 10 stores x 5 models, needed to decide which model wins
per product. If that's already been done (e.g. via Notebook 2, which
writes the same ml_benchmark_results.csv but never serializes models),
this script reuses the 100%-density row already in that CSV to make the
same selection decision, then fits only the winning model once per
product on the full history -- skipping the re-evaluation entirely.

Usage:
    python ml_experiments/scripts/serialize_from_results.py \
        --results ml_experiments/results/ml_benchmark_results.csv \
        --data ml_experiments/data/fmcg_rwanda_localized.csv \
        --artifact-dir ml_experiments/artifacts
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from run_experiment import _select_and_serialize_best_model  # noqa: E402
from app.ml.pipeline.rwanda_features import FMCG_PRODUCT_MAP, add_rwanda_features, subset_fmcg_products  # noqa: E402


def main(results_path: str, data_path: str, artifact_dir: str, horizon: int):
    results_df = pd.read_csv(results_path)
    at_100 = results_df[results_df["density_pct"] == 100]

    raw = pd.read_csv(data_path, parse_dates=["date"])
    fmcg = add_rwanda_features(subset_fmcg_products(raw))
    stores = sorted(fmcg["store"].unique())

    Path(artifact_dir).mkdir(parents=True, exist_ok=True)

    for _, meta in FMCG_PRODUCT_MAP.items():
        product_code = meta["code"]
        product_rows = at_100[at_100["product"] == product_code]
        if product_rows.empty:
            print(f"  [{product_code}] SKIPPED: no 100%-density rows in {results_path}")
            continue

        models = {}
        for _, row in product_rows.iterrows():
            entry = {"rmse": row["rmse"]}
            if row["model"] != "naive":
                sig = row["fraction_folds_significant"]
                entry["dm_vs_naive"] = {"fraction_folds_significant": float(sig) if pd.notna(sig) else 0.0}
            models[row["model"]] = entry

        rep_df = (
            fmcg[(fmcg["product_code"] == product_code) & (fmcg["store"] == stores[0])]
            .sort_values("date").reset_index(drop=True)
        )
        winning_kind = _select_and_serialize_best_model(
            product_code, rep_df, {"100": {"models": models}}, artifact_dir, horizon,
        )
        print(f"  [{product_code}] serialized as '{winning_kind}' (reused precomputed results, no re-run)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="ml_experiments/results/ml_benchmark_results.csv")
    parser.add_argument("--data", default="ml_experiments/data/fmcg_rwanda_localized.csv")
    parser.add_argument("--artifact-dir", default="ml_experiments/artifacts")
    parser.add_argument("--horizon", type=int, default=7)
    args = parser.parse_args()
    main(args.results, args.data, args.artifact_dir, args.horizon)
