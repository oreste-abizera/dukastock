"""
Full density-level benchmark runner — store-level (Duka-proxy) evaluation.

Trains the naive baseline, SARIMA, Prophet, XGBoost, and N-BEATS on each of
the five Rwanda-mapped FMCG products at all six cold-start density levels.
Evaluation is done at the individual Kaggle 'store' level (each store proxies
one Duka) — NOT as a national aggregate. Metrics are averaged across all 10
stores before the best model is selected for artifact serialization.

Scores each against walk-forward folds, runs the Diebold-Mariano test (with
Newey-West HAC variance for h=7) against the naive baseline, and writes a
JSON results file plus serialized joblib artifacts for ForecastService.

Usage:
    python ml_experiments/scripts/run_experiment.py \
        --data ml_experiments/data/fmcg_rwanda_localized.csv \
        --output-dir ml_experiments/results \
        --artifact-dir ml_experiments/artifacts

The notebooks (ml_experiments/notebooks/) mirror this evaluation logic
interactively with the same underlying app.ml modules.
"""
import argparse
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Fix random seeds for reproducibility
random.seed(42)
np.random.seed(42)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("run_experiment")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from app.ml.evaluation.metrics import compute_all_metrics, diebold_mariano_test  # noqa: E402
from app.ml.models.naive import NaiveBaseline  # noqa: E402
from app.ml.models.serializable import SerializableForecastModel  # noqa: E402
from app.ml.models.xgboost_model import (  # noqa: E402
    XGBoostDemandModel, add_lag_features, build_future_feature_template,
)
from app.ml.pipeline.cold_start import DENSITY_LEVELS, temporal_density_slice, walk_forward_folds  # noqa: E402
from app.ml.pipeline.rwanda_features import FMCG_PRODUCT_MAP, add_rwanda_features, subset_fmcg_products  # noqa: E402

try:
    from app.ml.models.sarima import SARIMAModel
    _SARIMA_AVAILABLE = True
except ImportError:
    _SARIMA_AVAILABLE = False

try:
    from app.ml.models.prophet_model import ProphetModel
    _PROPHET_AVAILABLE = True
except ImportError:
    _PROPHET_AVAILABLE = False

try:
    from app.ml.models.nbeats_model import NBEATSModel
    _NBEATS_AVAILABLE = True
except ImportError:
    _NBEATS_AVAILABLE = False


def run_for_product_and_density(
    product_df: pd.DataFrame, density_pct: int, horizon: int = 7, context: str = ""
) -> dict:
    """Run all five model classes for one product at one density level,
    aggregating metrics across all walk-forward folds.

    `context` (e.g. "SUGAR/store 3/density 30%") is only used for log
    messages when a model fit/predict fails and falls back to naive —
    without it, a failure is silently invisible in the output and the
    fallback is indistinguishable from naive genuinely winning."""
    sliced = temporal_density_slice(product_df, density_pct)
    folds = walk_forward_folds(sliced.train, horizon=horizon)

    if not folds:
        return {"density_pct": density_pct, "n_observations": sliced.n_observations, "folds_evaluated": 0, "models": {}}

    model_fold_metrics = {name: [] for name in ["naive", "sarima", "prophet", "xgboost", "nbeats"]}
    # Pooled across all folds, then ONE Diebold-Mariano test per model — not
    # one test per fold. A single 7-day fold gives the Newey-West HAC
    # variance estimator only n=7 points to fit h-1=6 lag terms, which
    # collapses to a degenerate (<=0) variance estimate most of the time,
    # forcing a "not significant" result regardless of true model quality
    # (verified: a model 3x more accurate than naive on every fold still
    # failed to reach significance ~60% of the time under the per-fold
    # design). Pooling the fold residuals first (n ~= 6*7 = 42) keeps the
    # same h=7 lag structure — still correct for 7-step-ahead forecasts —
    # but gives the estimator enough data to actually work.
    pooled_y_test = []
    pooled_naive_preds = []
    pooled_model_preds = {name: [] for name in ["sarima", "prophet", "xgboost", "nbeats"]}

    for fold_idx, (train, test) in enumerate(folds):
        y_train = train["sales"]
        y_test = test["sales"].values
        pooled_y_test.append(y_test)

        naive = NaiveBaseline().fit(y_train)
        naive_preds = naive.predict(len(test))
        model_fold_metrics["naive"].append(compute_all_metrics(y_test, naive_preds))
        pooled_naive_preds.append(naive_preds)

        try:
            sarima_preds = SARIMAModel().fit(y_train).predict(len(test)) if _SARIMA_AVAILABLE else naive_preds
        except Exception as exc:
            logger.warning("[%s] fold %d: sarima failed (%s), falling back to naive", context, fold_idx, exc)
            sarima_preds = naive_preds
        model_fold_metrics["sarima"].append(compute_all_metrics(y_test, sarima_preds))
        pooled_model_preds["sarima"].append(sarima_preds)

        try:
            if _PROPHET_AVAILABLE:
                prophet_preds, _, _ = ProphetModel().fit(train["date"], y_train).predict(len(test), train["date"].max())
            else:
                prophet_preds = naive_preds
        except Exception as exc:
            logger.warning("[%s] fold %d: prophet failed (%s), falling back to naive", context, fold_idx, exc)
            prophet_preds = naive_preds
        model_fold_metrics["prophet"].append(compute_all_metrics(y_test, prophet_preds))
        pooled_model_preds["prophet"].append(prophet_preds)

        try:
            xgb_model = XGBoostDemandModel().fit(train)
            future_features = add_lag_features(pd.concat([train, test]).reset_index(drop=True)).iloc[-len(test):]
            xgb_preds = xgb_model.predict(future_features)
        except Exception as exc:
            logger.warning("[%s] fold %d: xgboost failed (%s), falling back to naive", context, fold_idx, exc)
            xgb_preds = naive_preds
        model_fold_metrics["xgboost"].append(compute_all_metrics(y_test, xgb_preds))
        pooled_model_preds["xgboost"].append(xgb_preds)

        if _NBEATS_AVAILABLE:
            try:
                nbeats_preds = NBEATSModel(horizon=len(test)).fit(train["date"], y_train).predict()
            except Exception as exc:
                logger.warning("[%s] fold %d: nbeats failed (%s), falling back to naive", context, fold_idx, exc)
                nbeats_preds = naive_preds
        else:
            nbeats_preds = naive_preds
        model_fold_metrics["nbeats"].append(compute_all_metrics(y_test, nbeats_preds))
        pooled_model_preds["nbeats"].append(nbeats_preds)

    pooled_y_test_arr = np.concatenate(pooled_y_test)
    pooled_naive_arr = np.concatenate(pooled_naive_preds)
    dm_results = {
        name: diebold_mariano_test(pooled_y_test_arr, np.concatenate(preds), pooled_naive_arr, h=horizon)
        for name, preds in pooled_model_preds.items()
    }

    def _aggregate(metric_dicts: list[dict]) -> dict:
        keys = metric_dicts[0].keys()
        return {k: float(np.mean([m[k] for m in metric_dicts])) for k in keys}

    def _dm_to_dict(dm) -> dict:
        # Field names kept as "fraction_folds_significant" / "mean_p_value"
        # for backward compatibility with existing CSV columns and
        # notebook plotting cells, even though each is now a single pooled
        # result (0.0/1.0 and one p-value) rather than an average over
        # several per-fold tests.
        return {"fraction_folds_significant": float(dm.significant_at_05), "mean_p_value": float(dm.p_value)}

    return {
        "density_pct": density_pct,
        "n_observations": sliced.n_observations,
        "folds_evaluated": len(folds),
        "models": {
            "naive": _aggregate(model_fold_metrics["naive"]),
            "sarima": {**_aggregate(model_fold_metrics["sarima"]), "dm_vs_naive": _dm_to_dict(dm_results["sarima"])},
            "prophet": {**_aggregate(model_fold_metrics["prophet"]), "dm_vs_naive": _dm_to_dict(dm_results["prophet"])},
            "xgboost": {**_aggregate(model_fold_metrics["xgboost"]), "dm_vs_naive": _dm_to_dict(dm_results["xgboost"])},
            "nbeats": {**_aggregate(model_fold_metrics["nbeats"]), "dm_vs_naive": _dm_to_dict(dm_results["nbeats"])},
        },
    }


def _select_and_serialize_best_model(
    product_code: str,
    product_df: pd.DataFrame,
    product_results: dict,
    artifact_dir: str,
    horizon: int,
) -> str:
    """
    Pick the model with the lowest mean RMSE among those that are also
    Diebold-Mariano significant against naive on >=50% of walk-forward
    folds at 100% density (i.e. trained on everything available — what
    would actually be deployed once a real Duka has built up history), and
    serialize it wrapped in SerializableForecastModel, so ForecastService
    can call .predict_with_band() uniformly regardless of which of the
    five model kinds won. If no model clears that significance bar, naive
    is served — shipping a model with a lower RMSE but no proven
    statistical advantage over naive would misrepresent the primary
    research finding.

    Returns the winning model's kind, for logging.
    """
    full_density_metrics = product_results.get("100", {}).get("models", {})
    if not full_density_metrics:
        # No folds were evaluated even at 100% density (e.g. too few
        # observations in this dataset) — fall back to naive, which always
        # works with as little as a single observation.
        model = SerializableForecastModel(kind="naive", model=NaiveBaseline().fit(product_df["sales"]))
        joblib.dump(model, Path(artifact_dir) / f"{product_code.lower()}_best_model.joblib")
        return "naive (fallback: no folds evaluated at 100% density)"

    candidates = {name: metrics["rmse"] for name, metrics in full_density_metrics.items()}
    significant_candidates = {
        name: rmse
        for name, rmse in candidates.items()
        if name != "naive"
        and full_density_metrics.get(name, {}).get("dm_vs_naive", {}).get("fraction_folds_significant", 0.0) >= 0.5
    }
    if significant_candidates:
        best_kind = min(significant_candidates, key=significant_candidates.get)
    else:
        best_kind = "naive"

    y_full = product_df["sales"]
    dates_full = product_df["date"]

    if best_kind == "naive":
        wrapped = SerializableForecastModel(kind="naive", model=NaiveBaseline().fit(y_full))

    elif best_kind == "sarima" and _SARIMA_AVAILABLE:
        wrapped = SerializableForecastModel(kind="sarima", model=SARIMAModel().fit(y_full))

    elif best_kind == "prophet" and _PROPHET_AVAILABLE:
        fitted = ProphetModel().fit(dates_full, y_full)
        wrapped = SerializableForecastModel(kind="prophet", model=fitted, last_observed_date=dates_full.max())

    elif best_kind == "xgboost":
        fitted = XGBoostDemandModel().fit(product_df)
        future_with_lags = build_future_feature_template(product_df, horizon)
        wrapped = SerializableForecastModel(kind="xgboost", model=fitted, future_feature_template=future_with_lags)

    elif best_kind == "nbeats" and _NBEATS_AVAILABLE:
        fitted = NBEATSModel(horizon=horizon).fit(dates_full, y_full)
        wrapped = SerializableForecastModel(kind="nbeats", model=fitted)

    else:
        # Winning model's library wasn't available in THIS environment even
        # though it won during evaluation (e.g. evaluated on a machine with
        # neuralforecast installed, serialized on one without it). Fall back
        # to naive rather than crash the whole run.
        print(f"  [{product_code}] WARNING: best model '{best_kind}' unavailable in this "
              f"environment for serialization; falling back to naive.")
        wrapped = SerializableForecastModel(kind="naive", model=NaiveBaseline().fit(y_full))
        best_kind = "naive (fallback: winning library unavailable)"

    joblib.dump(wrapped, Path(artifact_dir) / f"{product_code.lower()}_best_model.joblib")
    return best_kind


def main(data_path: str, output_dir: str, artifact_dir: str, horizon: int):
    raw = pd.read_csv(data_path, parse_dates=["date"])
    fmcg = subset_fmcg_products(raw)
    fmcg = add_rwanda_features(fmcg)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(artifact_dir).mkdir(parents=True, exist_ok=True)

    # Evaluate at the individual-store (Duka-proxy) level.
    # Each (store, product) pair is one series — the unit of analysis matching
    # the deployment context (DukaStock serves individual shopkeepers).
    # Metrics are averaged across stores before artifact selection.
    all_results: dict[str, dict] = {}
    stores = sorted(fmcg["store"].unique())

    for item_id, meta in FMCG_PRODUCT_MAP.items():
        product_code = meta["code"]
        product_store_results: dict[int, dict] = {}

        for store in stores:
            store_df = (
                fmcg[(fmcg["product_code"] == product_code) & (fmcg["store"] == store)]
                .sort_values("date")
                .reset_index(drop=True)
            )
            store_results: dict[str, dict] = {}
            for density in DENSITY_LEVELS:
                context = f"{product_code}/store {store}/density {density}%"
                print(f"[{context}] ...")
                store_results[str(density)] = run_for_product_and_density(store_df, density, horizon, context=context)
            product_store_results[store] = store_results

        all_results[product_code] = product_store_results

        # For artifact serialization, use the full-history series for store 1
        # as a representative series. The best model type is decided by RMSE
        # AND Diebold-Mariano significance, both averaged across all stores
        # at 100% density.
        store_rmse: dict[str, list[float]] = {m: [] for m in ["naive","sarima","prophet","xgboost","nbeats"]}
        store_sig: dict[str, list[float]] = {m: [] for m in ["sarima","prophet","xgboost","nbeats"]}
        for store_res in product_store_results.values():
            for m_name, m_metrics in store_res.get("100", {}).get("models", {}).items():
                if "rmse" in m_metrics:
                    store_rmse[m_name].append(m_metrics["rmse"])
                dm = m_metrics.get("dm_vs_naive")
                if dm is not None:
                    store_sig[m_name].append(dm["fraction_folds_significant"])
        mean_rmse = {m: float(np.mean(v)) for m, v in store_rmse.items() if v}
        mean_sig = {m: float(np.mean(v)) for m, v in store_sig.items() if v}

        # Serialize using store 1's full history as the representative series
        rep_df = (
            fmcg[(fmcg["product_code"] == product_code) & (fmcg["store"] == stores[0])]
            .sort_values("date").reset_index(drop=True)
        )
        fake_100pct_models = {
            m: {"rmse": v, "dm_vs_naive": {"fraction_folds_significant": mean_sig.get(m, 0.0)}} if m != "naive"
            else {"rmse": v}
            for m, v in mean_rmse.items()
        }
        winning_kind = _select_and_serialize_best_model(
            product_code, rep_df,
            {"100": {"models": fake_100pct_models}},
            artifact_dir, horizon,
        )
        print(f"  [{product_code}] serialized as '{winning_kind}' "
              f"(mean RMSE at 100% density: {mean_rmse}, mean fraction significant: {mean_sig})")

    ts = datetime.utcnow().strftime('%Y%m%dT%H%M%S')
    output_path = Path(output_dir) / f"experiment_results_{ts}.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Also write a flat CSV matching the notebook's ml_benchmark_results.csv format
    rows = []
    for product_code, store_dict in all_results.items():
        for store, density_dict in store_dict.items():
            for density_str, density_data in density_dict.items():
                for model_name, metrics in density_data.get("models", {}).items():
                    row = {"store": store, "product": product_code,
                           "density_pct": int(density_str),
                           "model": model_name,
                           "n_observations": density_data.get("n_observations"),
                           "folds_evaluated": density_data.get("folds_evaluated"),
                           **{k: v for k, v in metrics.items() if k != "dm_vs_naive"}}
                    if "dm_vs_naive" in metrics:
                        row.update(metrics["dm_vs_naive"])
                    rows.append(row)
    raw_df = pd.DataFrame(rows)
    raw_path = Path(output_dir) / "ml_benchmark_results_raw.csv"
    raw_df.to_csv(raw_path, index=False)

    agg_cols = {c: "mean" for c in ["rmse","mae","mape","smape","fraction_folds_significant","mean_p_value"]}
    agg_cols.update({"n_observations": "first", "folds_evaluated": "first"})
    agg_df = raw_df.groupby(["product","density_pct","model"]).agg(agg_cols).reset_index()
    agg_path = Path(output_dir) / "ml_benchmark_results.csv"
    agg_df.to_csv(agg_path, index=False)

    # Threshold density: the direct, reproducible answer to the primary
    # research question — lowest density at which each (product, model)
    # pair reaches DM significance on >=50% of walk-forward folds,
    # store-averaged. Mirrors notebook 02's first_threshold() so both the
    # interactive and headless pipelines produce the same result.
    threshold_rows = []
    dm_agg = agg_df.dropna(subset=["fraction_folds_significant"])
    for (product, model), group in dm_agg.groupby(["product", "model"]):
        passing = group[group["fraction_folds_significant"] >= 0.5].sort_values("density_pct")
        threshold_rows.append({
            "product": product,
            "model": model,
            "threshold_density_pct": int(passing.iloc[0]["density_pct"]) if len(passing) else None,
        })
    threshold_df = pd.DataFrame(threshold_rows)
    threshold_path = Path(output_dir) / "threshold_density.csv"
    threshold_df.to_csv(threshold_path, index=False)

    print(f"\nJSON written to: {output_path}")
    print(f"Raw CSV: {raw_path} ({len(raw_df)} rows)")
    print(f"Aggregated CSV: {agg_path} ({len(agg_df)} rows)")
    print(f"Threshold density CSV: {threshold_path}")
    print(threshold_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="ml_experiments/data/fmcg_rwanda_localized.csv")
    parser.add_argument("--output-dir", default="ml_experiments/results")
    parser.add_argument("--artifact-dir", default="ml_experiments/artifacts")
    parser.add_argument("--horizon", type=int, default=7)
    args = parser.parse_args()
    main(args.data, args.output_dir, args.artifact_dir, args.horizon)
