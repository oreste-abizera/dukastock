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

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from app.ml.evaluation.metrics import compute_all_metrics, diebold_mariano_test  # noqa: E402
from app.ml.models.naive import NaiveBaseline  # noqa: E402
from app.ml.models.serializable import SerializableForecastModel  # noqa: E402
from app.ml.models.xgboost_model import XGBoostDemandModel, add_lag_features  # noqa: E402
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


def run_for_product_and_density(product_df: pd.DataFrame, density_pct: int, horizon: int = 7) -> dict:
    """Run all five model classes for one product at one density level,
    aggregating metrics across all walk-forward folds."""
    sliced = temporal_density_slice(product_df, density_pct)
    folds = walk_forward_folds(sliced.train, horizon=horizon)

    if not folds:
        return {"density_pct": density_pct, "n_observations": sliced.n_observations, "folds_evaluated": 0, "models": {}}

    model_fold_metrics = {name: [] for name in ["naive", "sarima", "prophet", "xgboost", "nbeats"]}
    dm_results = {name: [] for name in ["sarima", "prophet", "xgboost", "nbeats"]}

    for train, test in folds:
        y_train = train["sales"]
        y_test = test["sales"].values

        naive = NaiveBaseline().fit(y_train)
        naive_preds = naive.predict(len(test))
        model_fold_metrics["naive"].append(compute_all_metrics(y_test, naive_preds))

        try:
            sarima_preds = SARIMAModel().fit(y_train).predict(len(test)) if _SARIMA_AVAILABLE else naive_preds
        except Exception:
            sarima_preds = naive_preds
        model_fold_metrics["sarima"].append(compute_all_metrics(y_test, sarima_preds))
        dm_results["sarima"].append(diebold_mariano_test(y_test, sarima_preds, naive_preds, h=horizon))

        try:
            if _PROPHET_AVAILABLE:
                prophet_preds, _, _ = ProphetModel().fit(train["date"], y_train).predict(len(test), train["date"].max())
            else:
                prophet_preds = naive_preds
        except Exception:
            prophet_preds = naive_preds
        model_fold_metrics["prophet"].append(compute_all_metrics(y_test, prophet_preds))
        dm_results["prophet"].append(diebold_mariano_test(y_test, prophet_preds, naive_preds, h=horizon))

        try:
            xgb_model = XGBoostDemandModel().fit(train)
            future_features = add_lag_features(pd.concat([train, test]).reset_index(drop=True)).iloc[-len(test):]
            xgb_preds = xgb_model.predict(future_features)
        except Exception:
            xgb_preds = naive_preds
        model_fold_metrics["xgboost"].append(compute_all_metrics(y_test, xgb_preds))
        dm_results["xgboost"].append(diebold_mariano_test(y_test, xgb_preds, naive_preds, h=horizon))

        if _NBEATS_AVAILABLE:
            try:
                nbeats_preds = NBEATSModel(horizon=len(test)).fit(train["date"], y_train).predict()
            except Exception:
                nbeats_preds = naive_preds
        else:
            nbeats_preds = naive_preds
        model_fold_metrics["nbeats"].append(compute_all_metrics(y_test, nbeats_preds))
        dm_results["nbeats"].append(diebold_mariano_test(y_test, nbeats_preds, naive_preds, h=horizon))

    def _aggregate(metric_dicts: list[dict]) -> dict:
        keys = metric_dicts[0].keys()
        return {k: float(np.mean([m[k] for m in metric_dicts])) for k in keys}

    def _aggregate_dm(dm_list) -> dict:
        sig_fraction = float(np.mean([d.significant_at_05 for d in dm_list]))
        mean_p = float(np.mean([d.p_value for d in dm_list]))
        return {"fraction_folds_significant": sig_fraction, "mean_p_value": mean_p}

    return {
        "density_pct": density_pct,
        "n_observations": sliced.n_observations,
        "folds_evaluated": len(folds),
        "models": {
            "naive": _aggregate(model_fold_metrics["naive"]),
            "sarima": {**_aggregate(model_fold_metrics["sarima"]), "dm_vs_naive": _aggregate_dm(dm_results["sarima"])},
            "prophet": {**_aggregate(model_fold_metrics["prophet"]), "dm_vs_naive": _aggregate_dm(dm_results["prophet"])},
            "xgboost": {**_aggregate(model_fold_metrics["xgboost"]), "dm_vs_naive": _aggregate_dm(dm_results["xgboost"])},
            "nbeats": {**_aggregate(model_fold_metrics["nbeats"]), "dm_vs_naive": _aggregate_dm(dm_results["nbeats"])},
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
    Pick the model with the lowest mean RMSE at 100% density (i.e. trained
    on everything available — what would actually be deployed once a real
    Duka has built up history) and serialize it wrapped in
    SerializableForecastModel, so ForecastService can call
    .predict_with_band() uniformly regardless of which of the five model
    kinds won.

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
    best_kind = min(candidates, key=candidates.get)

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
        # Build a future feature template covering the next `horizon` days
        # past the end of the training data, so predict_next() has
        # calendar/seasonal features ready without needing the caller to
        # supply them at request time.
        future_dates = pd.date_range(dates_full.max() + pd.Timedelta(days=1), periods=horizon, freq="D")
        # Use np.nan, not None/Python-list-of-None: concatenating a column
        # of bare Nones degrades the whole 'sales' column (and therefore
        # every lag_* column derived from it) to object dtype, which XGBoost
        # rejects outright at predict time. np.nan keeps it float64.
        future_template = pd.DataFrame({"date": future_dates, "sales": np.full(horizon, np.nan)})
        future_template = add_rwanda_features(future_template)
        # XGBoost's lag features need real history immediately preceding
        # the forecast window; stitch training tail + future template so
        # add_lag_features can compute lag_7d/14d/28d correctly, then slice back
        # to just the future rows.
        combined = pd.concat([product_df, future_template], ignore_index=True)
        combined["sales"] = combined["sales"].astype(float)
        combined = add_lag_features(combined)
        future_with_lags = combined.iloc[-horizon:].copy()
        for lag_col in ("lag_7d", "lag_14d", "lag_28d"):
            future_with_lags[lag_col] = future_with_lags[lag_col].astype(float)
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
                print(f"[{product_code} / store {store}] density={density}% ...")
                store_results[str(density)] = run_for_product_and_density(store_df, density, horizon)
            product_store_results[store] = store_results

        all_results[product_code] = product_store_results

        # For artifact serialization, use the full-history series for store 1
        # as a representative series. The best model type is decided by the
        # metric averaged across all stores at 100 % density.
        store_rmse: dict[str, list[float]] = {m: [] for m in ["naive","sarima","prophet","xgboost","nbeats"]}
        for store_res in product_store_results.values():
            for m_name, m_metrics in store_res.get("100", {}).get("models", {}).items():
                if "rmse" in m_metrics:
                    store_rmse[m_name].append(m_metrics["rmse"])
        mean_rmse = {m: float(np.mean(v)) for m, v in store_rmse.items() if v}
        best_kind = min(mean_rmse, key=mean_rmse.get) if mean_rmse else "naive"
        print(f"  [{product_code}] best model at 100% density (avg across stores): {best_kind}")

        # Serialize using store 1's full history as the representative series
        rep_df = (
            fmcg[(fmcg["product_code"] == product_code) & (fmcg["store"] == stores[0])]
            .sort_values("date").reset_index(drop=True)
        )
        winning_kind = _select_and_serialize_best_model(
            product_code, rep_df,
            {"100": {"models": {m: {"rmse": v} for m, v in mean_rmse.items()}}},
            artifact_dir, horizon,
        )
        print(f"  [{product_code}] serialized as '{winning_kind}'.")

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

    print(f"\nJSON written to: {output_path}")
    print(f"Raw CSV: {raw_path} ({len(raw_df)} rows)")
    print(f"Aggregated CSV: {agg_path} ({len(agg_df)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="ml_experiments/data/fmcg_rwanda_localized.csv")
    parser.add_argument("--output-dir", default="ml_experiments/results")
    parser.add_argument("--artifact-dir", default="ml_experiments/artifacts")
    parser.add_argument("--horizon", type=int, default=7)
    args = parser.parse_args()
    main(args.data, args.output_dir, args.artifact_dir, args.horizon)
