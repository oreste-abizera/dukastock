"""
Per-shopkeeper personalized forecast training.

Reuses the exact evaluation methodology from
ml_experiments/scripts/run_experiment.py (walk-forward folds + a pooled
Diebold-Mariano test against naive), just scoped to one shopkeeper's own
sales history instead of one Kaggle-proxy store, and restricted to
naive/XGBoost -- SARIMA/Prophet/N-BEATS are too slow to fit inside a batch
job running across every shopkeeper, and in the offline benchmark
(ml_experiments/results/ml_benchmark_results.csv) only XGBoost ever beat
naive, so there is no accuracy case for including the other three here.

This module only trains and persists artifacts; see
app.services.forecast_service.ForecastService.forecast_for_shopkeeper for
the serving path that reads them back.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.ml.evaluation.metrics import diebold_mariano_test
from app.ml.models.naive import NaiveBaseline
from app.ml.models.serializable import SerializableForecastModel
from app.ml.models.xgboost_model import XGBoostDemandModel, add_lag_features, build_future_feature_template
from app.ml.pipeline.cold_start import walk_forward_folds
from app.ml.pipeline.rwanda_features import add_rwanda_features
from app.models.orm import ForecastResult
from app.services.sales_aggregation import get_daily_series

logger = get_logger(__name__)

HORIZON_DAYS = 7


def personalized_artifact_path(artifact_dir: str, shopkeeper_id: str, product_code: str) -> Path:
    """Shared with ForecastService.forecast_for_shopkeeper so the artifact
    naming convention lives in exactly one place."""
    return Path(artifact_dir) / "personalized" / f"{shopkeeper_id}_{product_code.lower()}_best_model.joblib"


def train_for_shopkeeper(db: Session, shopkeeper_id: str, product_code: str, artifact_dir: str) -> str:
    """Fit and serialize this shopkeeper's own best model for one product,
    and log a ForecastResult row. Returns the winning model kind ("naive"
    or "xgboost"), or a "skipped (...)" string when there's nothing to do
    yet, for batch-job logging."""
    series = get_daily_series(db, shopkeeper_id, product_code)
    if series.empty:
        return "skipped (no sales logged yet)"

    featured = add_rwanda_features(series)
    folds = walk_forward_folds(featured, horizon=HORIZON_DAYS)

    if not folds:
        # Not enough history yet for even one walk-forward fold -- serve
        # naive, exactly what the offline experiment's own cold-start
        # findings say to expect this early (see docs/RESEARCH_DESIGN.md).
        kind = "naive"
        wrapped = SerializableForecastModel(kind="naive", model=NaiveBaseline().fit(featured["sales"]))
    else:
        pooled_y_test: list[np.ndarray] = []
        pooled_naive_preds: list[np.ndarray] = []
        pooled_xgb_preds: list[np.ndarray] = []

        for train, test in folds:
            y_train, y_test = train["sales"], test["sales"].values
            pooled_y_test.append(y_test)

            naive_preds = NaiveBaseline().fit(y_train).predict(len(test))
            pooled_naive_preds.append(naive_preds)

            try:
                xgb_model = XGBoostDemandModel().fit(train)
                future_features = add_lag_features(
                    pd.concat([train, test]).reset_index(drop=True)
                ).iloc[-len(test):]
                xgb_preds = xgb_model.predict(future_features)
            except Exception as exc:
                logger.warning(
                    "personalized_xgboost_fold_failed",
                    shopkeeper_id=shopkeeper_id, product_code=product_code, error=str(exc),
                )
                xgb_preds = naive_preds
            pooled_xgb_preds.append(xgb_preds)

        dm = diebold_mariano_test(
            np.concatenate(pooled_y_test), np.concatenate(pooled_xgb_preds), np.concatenate(pooled_naive_preds),
            h=HORIZON_DAYS,
        )

        if dm.significant_at_05:
            kind = "xgboost"
            fitted = XGBoostDemandModel().fit(featured)
            future_template = build_future_feature_template(featured, HORIZON_DAYS)
            wrapped = SerializableForecastModel(kind="xgboost", model=fitted, future_feature_template=future_template)
        else:
            kind = "naive"
            wrapped = SerializableForecastModel(kind="naive", model=NaiveBaseline().fit(featured["sales"]))

    artifact_path = personalized_artifact_path(artifact_dir, shopkeeper_id, product_code)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(wrapped, artifact_path)

    band = wrapped.predict_with_band(HORIZON_DAYS)
    db.add(ForecastResult(
        shopkeeper_id=shopkeeper_id,
        product_code=product_code,
        predicted_quantity=band["predicted_quantity"],
        lower_bound=band["lower_bound"],
        upper_bound=band["upper_bound"],
        model_used=kind,
        # Repurposed: a live shopkeeper's series has no fixed total length to
        # take a percentage of (unlike the offline Kaggle-proxy experiment),
        # so this stores the absolute observation count instead.
        data_density_pct=float(len(featured)),
    ))
    db.commit()
    return kind
