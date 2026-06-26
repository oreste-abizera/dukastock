"""
Uniform forecast model wrapper.

The five model classes in app.ml.models have genuinely different fit/predict
signatures (NaiveBaseline and SARIMAModel take an int horizon; XGBoostDemandModel
takes a dataframe of future feature rows; ProphetModel takes a horizon AND a
last-seen date; NBEATSModel takes no predict() arguments at all, since its
horizon is fixed at construction time). That's fine while each lives inside
ml_experiments/scripts/run_experiment.py, which already builds whatever
inputs each one needs — but it becomes a real bug the moment something
downstream (ForecastService) tries to call .predict() on "whichever model
won" without knowing which of the five it is.

SerializableForecastModel is the single object actually written to
*_best_model.joblib. It snapshots everything its wrapped model needs to
produce a future forecast (the trained model, which kind it is, the future
feature rows for XGBoost, the last observed date for Prophet) at save time,
and exposes one method — predict_next(horizon_days) — that works
identically regardless of which of the five models is inside. This is what
ForecastService.forecast() should call; it should never touch
NaiveBaseline/SARIMAModel/ProphetModel/XGBoostDemandModel/NBEATSModel
directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

ModelKind = Literal["naive", "sarima", "prophet", "xgboost", "nbeats"]


@dataclass
class SerializableForecastModel:
    kind: ModelKind
    model: Any
    # Only populated for the model kinds that need it at predict time:
    last_observed_date: pd.Timestamp | None = None  # Prophet
    future_feature_template: pd.DataFrame | None = None  # XGBoost (is_holiday etc. for the *next* horizon_days)

    def predict_next(self, horizon_days: int) -> np.ndarray:
        """Return a point forecast array of length horizon_days, regardless
        of which underlying model kind this wraps."""
        if self.kind in ("naive", "sarima"):
            return self.model.predict(horizon_days)

        if self.kind == "prophet":
            if self.last_observed_date is None:
                raise ValueError("Prophet models require last_observed_date to be set before saving.")
            point, _lower, _upper = self.model.predict(horizon_days, self.last_observed_date)
            return point

        if self.kind == "xgboost":
            if self.future_feature_template is None:
                raise ValueError("XGBoost models require future_feature_template to be set before saving.")
            future_rows = self.future_feature_template.iloc[:horizon_days]
            if len(future_rows) < horizon_days:
                # Template was built for a shorter horizon than requested at
                # serve time — pad by repeating the last known row rather
                # than crashing the request. Degraded but not broken.
                pad_count = horizon_days - len(future_rows)
                padding = pd.concat([future_rows.iloc[-1:]] * pad_count, ignore_index=True) if len(future_rows) else future_rows
                future_rows = pd.concat([future_rows, padding], ignore_index=True)
            return self.model.predict(future_rows)

        if self.kind == "nbeats":
            # NBEATSModel's horizon is fixed at construction time. If the
            # caller asks for a different horizon than the model was built
            # for, that's a real mismatch worth surfacing rather than
            # silently truncating/padding.
            preds = self.model.predict()
            if len(preds) != horizon_days:
                raise ValueError(
                    f"This N-BEATS artifact was trained for a {len(preds)}-day horizon, "
                    f"but {horizon_days} days were requested. Re-train/serialize with the "
                    f"correct horizon, or request {len(preds)} days."
                )
            return preds

        raise ValueError(f"Unknown model kind: {self.kind}")

    def predict_with_band(self, horizon_days: int, band_pct: float = 0.2) -> dict:
        """Convenience wrapper returning the same
        {predicted_quantity, lower_bound, upper_bound} shape ForecastService
        needs, using Prophet's native interval when available and a simple
        +/-band_pct otherwise."""
        if self.kind == "prophet" and self.last_observed_date is not None:
            point, lower, upper = self.model.predict(horizon_days, self.last_observed_date)
            return {
                "predicted_quantity": float(np.sum(point)),
                "lower_bound": float(np.sum(lower)),
                "upper_bound": float(np.sum(upper)),
            }

        point = self.predict_next(horizon_days)
        total = float(np.sum(point))
        return {
            "predicted_quantity": total,
            "lower_bound": total * (1 - band_pct),
            "upper_bound": total * (1 + band_pct),
        }
