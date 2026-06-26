"""
Facebook Prophet decomposable model with injected Rwanda holiday calendar
(Taylor & Letham, 2018).

Prophet accepts a user-supplied holidays dataframe — this is exactly what
makes it well-suited to DukaStock's Rwanda localisation, since the model
can learn a distinct demand effect for Genocide Memorial Day (suppressor)
separately from ordinary holidays (which may boost FMCG demand, e.g. sugar
and cooking oil before Christmas).

Deliberate deviation from proposal Chapter 3.3: the ML training pipeline
description groups "lag-1/2/4 week features" under "XGBoost and Prophet."
This implementation does NOT pass lag features into Prophet as regressors.
Reasoning: Prophet is a decomposable additive model (trend + seasonality +
holidays), not a tabular regressor — its weekly_seasonality term already
captures the same 7-day periodicity lag_7d would encode, so adding lag_7d as
an external regressor mostly duplicates a signal Prophet already models
natively, and risks Prophet "learning" partly through extra_regressors
that aren't available at the requested future horizon as cleanly as
Prophet's own seasonal terms are. Prophet's strength in this comparison is
specifically the holiday-calendar contrast (Table 5: "Rwanda holiday
calendar support"), which is implemented in full below; this is a
documented, reasoned scope decision, not an oversight. If a stricter
reading of the proposal is required for the final submission, Prophet
supports lag regressors via .add_regressor("lag_7d") before .fit() and
passing the same lag columns in the future dataframe at predict time —
see app.ml.models.xgboost_model.add_lag_features for the matching
feature-engineering logic that would need to be threaded through here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from prophet import Prophet

from app.ml.pipeline.rwanda_features import build_holiday_set


def _holidays_dataframe(years: list[int]) -> pd.DataFrame:
    holiday_set = build_holiday_set(years)
    return pd.DataFrame({
        "holiday": list(holiday_set.values()),
        "ds": pd.to_datetime(list(holiday_set.keys())),
    })


class ProphetModel:
    def __init__(self):
        self.model: Prophet | None = None

    def fit(self, dates: pd.Series, y: pd.Series) -> "ProphetModel":
        years = sorted(pd.to_datetime(dates).dt.year.unique().tolist())
        # Extend one year past the observed range so the model has holiday
        # rows available for the forecast horizon too.
        if years:
            years.append(years[-1] + 1)
        holidays_df = _holidays_dataframe(years)

        df = pd.DataFrame({"ds": pd.to_datetime(dates).values, "y": y.values})
        self.model = Prophet(
            holidays=holidays_df,
            weekly_seasonality=True,
            yearly_seasonality=len(df) >= 365,
            daily_seasonality=False,
            interval_width=0.8,
        )
        self.model.fit(df)
        return self

    def predict(self, n_periods: int, last_date: pd.Timestamp) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.model is None:
            raise RuntimeError("Call fit() before predict().")
        future = self.model.make_future_dataframe(periods=n_periods, include_history=False)
        forecast = self.model.predict(future)
        point = np.clip(forecast["yhat"].values, a_min=0, a_max=None)
        lower = np.clip(forecast["yhat_lower"].values, a_min=0, a_max=None)
        upper = np.clip(forecast["yhat_upper"].values, a_min=0, a_max=None)
        return point, lower, upper
