"""
SARIMA classical statistical baseline (proposal Table 5).

Uses pmdarima's auto_arima to select (p,d,q)(P,D,Q,m) orders automatically,
since manually grid-searching SARIMA orders for every (product, store,
density-level) combination across the experiment matrix would be
impractical. Weekly seasonality (m=7) is assumed given daily retail data,
consistent with Falatouri et al. (2022) who found SARIMA competitive on
shorter series with clear seasonal structure.
"""
import numpy as np
import pandas as pd
import pmdarima as pm


class SARIMAModel:
    def __init__(self, seasonal_period: int = 7):
        self.seasonal_period = seasonal_period
        self.model = None

    def fit(self, y: pd.Series) -> "SARIMAModel":
        # auto_arima needs a minimum amount of data to fit a seasonal model;
        # fall back to a non-seasonal search for very short cold-start series.
        seasonal = len(y) >= 2 * self.seasonal_period
        self.model = pm.auto_arima(
            y.values,
            seasonal=seasonal,
            m=self.seasonal_period if seasonal else 1,
            suppress_warnings=True,
            error_action="ignore",
            stepwise=True,
            max_p=3, max_q=3, max_P=2, max_Q=2,
        )
        return self

    def predict(self, n_periods: int) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Call fit() before predict().")
        preds = self.model.predict(n_periods=n_periods)
        return np.clip(np.asarray(preds), a_min=0, a_max=None)  # demand can't be negative
