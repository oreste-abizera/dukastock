"""
XGBoost with Rwanda calendar and seasonal feature engineering (proposal
Table 5: "Best under sparse intermittent demand").

Fatima & Salam (2025) showed that incorporating external calendar features
reduced XGBoost MAE substantially across retail datasets, motivating the
Rwanda-specific holiday/season engineering applied here. A 2025 comparative
study (arXiv:2506.05941) found XGBoost outperforming several neural
alternatives on intermittent sparse retail data — directly the regime
DukaStock cares about (cold-start Duka shops).

Hyperparameter search: proposal Chapter 3.3 specifies "XGBoost with grid
search." GridSearchCV's default k-fold splitter shuffles rows, which would
leak future demand into training folds — the same time-series CV violation
flagged in Chapter 3.2 (Bergmeir & Benitez, 2012) for the cold-start splits
themselves. sklearn's TimeSeriesSplit is used instead, so the grid search
respects chronological order the same way the outer walk-forward evaluation
does. Grid search is enabled by default but auto-disables itself below
MIN_OBSERVATIONS_FOR_GRID_SEARCH, since at low cold-start density there
isn't enough data for TimeSeriesSplit to form even 3 valid folds — at that
point a fixed, reasonable hyperparameter set is used instead, which is
itself a reportable cold-start finding rather than a workaround to hide.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

FEATURE_COLUMNS = [
    "is_holiday", "is_memorial", "days_to_next_holiday", "season_flag",
    "rain_intensity", "day_of_week", "week_of_year", "month",
    "lag_7d", "lag_14d", "lag_28d",
]

# Small grid by design: this runs once per (product, density, fold) across
# the full experiment matrix in run_experiment.py, so an expansive grid
# multiplies an already SARIMA-dominated runtime. These four hyperparameters
# are the ones with the largest documented effect on tree-ensemble demand
# forecasting accuracy (depth/learning_rate trade bias vs. variance;
# n_estimators trades fit quality vs. overfitting on short series).
DEFAULT_PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [3, 4, 6],
    "learning_rate": [0.05, 0.1],
}

MIN_OBSERVATIONS_FOR_GRID_SEARCH = 60  # need enough rows for 3 TimeSeriesSplit folds to be meaningful


def add_lag_features(df: pd.DataFrame, target_col: str = "sales") -> pd.DataFrame:
    """Weekly lag features: sales 7, 14, and 28 days ago.
    Named lag_7d/14d/28d (not lag_1/2/4) to make the actual shift unambiguous
    in feature-importance plots and the thesis methodology section."""
    out = df.copy()
    out["lag_7d"] = out[target_col].shift(7)
    out["lag_14d"] = out[target_col].shift(14)
    out["lag_28d"] = out[target_col].shift(28)
    return out


class XGBoostDemandModel:
    def __init__(self, use_grid_search: bool = True, param_grid: dict | None = None, **xgb_kwargs):
        self.use_grid_search = use_grid_search
        self.param_grid = param_grid or DEFAULT_PARAM_GRID
        self._fixed_kwargs = dict(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror",
        )
        self._fixed_kwargs.update(xgb_kwargs)
        self.model = None
        self.best_params_: dict | None = None
        self._fallback_mean = 0.0

    def fit(self, df_features: pd.DataFrame, target_col: str = "sales") -> "XGBoostDemandModel":
        df = add_lag_features(df_features, target_col)
        df = df.dropna(subset=FEATURE_COLUMNS + [target_col])
        self._fallback_mean = float(df_features[target_col].mean()) if len(df_features) else 0.0

        if len(df) < 10:
            # Cold-start density too low for lagged features (need 28+ days
            # of history before lag_28d is even defined). Model degrades
            # gracefully to predicting the historical mean rather than
            # raising — this IS the expected and reportable behaviour at
            # the 5% density level for several products.
            self.model = None
            return self

        X, y = df[FEATURE_COLUMNS], df[target_col]

        if self.use_grid_search and len(df) >= MIN_OBSERVATIONS_FOR_GRID_SEARCH:
            base_model = xgb.XGBRegressor(
                subsample=self._fixed_kwargs["subsample"],
                colsample_bytree=self._fixed_kwargs["colsample_bytree"],
                objective=self._fixed_kwargs["objective"],
            )
            n_splits = min(3, max(2, len(df) // 30))
            tscv = TimeSeriesSplit(n_splits=n_splits)
            search = GridSearchCV(
                base_model, self.param_grid, cv=tscv,
                scoring="neg_root_mean_squared_error", n_jobs=1,
            )
            search.fit(X, y)
            self.model = search.best_estimator_
            self.best_params_ = search.best_params_
        else:
            # Not enough data for a meaningful time-series grid search at
            # this density level — use the fixed, documented default
            # hyperparameters instead of crashing or silently skipping.
            self.model = xgb.XGBRegressor(**self._fixed_kwargs)
            self.model.fit(X, y)
            self.best_params_ = None

        return self

    def predict(self, df_future_features: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            return np.full(len(df_future_features), self._fallback_mean)
        X = df_future_features[FEATURE_COLUMNS].fillna(0)
        preds = self.model.predict(X)
        return np.clip(preds, a_min=0, a_max=None)
