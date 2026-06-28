import numpy as np
import pandas as pd
import pytest

from app.ml.models.xgboost_model import (
    MIN_OBSERVATIONS_FOR_GRID_SEARCH,
    XGBoostDemandModel,
    add_lag_features,
)


@pytest.fixture
def sufficient_data_df():
    """A series long enough to clear MIN_OBSERVATIONS_FOR_GRID_SEARCH after
    dropna() removes the first 28 rows (lag_28d warm-up period)."""
    n = MIN_OBSERVATIONS_FOR_GRID_SEARCH + 40
    dates = pd.date_range("2013-01-01", periods=n, freq="D")
    rng = np.random.default_rng(0)
    sales = np.clip(10 + 3 * np.sin(2 * np.pi * dates.dayofweek / 7) + rng.normal(0, 1.0, n), 0, None)
    df = pd.DataFrame({"date": dates, "sales": sales})
    # Minimal Rwanda-feature-shaped columns so FEATURE_COLUMNS resolves without
    # importing the full rwanda_features module into this unit test.
    df["is_holiday"] = 0
    df["is_memorial"] = 0
    df["days_to_next_holiday"] = 30
    df["season_flag"] = 0
    df["rain_intensity"] = 0.5
    df["day_of_week"] = dates.dayofweek
    df["week_of_year"] = dates.isocalendar().week.astype(int).values
    df["month"] = dates.month
    return df


def test_grid_search_enabled_by_default():
    model = XGBoostDemandModel()
    assert model.use_grid_search is True


def test_grid_search_runs_and_sets_best_params(sufficient_data_df):
    model = XGBoostDemandModel(use_grid_search=True).fit(sufficient_data_df)
    assert model.best_params_ is not None
    assert set(model.best_params_.keys()) == {"n_estimators", "max_depth", "learning_rate"}


def test_grid_search_disabled_leaves_best_params_none(sufficient_data_df):
    model = XGBoostDemandModel(use_grid_search=False).fit(sufficient_data_df)
    assert model.best_params_ is None
    assert model.model is not None  # still fits a model, just without search


def test_grid_search_auto_skips_below_minimum_observations():
    # Need enough rows to survive add_lag_features' shift(28) dropna with
    # at least 10 remaining (the absolute floor before XGBoost gives up
    # entirely and falls back to predicting the mean) but fewer than
    # MIN_OBSERVATIONS_FOR_GRID_SEARCH remaining, to land in the
    # "fixed defaults, real model, no grid search" zone specifically.
    n = 28 + 20  # 20 rows survive shift(28)+dropna -- between 10 and 60
    dates = pd.date_range("2013-01-01", periods=n, freq="D")
    df = pd.DataFrame({"date": dates, "sales": np.arange(n, dtype=float)})
    df["is_holiday"] = 0
    df["is_memorial"] = 0
    df["days_to_next_holiday"] = 30
    df["season_flag"] = 0
    df["rain_intensity"] = 0.5
    df["day_of_week"] = dates.dayofweek
    df["week_of_year"] = dates.isocalendar().week.astype(int).values
    df["month"] = dates.month

    model = XGBoostDemandModel(use_grid_search=True).fit(df)
    assert model.best_params_ is None  # fixed defaults path, not grid search
    assert model.model is not None


def test_custom_param_grid_is_respected(sufficient_data_df):
    custom_grid = {"n_estimators": [50], "max_depth": [2], "learning_rate": [0.3]}
    model = XGBoostDemandModel(use_grid_search=True, param_grid=custom_grid).fit(sufficient_data_df)
    assert model.best_params_ == {"n_estimators": 50, "max_depth": 2, "learning_rate": 0.3}


def test_predict_still_works_after_grid_search(sufficient_data_df):
    model = XGBoostDemandModel(use_grid_search=True).fit(sufficient_data_df)
    future = add_lag_features(sufficient_data_df).iloc[-7:]
    preds = model.predict(future)
    assert len(preds) == 7
    assert all(p >= 0 for p in preds)  # demand can't be negative
