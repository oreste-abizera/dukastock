import numpy as np
import pandas as pd
import pytest

from app.ml.models.naive import NaiveBaseline
from app.ml.models.serializable import SerializableForecastModel


def test_naive_kind_predicts_via_int_horizon():
    model = NaiveBaseline().fit(pd.Series([1, 2, 3, 4, 5, 6, 7]))
    wrapped = SerializableForecastModel(kind="naive", model=model)
    preds = wrapped.predict_next(7)
    assert len(preds) == 7
    np.testing.assert_array_equal(preds, [1, 2, 3, 4, 5, 6, 7])


def test_xgboost_kind_requires_future_feature_template():
    class FakeXGBModel:
        def predict(self, df):
            return np.full(len(df), 9.0)

    wrapped = SerializableForecastModel(kind="xgboost", model=FakeXGBModel())
    with pytest.raises(ValueError, match="future_feature_template"):
        wrapped.predict_next(7)


def test_xgboost_kind_predicts_with_template():
    class FakeXGBModel:
        def predict(self, df):
            return np.full(len(df), 9.0)

    template = pd.DataFrame({"x": range(7)})
    wrapped = SerializableForecastModel(kind="xgboost", model=FakeXGBModel(), future_feature_template=template)
    preds = wrapped.predict_next(7)
    assert len(preds) == 7
    assert all(p == 9.0 for p in preds)


def test_xgboost_kind_pads_short_template_rather_than_crashing():
    class FakeXGBModel:
        def predict(self, df):
            return np.arange(len(df), dtype=float)

    # Template only covers 3 days, but caller asks for 7.
    template = pd.DataFrame({"x": range(3)})
    wrapped = SerializableForecastModel(kind="xgboost", model=FakeXGBModel(), future_feature_template=template)
    preds = wrapped.predict_next(7)
    assert len(preds) == 7


def test_prophet_kind_requires_last_observed_date():
    class FakeProphetModel:
        def predict(self, n_periods, last_date):
            return np.zeros(n_periods), np.zeros(n_periods), np.zeros(n_periods)

    wrapped = SerializableForecastModel(kind="prophet", model=FakeProphetModel())
    with pytest.raises(ValueError, match="last_observed_date"):
        wrapped.predict_next(7)


def test_prophet_kind_predicts_with_date():
    class FakeProphetModel:
        def predict(self, n_periods, last_date):
            point = np.full(n_periods, 5.0)
            lower = np.full(n_periods, 4.0)
            upper = np.full(n_periods, 6.0)
            return point, lower, upper

    wrapped = SerializableForecastModel(
        kind="prophet", model=FakeProphetModel(), last_observed_date=pd.Timestamp("2026-01-01")
    )
    preds = wrapped.predict_next(7)
    assert len(preds) == 7
    assert all(p == 5.0 for p in preds)


def test_prophet_predict_with_band_uses_native_interval():
    class FakeProphetModel:
        def predict(self, n_periods, last_date):
            point = np.full(n_periods, 10.0)
            lower = np.full(n_periods, 8.0)
            upper = np.full(n_periods, 12.0)
            return point, lower, upper

    wrapped = SerializableForecastModel(
        kind="prophet", model=FakeProphetModel(), last_observed_date=pd.Timestamp("2026-01-01")
    )
    band = wrapped.predict_with_band(7)
    assert band["predicted_quantity"] == 70.0
    assert band["lower_bound"] == 56.0
    assert band["upper_bound"] == 84.0


def test_nbeats_kind_predicts_when_horizon_matches():
    class FakeNBEATSModel:
        def predict(self):
            return np.full(7, 3.0)

    wrapped = SerializableForecastModel(kind="nbeats", model=FakeNBEATSModel())
    preds = wrapped.predict_next(7)
    assert len(preds) == 7


def test_nbeats_kind_raises_clear_error_on_horizon_mismatch():
    class FakeNBEATSModel:
        def predict(self):
            return np.full(7, 3.0)

    wrapped = SerializableForecastModel(kind="nbeats", model=FakeNBEATSModel())
    with pytest.raises(ValueError, match="7-day horizon"):
        wrapped.predict_next(14)


def test_predict_with_band_default_uses_pct_band_for_non_prophet():
    model = NaiveBaseline().fit(pd.Series([10, 10, 10, 10, 10, 10, 10]))
    wrapped = SerializableForecastModel(kind="naive", model=model)
    band = wrapped.predict_with_band(7, band_pct=0.2)
    assert band["predicted_quantity"] == 70.0
    assert band["lower_bound"] == 56.0
    assert band["upper_bound"] == 84.0


def test_unknown_kind_raises():
    wrapped = SerializableForecastModel(kind="bogus", model=object())
    with pytest.raises(ValueError, match="Unknown model kind"):
        wrapped.predict_next(7)
