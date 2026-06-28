import numpy as np
import pandas as pd

from app.ml.models.naive import NaiveBaseline


def test_naive_predicts_last_week_pattern():
    y = pd.Series([1, 2, 3, 4, 5, 6, 7])
    model = NaiveBaseline().fit(y)
    preds = model.predict(7)
    np.testing.assert_array_equal(preds, [1, 2, 3, 4, 5, 6, 7])


def test_naive_tiles_for_longer_horizon():
    y = pd.Series([1, 2, 3, 4, 5, 6, 7])
    model = NaiveBaseline().fit(y)
    preds = model.predict(10)
    assert len(preds) == 10
    np.testing.assert_array_equal(preds[:7], [1, 2, 3, 4, 5, 6, 7])


def test_naive_falls_back_to_mean_under_one_week_history():
    y = pd.Series([2, 4, 6])
    model = NaiveBaseline().fit(y)
    preds = model.predict(5)
    assert np.allclose(preds, 4.0)


def test_naive_empty_history_returns_zeros():
    model = NaiveBaseline().fit(pd.Series([], dtype=float))
    preds = model.predict(3)
    np.testing.assert_array_equal(preds, [0.0, 0.0, 0.0])
