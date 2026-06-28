import pandas as pd
import pytest

from app.ml.pipeline.cold_start import DENSITY_LEVELS, temporal_density_slice, walk_forward_folds


@pytest.fixture
def daily_series():
    return pd.DataFrame({
        "date": pd.date_range("2013-01-01", periods=1000, freq="D"),
        "sales": range(1000),
    })


def test_all_density_levels_produce_increasing_sizes(daily_series):
    sizes = [temporal_density_slice(daily_series, d).n_observations for d in DENSITY_LEVELS]
    assert sizes == sorted(sizes)
    assert sizes[-1] == len(daily_series)


def test_five_percent_density_is_smallest(daily_series):
    slice_5 = temporal_density_slice(daily_series, 5)
    assert slice_5.n_observations == 50  # 5% of 1000


def test_density_slice_is_chronologically_first(daily_series):
    slice_5 = temporal_density_slice(daily_series, 5)
    assert slice_5.start_date == daily_series["date"].min()


def test_invalid_density_raises():
    df = pd.DataFrame({"date": pd.date_range("2013-01-01", periods=10), "sales": range(10)})
    with pytest.raises(ValueError):
        temporal_density_slice(df, 42)


def test_walk_forward_folds_respects_minimum_count(daily_series):
    folds = walk_forward_folds(daily_series, horizon=7, min_folds=6)
    assert len(folds) <= 6
    assert len(folds) > 0


def test_walk_forward_folds_train_grows_each_fold(daily_series):
    folds = walk_forward_folds(daily_series, horizon=7, min_folds=6)
    train_sizes = [len(train) for train, _ in folds]
    assert train_sizes == sorted(train_sizes)


def test_walk_forward_test_fold_size_matches_horizon(daily_series):
    folds = walk_forward_folds(daily_series, horizon=7, min_folds=6)
    for _, test in folds:
        assert len(test) == 7
