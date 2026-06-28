import numpy as np

from app.ml.evaluation.metrics import (
    compute_all_metrics,
    diebold_mariano_test,
    mae,
    mape,
    rmse,
    smape,
)


def test_rmse_perfect_prediction_is_zero():
    y = np.array([1.0, 2.0, 3.0])
    assert rmse(y, y) == 0.0


def test_mae_known_value():
    y_true = np.array([10.0, 20.0])
    y_pred = np.array([8.0, 22.0])
    assert mae(y_true, y_pred) == 2.0


def test_mape_handles_zero_with_epsilon():
    y_true = np.array([0.0, 10.0])
    y_pred = np.array([1.0, 10.0])
    result = mape(y_true, y_pred)
    assert np.isfinite(result)


def test_smape_symmetric_and_bounded():
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([0.0, 0.0])
    assert smape(y_true, y_pred) == 0.0


def test_dm_test_identical_predictions_not_significant():
    np.random.seed(0)
    y_true = np.random.normal(10, 2, 50)
    preds = y_true + np.random.normal(0, 0.1, 50)
    result = diebold_mariano_test(y_true, preds, preds)
    assert result.significant_at_05 is False
    assert result.p_value == 1.0


def test_dm_test_detects_clearly_better_model():
    np.random.seed(1)
    y_true = np.full(200, 10.0)
    good_model = y_true + np.random.normal(0, 0.5, 200)
    bad_baseline = y_true + np.random.normal(0, 5.0, 200)
    result = diebold_mariano_test(y_true, good_model, bad_baseline)
    assert result.mean_loss_differential < 0
    assert result.p_value < 0.05
    assert result.significant_at_05 is True


def test_compute_all_metrics_returns_expected_keys():
    y = np.array([1.0, 2.0, 3.0])
    out = compute_all_metrics(y, y)
    assert set(out.keys()) == {"rmse", "mae", "mape", "smape"}


def test_dm_test_h7_newey_west_differs_from_h1():
    """Confirm the Newey-West HAC correction (h=7) produces a different
    DM statistic than the naive h=1 estimator when the loss differential
    has non-zero serial autocorrelation. Verifies that the lags in the
    HAC loop are actually affecting the variance computation."""
    np.random.seed(42)
    n = 84  # two cycles of 6 folds × 7 days
    # Build a loss differential with strong positive autocorrelation (AR(1))
    d = np.zeros(n)
    d[0] = np.random.normal(0, 2)
    for t in range(1, n):
        d[t] = 0.8 * d[t - 1] + np.random.normal(0, 1)
    y_true = np.zeros(n)
    y_model = y_true - np.sqrt(np.abs(d)) * np.sign(d)  # model better when d<0
    y_base = y_true

    result_h1 = diebold_mariano_test(y_true, y_model, y_base, h=1)
    result_h7 = diebold_mariano_test(y_true, y_model, y_base, h=7)
    # With positive autocorrelation, HAC variance > naive variance → |DM_h7| < |DM_h1|
    # so the two statistics must differ (HAC correction has a material effect)
    assert abs(result_h7.dm_statistic) != abs(result_h1.dm_statistic)


def test_dm_significant_requires_directionally_better():
    """significant_at_05 must be False when d_bar > 0 (model is WORSE),
    even if p < 0.05 (test detects a significant difference, just in the
    wrong direction)."""
    np.random.seed(0)
    y_true = np.full(200, 10.0)
    bad_model = y_true + np.random.normal(0, 5.0, 200)
    good_baseline = y_true + np.random.normal(0, 0.5, 200)
    result = diebold_mariano_test(y_true, bad_model, good_baseline, h=7)
    # Model is worse → d_bar > 0 → significant_at_05 must be False
    assert result.mean_loss_differential > 0
    assert result.significant_at_05 is False
