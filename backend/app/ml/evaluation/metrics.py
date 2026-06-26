"""
Evaluation metrics framework (proposal Table 6).

RMSE, MAE, MAPE, sMAPE for raw accuracy comparison, plus the
Diebold-Mariano (1995) test for statistical significance of forecast
accuracy differences — the core inferential tool behind the primary
research question ("at what minimum data density does each model first
achieve statistically significant improvement over the naive baseline?").
"""
from dataclasses import dataclass

import numpy as np
from scipy import stats


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """Undefined at y_true=0; epsilon guards against division by zero but the
    metric should be treated as unreliable whenever true demand is sparse —
    this is exactly why sMAPE is the preferred metric in DukaStock (proposal
    Table 6)."""
    denom = np.where(np.abs(y_true) < epsilon, epsilon, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """Symmetric MAPE — handles zero-demand periods gracefully, which is
    common for informal Duka FMCG items on slow days."""
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.where(denom < epsilon, epsilon, denom)
    return float(np.mean(2 * np.abs(y_true - y_pred) / denom) * 100)


@dataclass
class DMTestResult:
    dm_statistic: float
    p_value: float
    significant_at_05: bool
    mean_loss_differential: float


def diebold_mariano_test(
    y_true: np.ndarray,
    y_pred_model: np.ndarray,
    y_pred_baseline: np.ndarray,
    loss: str = "squared",
    h: int = 1,
) -> DMTestResult:
    """
    Diebold & Mariano (1995) test for equal predictive accuracy.

    H0: the model and the naive baseline have equal expected loss.
    H1 (two-sided): they differ.

    A significant result with mean_loss_differential < 0 means the model's
    errors are smaller than the baseline's — i.e. the model is genuinely
    better, not just better by chance. This is the test referenced
    throughout the proposal as "the DM test at p < 0.05" and is what
    determines the threshold density for each model class.
    """
    e_model = y_true - y_pred_model
    e_base = y_true - y_pred_baseline

    if loss == "squared":
        loss_model = e_model ** 2
        loss_base = e_base ** 2
    elif loss == "absolute":
        loss_model = np.abs(e_model)
        loss_base = np.abs(e_base)
    else:
        raise ValueError("loss must be 'squared' or 'absolute'")

    d = loss_model - loss_base
    n = len(d)
    d_bar = float(np.mean(d))

    # Newey-West style variance estimator with h-1 lag autocovariance terms,
    # standard for the DM test when h > 1 (here h=1 reduces to plain sample
    # variance of the loss differential, dividing by n for the variance of
    # the mean).
    gamma0 = float(np.var(d, ddof=0))
    var_d = gamma0
    for lag in range(1, h):
        cov = float(np.cov(d[lag:], d[:-lag])[0, 1]) if n > lag else 0.0
        var_d += 2 * cov
    var_d_bar = var_d / n if n > 0 else np.nan

    if var_d_bar <= 0 or np.isnan(var_d_bar):
        # Degenerate case: identical predictions, zero variance in the loss
        # differential. No statistically meaningful difference can be claimed.
        return DMTestResult(dm_statistic=0.0, p_value=1.0, significant_at_05=False, mean_loss_differential=d_bar)

    dm_stat = d_bar / np.sqrt(var_d_bar)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

    return DMTestResult(
        dm_statistic=float(dm_stat),
        p_value=float(p_value),
        significant_at_05=bool(p_value < 0.05 and d_bar < 0),  # must ALSO be better, not just different
        mean_loss_differential=d_bar,
    )


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
    }
