"""
Temporal cold-start data density splitting.

Per proposal Chapter 3.2 ("Research Design"): the ML experiment compares
four forecasting model classes across six temporal cold-start data density
levels — 5, 15, 30, 50, 75, and 100 percent of available chronological
history. At 5% density the model sees only the first 5% of the timeline,
simulating a Duka that recently began recording sales. This is a temporal
cut, not random record masking, because Bergmeir & Benitez (2012) and
Suradhaniwar et al. (2021) established that random splits leak future
information into time series cross-validation.
"""
from dataclasses import dataclass

import pandas as pd

DENSITY_LEVELS: list[int] = [5, 15, 30, 50, 75, 100]
MIN_WALK_FORWARD_FOLDS = 6


@dataclass
class ColdStartSlice:
    density_pct: int
    train: pd.DataFrame
    n_observations: int
    start_date: pd.Timestamp
    end_date: pd.Timestamp


def temporal_density_slice(series: pd.DataFrame, density_pct: int, date_col: str = "date") -> ColdStartSlice:
    """Return the first `density_pct`% of a chronologically sorted series."""
    if density_pct not in DENSITY_LEVELS:
        raise ValueError(f"density_pct must be one of {DENSITY_LEVELS}, got {density_pct}")
    ordered = series.sort_values(date_col).reset_index(drop=True)
    cutoff = max(1, int(len(ordered) * density_pct / 100))
    sliced = ordered.iloc[:cutoff].copy()
    return ColdStartSlice(
        density_pct=density_pct,
        train=sliced,
        n_observations=len(sliced),
        start_date=sliced[date_col].min(),
        end_date=sliced[date_col].max(),
    )


def walk_forward_folds(
    series: pd.DataFrame,
    date_col: str = "date",
    target_col: str = "sales",
    horizon: int = 7,
    min_folds: int = MIN_WALK_FORWARD_FOLDS,
):
    """
    Generate walk-forward (expanding window) train/test folds.

    Each fold trains on everything up to time t and tests on the next
    `horizon` days, then advances. This directly implements the validation
    methodology grounded in Bergmeir & Benitez (2012): standard k-fold CV is
    invalid for time series because it allows the model to "see the future"
    during training; walk-forward validation never does.
    """
    ordered = series.sort_values(date_col).reset_index(drop=True)
    n = len(ordered)
    # Reserve enough tail observations to produce at least `min_folds` folds
    # of size `horizon`, leaving a minimum initial training window.
    usable_for_folds = min_folds * horizon
    min_train_size = max(14, n - usable_for_folds)  # at least 2 weeks to start
    if min_train_size >= n:
        min_train_size = max(7, n // 2)

    folds = []
    cursor = min_train_size
    while cursor + horizon <= n and len(folds) < min_folds:
        train = ordered.iloc[:cursor]
        test = ordered.iloc[cursor: cursor + horizon]
        folds.append((train, test))
        cursor += horizon
    return folds
