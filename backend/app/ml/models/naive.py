"""
Naive last-week-sales baseline.

This is H0 in the Diebold-Mariano test: "this week's demand = same weekday's
demand from last week". It requires zero training data beyond one prior
cycle and is always available, which is exactly why it is the right floor
for a cold-start study — every other model has to beat THIS, not zero.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class NaiveBaseline:
    """Predicts demand for a given day as equal to demand on the same
    weekday one week prior. Falls back to the overall mean of available
    history when fewer than 7 observations exist (extreme cold-start)."""

    def __init__(self):
        self.history: pd.Series | None = None

    def fit(self, y: pd.Series) -> "NaiveBaseline":
        self.history = y.reset_index(drop=True)
        return self

    def predict(self, n_periods: int) -> np.ndarray:
        if self.history is None or len(self.history) == 0:
            return np.zeros(n_periods)
        if len(self.history) < 7:
            mean_val = float(self.history.mean())
            return np.full(n_periods, mean_val)
        last_week = self.history.iloc[-7:].values
        # Tile the last observed week to cover the requested horizon.
        reps = int(np.ceil(n_periods / 7))
        tiled = np.tile(last_week, reps)
        return tiled[:n_periods]
