"""
N-BEATS lightweight neural baseline (Oreshkin et al., ICLR 2020).

N-BEATS improved M4 competition accuracy by 11% over the statistical
benchmark without any domain-specific feature engineering, making it the
proposal's "interpretable, no feature engineering needed" neural baseline
(Table 5). Implemented via Nixtla's neuralforecast library as specified in
proposal Table 7 (Development Tools).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBEATS
    _NEURALFORECAST_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when the heavy dep is absent
    _NEURALFORECAST_AVAILABLE = False


class NBEATSModel:
    """Thin wrapper around neuralforecast's NBEATS so the rest of the
    benchmarking pipeline can treat it identically to the other four model
    classes (fit(series) -> predict(horizon))."""

    def __init__(self, horizon: int = 7, input_size: int | None = None, max_steps: int = 500):
        self.horizon = horizon
        self.input_size = input_size
        self.max_steps = max_steps
        self.nf: "NeuralForecast | None" = None
        self._fallback_mean = 0.0

    def fit(self, dates: pd.Series, y: pd.Series) -> "NBEATSModel":
        self._fallback_mean = float(y.mean()) if len(y) else 0.0
        n = len(y)
        # N-BEATS needs enough history to form at least one input window;
        # below that, the cold-start condition is too severe for a neural
        # model and we fall back to predicting the historical mean — this
        # degradation is itself a reportable finding at low density levels.
        min_required = (self.input_size or self.horizon * 2) + self.horizon
        if not _NEURALFORECAST_AVAILABLE or n < min_required:
            self.nf = None
            return self

        input_size = self.input_size or min(2 * self.horizon, n - self.horizon)
        df = pd.DataFrame({
            "unique_id": "series",
            "ds": pd.to_datetime(dates).values,
            "y": y.values,
        })
        # devices=1: without an explicit device count, PyTorch Lightning
        # auto-detects every visible GPU and, on a multi-GPU machine,
        # defaults to a distributed strategy that spawns worker
        # subprocesses. Spawning fails outright once CUDA is already
        # initialized in the parent process ("Lightning can't create new
        # processes if CUDA is already initialized") -- irrelevant for a
        # model this size anyway, so pin it to a single device.
        model = NBEATS(h=self.horizon, input_size=input_size, max_steps=self.max_steps, devices=1)
        self.nf = NeuralForecast(models=[model], freq="D")
        self.nf.fit(df=df)
        return self

    def predict(self) -> np.ndarray:
        if self.nf is None:
            return np.full(self.horizon, self._fallback_mean)
        forecast = self.nf.predict()
        preds = forecast["NBEATS"].values[: self.horizon]
        return np.clip(preds, a_min=0, a_max=None)
