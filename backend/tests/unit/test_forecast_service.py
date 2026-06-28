import numpy as np
import pandas as pd
import joblib
import pytest

from app.ml.models.naive import NaiveBaseline
from app.ml.models.serializable import SerializableForecastModel
from app.services.forecast_service import ForecastService


class _FakeNBEATSModelForPickling:
    """Defined at module level (not nested in a test function) so joblib/pickle
    can actually serialize it — pickle requires a class be importable by its
    qualified name, which a function-local class is not."""
    def predict(self):
        return np.full(7, 3.0)


@pytest.fixture
def artifact_dir(tmp_path):
    return str(tmp_path)


def test_forecast_returns_no_model_when_no_artifact(artifact_dir):
    service = ForecastService(artifact_dir=artifact_dir)
    result = service.forecast("NONEXISTENT")
    assert result["model_used"] == "no_model"
    assert result["status"] == "no_model_available"
    assert result["predicted_quantity"] is None


def test_forecast_loads_naive_artifact_correctly(artifact_dir, tmp_path):
    model = NaiveBaseline().fit(pd.Series([10.0] * 7))
    wrapped = SerializableForecastModel(kind="naive", model=model)
    joblib.dump(wrapped, tmp_path / "sugar_best_model.joblib")

    service = ForecastService(artifact_dir=artifact_dir)
    result = service.forecast("SUGAR", horizon_days=7)
    assert result["model_used"] == "naive"
    assert result["predicted_quantity"] == 70.0


def test_forecast_rejects_legacy_unwrapped_artifact(artifact_dir, tmp_path):
    """A raw model pickled directly (the old, buggy behavior) should be
    treated as unavailable rather than crash with an AttributeError deep
    inside whichever predict() signature doesn't match."""
    raw_model = NaiveBaseline().fit(pd.Series([10.0] * 7))
    joblib.dump(raw_model, tmp_path / "legacy_best_model.joblib")

    service = ForecastService(artifact_dir=artifact_dir)
    result = service.forecast("LEGACY", horizon_days=7)
    assert result["model_used"] == "no_model"
    assert result["predicted_quantity"] is None


def test_forecast_caches_loaded_model(artifact_dir, tmp_path):
    model = NaiveBaseline().fit(pd.Series([5.0] * 7))
    wrapped = SerializableForecastModel(kind="naive", model=model)
    artifact_path = tmp_path / "rice_best_model.joblib"
    joblib.dump(wrapped, artifact_path)

    service = ForecastService(artifact_dir=artifact_dir)
    service.forecast("RICE")
    assert "RICE" in service._cache

    # Delete the file on disk; cached result should still serve fine.
    artifact_path.unlink()
    result = service.forecast("RICE")
    assert result["model_used"] == "naive"


def test_forecast_handles_horizon_mismatch_gracefully(artifact_dir, tmp_path):
    wrapped = SerializableForecastModel(kind="nbeats", model=_FakeNBEATSModelForPickling())
    joblib.dump(wrapped, tmp_path / "soap_best_model.joblib")

    service = ForecastService(artifact_dir=artifact_dir)
    result = service.forecast("SOAP", horizon_days=14)  # model trained for 7
    assert "horizon_mismatch" in result["model_used"]
    assert result["predicted_quantity"] == 0.0
