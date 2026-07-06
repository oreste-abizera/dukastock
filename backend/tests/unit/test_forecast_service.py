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


def test_forecast_for_shopkeeper_falls_back_to_global_when_no_personalized_artifact(artifact_dir, tmp_path):
    global_model = SerializableForecastModel(kind="naive", model=NaiveBaseline().fit(pd.Series([10.0] * 7)))
    joblib.dump(global_model, tmp_path / "sugar_best_model.joblib")

    service = ForecastService(artifact_dir=artifact_dir)
    result = service.forecast_for_shopkeeper("shopkeeper-1", "SUGAR", horizon_days=7)

    assert result["model_used"] == "naive"
    assert result["predicted_quantity"] == 70.0


def test_forecast_for_shopkeeper_prefers_personalized_artifact(artifact_dir, tmp_path):
    joblib.dump(
        SerializableForecastModel(kind="naive", model=NaiveBaseline().fit(pd.Series([10.0] * 7))),
        tmp_path / "sugar_best_model.joblib",
    )
    personalized_dir = tmp_path / "personalized"
    personalized_dir.mkdir()
    joblib.dump(
        SerializableForecastModel(kind="naive", model=NaiveBaseline().fit(pd.Series([50.0] * 7))),
        personalized_dir / "shopkeeper-1_sugar_best_model.joblib",
    )

    service = ForecastService(artifact_dir=artifact_dir)
    result = service.forecast_for_shopkeeper("shopkeeper-1", "SUGAR", horizon_days=7)

    assert result["predicted_quantity"] == 350.0  # 50 * 7, not the global model's 10 * 7


def test_forecast_for_shopkeeper_caches_are_bounded_and_evict_oldest(artifact_dir, tmp_path):
    personalized_dir = tmp_path / "personalized"
    personalized_dir.mkdir()
    service = ForecastService(artifact_dir=artifact_dir)
    from app.services.forecast_service import _MAX_PERSONALIZED_CACHE_ENTRIES
    service._personalized_cache["evict-me"] = "placeholder"  # inserted first -> oldest
    for i in range(_MAX_PERSONALIZED_CACHE_ENTRIES - 1):
        service._personalized_cache[f"key-{i}"] = "placeholder"
    assert len(service._personalized_cache) == _MAX_PERSONALIZED_CACHE_ENTRIES  # cache is exactly at cap

    joblib.dump(
        SerializableForecastModel(kind="naive", model=NaiveBaseline().fit(pd.Series([1.0] * 7))),
        personalized_dir / "shopkeeper-new_sugar_best_model.joblib",
    )
    service.forecast_for_shopkeeper("shopkeeper-new", "SUGAR", horizon_days=7)

    assert "evict-me" not in service._personalized_cache
    assert len(service._personalized_cache) == _MAX_PERSONALIZED_CACHE_ENTRIES
