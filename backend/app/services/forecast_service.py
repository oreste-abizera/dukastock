"""
Forecast service.

Loads joblib-serialized per-product models (trained offline by
ml_experiments/scripts/run_experiment.py) and serves predictions to the
API/channel layer. This is the bridge between the research pipeline
(notebooks + run_experiment.py, which decide WHICH model wins at which
density) and the live prototype (which just needs to answer "how much
sugar will I sell next week?" in under 3 seconds for the USSD SLA).

Every artifact loaded here is expected to be a
app.ml.models.serializable.SerializableForecastModel, not a raw
NaiveBaseline/SARIMAModel/ProphetModel/XGBoostDemandModel/NBEATSModel
instance. The five underlying model classes have genuinely different
predict() signatures (see serializable.py's module docstring), and
SerializableForecastModel.predict_with_band() is what normalizes them into
one call this service can make without caring which model won.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import joblib

from app.core.config import get_settings
from app.core.logging import get_logger
from app.ml.models.serializable import SerializableForecastModel
from app.services.personalized_forecast_service import personalized_artifact_path

logger = get_logger(__name__)
settings = get_settings()

_MAX_PERSONALIZED_CACHE_ENTRIES = 500


class ForecastService:
    def __init__(self, artifact_dir: str | None = None):
        self.artifact_dir = Path(artifact_dir or settings.model_artifact_dir)
        self._cache: dict[str, SerializableForecastModel] = {}
        # Separate, bounded cache: unlike the 5 global per-product models
        # above, there's one personalized model per (shopkeeper, product),
        # so keeping every one in memory forever isn't viable. FIFO eviction
        # via OrderedDict is enough here -- this isn't a hot enough path to
        # need real LRU bookkeeping.
        self._personalized_cache: OrderedDict[tuple[str, str], SerializableForecastModel] = OrderedDict()

    def _model_path(self, product_code: str) -> Path:
        return self.artifact_dir / f"{product_code.lower()}_best_model.joblib"

    def _load_model(self, product_code: str) -> SerializableForecastModel | None:
        if product_code in self._cache:
            return self._cache[product_code]
        path = self._model_path(product_code)
        if not path.exists():
            logger.warning("model_artifact_missing", product_code=product_code, path=str(path))
            return None
        model = joblib.load(path)
        if not isinstance(model, SerializableForecastModel):
            # Defensive check for stale artifacts saved before this wrapper
            # existed (a raw model object pickled directly). Treat as
            # missing rather than crashing the request on an AttributeError
            # deep inside whichever predict() signature happens not to match.
            logger.warning("model_artifact_legacy_format", product_code=product_code, path=str(path))
            return None
        self._cache[product_code] = model
        return model

    def _load_personalized_model(self, shopkeeper_id: str, product_code: str) -> SerializableForecastModel | None:
        key = (shopkeeper_id, product_code)
        if key in self._personalized_cache:
            return self._personalized_cache[key]
        path = personalized_artifact_path(str(self.artifact_dir), shopkeeper_id, product_code)
        if not path.exists():
            return None
        model = joblib.load(path)
        if not isinstance(model, SerializableForecastModel):
            logger.warning(
                "personalized_model_legacy_format", shopkeeper_id=shopkeeper_id, product_code=product_code
            )
            return None
        if len(self._personalized_cache) >= _MAX_PERSONALIZED_CACHE_ENTRIES:
            self._personalized_cache.popitem(last=False)
        self._personalized_cache[key] = model
        return model

    def forecast(self, product_code: str, horizon_days: int | None = None) -> dict:
        horizon_days = horizon_days or settings.forecast_horizon_days
        model = self._load_model(product_code)

        if model is None:
            # No trained artifact yet for this product.  Return a clearly
            # labelled "no_model" status so callers (channel handlers, tests,
            # the demo operator) can distinguish "model says zero demand" from
            # "no model exists yet".  predicted_quantity is None (not 0.0)
            # so formatters render a human-readable placeholder instead of an
            # authoritative-looking number.
            return {
                "product_code": product_code,
                "predicted_quantity": None,
                "lower_bound": None,
                "upper_bound": None,
                "model_used": "no_model",
                "status": "no_model_available",
                "message": (
                    "Forecast model not yet trained for this product. "
                    "Run ml_experiments/scripts/run_experiment.py first."
                ),
                "horizon_days": horizon_days,
            }

        return self._build_result(product_code, model, horizon_days)

    def forecast_for_shopkeeper(
        self, shopkeeper_id: str, product_code: str, horizon_days: int | None = None
    ) -> dict:
        """Same as forecast(), but prefers a model trained on this specific
        shopkeeper's own sales history (see
        app.services.personalized_forecast_service.train_for_shopkeeper)
        when one exists. Falls back to the shared global per-product model
        otherwise, so a brand-new shopkeeper -- or one the nightly batch job
        hasn't reached yet -- still gets a forecast, just not a personalized
        one."""
        horizon_days = horizon_days or settings.forecast_horizon_days
        model = self._load_personalized_model(shopkeeper_id, product_code)
        if model is not None:
            return self._build_result(product_code, model, horizon_days)
        return self.forecast(product_code, horizon_days)

    def _build_result(self, product_code: str, model: SerializableForecastModel, horizon_days: int) -> dict:
        try:
            band = model.predict_with_band(horizon_days)
        except ValueError as exc:
            # E.g. an N-BEATS artifact whose fixed training horizon doesn't
            # match what was requested. Log it and degrade rather than 500.
            logger.warning("forecast_horizon_mismatch", product_code=product_code, error=str(exc))
            return {
                "product_code": product_code,
                "predicted_quantity": 0.0,
                "lower_bound": 0.0,
                "upper_bound": 0.0,
                "model_used": f"{model.kind}_horizon_mismatch",
                "horizon_days": horizon_days,
            }

        return {
            "product_code": product_code,
            "predicted_quantity": round(band["predicted_quantity"], 2),
            "lower_bound": round(band["lower_bound"], 2),
            "upper_bound": round(band["upper_bound"], 2),
            "model_used": model.kind,
            "horizon_days": horizon_days,
        }
