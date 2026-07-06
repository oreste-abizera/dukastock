"""
Batch job: train each shopkeeper's own personalized forecast model per
product, using app.services.personalized_forecast_service.

Run periodically (e.g. nightly, via Coolify's Scheduled Tasks) against the
running container:

    python scripts/train_personalized_forecasts.py

Idempotent -- safe to re-run on any schedule. Shopkeepers/products with no
logged sales yet are skipped; ForecastService.forecast_for_shopkeeper falls
back to the shared global model until a shopkeeper's first personalized
artifact exists, so skipping here never removes a forecast a shopkeeper
already had.
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.db.session import db_session  # noqa: E402
from app.ml.pipeline.rwanda_features import FMCG_PRODUCT_MAP  # noqa: E402
from app.models.orm import ShopkeeperProfile  # noqa: E402
from app.services.personalized_forecast_service import train_for_shopkeeper  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_personalized_forecasts")

PRODUCT_CODES = [meta["code"] for meta in FMCG_PRODUCT_MAP.values()]


def main() -> None:
    settings = get_settings()
    with db_session() as db:
        shopkeepers = db.query(ShopkeeperProfile).all()
        logger.info(
            "starting personalized training for %d shopkeepers x %d products",
            len(shopkeepers), len(PRODUCT_CODES),
        )
        for shopkeeper in shopkeepers:
            for product_code in PRODUCT_CODES:
                result = train_for_shopkeeper(db, shopkeeper.uuid, product_code, settings.model_artifact_dir)
                logger.info("[%s/%s] %s", shopkeeper.uuid, product_code, result)


if __name__ == "__main__":
    main()
