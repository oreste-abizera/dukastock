import uuid
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.ml.evaluation.metrics import DMTestResult
from app.models.orm import Base, ChannelEnum, ForecastResult, SalesLog
from app.services.personalized_forecast_service import personalized_artifact_path, train_for_shopkeeper

SHOPKEEPER_ID = str(uuid.uuid4())


@pytest.fixture
def db():
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(autocommit=False, autoflush=False, bind=engine)()
    yield session
    session.close()


def _seed_sales(db, n_days: int, start="2026-01-01"):
    """A mildly-seasonal, noisy synthetic history -- enough variation that
    fitting XGBoost/naive on it doesn't hit any degenerate (zero-variance)
    edge case, matching the style of test_xgboost_grid_search.py's fixture."""
    dates = pd.date_range(start, periods=n_days, freq="D")
    rng = np.random.default_rng(0)
    sales = np.clip(10 + 3 * np.sin(2 * np.pi * dates.dayofweek / 7) + rng.normal(0, 1.0, n_days), 0, None)
    for d, qty in zip(dates, sales):
        db.add(
            SalesLog(
                shopkeeper_id=SHOPKEEPER_ID, product_code="SUGAR", quantity=float(qty),
                unit="kg", channel=ChannelEnum.ussd, logged_at=d,
            )
        )
    db.commit()


def test_skips_when_no_sales_logged(db, tmp_path):
    result = train_for_shopkeeper(db, SHOPKEEPER_ID, "SUGAR", str(tmp_path))

    assert result == "skipped (no sales logged yet)"
    assert not personalized_artifact_path(str(tmp_path), SHOPKEEPER_ID, "SUGAR").exists()
    assert db.query(ForecastResult).count() == 0


def test_too_little_history_serves_naive(db, tmp_path):
    _seed_sales(db, n_days=5)

    result = train_for_shopkeeper(db, SHOPKEEPER_ID, "SUGAR", str(tmp_path))

    assert result == "naive"
    assert personalized_artifact_path(str(tmp_path), SHOPKEEPER_ID, "SUGAR").exists()
    saved = db.query(ForecastResult).one()
    assert saved.model_used == "naive"
    assert saved.shopkeeper_id == SHOPKEEPER_ID
    assert saved.product_code == "SUGAR"


@patch("app.services.personalized_forecast_service.diebold_mariano_test")
def test_significant_xgboost_result_is_persisted_as_xgboost(mock_dm, db, tmp_path):
    mock_dm.return_value = DMTestResult(
        dm_statistic=3.0, p_value=0.01, significant_at_05=True, mean_loss_differential=1.0
    )
    _seed_sales(db, n_days=90)

    result = train_for_shopkeeper(db, SHOPKEEPER_ID, "SUGAR", str(tmp_path))

    assert result == "xgboost"
    model = joblib.load(personalized_artifact_path(str(tmp_path), SHOPKEEPER_ID, "SUGAR"))
    assert model.kind == "xgboost"
    saved = db.query(ForecastResult).one()
    assert saved.model_used == "xgboost"


@patch("app.services.personalized_forecast_service.diebold_mariano_test")
def test_insignificant_result_falls_back_to_naive(mock_dm, db, tmp_path):
    mock_dm.return_value = DMTestResult(
        dm_statistic=0.1, p_value=0.9, significant_at_05=False, mean_loss_differential=0.0
    )
    _seed_sales(db, n_days=90)

    result = train_for_shopkeeper(db, SHOPKEEPER_ID, "SUGAR", str(tmp_path))

    assert result == "naive"
    model = joblib.load(personalized_artifact_path(str(tmp_path), SHOPKEEPER_ID, "SUGAR"))
    assert model.kind == "naive"
