import uuid

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.models.orm import Base, ChannelEnum, SalesLog
from app.services.sales_aggregation import get_daily_series

SHOP_1 = str(uuid.uuid4())
SHOP_2 = str(uuid.uuid4())


@pytest.fixture
def db():
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(autocommit=False, autoflush=False, bind=engine)()
    yield session
    session.close()


def _add_sale(db, shopkeeper_id, product_code, quantity, logged_at):
    db.add(
        SalesLog(
            shopkeeper_id=shopkeeper_id, product_code=product_code, quantity=quantity,
            unit="kg", channel=ChannelEnum.ussd, logged_at=logged_at,
        )
    )
    db.commit()


def test_returns_empty_frame_when_no_sales(db):
    result = get_daily_series(db, SHOP_1, "SUGAR")
    assert result.empty
    assert list(result.columns) == ["date", "sales"]


def test_sums_same_day_sales(db):
    day = pd.Timestamp("2026-06-01")
    _add_sale(db, SHOP_1, "SUGAR", 2.0, day)
    _add_sale(db, SHOP_1, "SUGAR", 3.0, day)

    result = get_daily_series(db, SHOP_1, "SUGAR")

    assert len(result) == 1
    assert result.iloc[0]["sales"] == 5.0


def test_zero_fills_gap_days(db):
    _add_sale(db, SHOP_1, "SUGAR", 2.0, pd.Timestamp("2026-06-01"))
    _add_sale(db, SHOP_1, "SUGAR", 4.0, pd.Timestamp("2026-06-04"))

    result = get_daily_series(db, SHOP_1, "SUGAR")

    assert len(result) == 4  # 06-01, 02, 03, 04
    assert result["sales"].tolist() == [2.0, 0.0, 0.0, 4.0]


def test_filters_by_shopkeeper_and_product(db):
    _add_sale(db, SHOP_1, "SUGAR", 2.0, pd.Timestamp("2026-06-01"))
    _add_sale(db, SHOP_2, "SUGAR", 9.0, pd.Timestamp("2026-06-01"))
    _add_sale(db, SHOP_1, "RICE", 9.0, pd.Timestamp("2026-06-01"))

    result = get_daily_series(db, SHOP_1, "SUGAR")

    assert len(result) == 1
    assert result.iloc[0]["sales"] == 2.0
