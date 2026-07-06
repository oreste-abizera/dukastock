"""
Daily-series aggregation from SalesLog, for per-shopkeeper personalized
forecasting.

SalesLog rows are individual transactions at arbitrary timestamps -- not a
pre-aggregated daily series like the Kaggle benchmark data the offline
models (app.ml.models.*) were validated against. This module bridges that
gap so a shopkeeper's own history can be fed into the same feature
engineering (app.ml.pipeline.rwanda_features) and evaluation
(app.ml.pipeline.cold_start) code the offline experiment already uses.
"""
from __future__ import annotations

import pandas as pd
from sqlalchemy.orm import Session

from app.models.orm import SalesLog


def get_daily_series(db: Session, shopkeeper_id: str, product_code: str) -> pd.DataFrame:
    """Return a complete daily (date, sales) series for one shopkeeper's
    product: same-day transactions summed, gap days zero-filled.

    Zero-filling treats a day with no SalesLog row as "sold zero," not "the
    operator forgot to log" -- a documented simplification. Early in
    USSD/WhatsApp adoption, inconsistent logging is at least as likely an
    explanation for a gap as genuine zero sales; revisit once real logging
    habits can be observed, the same way docs/RESEARCH_DESIGN.md flags
    other proxy-data limitations rather than leaving them implicit.

    Returns an empty (date, sales) frame if the shopkeeper has no logged
    sales for this product yet.
    """
    rows = (
        db.query(SalesLog)
        .filter(SalesLog.shopkeeper_id == shopkeeper_id, SalesLog.product_code == product_code)
        .all()
    )
    if not rows:
        return pd.DataFrame(columns=["date", "sales"])

    raw = pd.DataFrame({"date": [r.logged_at.date() for r in rows], "sales": [r.quantity for r in rows]})
    daily = raw.groupby("date", as_index=False)["sales"].sum()
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)

    full_range = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    daily = (
        daily.set_index("date")
        .reindex(full_range, fill_value=0.0)
        .rename_axis("date")
        .reset_index()
    )
    return daily
