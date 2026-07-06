"""
Sales service.

Single entry point for recording a sale regardless of which channel it
arrived through (WhatsApp via NLP parse, or USSD via direct FSM menu
selection — see proposal's Use Case Diagram note: "WhatsApp input routes
through the NLP parser while USSD input routes through the FSM menu
directly to the sales log without NLP processing").
"""
from datetime import datetime

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.security import hash_phone_number
from app.models.orm import ChannelEnum, SalesLog, ShopkeeperProfile


def get_or_create_shopkeeper(db: Session, raw_phone: str, channel: ChannelEnum) -> ShopkeeperProfile:
    shopkeeper_id = hash_phone_number(raw_phone)
    shopkeeper = db.get(ShopkeeperProfile, shopkeeper_id)
    if shopkeeper is not None:
        return shopkeeper

    shopkeeper = ShopkeeperProfile(uuid=shopkeeper_id, channel_preference=channel)
    db.add(shopkeeper)
    try:
        db.commit()
    except IntegrityError:
        # Two concurrent first-time requests from the same phone number both
        # passed the check above; whichever committed first wins the row.
        db.rollback()
        shopkeeper = db.get(ShopkeeperProfile, shopkeeper_id)
        if shopkeeper is None:
            raise
    return shopkeeper


def get_recent_sales(db: Session, shopkeeper_id: str, limit: int = 3) -> list[SalesLog]:
    return (
        db.query(SalesLog)
        .filter(SalesLog.shopkeeper_id == shopkeeper_id)
        .order_by(SalesLog.logged_at.desc())
        .limit(limit)
        .all()
    )


def record_sale(
    db: Session,
    raw_phone: str,
    product_code: str,
    quantity: float,
    unit: str,
    channel: ChannelEnum,
) -> SalesLog:
    shopkeeper = get_or_create_shopkeeper(db, raw_phone, channel)
    log = SalesLog(
        shopkeeper_id=shopkeeper.uuid,
        product_code=product_code,
        quantity=quantity,
        unit=unit,
        channel=channel,
        logged_at=datetime.utcnow(),
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return log
