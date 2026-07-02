"""
ORM models.

These mirror the class diagram in Chapter 3 of the capstone proposal
exactly: ShopkeeperProfile, SalesLog, ForecastResult, NLPParseResult,
USSDSession. shopkeeper_id is always a UUID derived from a hashed phone
number (see app.core.security.hash_phone_number) — raw phone numbers are
never persisted, per Rwanda's Law No. 058/2021.
"""
from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, Enum, Float, ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def _uuid_column(primary_key: bool = False, foreign_key: str | None = None):
    """SQLite (used in local dev/tests) has no native UUID type, so we store
    UUIDs as 36-char strings there while using the native PG UUID type in
    production Postgres/Supabase. SQLAlchemy's UUID(as_uuid=False) gives us
    one column definition that works against both backends."""
    kwargs = {"default": lambda: str(uuid.uuid4())} if primary_key else {}
    if foreign_key:
        return Column(UUID(as_uuid=False), ForeignKey(foreign_key), nullable=False)
    return Column(UUID(as_uuid=False), primary_key=primary_key, **kwargs)


class ChannelEnum(str, enum.Enum):
    whatsapp = "whatsapp"
    ussd = "ussd"
    sms = "sms"


class LocaleEnum(str, enum.Enum):
    kinyarwanda = "kinyarwanda"
    english = "english"


class ShopkeeperProfile(Base):
    __tablename__ = "shopkeeper_profiles"

    uuid = _uuid_column(primary_key=True)
    channel_preference = Column(Enum(ChannelEnum), nullable=False, default=ChannelEnum.ussd)
    locale = Column(Enum(LocaleEnum), nullable=False, default=LocaleEnum.kinyarwanda)
    created_at = Column(DateTime, default=datetime.utcnow)

    sales_logs = relationship("SalesLog", back_populates="shopkeeper")
    forecast_results = relationship("ForecastResult", back_populates="shopkeeper")
    ussd_sessions = relationship("USSDSession", back_populates="shopkeeper")
    nlp_parse_results = relationship("NLPParseResult", back_populates="shopkeeper")


class SalesLog(Base):
    __tablename__ = "sales_logs"

    uuid = _uuid_column(primary_key=True)
    shopkeeper_id = _uuid_column(foreign_key="shopkeeper_profiles.uuid")
    product_code = Column(String, nullable=False)
    quantity = Column(Float, nullable=False)
    unit = Column(String, nullable=False)
    channel = Column(Enum(ChannelEnum), nullable=False)
    logged_at = Column(DateTime, default=datetime.utcnow)

    shopkeeper = relationship("ShopkeeperProfile", back_populates="sales_logs")


class ForecastResult(Base):
    __tablename__ = "forecast_results"

    uuid = _uuid_column(primary_key=True)
    shopkeeper_id = _uuid_column(foreign_key="shopkeeper_profiles.uuid")
    product_code = Column(String, nullable=False)
    predicted_quantity = Column(Float, nullable=False)
    lower_bound = Column(Float, nullable=False)
    upper_bound = Column(Float, nullable=False)
    model_used = Column(String, nullable=False)
    data_density_pct = Column(Float, nullable=False)
    generated_at = Column(DateTime, default=datetime.utcnow)

    shopkeeper = relationship("ShopkeeperProfile", back_populates="forecast_results")


class NLPParseResult(Base):
    __tablename__ = "nlp_parse_results"

    uuid = _uuid_column(primary_key=True)
    shopkeeper_id = _uuid_column(foreign_key="shopkeeper_profiles.uuid")
    raw_message = Column(String, nullable=False)
    product_name = Column(String, nullable=True)
    quantity = Column(Float, nullable=True)
    unit = Column(String, nullable=True)
    confidence = Column(Float, nullable=False)
    parsed_at = Column(DateTime, default=datetime.utcnow)

    shopkeeper = relationship("ShopkeeperProfile", back_populates="nlp_parse_results")


class USSDSession(Base):
    __tablename__ = "ussd_sessions"

    session_id = Column(String, primary_key=True)
    shopkeeper_id = _uuid_column(foreign_key="shopkeeper_profiles.uuid")
    state = Column(String, nullable=False, default="MAIN_MENU")
    last_active = Column(DateTime, default=datetime.utcnow)

    shopkeeper = relationship("ShopkeeperProfile", back_populates="ussd_sessions")
