"""
DukaStock FastAPI application entry point.

A single backend serves all three channels (WhatsApp, USSD, SMS) through
one shared forecasting pipeline, per the proposal's modular,
channel-agnostic architecture (Chapter 3.3).
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import router as v1_router
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.models.orm import Base
from app.db.session import engine

settings = get_settings()
configure_logging()
logger = get_logger(__name__)

app = FastAPI(
    title=settings.app_name,
    description=(
        "Understands how Rwandan Duka shopkeepers text about their sales "
        "(Kinyarwanda-English commerce NER) and forecasts demand around it."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.environment != "production" else [],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(v1_router, prefix=settings.api_v1_prefix)


@app.on_event("startup")
def on_startup():
    # Local dev / CI convenience: create tables on SQLite automatically.
    # Production (Supabase/Postgres) is migrated via Alembic instead — see
    # backend/alembic.ini and docs/ARCHITECTURE.md.
    if settings.database_url.startswith("sqlite"):
        Base.metadata.create_all(bind=engine)
    logger.info("dukastock_startup", environment=settings.environment)


@app.get("/")
def root():
    return {"service": settings.app_name, "status": "running"}
