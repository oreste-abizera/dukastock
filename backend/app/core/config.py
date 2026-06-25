"""
Application configuration.

All values are loaded from environment variables (see config/.env.example).
No secrets are hard-coded. Settings are validated at import time via Pydantic
so misconfiguration fails fast at startup rather than at first request.
"""
from __future__ import annotations

import os
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_ner_model_dir() -> str:
    # The fine-tuned XLM-R artifact lives in the ml_experiments tree
    # (trained by Notebook 03 / train_xlmr_ner.py).  The backend's own
    # app/nlp/xlmr_commerce_ner/ directory is intentionally empty
    # (placeholder .gitkeep) because the 1.1 GB safetensors file is not
    # committed to git.  At runtime we resolve to the actual artifact
    # directory, falling back to the backend-relative path so any
    # NER_MODEL_DIR env override still works.
    candidates = [
        "ml_experiments/notebooks/xlmr_commerce_ner_output",
        "app/nlp/xlmr_commerce_ner",
    ]
    for c in candidates:
        if os.path.isfile(os.path.join(c, "config.json")):
            return c
    return candidates[-1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- App ---
    app_name: str = "DukaStock"
    environment: str = "development"
    debug: bool = False
    api_v1_prefix: str = "/api/v1"

    # --- Database (Supabase / PostgreSQL) ---
    supabase_url: str = ""
    supabase_key: str = ""
    database_url: str = "sqlite:///./dukastock_dev.db"

    # --- Redis (Upstash) — USSD session persistence ---
    redis_url: str = "redis://localhost:6379/0"
    ussd_session_ttl_seconds: int = 180  # matches MTN/Airtel Rwanda USSD timeout window

    # --- Twilio (WhatsApp sandbox) ---
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_whatsapp_number: str = "whatsapp:+14155238886"  # Twilio sandbox default

    # --- Africa's Talking (USSD + SMS, Rwanda: MTN & Airtel) ---
    at_username: str = "sandbox"
    at_api_key: str = ""
    at_ussd_code: str = "*384*00#"
    at_sender_id: str = "DukaStock"

    # --- Security / privacy (Rwanda Law No. 058/2021) ---
    phone_hash_salt: str = "change-me-in-production"

    # --- Channel response constraints ---
    ussd_max_chars: int = 182
    sms_max_chars: int = 160

    # --- ML ---
    model_artifact_dir: str = "ml_experiments/artifacts"
    forecast_horizon_days: int = 7

    # --- NLP ---
    ner_model_dir: str = ""  # resolved at startup by _default_ner_model_dir()
    ner_confidence_threshold: float = 0.55

    def model_post_init(self, __context: object) -> None:
        if not self.ner_model_dir:
            object.__setattr__(self, "ner_model_dir", _default_ner_model_dir())
        if (
            self.phone_hash_salt == "change-me-in-production"
            and self.environment != "development"
        ):
            raise ValueError(
                "PHONE_HASH_SALT must be set to a random secret in non-development "
                "environments (Rwanda Law No. 058/2021 compliance requirement). "
                "Set PHONE_HASH_SALT in your .env or Railway environment variables."
            )


@lru_cache
def get_settings() -> Settings:
    return Settings()
