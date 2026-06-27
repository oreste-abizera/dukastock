"""
Outbound SMS handler (Africa's Talking).

Delivers weekly restock summaries to shopkeepers who have no WhatsApp
access and prefer not to dial the USSD menu proactively — the "nudge"
channel described in the proposal's interface stack. Messages are
truncated to settings.sms_max_chars (160), the GSM-7 single-segment SMS
limit, to keep delivery costs predictable on the Africa's Talking sandbox.
"""
from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.forecast_service import ForecastService

logger = get_logger(__name__)
settings = get_settings()
_forecast_service = ForecastService()


def build_weekly_summary_sms(shopkeeper_locale_is_kinyarwanda: bool, product_codes: list[str]) -> str:
    lines = []
    for code in product_codes:
        forecast = _forecast_service.forecast(code)
        qty = forecast.get("predicted_quantity")
        if qty is None:
            qty_str = "N/A"
        else:
            qty_str = str(round(qty, 1))
        if shopkeeper_locale_is_kinyarwanda:
            lines.append(f"{code}: {qty_str}")
        else:
            lines.append(f"{code}: {qty_str} predicted")

    header = "DukaStock - icyumweru gitaha:" if shopkeeper_locale_is_kinyarwanda else "DukaStock - next week:"
    message = header + " " + "; ".join(lines)
    return message[: settings.sms_max_chars]


def send_sms(client, to_number: str, message: str) -> dict:
    """Thin wrapper around the Africa's Talking SMS client so callers (and
    tests) don't need to import the SDK directly."""
    try:
        response = client.SMS.send(message, [to_number], sender_id=settings.at_sender_id)
        return {"status": "sent", "raw": response}
    except Exception as exc:  # pragma: no cover - network dependent
        logger.error("sms_send_failed", error=str(exc), to=to_number)
        return {"status": "failed", "error": str(exc)}
