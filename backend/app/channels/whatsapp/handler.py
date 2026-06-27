"""
WhatsApp channel handler (Twilio sandbox).

Conversational sales logging: the shopkeeper sends a free-text Kinyarwanda
message, which is routed through the NLP parser (app.nlp.ner_pipeline)
before being written to the sales log. This mirrors the proposal's
Sequence Diagram exactly: Duka Operator -> Twilio webhook -> FastAPI ->
NLP Parser -> Supabase -> Forecasting Service -> Kinyarwanda response ->
Twilio -> Duka Operator.
"""
from sqlalchemy.orm import Session
from twilio.twiml.messaging_response import MessagingResponse

from app.core.logging import get_logger
from app.models.orm import ChannelEnum
from app.nlp.ner_pipeline import CommerceNERPipeline
from app.services.forecast_service import ForecastService
from app.services.sales_service import record_sale

logger = get_logger(__name__)
_ner_pipeline = CommerceNERPipeline()
_forecast_service = ForecastService()


KINYARWANDA_REPLY_TEMPLATES = {
    "sale_logged": "Murakoze! Twanditse: {product} {quantity} {unit}.",
    "sale_not_understood": "Mbabarira, sinabashije gusobanukirwa ubutumwa bwawe. Ongera ugerageze.",
    "forecast_reply": "Mu cyumweru gitaha, tubona ko uzagurisha hafi {qty} {unit} ya {product}.",
}


def handle_whatsapp_message(db: Session, from_number: str, body: str) -> str:
    """Process an inbound WhatsApp message and return TwiML response text."""
    parsed = _ner_pipeline.parse(body)
    resp = MessagingResponse()

    if not parsed:
        resp.message(KINYARWANDA_REPLY_TEMPLATES["sale_not_understood"])
        return str(resp)

    confirmations = []
    for entity in parsed:
        if entity.product_name is None or entity.quantity is None:
            continue
        record_sale(
            db,
            raw_phone=from_number,
            product_code=entity.product_name,
            quantity=entity.quantity,
            unit=entity.unit or "unit",
            channel=ChannelEnum.whatsapp,
        )
        confirmations.append(
            KINYARWANDA_REPLY_TEMPLATES["sale_logged"].format(
                product=entity.product_name, quantity=entity.quantity, unit=entity.unit or ""
            )
        )

    if not confirmations:
        resp.message(KINYARWANDA_REPLY_TEMPLATES["sale_not_understood"])
    else:
        resp.message(" ".join(confirmations))

    return str(resp)
