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

from app.channels.messages import sale_logged_message, sale_not_understood_message
from app.core.logging import get_logger
from app.models.orm import ChannelEnum, NLPParseResult
from app.nlp.ner_pipeline import CommerceNERPipeline
from app.services.forecast_service import ForecastService
from app.services.sales_service import get_or_create_shopkeeper, record_sale

logger = get_logger(__name__)
_ner_pipeline = CommerceNERPipeline()
_forecast_service = ForecastService()


def _log_nlp_parse(db: Session, shopkeeper_id: str, raw_message: str, entity=None) -> None:
    """Persist every parse attempt (including failed ones) for later NER
    error analysis — this table previously existed in the schema but was
    never written to."""
    db.add(
        NLPParseResult(
            shopkeeper_id=shopkeeper_id,
            raw_message=raw_message,
            product_name=entity.product_name if entity else None,
            quantity=entity.quantity if entity else None,
            unit=entity.unit if entity else None,
            confidence=entity.confidence if entity else 0.0,
        )
    )
    db.commit()


def handle_whatsapp_message(db: Session, from_number: str, body: str) -> str:
    """Process an inbound WhatsApp message and return TwiML response text."""
    shopkeeper = get_or_create_shopkeeper(db, from_number, ChannelEnum.whatsapp)
    parsed = _ner_pipeline.parse(body)
    resp = MessagingResponse()

    if not parsed:
        _log_nlp_parse(db, shopkeeper.uuid, body)
        resp.message(sale_not_understood_message())
        return str(resp)

    confirmations = []
    for entity in parsed:
        _log_nlp_parse(db, shopkeeper.uuid, body, entity)
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
            sale_logged_message(entity.product_name, entity.quantity, entity.unit or "")
        )

    if not confirmations:
        resp.message(sale_not_understood_message())
    else:
        resp.message(" ".join(confirmations))

    return str(resp)
