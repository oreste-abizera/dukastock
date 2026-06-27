"""
API v1 router.

Five endpoints, matching the channels and use cases in the proposal:
  POST /webhooks/whatsapp   - Twilio inbound message webhook
  POST /webhooks/ussd       - Africa's Talking inbound USSD webhook
  GET  /forecast/{product}  - direct forecast lookup (used by SMS job + tests)
  POST /sales                - direct sale recording (used by integration tests)
  GET  /health                - liveness/readiness probe for Railway.app
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Form
from fastapi.responses import PlainTextResponse
from sqlalchemy.orm import Session

from app.channels.ussd.fsm import handle_ussd_request
from app.channels.whatsapp.handler import handle_whatsapp_message
from app.db.session import get_db
from app.models.orm import ChannelEnum
from app.services.forecast_service import ForecastService
from app.services.sales_service import record_sale

router = APIRouter()
_forecast_service = ForecastService()


@router.get("/health")
def health_check():
    return {"status": "ok", "service": "dukastock-backend"}


@router.post("/webhooks/whatsapp", response_class=PlainTextResponse)
def whatsapp_webhook(
    From: str = Form(...),
    Body: str = Form(...),
    db: Session = Depends(get_db),
):
    twiml = handle_whatsapp_message(db, from_number=From, body=Body)
    return PlainTextResponse(content=twiml, media_type="application/xml")


@router.post("/webhooks/ussd", response_class=PlainTextResponse)
def ussd_webhook(
    sessionId: str = Form(...),
    phoneNumber: str = Form(...),
    text: str = Form(""),
    serviceCode: str = Form(""),   # Africa's Talking also sends these; accept
    networkCode: str = Form(""),   # them to avoid 422 on real AT webhooks
    db: Session = Depends(get_db),
):
    response = handle_ussd_request(db, session_id=sessionId, phone_number=phoneNumber, text=text)
    return PlainTextResponse(content=response.render())


@router.get("/forecast/{product_code}")
def get_forecast(product_code: str, horizon_days: int | None = None):
    return _forecast_service.forecast(product_code.upper(), horizon_days)


@router.post("/sales")
def post_sale(
    phone_number: str,
    product_code: str,
    quantity: float,
    unit: str,
    channel: ChannelEnum,
    db: Session = Depends(get_db),
):
    log = record_sale(db, raw_phone=phone_number, product_code=product_code, quantity=quantity, unit=unit, channel=channel)
    return {"uuid": log.uuid, "logged_at": log.logged_at}
