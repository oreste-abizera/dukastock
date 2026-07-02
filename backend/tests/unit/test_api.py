"""
API-level tests through FastAPI's TestClient — previously the test suite
had zero coverage of the actual HTTP layer (router, webhook parsing,
request/response shapes), only unit tests that called service functions
directly with mocked dependencies. These exercise the same code paths a
real Twilio/Africa's Talking webhook or /sales client would hit.
"""
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.session import get_db
from app.main import app
from app.models.orm import Base, ShopkeeperProfile


@pytest.fixture()
def client():
    """Isolated in-memory SQLite DB per test, independent of whatever
    DATABASE_URL happens to be set in the environment."""
    # StaticPool: every checkout shares the one connection, so all requests
    # see the same in-memory DB. Without it, SQLAlchemy hands each request a
    # fresh connection, which SQLite maps to a brand-new (tableless) memory
    # database — "no such table" even though create_all ran successfully.
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_health_check(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_post_sale_creates_shopkeeper_and_log(client):
    response = client.post(
        "/api/v1/sales",
        json={
            "phone_number": "+250788111222",
            "product_code": "SUGAR",
            "quantity": 3.0,
            "unit": "kg",
            "channel": "whatsapp",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert "uuid" in body and "logged_at" in body


def test_post_sale_phone_number_never_appears_in_request_url(client):
    # Regression test: phone_number used to be a query/body-form param that
    # FastAPI treated as a query parameter, landing it in the request line
    # (and therefore access logs) before sales_service ever hashes it.
    with patch("app.api.v1.router.record_sale") as mock_record:
        mock_record.return_value.uuid = "fake-uuid"
        mock_record.return_value.logged_at = "2026-01-01T00:00:00"
        response = client.post(
            "/api/v1/sales",
            json={
                "phone_number": "+250788999888",
                "product_code": "OIL",
                "quantity": 1.0,
                "unit": "litre",
                "channel": "ussd",
            },
        )
    assert response.status_code == 200
    # The phone number must have travelled in the JSON body, not the URL.
    assert "788999888" not in str(response.request.url)


def test_forecast_endpoint_reports_no_model_when_artifact_missing(client, tmp_path):
    with patch("app.api.v1.router._forecast_service") as mock_service:
        mock_service.forecast.return_value = {
            "product_code": "SUGAR",
            "predicted_quantity": None,
            "model_used": "no_model",
            "status": "no_model_available",
        }
        response = client.get("/api/v1/forecast/SUGAR")
    assert response.status_code == 200
    assert response.json()["status"] == "no_model_available"


def test_whatsapp_webhook_returns_twiml(client):
    with patch("app.api.v1.router.handle_whatsapp_message", return_value="<Response></Response>"):
        response = client.post(
            "/api/v1/webhooks/whatsapp",
            data={"From": "whatsapp:+250788123456", "Body": "nagurishije isukari kilo bitatu"},
        )
    assert response.status_code == 200
    assert "<Response>" in response.text


@pytest.fixture()
def fake_redis_session_store():
    """In-memory stand-in for app.db.redis_client, matching the pattern
    already used in test_ussd_fsm.py — real Redis isn't a hard dependency
    for the test suite (CI provides one, but plain local `pytest` doesn't
    require it running)."""
    store: dict = {}
    with (
        patch("app.channels.ussd.fsm.load_ussd_session", side_effect=lambda sid: store.get(sid)),
        patch("app.channels.ussd.fsm.save_ussd_session", side_effect=lambda sid, data: store.__setitem__(sid, data)),
        patch("app.channels.ussd.fsm.clear_ussd_session", side_effect=lambda sid: store.pop(sid, None)),
    ):
        yield store


def test_ussd_webhook_initial_dial_returns_main_menu(client, fake_redis_session_store):
    response = client.post(
        "/api/v1/webhooks/ussd",
        data={"sessionId": "sess-api-test", "phoneNumber": "+250788123456", "text": ""},
    )
    assert response.status_code == 200
    assert response.text.startswith("CON ")
    assert "DukaStock" in response.text


def test_ussd_webhook_full_sale_flow_persists_shopkeeper(client, fake_redis_session_store):
    session_id = "sess-full-flow"
    phone = "+250788555444"
    client.post("/api/v1/webhooks/ussd", data={"sessionId": session_id, "phoneNumber": phone, "text": ""})
    client.post("/api/v1/webhooks/ussd", data={"sessionId": session_id, "phoneNumber": phone, "text": "1"})
    client.post("/api/v1/webhooks/ussd", data={"sessionId": session_id, "phoneNumber": phone, "text": "1*1"})
    final = client.post(
        "/api/v1/webhooks/ussd", data={"sessionId": session_id, "phoneNumber": phone, "text": "1*1*3"}
    )
    assert final.text.startswith("END ")
    assert "Murakoze" in final.text

    # Confirm a shopkeeper row actually exists — this is the real DB path,
    # not a mock, so it also proves get_or_create_shopkeeper + record_sale
    # committed successfully through the full HTTP -> FSM -> ORM chain.
    from app.core.security import hash_phone_number

    override = app.dependency_overrides[get_db]
    db = next(override())
    shopkeeper_id = hash_phone_number(phone)
    assert db.get(ShopkeeperProfile, shopkeeper_id) is not None
