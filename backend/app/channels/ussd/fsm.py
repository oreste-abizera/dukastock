"""
USSD finite-state-machine menu handler (Africa's Talking sandbox, MTN/Airtel
Rwanda).

USSD bypasses NLP entirely (per the proposal's Use Case Diagram note) —
the shopkeeper navigates a structured Kinyarwanda menu by pressing digits,
so there is no free text to parse. This is the inclusive fallback channel
for the 66% of Rwandan households (EICV7, 2025) without a smartphone.

Africa's Talking calls our webhook on every single keypress with the FULL
accumulated input string (e.g. "1*2*3"), not just the latest key. The FSM
below re-derives current state from that string every time rather than
trusting any client-side state, which is also why Redis-backed session
persistence (app.db.redis_client) exists: to remember which product the
shopkeeper is mid-transaction on across those stateless round-trips within
the 180-second MTN/Airtel timeout window.
"""
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from sqlalchemy.orm import Session

from app.channels.messages import forecast_message, recent_sales_message, sale_logged_message
from app.core.config import get_settings
from app.db.redis_client import clear_ussd_session, load_ussd_session, save_ussd_session
from app.models.orm import ChannelEnum, LocaleEnum, USSDSession
from app.services.forecast_service import ForecastService
from app.services.sales_service import get_or_create_shopkeeper, get_recent_sales, record_sale

settings = get_settings()
_forecast_service = ForecastService()


def _persist_ussd_session_record(db: Session, session_id: str, shopkeeper_id: str, state: str) -> None:
    """Write-through audit copy of USSD session state into Postgres/Supabase.

    Redis (app.db.redis_client) remains the authoritative fast path the FSM
    reads from on every keypress — this table previously existed in the
    schema but nothing ever wrote to it, even though the architecture
    claims Supabase stores USSD session data alongside Redis."""
    record = db.get(USSDSession, session_id)
    if record is None:
        db.add(USSDSession(session_id=session_id, shopkeeper_id=shopkeeper_id, state=state))
    else:
        record.state = state
        record.last_active = datetime.utcnow()
    db.commit()


class USSDState(str, Enum):
    MAIN_MENU = "MAIN_MENU"
    SELECT_PRODUCT = "SELECT_PRODUCT"
    ENTER_QUANTITY = "ENTER_QUANTITY"
    FORECAST_SELECT_PRODUCT = "FORECAST_SELECT_PRODUCT"
    CHANGE_LANGUAGE_SELECT = "CHANGE_LANGUAGE_SELECT"
    DONE = "DONE"


PRODUCT_MENU = {
    "1": ("SUGAR", "kg"),
    "2": ("OIL", "litre"),
    "3": ("FLOUR", "kg"),
    "4": ("RICE", "kg"),
    "5": ("SOAP", "bar"),
}

# Menu/prompt copy, keyed by ShopkeeperProfile.locale ("kinyarwanda"/"english").
# Event copy (sale logged, forecast result) lives in app.channels.messages
# instead, since WhatsApp shares those; this is USSD-menu-specific navigation
# text, so it stays local to the FSM.
_STRINGS = {
    "kinyarwanda": {
        "main_menu": (
            "Murakaza neza kuri DukaStock\n"
            "1. Andika igurisha\n"
            "2. Saba inama yo kongera ibicuruzwa\n"
            "3. Reba amagurisha yanjye\n"
            "4. Hindura ururimi"
        ),
        "product_menu": "Hitamo igicuruzwa:\n1. Isukari\n2. Amavuta\n3. Ifu\n4. Umuceri\n5. Isabune",
        "invalid_option": "Hitamo ikibazo cyemewe.",
        "product_not_found": "Igicuruzwa ntikibonetse.",
        "quantity_prompt": "Andika umubare w'ibyo wagurishije ({unit}):",
        "invalid_quantity": "Andika umubare gusa (urugero: 3).",
        "language_menu": "Hitamo ururimi:\n1. Ikinyarwanda\n2. Icyongereza",
        "language_changed": "Ururimi rwahinduwe ku Kinyarwanda.",
        "generic_error": "Ikosa ryabaye. Ongera ugerageze.",
    },
    "english": {
        "main_menu": (
            "Welcome to DukaStock\n"
            "1. Log a sale\n"
            "2. Request restocking advice\n"
            "3. View my sales\n"
            "4. Change language"
        ),
        "product_menu": "Choose a product:\n1. Sugar\n2. Cooking oil\n3. Flour\n4. Rice\n5. Soap",
        "invalid_option": "Please choose a valid option.",
        "product_not_found": "Product not found.",
        "quantity_prompt": "Enter the quantity sold ({unit}):",
        "invalid_quantity": "Please enter a number only (e.g. 3).",
        "language_menu": "Choose your language:\n1. Kinyarwanda\n2. English",
        "language_changed": "Language changed to English.",
        "generic_error": "An error occurred. Please try again.",
    },
}


def _t(locale, key: str, **kwargs) -> str:
    strings = _STRINGS.get(locale, _STRINGS["kinyarwanda"])
    text = strings[key]
    return text.format(**kwargs) if kwargs else text


@dataclass
class USSDResponse:
    text: str
    continue_session: bool

    def render(self) -> str:
        # Enforced here, once, so every USSDResponse constructed anywhere in
        # the FSM below respects Africa's Talking's 182-char USSD limit —
        # no call site has to remember to truncate its own message text.
        prefix = "CON" if self.continue_session else "END"
        return f"{prefix} {self.text}"[: settings.ussd_max_chars]


def handle_ussd_request(db: Session, session_id: str, phone_number: str, text: str) -> USSDResponse:
    """
    `text` is the FULL accumulated input for this USSD session as sent by
    Africa's Talking (e.g. "" on first dial, "1" after first keypress,
    "1*2" after second, etc).
    """
    shopkeeper = get_or_create_shopkeeper(db, phone_number, ChannelEnum.ussd)
    # Falls back to Kinyarwanda for any locale value not in _STRINGS (e.g. a
    # mocked shopkeeper in unit tests, or a future locale not yet translated).
    locale = shopkeeper.locale if shopkeeper.locale in _STRINGS else "kinyarwanda"
    inputs = text.split("*") if text else []
    state = load_ussd_session(session_id) or {"state": USSDState.MAIN_MENU.value}

    if not inputs:
        save_ussd_session(session_id, {"state": USSDState.MAIN_MENU.value})
        _persist_ussd_session_record(db, session_id, shopkeeper.uuid, USSDState.MAIN_MENU.value)
        return USSDResponse(text=_t(locale, "main_menu"), continue_session=True)

    last_input = inputs[-1]

    if state["state"] == USSDState.MAIN_MENU.value:
        if last_input == "1":
            save_ussd_session(session_id, {"state": USSDState.SELECT_PRODUCT.value, "flow": "sale"})
            _persist_ussd_session_record(db, session_id, shopkeeper.uuid, USSDState.SELECT_PRODUCT.value)
            return USSDResponse(text=_t(locale, "product_menu"), continue_session=True)
        elif last_input == "2":
            save_ussd_session(session_id, {"state": USSDState.FORECAST_SELECT_PRODUCT.value})
            _persist_ussd_session_record(db, session_id, shopkeeper.uuid, USSDState.FORECAST_SELECT_PRODUCT.value)
            return USSDResponse(text=_t(locale, "product_menu"), continue_session=True)
        elif last_input == "3":
            sales = get_recent_sales(db, shopkeeper.uuid)
            clear_ussd_session(session_id)
            _persist_ussd_session_record(db, session_id, shopkeeper.uuid, USSDState.DONE.value)
            return USSDResponse(text=recent_sales_message(sales, locale), continue_session=False)
        elif last_input == "4":
            save_ussd_session(session_id, {"state": USSDState.CHANGE_LANGUAGE_SELECT.value})
            _persist_ussd_session_record(db, session_id, shopkeeper.uuid, USSDState.CHANGE_LANGUAGE_SELECT.value)
            return USSDResponse(text=_t(locale, "language_menu"), continue_session=True)
        return USSDResponse(text=_t(locale, "invalid_option") + "\n" + _t(locale, "main_menu"), continue_session=True)

    if state["state"] == USSDState.SELECT_PRODUCT.value:
        if last_input not in PRODUCT_MENU:
            return USSDResponse(
                text=_t(locale, "product_not_found") + " " + _t(locale, "product_menu"), continue_session=True
            )
        product_code, unit = PRODUCT_MENU[last_input]
        save_ussd_session(session_id, {"state": USSDState.ENTER_QUANTITY.value, "product_code": product_code, "unit": unit})
        _persist_ussd_session_record(db, session_id, shopkeeper.uuid, USSDState.ENTER_QUANTITY.value)
        return USSDResponse(text=_t(locale, "quantity_prompt", unit=unit), continue_session=True)

    if state["state"] == USSDState.ENTER_QUANTITY.value:
        try:
            quantity = float(last_input)
        except ValueError:
            return USSDResponse(text=_t(locale, "invalid_quantity"), continue_session=True)
        record_sale(
            db, raw_phone=phone_number, product_code=state["product_code"],
            quantity=quantity, unit=state["unit"], channel=ChannelEnum.ussd,
        )
        clear_ussd_session(session_id)
        _persist_ussd_session_record(db, session_id, shopkeeper.uuid, USSDState.DONE.value)
        return USSDResponse(
            text=sale_logged_message(state["product_code"], quantity, state["unit"], locale),
            continue_session=False,
        )

    if state["state"] == USSDState.FORECAST_SELECT_PRODUCT.value:
        if last_input not in PRODUCT_MENU:
            return USSDResponse(
                text=_t(locale, "product_not_found") + " " + _t(locale, "product_menu"), continue_session=True
            )
        product_code, unit = PRODUCT_MENU[last_input]
        forecast = _forecast_service.forecast(product_code)
        clear_ussd_session(session_id)
        _persist_ussd_session_record(db, session_id, shopkeeper.uuid, USSDState.DONE.value)
        qty = forecast.get("predicted_quantity")
        return USSDResponse(text=forecast_message(product_code, unit, qty, locale), continue_session=False)

    if state["state"] == USSDState.CHANGE_LANGUAGE_SELECT.value:
        if last_input == "1":
            new_locale = "kinyarwanda"
        elif last_input == "2":
            new_locale = "english"
        else:
            return USSDResponse(text=_t(locale, "language_menu"), continue_session=True)
        shopkeeper.locale = LocaleEnum(new_locale)
        db.commit()
        clear_ussd_session(session_id)
        _persist_ussd_session_record(db, session_id, shopkeeper.uuid, USSDState.DONE.value)
        return USSDResponse(text=_t(new_locale, "language_changed"), continue_session=False)

    clear_ussd_session(session_id)
    _persist_ussd_session_record(db, session_id, shopkeeper.uuid, USSDState.DONE.value)
    return USSDResponse(text=_t(locale, "generic_error"), continue_session=False)
