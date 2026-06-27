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
from enum import Enum

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.redis_client import clear_ussd_session, load_ussd_session, save_ussd_session
from app.models.orm import ChannelEnum
from app.services.forecast_service import ForecastService
from app.services.sales_service import record_sale

settings = get_settings()
_forecast_service = ForecastService()


class USSDState(str, Enum):
    MAIN_MENU = "MAIN_MENU"
    SELECT_PRODUCT = "SELECT_PRODUCT"
    ENTER_QUANTITY = "ENTER_QUANTITY"
    FORECAST_SELECT_PRODUCT = "FORECAST_SELECT_PRODUCT"
    DONE = "DONE"


PRODUCT_MENU = {
    "1": ("SUGAR", "kg"),
    "2": ("OIL", "litre"),
    "3": ("FLOUR", "kg"),
    "4": ("RICE", "kg"),
    "5": ("SOAP", "bar"),
}

MAIN_MENU_TEXT = (
    "Murakaza neza kuri DukaStock\n"
    "1. Andika igurisha\n"
    "2. Saba inama yo kongera ibicuruzwa"
)
PRODUCT_MENU_TEXT = "Hitamo igicuruzwa:\n1. Isukari\n2. Amavuta\n3. Ifu\n4. Umuceri\n5. Isabune"


@dataclass
class USSDResponse:
    text: str
    continue_session: bool

    def render(self) -> str:
        prefix = "CON" if self.continue_session else "END"
        return f"{prefix} {self.text}"


def handle_ussd_request(db: Session, session_id: str, phone_number: str, text: str) -> USSDResponse:
    """
    `text` is the FULL accumulated input for this USSD session as sent by
    Africa's Talking (e.g. "" on first dial, "1" after first keypress,
    "1*2" after second, etc).
    """
    inputs = text.split("*") if text else []
    state = load_ussd_session(session_id) or {"state": USSDState.MAIN_MENU.value}

    if not inputs:
        save_ussd_session(session_id, {"state": USSDState.MAIN_MENU.value})
        return USSDResponse(text=MAIN_MENU_TEXT, continue_session=True)

    last_input = inputs[-1]

    if state["state"] == USSDState.MAIN_MENU.value:
        if last_input == "1":
            save_ussd_session(session_id, {"state": USSDState.SELECT_PRODUCT.value, "flow": "sale"})
            return USSDResponse(text=PRODUCT_MENU_TEXT, continue_session=True)
        elif last_input == "2":
            save_ussd_session(session_id, {"state": USSDState.FORECAST_SELECT_PRODUCT.value})
            return USSDResponse(text=PRODUCT_MENU_TEXT, continue_session=True)
        return USSDResponse(text="Hitamo ikibazo cyemewe.\n" + MAIN_MENU_TEXT, continue_session=True)

    if state["state"] == USSDState.SELECT_PRODUCT.value:
        if last_input not in PRODUCT_MENU:
            return USSDResponse(text="Igicuruzwa ntikibonetse. " + PRODUCT_MENU_TEXT, continue_session=True)
        product_code, unit = PRODUCT_MENU[last_input]
        save_ussd_session(session_id, {"state": USSDState.ENTER_QUANTITY.value, "product_code": product_code, "unit": unit})
        return USSDResponse(text=f"Andika umubare w'ibyo wagurishije ({unit}):", continue_session=True)

    if state["state"] == USSDState.ENTER_QUANTITY.value:
        try:
            quantity = float(last_input)
        except ValueError:
            return USSDResponse(text="Andika umubare gusa (urugero: 3).", continue_session=True)
        record_sale(
            db, raw_phone=phone_number, product_code=state["product_code"],
            quantity=quantity, unit=state["unit"], channel=ChannelEnum.ussd,
        )
        clear_ussd_session(session_id)
        return USSDResponse(text="Murakoze! Igurisha ryanditswe.", continue_session=False)

    if state["state"] == USSDState.FORECAST_SELECT_PRODUCT.value:
        if last_input not in PRODUCT_MENU:
            return USSDResponse(text="Igicuruzwa ntikibonetse. " + PRODUCT_MENU_TEXT, continue_session=True)
        product_code, unit = PRODUCT_MENU[last_input]
        forecast = _forecast_service.forecast(product_code)
        clear_ussd_session(session_id)
        qty = forecast.get("predicted_quantity")
        if qty is None:
            msg = f"Nta modeli y'{product_code} irahari. Ongera ugerageze nyuma."
        else:
            msg = f"Mu cyumweru gitaha: {round(qty, 1)} {unit} ya {product_code}."
        return USSDResponse(text=msg[: settings.ussd_max_chars], continue_session=False)

    clear_ussd_session(session_id)
    return USSDResponse(text="Ikosa ryabaye. Ongera ugerageze.", continue_session=False)
