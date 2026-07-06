from datetime import datetime
from unittest.mock import MagicMock, patch

from app.channels.ussd.fsm import handle_ussd_request
from app.models.orm import LocaleEnum


@patch("app.channels.ussd.fsm.load_ussd_session", return_value=None)
@patch("app.channels.ussd.fsm.save_ussd_session")
def test_initial_dial_shows_main_menu(mock_save, mock_load):
    db = MagicMock()
    response = handle_ussd_request(db, session_id="sess1", phone_number="+250788123456", text="")
    assert response.continue_session is True
    assert "DukaStock" in response.text


@patch("app.channels.ussd.fsm.load_ussd_session", return_value={"state": "MAIN_MENU"})
@patch("app.channels.ussd.fsm.save_ussd_session")
def test_selecting_log_sale_shows_product_menu(mock_save, mock_load):
    db = MagicMock()
    response = handle_ussd_request(db, session_id="sess1", phone_number="+250788123456", text="1")
    assert response.continue_session is True
    assert "Isukari" in response.text


@patch("app.channels.ussd.fsm.load_ussd_session", return_value={"state": "SELECT_PRODUCT", "flow": "sale"})
@patch("app.channels.ussd.fsm.save_ussd_session")
def test_selecting_product_prompts_for_quantity(mock_save, mock_load):
    db = MagicMock()
    response = handle_ussd_request(db, session_id="sess1", phone_number="+250788123456", text="1*1")
    assert response.continue_session is True
    assert "kg" in response.text


@patch("app.channels.ussd.fsm.clear_ussd_session")
@patch("app.channels.ussd.fsm.load_ussd_session", return_value={"state": "ENTER_QUANTITY", "product_code": "SUGAR", "unit": "kg"})
@patch("app.channels.ussd.fsm.save_ussd_session")
@patch("app.channels.ussd.fsm.record_sale")
def test_entering_quantity_records_sale_and_ends_session(mock_record, mock_save, mock_load, mock_clear):
    db = MagicMock()
    response = handle_ussd_request(db, session_id="sess1", phone_number="+250788123456", text="1*1*3")
    assert response.continue_session is False
    mock_record.assert_called_once()
    mock_clear.assert_called_once()


@patch("app.channels.ussd.fsm.load_ussd_session", return_value={"state": "ENTER_QUANTITY", "product_code": "SUGAR", "unit": "kg"})
@patch("app.channels.ussd.fsm.save_ussd_session")
def test_invalid_quantity_input_reprompts(mock_save, mock_load):
    db = MagicMock()
    response = handle_ussd_request(db, session_id="sess1", phone_number="+250788123456", text="1*1*abc")
    assert response.continue_session is True
    assert "umubare" in response.text.lower()


@patch("app.channels.ussd.fsm.clear_ussd_session")
@patch("app.channels.ussd.fsm.get_recent_sales")
@patch("app.channels.ussd.fsm.load_ussd_session", return_value={"state": "MAIN_MENU"})
@patch("app.channels.ussd.fsm.save_ussd_session")
def test_view_sales_lists_recent_entries(mock_save, mock_load, mock_get_sales, mock_clear):
    db = MagicMock()
    db.get.return_value.locale = "kinyarwanda"
    mock_get_sales.return_value = [
        MagicMock(logged_at=datetime(2026, 7, 4), product_code="SUGAR", quantity=3.0, unit="kg"),
        MagicMock(logged_at=datetime(2026, 7, 3), product_code="RICE", quantity=5.0, unit="kg"),
    ]
    response = handle_ussd_request(db, session_id="sess1", phone_number="+250788123456", text="3")
    assert response.continue_session is False
    assert "SUGAR" in response.text
    assert "RICE" in response.text


@patch("app.channels.ussd.fsm.clear_ussd_session")
@patch("app.channels.ussd.fsm.get_recent_sales", return_value=[])
@patch("app.channels.ussd.fsm.load_ussd_session", return_value={"state": "MAIN_MENU"})
@patch("app.channels.ussd.fsm.save_ussd_session")
def test_view_sales_with_no_history(mock_save, mock_load, mock_get_sales, mock_clear):
    db = MagicMock()
    db.get.return_value.locale = "kinyarwanda"
    response = handle_ussd_request(db, session_id="sess1", phone_number="+250788123456", text="3")
    assert response.continue_session is False
    assert "kintu na kimwe" in response.text


@patch("app.channels.ussd.fsm.load_ussd_session", return_value={"state": "MAIN_MENU"})
@patch("app.channels.ussd.fsm.save_ussd_session")
def test_change_language_option_shows_language_menu(mock_save, mock_load):
    db = MagicMock()
    db.get.return_value.locale = "kinyarwanda"
    response = handle_ussd_request(db, session_id="sess1", phone_number="+250788123456", text="4")
    assert response.continue_session is True
    assert "English" in response.text or "Icyongereza" in response.text


@patch("app.channels.ussd.fsm.clear_ussd_session")
@patch("app.channels.ussd.fsm.load_ussd_session", return_value={"state": "CHANGE_LANGUAGE_SELECT"})
@patch("app.channels.ussd.fsm.save_ussd_session")
def test_selecting_english_persists_locale_and_confirms_in_english(mock_save, mock_load, mock_clear):
    db = MagicMock()
    db.get.return_value.locale = "kinyarwanda"
    response = handle_ussd_request(db, session_id="sess1", phone_number="+250788123456", text="4*2")
    assert response.continue_session is False
    assert response.text == "Language changed to English."
    assert db.get.return_value.locale == LocaleEnum.english
    db.commit.assert_called()


@patch("app.channels.ussd.fsm.load_ussd_session", return_value={"state": "MAIN_MENU"})
@patch("app.channels.ussd.fsm.save_ussd_session")
def test_main_menu_renders_in_english_once_locale_is_english(mock_save, mock_load):
    db = MagicMock()
    db.get.return_value.locale = "english"
    response = handle_ussd_request(db, session_id="sess1", phone_number="+250788123456", text="")
    assert "Welcome to DukaStock" in response.text
