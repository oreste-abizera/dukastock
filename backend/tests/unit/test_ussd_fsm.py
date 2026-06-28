from unittest.mock import MagicMock, patch

from app.channels.ussd.fsm import handle_ussd_request


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
