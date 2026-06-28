from app.core.security import _normalize_phone, hash_phone_number


def test_hash_is_deterministic():
    h1 = hash_phone_number("+250788123456")
    h2 = hash_phone_number("+250788123456")
    assert h1 == h2


def test_hash_local_and_international_format_match():
    h_local = hash_phone_number("0788123456")
    h_intl = hash_phone_number("+250788123456")
    assert h_local == h_intl


def test_hash_output_looks_like_uuid():
    h = hash_phone_number("+250788123456")
    parts = h.split("-")
    assert len(parts) == 5


def test_hash_does_not_contain_raw_digits():
    raw = "250788123456"
    h = hash_phone_number(raw)
    assert raw not in h


def test_normalize_local_rwandan_format():
    assert _normalize_phone("0788123456") == "250788123456"


def test_different_numbers_hash_differently():
    h1 = hash_phone_number("+250788123456")
    h2 = hash_phone_number("+250788999999")
    assert h1 != h2
