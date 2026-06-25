"""
Privacy / security utilities.

Rwanda's Law No. 058/2021 relating to the protection of personal data and
privacy requires that personal data (phone numbers are personal data under
Article 3 of the law) be processed lawfully and stored no longer than
necessary, with appropriate technical safeguards.

DukaStock never stores raw phone numbers. Every inbound phone number is
hashed to a deterministic UUID-shaped identifier at the point of ingestion,
before it touches the database layer. The salt is an application secret
(see core.config.Settings.phone_hash_salt) so the hash cannot be reversed
by rainbow-table lookup even if the database is compromised.
"""
import hashlib
import uuid

from app.core.config import get_settings


def hash_phone_number(raw_phone: str) -> str:
    """
    Deterministically hash a raw phone number (e.g. "+250788123456") into a
    UUID5-shaped string. Deterministic so the same shopkeeper always maps to
    the same shopkeeper_id, but the original number cannot be recovered.
    """
    settings = get_settings()
    normalized = _normalize_phone(raw_phone)
    salted = f"{settings.phone_hash_salt}:{normalized}".encode("utf-8")
    digest = hashlib.sha256(salted).hexdigest()
    # Fold the SHA-256 digest into a UUID5 for a stable, storage-friendly shape.
    return str(uuid.uuid5(uuid.NAMESPACE_OID, digest))


def _normalize_phone(raw_phone: str) -> str:
    """Strip whitespace/formatting so the same number always hashes the same way."""
    digits = "".join(ch for ch in raw_phone if ch.isdigit())
    if digits.startswith("0") and len(digits) == 10:
        # Local Rwandan format (07XXXXXXXX) -> E.164 without '+'
        digits = "250" + digits[1:]
    return digits
