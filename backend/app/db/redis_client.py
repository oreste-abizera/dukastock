"""
Redis (Upstash) client for USSD session state.

MTN and Airtel Rwanda terminate USSD sessions after a network-level
inactivity window (~180 seconds). Because USSD is stateless at the
telecom layer (every keypress is a brand-new HTTP request to our webhook),
DukaStock must persist "where the user is in the FSM menu" itself. Redis
with a matching TTL is the natural fit: state expires automatically at the
same moment the telecom session would have expired anyway, so there's
nothing to garbage-collect.
"""
import json
from typing import Any, Optional

import redis

from app.core.config import get_settings

settings = get_settings()

_redis_client: Optional[redis.Redis] = None


def get_redis() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(settings.redis_url, decode_responses=True)
    return _redis_client


def save_ussd_session(session_id: str, state: dict[str, Any]) -> None:
    client = get_redis()
    client.set(
        f"ussd:session:{session_id}",
        json.dumps(state),
        ex=settings.ussd_session_ttl_seconds,
    )


def load_ussd_session(session_id: str) -> Optional[dict[str, Any]]:
    client = get_redis()
    raw = client.get(f"ussd:session:{session_id}")
    if raw is None:
        return None
    return json.loads(raw)


def clear_ussd_session(session_id: str) -> None:
    client = get_redis()
    client.delete(f"ussd:session:{session_id}")
