import json
from datetime import datetime, timezone
from typing import Any


def get_utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_server_sent_event(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, default=str)}\n\n"


def get_server_sent_heartbeat() -> str:
    return ": heartbeat\n\n"


def get_non_empty_values(values: dict[str, Any]) -> dict[str, Any]:
    return {name: value for name, value in values.items() if value is not None}
