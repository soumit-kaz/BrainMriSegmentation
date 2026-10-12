import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO

COPY_CHUNK_SIZE = 1024 * 1024


class StreamSizeLimitError(Exception):
    pass


def get_utc_now() -> datetime:
    return datetime.now(timezone.utc)


def get_new_identifier() -> str:
    return uuid.uuid4().hex


def get_server_sent_event(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, default=str)}\n\n"


def get_server_sent_heartbeat() -> str:
    return ": heartbeat\n\n"


def get_copied_stream_digest(source_stream: BinaryIO, target_path: Path, maximum_byte_count: int) -> str:
    digest = hashlib.sha256()
    written_byte_count = 0
    with target_path.open("wb") as target_file:
        while chunk := source_stream.read(COPY_CHUNK_SIZE):
            written_byte_count += len(chunk)
            if written_byte_count > maximum_byte_count:
                raise StreamSizeLimitError(f"Stream exceeds {maximum_byte_count} bytes.")
            digest.update(chunk)
            target_file.write(chunk)
    return digest.hexdigest()


def get_combined_digest(digests: list[str]) -> str:
    return hashlib.sha256("".join(digests).encode("ascii")).hexdigest()
