import os
import socket
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

from segmentation_core.settings import get_parsed_setting_value

ENVIRONMENT_VARIABLE_PREFIX = "INFERENCE_"


@dataclass(frozen=True)
class InferenceServiceSettings:
    database_url: str = "postgresql+psycopg://segmentation:segmentation@postgres:5432/segmentation"
    database_pool_size: int = 10
    redis_url: str = "redis://redis:6379/0"
    storage_directory: Path = Path("/storage")

    maximum_upload_size_megabytes: int = 200
    maximum_queue_length: int = 200
    rate_limit_requests_per_minute: int = 30
    trusted_proxy_hop_count: int = 1

    job_stream_name: str = "segmentation-jobs"
    job_consumer_group: str = "segmentation-workers"
    job_event_channel_prefix: str = "segmentation-job-events"
    maximum_stream_length: int = 100_000
    worker_name: str = socket.gethostname()
    worker_block_milliseconds: int = 5000
    job_reclaim_idle_seconds: int = 900
    maximum_job_attempt_count: int = 3
    worker_status_ttl_seconds: int = 60
    worker_heartbeat_path: Path = Path("/tmp/inference-worker-heartbeat")
    gpu_memory_fraction: float = 0.35
    default_processing_seconds: float = 30.0

    job_retention_days: int = 30
    retention_check_interval_seconds: int = 3600

    @classmethod
    def from_environment(cls) -> "InferenceServiceSettings":
        environment_values: dict[str, Any] = {}
        for settings_field in fields(cls):
            raw_value = os.getenv(f"{ENVIRONMENT_VARIABLE_PREFIX}{settings_field.name.upper()}")
            if raw_value is not None:
                environment_values[settings_field.name] = get_parsed_setting_value(raw_value, settings_field.default)
        return replace(cls(), **environment_values)

    @property
    def maximum_upload_size_bytes(self) -> int:
        return self.maximum_upload_size_megabytes * 1024 * 1024

    @property
    def staging_directory(self) -> Path:
        return self.storage_directory / "staging"

    @property
    def jobs_directory(self) -> Path:
        return self.storage_directory / "jobs"
