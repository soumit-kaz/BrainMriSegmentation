from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, BinaryIO


class JobStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


TERMINAL_JOB_STATUSES: frozenset[JobStatus] = frozenset({JobStatus.COMPLETED, JobStatus.FAILED})
ACTIVE_JOB_STATUSES: frozenset[JobStatus] = frozenset({JobStatus.QUEUED, JobStatus.PROCESSING})


class MaskSource(StrEnum):
    PREDICTION = "prediction"
    GROUND_TRUTH = "ground_truth"


class SegmentationNotFoundError(Exception):
    pass


class InvalidScanError(Exception):
    pass


class JobNotReadyError(Exception):
    pass


class JobConflictError(Exception):
    pass


class QueueUnavailableError(Exception):
    pass


class ModelNotReadyError(Exception):
    pass


class RetryLaterError(Exception):
    def __init__(self, message: str, retry_after_seconds: int):
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class RateLimitExceededError(RetryLaterError):
    pass


class QueueFullError(RetryLaterError):
    pass


@dataclass(frozen=True)
class UploadedScan:
    file_name: str
    content_stream: BinaryIO


@dataclass(frozen=True)
class ScanSubmission:
    modality_scans: dict[str, UploadedScan]
    ground_truth_scan: UploadedScan | None


@dataclass(frozen=True)
class StagedScans:
    staging_directory_name: str
    case_name: str
    input_hash: str
    has_ground_truth: bool


@dataclass(frozen=True)
class SegmentationSummary:
    volume_shape: list[int]
    voxel_spacing_millimeters: list[float]
    slice_count: int
    tumor_voxel_count: int
    tumor_volume_cubic_centimeters: float
    tumor_area_per_slice: list[int]
    tumor_slice_range: list[int] | None
    largest_tumor_slice_index: int
    dice_score: float | None
    iou_score: float | None
    inference_seconds: float
    device: str
    model_name: str
    model_epoch: int | None


@dataclass(frozen=True)
class SegmentationJob:
    job_id: str
    case_name: str
    status: JobStatus
    input_hash: str
    has_ground_truth: bool
    attempt_count: int
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    error_message: str | None
    summary: SegmentationSummary | None

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_JOB_STATUSES

    @property
    def is_completed(self) -> bool:
        return self.status == JobStatus.COMPLETED


@dataclass(frozen=True)
class JobSubmissionResult:
    job: SegmentationJob
    queue_position: int
    is_reused: bool


@dataclass(frozen=True)
class ModelInformation:
    is_loaded: bool
    model_name: str | None
    device: str
    device_name: str
    precision: str
    is_test_time_augmentation_enabled: bool
    epoch: int | None
    metrics: dict[str, float] | None
    architecture: dict[str, Any] | None
    parameter_count: int | None
    loaded_at: str | None
    message: str | None


@dataclass(frozen=True)
class QueueStatus:
    active_job_count: int
    queued_job_count: int
    maximum_queue_length: int
    average_processing_seconds: float
    is_worker_online: bool
