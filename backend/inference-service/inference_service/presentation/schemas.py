from dataclasses import asdict
from datetime import datetime
from typing import Any

from pydantic import BaseModel

from inference_service.domain.segmentation_job import JobStatus, ModelInformation, QueueStatus, SegmentationJob


class SegmentationSummaryResponse(BaseModel):
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


class SegmentationJobResponse(BaseModel):
    job_id: str
    case_name: str
    status: JobStatus
    has_ground_truth: bool
    attempt_count: int
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    error_message: str | None
    queue_position: int
    is_reused: bool
    summary: SegmentationSummaryResponse | None

    @classmethod
    def from_domain(cls, job: SegmentationJob, queue_position: int = 0, is_reused: bool = False) -> "SegmentationJobResponse":
        return cls(
            job_id=job.job_id,
            case_name=job.case_name,
            status=job.status,
            has_ground_truth=job.has_ground_truth,
            attempt_count=job.attempt_count,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            error_message=job.error_message,
            queue_position=queue_position,
            is_reused=is_reused,
            summary=SegmentationSummaryResponse(**asdict(job.summary)) if job.summary else None,
        )


class ModelInformationResponse(BaseModel):
    is_worker_online: bool
    is_loaded: bool
    model_name: str | None = None
    device: str | None = None
    device_name: str | None = None
    precision: str | None = None
    is_test_time_augmentation_enabled: bool | None = None
    epoch: int | None = None
    metrics: dict[str, float] | None = None
    architecture: dict[str, Any] | None = None
    parameter_count: int | None = None
    loaded_at: str | None = None
    message: str | None = None

    @classmethod
    def from_domain(cls, model_information: ModelInformation | None) -> "ModelInformationResponse":
        if model_information is None:
            return cls(is_worker_online=False, is_loaded=False, message="No inference worker is online. Jobs wait in the queue.")
        return cls(is_worker_online=True, **asdict(model_information))


class QueueStatusResponse(BaseModel):
    active_job_count: int
    queued_job_count: int
    maximum_queue_length: int
    average_processing_seconds: float
    estimated_wait_seconds: float
    is_worker_online: bool

    @classmethod
    def from_domain(cls, queue_status: QueueStatus) -> "QueueStatusResponse":
        return cls(**asdict(queue_status),
                   estimated_wait_seconds=queue_status.active_job_count * queue_status.average_processing_seconds)


class HealthResponse(BaseModel):
    status: str
