import asyncio
import logging
import math
from collections.abc import AsyncGenerator
from pathlib import Path

import numpy as np
from segmentation_core.settings import MASK_SUFFIX, MODALITY_SUFFIXES, is_nifti_file

from inference_service.application.ports import (
    JobEventPublisher,
    JobEventSubscriber,
    JobQueue,
    RateLimiter,
    ScanStorage,
    SegmentationJobRepository,
    SliceRenderer,
    WorkerStatusStore,
)
from inference_service.configuration import InferenceServiceSettings
from inference_service.domain.segmentation_job import (
    ACTIVE_JOB_STATUSES,
    InvalidScanError,
    JobConflictError,
    JobNotReadyError,
    JobStatus,
    JobSubmissionResult,
    MaskSource,
    ModelInformation,
    QueueFullError,
    QueueStatus,
    QueueUnavailableError,
    RateLimitExceededError,
    ScanSubmission,
    SegmentationJob,
    StagedScans,
    UploadedScan,
)
from inference_service.helpers import get_case_name, get_nifti_extension, get_nifti_shape
from inference_service.utils import (
    StreamSizeLimitError,
    get_combined_digest,
    get_copied_stream_digest,
    get_new_identifier,
    get_utc_now,
)

logger = logging.getLogger("segmentation.inference.api")

MASK_COLORS: dict[MaskSource, tuple[int, int, int]] = {
    MaskSource.PREDICTION: (255, 64, 64),
    MaskSource.GROUND_TRUTH: (64, 220, 120),
}
RATE_LIMIT_WINDOW_SECONDS = 60
JOB_LIST_LIMIT = 100
VOLUME_DIMENSION_COUNT = 3


class SegmentationJobService:
    def __init__(self, settings: InferenceServiceSettings, job_repository: SegmentationJobRepository, job_queue: JobQueue,
                 event_publisher: JobEventPublisher, event_subscriber: JobEventSubscriber, rate_limiter: RateLimiter,
                 worker_status_store: WorkerStatusStore, scan_storage: ScanStorage, slice_renderer: SliceRenderer):
        self.settings = settings
        self.job_repository = job_repository
        self.job_queue = job_queue
        self.event_publisher = event_publisher
        self.event_subscriber = event_subscriber
        self.rate_limiter = rate_limiter
        self.worker_status_store = worker_status_store
        self.scan_storage = scan_storage
        self.slice_renderer = slice_renderer

    def is_ready(self) -> bool:
        return self.job_repository.is_available() and self.job_queue.is_available()

    def get_average_processing_seconds(self) -> float:
        return self.worker_status_store.get_average_processing_seconds(self.settings.default_processing_seconds)

    def get_queue_status(self) -> QueueStatus:
        return QueueStatus(
            active_job_count=self.job_repository.get_job_count(ACTIVE_JOB_STATUSES),
            queued_job_count=self.job_repository.get_job_count(frozenset({JobStatus.QUEUED})),
            maximum_queue_length=self.settings.maximum_queue_length,
            average_processing_seconds=self.get_average_processing_seconds(),
            is_worker_online=self.worker_status_store.get_model_information() is not None,
        )

    def get_model_information(self) -> ModelInformation | None:
        return self.worker_status_store.get_model_information()

    def ensure_capacity(self, client_key: str) -> None:
        if not self.rate_limiter.is_allowed(client_key, self.settings.rate_limit_requests_per_minute):
            raise RateLimitExceededError("Too many submissions. Try again shortly.", RATE_LIMIT_WINDOW_SECONDS)
        active_job_count = self.job_repository.get_job_count(ACTIVE_JOB_STATUSES)
        if active_job_count >= self.settings.maximum_queue_length:
            retry_after_seconds = math.ceil(active_job_count * self.get_average_processing_seconds())
            raise QueueFullError("The segmentation queue is full. Try again later.", retry_after_seconds)

    def save_scan(self, uploaded_scan: UploadedScan, staging_directory_name: str, suffix: str) -> tuple[str, tuple[int, ...]]:
        if not is_nifti_file(uploaded_scan.file_name):
            raise InvalidScanError(f"'{suffix}' must be a NIfTI file (.nii or .nii.gz), got '{uploaded_scan.file_name}'.")
        target_path = self.scan_storage.get_staging_path(staging_directory_name, f"{suffix}{get_nifti_extension(uploaded_scan.file_name)}")
        try:
            digest = get_copied_stream_digest(uploaded_scan.content_stream, target_path, self.settings.maximum_upload_size_bytes)
            volume_shape = get_nifti_shape(target_path)
        except StreamSizeLimitError as error:
            raise InvalidScanError(f"'{suffix}' exceeds {self.settings.maximum_upload_size_megabytes} MB.") from error
        except Exception as error:
            raise InvalidScanError(f"'{suffix}' could not be read as a NIfTI volume.") from error
        if len(volume_shape) != VOLUME_DIMENSION_COUNT:
            raise InvalidScanError(f"'{suffix}' must be a 3D volume, got shape {volume_shape}.")
        return digest, volume_shape

    def get_staged_scans(self, submission: ScanSubmission, staging_directory_name: str) -> StagedScans:
        scan_results = [self.save_scan(submission.modality_scans[suffix], staging_directory_name, suffix) for suffix in MODALITY_SUFFIXES]
        if submission.ground_truth_scan is not None:
            scan_results.append(self.save_scan(submission.ground_truth_scan, staging_directory_name, MASK_SUFFIX))
        if len({volume_shape for _, volume_shape in scan_results}) != 1:
            raise InvalidScanError("All uploaded volumes must have the same dimensions.")
        return StagedScans(
            staging_directory_name=staging_directory_name,
            case_name=get_case_name(submission.modality_scans[MODALITY_SUFFIXES[0]].file_name),
            input_hash=get_combined_digest([digest for digest, _ in scan_results]),
            has_ground_truth=submission.ground_truth_scan is not None,
        )

    def get_reused_submission(self, staged_scans: StagedScans) -> JobSubmissionResult | None:
        reusable_job = self.job_repository.get_reusable_job(staged_scans.input_hash)
        if reusable_job is None:
            return None
        self.scan_storage.remove_staging_directory(staged_scans.staging_directory_name)
        return JobSubmissionResult(reusable_job, self.job_repository.get_queue_position(reusable_job), is_reused=True)

    def add_queued_job(self, staged_scans: StagedScans) -> SegmentationJob:
        job = SegmentationJob(
            job_id=get_new_identifier(), case_name=staged_scans.case_name, status=JobStatus.QUEUED,
            input_hash=staged_scans.input_hash, has_ground_truth=staged_scans.has_ground_truth, attempt_count=0,
            created_at=get_utc_now(), started_at=None, completed_at=None, error_message=None, summary=None,
        )
        self.scan_storage.promote_staging_directory(staged_scans.staging_directory_name, job.job_id)
        self.job_repository.add_job(job)
        try:
            self.job_queue.add_job(job.job_id)
        except Exception as error:
            self.job_repository.mark_failed(job.job_id, "The job could not be queued.")
            raise QueueUnavailableError("The job queue is unavailable.") from error
        return job

    def submit_job(self, submission: ScanSubmission, client_key: str) -> JobSubmissionResult:
        self.ensure_capacity(client_key)
        staging_directory_name = self.scan_storage.create_staging_directory()
        try:
            staged_scans = self.get_staged_scans(submission, staging_directory_name)
        except Exception:
            self.scan_storage.remove_staging_directory(staging_directory_name)
            raise
        reused_submission = self.get_reused_submission(staged_scans)
        if reused_submission is not None:
            logger.info("Reused job %s for identical input.", reused_submission.job.job_id)
            return reused_submission
        job = self.add_queued_job(staged_scans)
        self.event_publisher.publish(job.job_id, {"status": job.status})
        logger.info("Queued job %s (%s).", job.job_id, job.case_name)
        return JobSubmissionResult(job, self.job_repository.get_queue_position(job), is_reused=False)

    def get_job(self, job_id: str) -> SegmentationJob:
        return self.job_repository.get_job(job_id)

    def get_queue_position(self, job: SegmentationJob) -> int:
        return self.job_repository.get_queue_position(job)

    def get_jobs(self) -> list[SegmentationJob]:
        return self.job_repository.get_jobs(JOB_LIST_LIMIT)

    def get_completed_job(self, job_id: str) -> SegmentationJob:
        job = self.job_repository.get_job(job_id)
        if not job.is_completed:
            raise JobNotReadyError(f"Job '{job_id}' is {job.status}.")
        return job

    def delete_job(self, job_id: str) -> None:
        job = self.job_repository.get_job(job_id)
        if not job.is_terminal:
            raise JobConflictError(f"Job '{job_id}' is {job.status} and cannot be deleted yet.")
        self.job_repository.delete_job(job_id)
        self.scan_storage.delete_job_files(job_id)

    def get_slice_image_png(self, job_id: str, modality: str, slice_index: int) -> bytes:
        self.get_completed_job(job_id)
        return self.slice_renderer.get_image_png(self.scan_storage.get_preview_volume(job_id, modality), slice_index)

    def get_slice_mask_png(self, job_id: str, mask_source: MaskSource, slice_index: int) -> bytes:
        job = self.get_completed_job(job_id)
        if mask_source == MaskSource.GROUND_TRUTH and not job.has_ground_truth:
            raise JobNotReadyError(f"Job '{job_id}' has no ground truth.")
        mask_volume: np.ndarray = self.scan_storage.get_preview_volume(job_id, mask_source)
        return self.slice_renderer.get_mask_png(mask_volume, slice_index, MASK_COLORS[mask_source])

    def get_mask_path(self, job_id: str) -> tuple[SegmentationJob, Path]:
        job = self.get_completed_job(job_id)
        return job, self.scan_storage.get_mask_path(job_id)

    async def get_job_with_queue_position(self, job_id: str) -> tuple[SegmentationJob, int]:
        job = await asyncio.to_thread(self.job_repository.get_job, job_id)
        return job, await asyncio.to_thread(self.job_repository.get_queue_position, job)

    async def get_job_updates(self, job_id: str) -> AsyncGenerator[tuple[SegmentationJob, int], None]:
        job_update = await self.get_job_with_queue_position(job_id)
        yield job_update
        if job_update[0].is_terminal:
            return
        event_stream = self.event_subscriber.get_events(job_id)
        try:
            async for _ in event_stream:
                job_update = await self.get_job_with_queue_position(job_id)
                yield job_update
                if job_update[0].is_terminal:
                    return
        finally:
            await event_stream.aclose()
