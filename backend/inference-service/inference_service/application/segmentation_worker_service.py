import logging
import threading
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
from nibabel.nifti1 import Nifti1Image
from segmentation_core.data import get_display_volume, get_restored_volume, get_stacked_image, get_whole_tumor_mask, load_nifti_volume
from segmentation_core.evaluation import get_dice_score, get_iou_score
from segmentation_core.settings import MASK_SUFFIX, MODALITY_SUFFIXES

from inference_service.application.ports import (
    JobEventPublisher,
    JobQueue,
    ModelProvider,
    ScanStorage,
    SegmentationJobRepository,
    WorkerStatusStore,
)
from inference_service.configuration import InferenceServiceSettings
from inference_service.domain.segmentation_job import (
    InvalidScanError,
    JobStatus,
    MaskSource,
    SegmentationJob,
    SegmentationNotFoundError,
    SegmentationSummary,
)
from inference_service.helpers import get_largest_tumor_slice_index, get_tumor_slice_range, get_volume_cubic_centimeters
from inference_service.utils import get_utc_now

logger = logging.getLogger("segmentation.inference.worker")

MILLISECONDS_PER_SECOND = 1000


class SegmentationWorkerService:
    def __init__(self, settings: InferenceServiceSettings, job_repository: SegmentationJobRepository, job_queue: JobQueue,
                 event_publisher: JobEventPublisher, worker_status_store: WorkerStatusStore, scan_storage: ScanStorage,
                 model_provider: ModelProvider):
        self.settings = settings
        self.job_repository = job_repository
        self.job_queue = job_queue
        self.event_publisher = event_publisher
        self.worker_status_store = worker_status_store
        self.scan_storage = scan_storage
        self.model_provider = model_provider
        self.last_retention_check = 0.0

    def is_model_ready(self) -> bool:
        self.settings.worker_heartbeat_path.touch()
        model_information = self.model_provider.get_information()
        self.worker_status_store.set_model_information(model_information)
        return model_information.is_loaded

    def remove_expired_jobs(self) -> None:
        if time.monotonic() - self.last_retention_check < self.settings.retention_check_interval_seconds:
            return
        self.last_retention_check = time.monotonic()
        expired_job_ids = self.job_repository.get_expired_job_ids(get_utc_now() - timedelta(days=self.settings.job_retention_days))
        for job_id in expired_job_ids:
            self.job_repository.delete_job(job_id)
            self.scan_storage.delete_job_files(job_id)
        if expired_job_ids:
            logger.info("Removed %d expired jobs.", len(expired_job_ids))

    def get_next_message(self) -> tuple[str, str] | None:
        stalled_message = self.job_queue.get_stalled_job(self.settings.job_reclaim_idle_seconds * MILLISECONDS_PER_SECOND)
        if stalled_message is not None:
            logger.warning("Reclaimed stalled job %s.", stalled_message[1])
            return stalled_message
        return self.job_queue.get_next_job(self.settings.worker_block_milliseconds)

    def run(self, stop_event: threading.Event) -> None:
        self.job_queue.ensure_consumer_group()
        logger.info("Worker %s started.", self.settings.worker_name)
        while not stop_event.is_set():
            if not self.is_model_ready():
                stop_event.wait(self.settings.worker_block_milliseconds / MILLISECONDS_PER_SECOND)
                continue
            self.remove_expired_jobs()
            next_message = self.get_next_message()
            if next_message is not None:
                message_id, job_id = next_message
                self.process_message(message_id, job_id)
        logger.info("Worker %s stopped.", self.settings.worker_name)

    def get_processable_job(self, job_id: str) -> SegmentationJob | None:
        try:
            job = self.job_repository.get_job(job_id)
        except SegmentationNotFoundError:
            logger.warning("Job %s no longer exists.", job_id)
            return None
        if job.is_terminal:
            return None
        if job.attempt_count >= self.settings.maximum_job_attempt_count:
            self.job_repository.mark_failed(job_id, "The job exceeded the maximum number of attempts.")
            self.event_publisher.publish(job_id, {"status": JobStatus.FAILED})
            return None
        return job

    def process_message(self, message_id: str, job_id: str) -> None:
        job = self.get_processable_job(job_id)
        if job is None:
            self.job_queue.acknowledge(message_id)
            return
        self.job_repository.mark_processing(job_id)
        self.event_publisher.publish(job_id, {"status": JobStatus.PROCESSING})
        started_at = time.perf_counter()
        try:
            summary = self.get_segmentation_summary(job, started_at)
            self.job_repository.mark_completed(job_id, summary)
            self.worker_status_store.add_processing_duration(summary.inference_seconds)
            self.event_publisher.publish(job_id, {"status": JobStatus.COMPLETED})
            logger.info("Completed job %s in %.1fs.", job_id, summary.inference_seconds)
        except (InvalidScanError, ValueError) as error:
            self.mark_job_failed(job_id, str(error))
        except Exception:
            logger.exception("Job %s failed unexpectedly.", job_id)
            self.mark_job_failed(job_id, "Unexpected processing error.")
        finally:
            self.job_queue.acknowledge(message_id)

    def mark_job_failed(self, job_id: str, error_message: str) -> None:
        self.job_repository.mark_failed(job_id, error_message)
        self.event_publisher.publish(job_id, {"status": JobStatus.FAILED})

    def get_input_volume(self, job_id: str, suffix: str) -> tuple[np.ndarray, Nifti1Image]:
        input_path: Path = self.scan_storage.get_input_path(job_id, suffix)
        return load_nifti_volume(input_path)

    def get_predicted_mask(self, modality_volumes: list[np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:
        predictor, metadata = self.model_provider.get_predictor()
        preprocessing = metadata["preprocessing"]
        image = torch.from_numpy(get_stacked_image(modality_volumes, tuple(preprocessing["cropped_volume_shape"])))
        probability_volume = torch.sigmoid(predictor.get_logit_volume(image)).cpu().numpy()
        cropped_mask = (probability_volume >= preprocessing["prediction_threshold"]).astype(np.uint8)
        return get_restored_volume(cropped_mask, modality_volumes[0].shape), metadata

    def get_segmentation_summary(self, job: SegmentationJob, started_at: float) -> SegmentationSummary:
        loaded_modalities = [self.get_input_volume(job.job_id, suffix) for suffix in MODALITY_SUFFIXES]
        modality_volumes = [volume for volume, _ in loaded_modalities]
        reference_image = loaded_modalities[-1][1]
        ground_truth_mask = get_whole_tumor_mask(self.get_input_volume(job.job_id, MASK_SUFFIX)[0]) if job.has_ground_truth else None

        predicted_mask, metadata = self.get_predicted_mask(modality_volumes)
        preview_volumes = {suffix: get_display_volume(volume) for suffix, volume in zip(MODALITY_SUFFIXES, modality_volumes)}
        preview_volumes[MaskSource.PREDICTION] = predicted_mask
        if ground_truth_mask is not None:
            preview_volumes[MaskSource.GROUND_TRUTH] = ground_truth_mask
        self.scan_storage.save_results(job.job_id, preview_volumes, predicted_mask, reference_image)

        voxel_spacing = [float(spacing) for spacing in reference_image.header.get_zooms()[:3]]
        tumor_area_per_slice = predicted_mask.sum(axis=(0, 1)).astype(int).tolist()
        tumor_voxel_count = int(predicted_mask.sum())
        return SegmentationSummary(
            volume_shape=list(predicted_mask.shape),
            voxel_spacing_millimeters=voxel_spacing,
            slice_count=predicted_mask.shape[2],
            tumor_voxel_count=tumor_voxel_count,
            tumor_volume_cubic_centimeters=get_volume_cubic_centimeters(tumor_voxel_count, voxel_spacing),
            tumor_area_per_slice=tumor_area_per_slice,
            tumor_slice_range=get_tumor_slice_range(tumor_area_per_slice),
            largest_tumor_slice_index=get_largest_tumor_slice_index(tumor_area_per_slice),
            dice_score=get_dice_score(predicted_mask, ground_truth_mask) if ground_truth_mask is not None else None,
            iou_score=get_iou_score(predicted_mask, ground_truth_mask) if ground_truth_mask is not None else None,
            inference_seconds=round(time.perf_counter() - started_at, 2),
            device=self.model_provider.get_device_name(),
            model_name=str(metadata["model_name"]),
            model_epoch=metadata.get("epoch"),
        )
