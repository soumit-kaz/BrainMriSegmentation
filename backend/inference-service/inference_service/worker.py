import logging
import signal
import threading
from types import FrameType

import torch
from segmentation_core.settings import SegmentationSettings

from inference_service.application.segmentation_worker_service import SegmentationWorkerService
from inference_service.configuration import InferenceServiceSettings
from inference_service.helpers import configure_logging
from inference_service.infrastructure.checkpoint_model_provider import CheckpointModelProvider
from inference_service.infrastructure.file_scan_storage import FileScanStorage
from inference_service.infrastructure.postgres_job_repository import PostgresSegmentationJobRepository
from inference_service.infrastructure.redis_messaging import (
    RedisJobEventPublisher,
    RedisJobQueue,
    RedisWorkerStatusStore,
    create_redis_client,
)

logger = logging.getLogger("segmentation.inference.worker")


def limit_gpu_memory(settings: InferenceServiceSettings) -> None:
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(settings.gpu_memory_fraction)
        torch.backends.cudnn.benchmark = True


def create_worker_service(settings: InferenceServiceSettings) -> SegmentationWorkerService:
    redis_client = create_redis_client(settings.redis_url)
    job_repository = PostgresSegmentationJobRepository(settings.database_url, settings.database_pool_size)
    job_repository.initialize_schema()
    return SegmentationWorkerService(
        settings=settings,
        job_repository=job_repository,
        job_queue=RedisJobQueue(redis_client, settings),
        event_publisher=RedisJobEventPublisher(redis_client, settings.job_event_channel_prefix),
        worker_status_store=RedisWorkerStatusStore(redis_client, settings.worker_status_ttl_seconds),
        scan_storage=FileScanStorage(settings),
        model_provider=CheckpointModelProvider(SegmentationSettings.from_environment()),
    )


def run_worker() -> None:
    configure_logging()
    settings = InferenceServiceSettings.from_environment()
    limit_gpu_memory(settings)
    stop_event = threading.Event()

    def request_stop(signal_number: int, _: FrameType | None) -> None:
        logger.info("Received signal %d; finishing the current job before stopping.", signal_number)
        stop_event.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    create_worker_service(settings).run(stop_event)


if __name__ == "__main__":
    run_worker()
