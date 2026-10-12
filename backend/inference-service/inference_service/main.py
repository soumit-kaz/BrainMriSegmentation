from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from inference_service.application.segmentation_job_service import SegmentationJobService
from inference_service.configuration import InferenceServiceSettings
from inference_service.helpers import configure_logging
from inference_service.infrastructure.file_scan_storage import FileScanStorage
from inference_service.infrastructure.png_slice_renderer import PngSliceRenderer
from inference_service.infrastructure.postgres_job_repository import PostgresSegmentationJobRepository
from inference_service.infrastructure.redis_messaging import (
    RedisJobEventPublisher,
    RedisJobEventSubscriber,
    RedisJobQueue,
    RedisRateLimiter,
    RedisWorkerStatusStore,
    create_async_redis_client,
    create_redis_client,
)
from inference_service.presentation.controllers.segmentation_controller import segmentation_router
from inference_service.presentation.controllers.system_controller import system_router
from inference_service.presentation.error_handlers import add_error_handlers


def create_application() -> FastAPI:
    configure_logging()
    settings = InferenceServiceSettings.from_environment()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
        redis_client = create_redis_client(settings.redis_url)
        async_redis_client = create_async_redis_client(settings.redis_url)
        job_repository = PostgresSegmentationJobRepository(settings.database_url, settings.database_pool_size)
        job_repository.initialize_schema()
        application.state.trusted_proxy_hop_count = settings.trusted_proxy_hop_count
        application.state.segmentation_job_service = SegmentationJobService(
            settings=settings,
            job_repository=job_repository,
            job_queue=RedisJobQueue(redis_client, settings),
            event_publisher=RedisJobEventPublisher(redis_client, settings.job_event_channel_prefix),
            event_subscriber=RedisJobEventSubscriber(async_redis_client, settings.job_event_channel_prefix),
            rate_limiter=RedisRateLimiter(redis_client),
            worker_status_store=RedisWorkerStatusStore(redis_client, settings.worker_status_ttl_seconds),
            scan_storage=FileScanStorage(settings),
            slice_renderer=PngSliceRenderer(),
        )
        yield
        await async_redis_client.aclose()
        redis_client.close()
        job_repository.engine.dispose()

    application = FastAPI(title="Glioma Segmentation Inference API", version="3.0.0", lifespan=lifespan)
    add_error_handlers(application)
    application.include_router(system_router)
    application.include_router(segmentation_router)
    return application


app = create_application()
