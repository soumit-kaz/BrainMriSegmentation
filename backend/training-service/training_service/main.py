import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from segmentation_core.settings import SegmentationSettings

from training_service.application.training_session_service import TrainingSessionService
from training_service.helpers import configure_logging
from training_service.infrastructure.training_process import SpawnedTrainingProcessLauncher
from training_service.presentation.controllers.system_controller import system_router
from training_service.presentation.controllers.training_controller import training_router


def create_application() -> FastAPI:
    configure_logging()
    training_session_service = TrainingSessionService(SegmentationSettings.from_environment(), SpawnedTrainingProcessLauncher())

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
        training_session_service.attach_event_loop(asyncio.get_running_loop())
        application.state.training_session_service = training_session_service
        yield
        training_session_service.shutdown()

    application = FastAPI(title="Glioma Segmentation Training Service", version="2.0.0", lifespan=lifespan)
    application.include_router(system_router)
    application.include_router(training_router)
    return application


app = create_application()
