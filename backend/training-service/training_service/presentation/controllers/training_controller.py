from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from training_service.application.training_session_service import TrainingSessionService
from training_service.domain.training_run import TrainingConflictError
from training_service.presentation.dependencies import get_training_session_service
from training_service.presentation.schemas import TrainingSnapshotResponse, TrainingStartRequest

training_router = APIRouter(prefix="/api/training", tags=["training"])
TrainingSessionServiceDependency = Annotated[TrainingSessionService, Depends(get_training_session_service)]
SERVER_SENT_EVENT_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


@training_router.get("/status")
def get_training_status(training_session_service: TrainingSessionServiceDependency) -> TrainingSnapshotResponse:
    return training_session_service.get_snapshot()


@training_router.get("/events")
async def get_training_events(training_session_service: TrainingSessionServiceDependency) -> StreamingResponse:
    return StreamingResponse(training_session_service.get_event_stream(), media_type="text/event-stream",
                             headers=SERVER_SENT_EVENT_HEADERS)


@training_router.post("/start", status_code=status.HTTP_202_ACCEPTED)
def start_training(start_request: TrainingStartRequest,
                   training_session_service: TrainingSessionServiceDependency) -> TrainingSnapshotResponse:
    try:
        return training_session_service.start_run(start_request.to_run_request())
    except TrainingConflictError as error:
        raise HTTPException(status.HTTP_409_CONFLICT, str(error)) from error
    except ValueError as error:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(error)) from error


@training_router.post("/stop")
def stop_training(training_session_service: TrainingSessionServiceDependency) -> TrainingSnapshotResponse:
    try:
        return training_session_service.request_stop()
    except TrainingConflictError as error:
        raise HTTPException(status.HTTP_409_CONFLICT, str(error)) from error
