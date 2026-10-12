from fastapi import APIRouter, Response, status

from inference_service.presentation.dependencies import SegmentationJobServiceDependency
from inference_service.presentation.schemas import HealthResponse, ModelInformationResponse, QueueStatusResponse

system_router = APIRouter(prefix="/api", tags=["system"])


@system_router.get("/health/live")
def get_liveness() -> HealthResponse:
    return HealthResponse(status="alive")


@system_router.get("/health/ready")
def get_readiness(response: Response, segmentation_job_service: SegmentationJobServiceDependency) -> HealthResponse:
    if segmentation_job_service.is_ready():
        return HealthResponse(status="ready")
    response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(status="unavailable")


@system_router.get("/model")
def get_model_information(segmentation_job_service: SegmentationJobServiceDependency) -> ModelInformationResponse:
    return ModelInformationResponse.from_domain(segmentation_job_service.get_model_information())


@system_router.get("/queue")
def get_queue_status(segmentation_job_service: SegmentationJobServiceDependency) -> QueueStatusResponse:
    return QueueStatusResponse.from_domain(segmentation_job_service.get_queue_status())
