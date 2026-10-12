from typing import Annotated

import torch
from fastapi import APIRouter, Depends
from segmentation_core.runtime import get_device, get_device_name

from training_service.application.training_session_service import TrainingSessionService
from training_service.presentation.dependencies import get_training_session_service
from training_service.presentation.schemas import DatasetInformationResponse, HealthResponse

system_router = APIRouter(prefix="/api/training", tags=["system"])
TrainingSessionServiceDependency = Annotated[TrainingSessionService, Depends(get_training_session_service)]


@system_router.get("/health")
def get_health() -> HealthResponse:
    device = get_device()
    return HealthResponse(status="healthy", device=device.type, device_name=get_device_name(device),
                          gpu_count=torch.cuda.device_count())


@system_router.get("/dataset")
def get_dataset_information(training_session_service: TrainingSessionServiceDependency) -> DatasetInformationResponse:
    dataset_directory = str(training_session_service.settings.dataset_directory)
    try:
        case_count = training_session_service.get_dataset_case_count()
    except FileNotFoundError as error:
        return DatasetInformationResponse(dataset_directory=dataset_directory, is_available=False, case_count=0, message=str(error))
    return DatasetInformationResponse(dataset_directory=dataset_directory, is_available=case_count > 0, case_count=case_count)
