from typing import Any

from pydantic import BaseModel, Field

from training_service.domain.training_run import TrainingCommand, TrainingRunRequest
from training_service.utils import get_non_empty_values


class TrainingStartRequest(BaseModel):
    command: TrainingCommand = TrainingCommand.TRAIN
    is_resume: bool = False
    epoch_count: int | None = Field(default=None, ge=1, le=1000)
    batch_size: int | None = Field(default=None, ge=1, le=32)
    learning_rate: float | None = Field(default=None, gt=0, le=1)
    maximum_case_count: int | None = Field(default=None, ge=0)

    def to_run_request(self) -> TrainingRunRequest:
        setting_overrides = get_non_empty_values({
            "epoch_count": self.epoch_count,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "maximum_case_count": self.maximum_case_count,
        })
        return TrainingRunRequest(self.command, self.is_resume, setting_overrides)


class DatasetInformationResponse(BaseModel):
    dataset_directory: str
    is_available: bool
    case_count: int
    message: str | None = None


class HealthResponse(BaseModel):
    status: str
    device: str
    device_name: str
    gpu_count: int


TrainingSnapshotResponse = dict[str, Any]
