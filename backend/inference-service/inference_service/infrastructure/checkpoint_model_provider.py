import logging
import threading
from typing import Any

from segmentation_core.evaluation import SlidingWindowPredictor
from segmentation_core.models import SegmentationModel, get_parameter_count, load_model_checkpoint
from segmentation_core.runtime import get_autocast_dtype, get_device, get_device_name, get_precision_name
from segmentation_core.settings import SegmentationSettings

from inference_service.domain.segmentation_job import ModelInformation, ModelNotReadyError
from inference_service.utils import get_utc_now

logger = logging.getLogger("segmentation.inference.worker")


class CheckpointModelProvider:
    def __init__(self, settings: SegmentationSettings):
        self.settings = settings
        self.model_path = settings.best_model_path
        self.device = get_device()
        self.autocast_dtype = get_autocast_dtype(self.device, settings.is_mixed_precision_enabled)
        self.lock = threading.Lock()
        self.model: SegmentationModel | None = None
        self.metadata: dict[str, Any] = {}
        self.loaded_modified_time: float | None = None
        self.loaded_at: str | None = None

    def get_device_name(self) -> str:
        return get_device_name(self.device)

    def has_newer_model_file(self) -> bool:
        return self.model_path.exists() and self.model_path.stat().st_mtime != self.loaded_modified_time

    def get_model(self) -> tuple[SegmentationModel, dict[str, Any]]:
        with self.lock:
            if self.has_newer_model_file():
                modified_time = self.model_path.stat().st_mtime
                self.model, self.metadata = load_model_checkpoint(self.model_path, self.device)
                self.loaded_modified_time = modified_time
                self.loaded_at = get_utc_now().isoformat()
                logger.info("Loaded model from epoch %s on %s.", self.metadata.get("epoch"), self.device)
            if self.model is None:
                raise ModelNotReadyError(f"No trained model found at {self.model_path}. Train the model first.")
            return self.model, self.metadata

    def get_predictor(self) -> tuple[SlidingWindowPredictor, dict[str, Any]]:
        model, metadata = self.get_model()
        predictor = SlidingWindowPredictor(
            model, tuple(metadata["preprocessing"]["patch_shape"]), self.device, self.autocast_dtype,
            self.settings.sliding_window_batch_size, self.settings.is_test_time_augmentation_enabled,
        )
        return predictor, metadata

    def get_information(self) -> ModelInformation:
        try:
            model, metadata = self.get_model()
        except ModelNotReadyError as error:
            model, metadata, message = None, {}, str(error)
        else:
            message = None
        return ModelInformation(
            is_loaded=model is not None,
            model_name=metadata.get("model_name"),
            device=self.device.type,
            device_name=self.get_device_name(),
            precision=get_precision_name(self.autocast_dtype),
            is_test_time_augmentation_enabled=self.settings.is_test_time_augmentation_enabled,
            epoch=metadata.get("epoch"),
            metrics=metadata.get("metrics"),
            architecture=metadata.get("architecture"),
            parameter_count=get_parameter_count(model) if model is not None else None,
            loaded_at=self.loaded_at,
            message=message,
        )
