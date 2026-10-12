import json
import os
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

MODALITY_SUFFIXES: tuple[str, ...] = ("t1n", "t1c", "t2w", "t2f")
MASK_SUFFIX = "seg"
MODEL_NAME = "modality_aware_attention_unet"
NIFTI_EXTENSIONS: tuple[str, ...] = (".nii.gz", ".nii")
ENVIRONMENT_VARIABLE_PREFIX = "SEGMENTATION_"


def get_parsed_setting_value(raw_value: str, default_value: Any) -> Any:
    if isinstance(default_value, bool):
        return raw_value.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(default_value, tuple):
        return tuple(int(part) for part in raw_value.split(","))
    if isinstance(default_value, Path):
        return Path(raw_value)
    return type(default_value)(raw_value)


@dataclass(frozen=True)
class SegmentationSettings:
    dataset_directory: Path = Path("/data")
    artifacts_directory: Path = Path("/artifacts")
    cache_directory: Path = Path("/cache")

    cropped_volume_shape: tuple[int, ...] = (140, 190, 140)
    patch_shape: tuple[int, ...] = (128, 128, 128)

    attention_reduction_ratio: int = 8
    spatial_attention_kernel_size: int = 7
    modality_channel_count: int = 8
    encoder_channel_counts: tuple[int, ...] = (24, 48, 96, 192, 320)
    stage_depths: tuple[int, ...] = (1, 1, 2, 2, 1)
    attention_block_count: int = 3
    area_count: int = 4
    attention_head_dimension: int = 32
    drop_path_rate: float = 0.1
    is_deep_supervision_enabled: bool = True
    is_gradient_checkpointing_enabled: bool = True

    training_case_count: int = 800
    validation_case_count: int = 200
    test_case_count: int = 251
    maximum_case_count: int = 0

    foreground_sampling_probability: float = 0.33
    modality_dropout_probability: float = 0.1
    is_augmentation_enabled: bool = True
    augmentation_close_epoch_count: int = 5
    iterations_per_epoch: int = 0

    epoch_count: int = 60
    batch_size: int = 2
    gradient_accumulation_step_count: int = 1
    is_auto_batch_fitting_enabled: bool = True
    early_stopping_patience: int = 15
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    warmup_epoch_count: int = 2
    maximum_gradient_norm: float = 1.0
    ema_decay: float = 0.999
    tversky_false_negative_weight: float = 0.7
    tversky_focal_exponent: float = 0.75
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    maximum_boundary_weight: float = 4.0
    boundary_band_width: int = 2
    volume_consistency_weight: float = 0.5
    prediction_threshold: float = 0.5

    data_loader_worker_count: int = 4
    random_seed: int = 42
    is_mixed_precision_enabled: bool = True
    is_case_cache_enabled: bool = True
    is_test_time_augmentation_enabled: bool = True
    sliding_window_batch_size: int = 2

    @classmethod
    def from_environment(cls) -> "SegmentationSettings":
        environment_values: dict[str, Any] = {}
        for settings_field in fields(cls):
            raw_value = os.getenv(f"{ENVIRONMENT_VARIABLE_PREFIX}{settings_field.name.upper()}")
            if raw_value is not None:
                environment_values[settings_field.name] = get_parsed_setting_value(raw_value, settings_field.default)
        return cls().get_updated(**environment_values)

    def get_updated(self, **changed_values: Any) -> "SegmentationSettings":
        updated_settings = replace(self, **{name: value for name, value in changed_values.items() if value is not None})
        updated_settings.validate()
        return updated_settings

    def validate(self) -> None:
        if len(self.cropped_volume_shape) != 3 or len(self.patch_shape) != 3:
            raise ValueError("cropped_volume_shape and patch_shape must have three dimensions.")
        if any(patch_size % 16 != 0 for patch_size in self.patch_shape):
            raise ValueError("Every patch_shape dimension must be divisible by 16.")
        if any(patch_size > cropped_size for patch_size, cropped_size in zip(self.patch_shape, self.cropped_volume_shape)):
            raise ValueError("patch_shape must not be larger than cropped_volume_shape.")
        if len(self.encoder_channel_counts) != 5 or len(self.stage_depths) != 5:
            raise ValueError("encoder_channel_counts and stage_depths need five values.")
        if any(channel_count % self.attention_head_dimension for channel_count in self.encoder_channel_counts[3:]):
            raise ValueError("The two deepest encoder channel counts must be divisible by attention_head_dimension.")
        if not 0.0 < self.tversky_false_negative_weight < 1.0:
            raise ValueError("tversky_false_negative_weight must be between 0 and 1.")
        if self.batch_size < 1 or self.epoch_count < 1 or self.gradient_accumulation_step_count < 1:
            raise ValueError("batch_size, epoch_count and gradient_accumulation_step_count must be positive.")
        if not 0.0 <= self.foreground_sampling_probability <= 1.0 or not 0.0 <= self.modality_dropout_probability <= 1.0:
            raise ValueError("foreground_sampling_probability and modality_dropout_probability must be in [0, 1].")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1).")

    @property
    def best_model_path(self) -> Path:
        return self.artifacts_directory / "best_model.pt"

    @property
    def last_checkpoint_path(self) -> Path:
        return self.artifacts_directory / "last_checkpoint.pt"

    @property
    def training_history_path(self) -> Path:
        return self.artifacts_directory / "training_history.json"

    @property
    def data_split_path(self) -> Path:
        return self.artifacts_directory / "data_split.json"

    @property
    def test_report_path(self) -> Path:
        return self.artifacts_directory / "test_report.json"

    @property
    def training_log_path(self) -> Path:
        return self.artifacts_directory / "training.log"


def is_nifti_file(file_name: str) -> bool:
    return file_name.lower().endswith(NIFTI_EXTENSIONS)


def read_json_file(file_path: Path) -> Any:
    return json.loads(file_path.read_text(encoding="utf-8"))


def write_json_file(file_path: Path, content: Any) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = file_path.with_suffix(f"{file_path.suffix}.{os.getpid()}.tmp")
    temporary_path.write_text(json.dumps(content, indent=2, default=str), encoding="utf-8")
    temporary_path.replace(file_path)
