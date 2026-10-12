import logging
import math
import sys

import torch
from segmentation_core.models import SegmentationModel
from segmentation_core.settings import SegmentationSettings

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
MINIMUM_LEARNING_RATE_RATIO = 0.01
WARMUP_START_RATIO = 0.01
LAYER_SCALE_PARAMETER_SUFFIX = ".scale"


def get_learning_rate_factor(optimizer_step: int, settings: SegmentationSettings, steps_per_epoch: int) -> float:
    warmup_step_count = settings.warmup_epoch_count * steps_per_epoch
    if optimizer_step < warmup_step_count:
        return WARMUP_START_RATIO + (1.0 - WARMUP_START_RATIO) * optimizer_step / warmup_step_count
    decay_step_count = max(1, settings.epoch_count * steps_per_epoch - warmup_step_count)
    decay_progress = min(1.0, (optimizer_step - warmup_step_count) / decay_step_count)
    return MINIMUM_LEARNING_RATE_RATIO + (1.0 - MINIMUM_LEARNING_RATE_RATIO) * 0.5 * (1.0 + math.cos(math.pi * decay_progress))


class LearningRateSchedule:
    def __init__(self, settings: SegmentationSettings, steps_per_epoch: int):
        self.settings = settings
        self.steps_per_epoch = steps_per_epoch

    def __call__(self, optimizer_step: int) -> float:
        return get_learning_rate_factor(optimizer_step, self.settings, self.steps_per_epoch)


def get_optimizer(model: SegmentationModel, settings: SegmentationSettings, device: torch.device) -> torch.optim.Optimizer:
    decayed_parameters: list[torch.nn.Parameter] = []
    undecayed_parameters: list[torch.nn.Parameter] = []
    for parameter_name, parameter in model.named_parameters():
        is_decayed = parameter.ndim > 1 and not parameter_name.endswith(LAYER_SCALE_PARAMETER_SUFFIX)
        if is_decayed:
            decayed_parameters.append(parameter)
        else:
            undecayed_parameters.append(parameter)
    parameter_groups = [
        {"params": decayed_parameters, "weight_decay": settings.weight_decay},
        {"params": undecayed_parameters, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(parameter_groups, lr=settings.learning_rate, fused=device.type == "cuda")


def configure_logging(extra_handlers: list[logging.Handler] | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout), *(extra_handlers or [])]
    for handler in handlers:
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True)


def configure_training_logging(settings: SegmentationSettings, extra_handler: logging.Handler) -> None:
    settings.artifacts_directory.mkdir(parents=True, exist_ok=True)
    configure_logging([logging.FileHandler(settings.training_log_path), extra_handler])


def configure_torch_performance() -> None:
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
