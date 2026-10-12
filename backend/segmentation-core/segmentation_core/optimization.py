import copy
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as functional

from segmentation_core.models import ModelOutput, SegmentationModel
from segmentation_core.settings import SegmentationSettings

SMOOTHING_FACTOR = 1.0
MORPHOLOGY_KERNEL_SIZE = 3


class TumorAwareSegmentationLoss(nn.Module):
    def __init__(self, settings: SegmentationSettings):
        super().__init__()
        self.false_negative_weight = settings.tversky_false_negative_weight
        self.false_positive_weight = 1.0 - settings.tversky_false_negative_weight
        self.tversky_focal_exponent = settings.tversky_focal_exponent
        self.focal_alpha = settings.focal_alpha
        self.focal_gamma = settings.focal_gamma
        self.maximum_boundary_weight = settings.maximum_boundary_weight
        self.boundary_band_width = settings.boundary_band_width
        self.volume_consistency_weight = settings.volume_consistency_weight
        self.training_progress = 0.0

    def set_training_progress(self, training_progress: float) -> None:
        self.training_progress = min(max(training_progress, 0.0), 1.0)

    @staticmethod
    def get_dilated(mask: torch.Tensor) -> torch.Tensor:
        return functional.max_pool3d(mask, MORPHOLOGY_KERNEL_SIZE, stride=1, padding=MORPHOLOGY_KERNEL_SIZE // 2)

    def get_boundary_band(self, targets: torch.Tensor) -> torch.Tensor:
        dilated_targets, eroded_targets = targets, targets
        for _ in range(self.boundary_band_width):
            dilated_targets = self.get_dilated(dilated_targets)
            eroded_targets = 1.0 - self.get_dilated(1.0 - eroded_targets)
        return dilated_targets - eroded_targets

    def get_focal_tversky_loss(self, probabilities: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        reduction_dimensions = tuple(range(1, probabilities.ndim))
        true_positive = (probabilities * targets).sum(reduction_dimensions)
        false_negative = ((1.0 - probabilities) * targets).sum(reduction_dimensions)
        false_positive = (probabilities * (1.0 - targets)).sum(reduction_dimensions)
        tversky_index = (true_positive + SMOOTHING_FACTOR) / (
            true_positive + self.false_negative_weight * false_negative + self.false_positive_weight * false_positive + SMOOTHING_FACTOR
        )
        return (1.0 - tversky_index).clamp(min=0.0).pow(self.tversky_focal_exponent).mean()

    def get_boundary_focal_loss(self, logits: torch.Tensor, probabilities: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        cross_entropy = functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        target_probabilities = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
        alpha_weights = self.focal_alpha * targets + (1.0 - self.focal_alpha) * (1.0 - targets)
        focal_loss = alpha_weights * (1.0 - target_probabilities).pow(self.focal_gamma) * cross_entropy
        boundary_weights = 1.0 + self.maximum_boundary_weight * self.training_progress * self.get_boundary_band(targets)
        return (boundary_weights * focal_loss).sum() / boundary_weights.sum()

    def get_volume_consistency_loss(self, probabilities: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        reduction_dimensions = tuple(range(1, probabilities.ndim))
        predicted_volume = probabilities.sum(reduction_dimensions)
        target_volume = targets.sum(reduction_dimensions)
        has_tumor = target_volume > 0
        if not bool(has_tumor.any()):
            return probabilities.new_zeros(())
        relative_volume_error = (predicted_volume - target_volume).abs() / (predicted_volume + target_volume + SMOOTHING_FACTOR)
        return relative_volume_error[has_tumor].mean()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        targets = targets.float()
        probabilities = torch.sigmoid(logits)
        return (self.get_focal_tversky_loss(probabilities, targets)
                + self.get_boundary_focal_loss(logits, probabilities, targets)
                + self.volume_consistency_weight * self.get_volume_consistency_loss(probabilities, targets))


class DeepSupervisionLoss(nn.Module):
    def __init__(self, base_loss: TumorAwareSegmentationLoss):
        super().__init__()
        self.base_loss = base_loss

    def set_training_progress(self, training_progress: float) -> None:
        self.base_loss.set_training_progress(training_progress)

    def forward(self, outputs: ModelOutput, targets: torch.Tensor) -> torch.Tensor:
        if isinstance(outputs, torch.Tensor):
            return self.base_loss(outputs, targets)
        weights = [0.5 ** index for index in range(len(outputs))]
        weighted_losses: list[torch.Tensor] = []
        for weight, output in zip(weights, outputs):
            scaled_targets = targets if output.shape[2:] == targets.shape[2:] else \
                functional.interpolate(targets, size=output.shape[2:], mode="nearest")
            weighted_losses.append(weight * self.base_loss(output, scaled_targets))
        return torch.stack(weighted_losses).sum() / sum(weights)


class ModelExponentialMovingAverage:
    WARMUP_UPDATE_COUNT = 2000

    def __init__(self, model: SegmentationModel, decay: float):
        self.averaged_model: SegmentationModel = copy.deepcopy(model).eval()
        for parameter in self.averaged_model.parameters():
            parameter.requires_grad_(False)
        self.decay = decay
        self.update_count = 0

    def get_current_decay(self) -> float:
        return self.decay * (1.0 - math.exp(-self.update_count / self.WARMUP_UPDATE_COUNT))

    @torch.no_grad()
    def update(self, model: SegmentationModel) -> None:
        self.update_count += 1
        current_decay = self.get_current_decay()
        model_state = model.state_dict()
        for name, averaged_value in self.averaged_model.state_dict().items():
            if averaged_value.dtype.is_floating_point:
                averaged_value.lerp_(model_state[name].detach(), 1.0 - current_decay)
            else:
                averaged_value.copy_(model_state[name])

    def get_state(self) -> dict[str, Any]:
        return {"model_state": self.averaged_model.state_dict(), "update_count": self.update_count}

    def set_state(self, state: dict[str, Any]) -> None:
        self.averaged_model.load_state_dict(state["model_state"])
        self.update_count = state["update_count"]
