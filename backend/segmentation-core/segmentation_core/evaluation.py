import math

import numpy as np
import torch

from segmentation_core.models import SegmentationModel, get_logits
from segmentation_core.runtime import get_autocast_context

TEST_TIME_FLIP_AXES: tuple[tuple[int, ...], ...] = ((), (2,), (3,), (4,))
GAUSSIAN_SIGMA_DIVISOR = 8
MINIMUM_IMPORTANCE = 1e-3

MaskLike = torch.Tensor | np.ndarray


def get_dice_score(predicted_mask: MaskLike, target_mask: MaskLike) -> float:
    predicted_values, target_values = torch.as_tensor(predicted_mask).bool(), torch.as_tensor(target_mask).bool()
    total = predicted_values.sum().item() + target_values.sum().item()
    if total == 0:
        return 1.0
    return 2.0 * (predicted_values & target_values).sum().item() / total


def get_iou_score(predicted_mask: MaskLike, target_mask: MaskLike) -> float:
    predicted_values, target_values = torch.as_tensor(predicted_mask).bool(), torch.as_tensor(target_mask).bool()
    union = (predicted_values | target_values).sum().item()
    if union == 0:
        return 1.0
    return (predicted_values & target_values).sum().item() / union


def get_metric_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    value_array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(value_array.mean()),
        "median": float(np.median(value_array)),
        "standard_deviation": float(value_array.std()),
        "percentile_25": float(np.percentile(value_array, 25)),
        "percentile_75": float(np.percentile(value_array, 75)),
    }


def get_window_starts(volume_size: int, window_size: int) -> list[int]:
    if volume_size <= window_size:
        return [0]
    window_count = math.ceil((volume_size - window_size) / (window_size / 2)) + 1
    return sorted({round(index * (volume_size - window_size) / (window_count - 1)) for index in range(window_count)})


def get_gaussian_importance_map(patch_shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    axis_coordinates = [
        (torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2) / (size / GAUSSIAN_SIGMA_DIVISOR)
        for size in patch_shape
    ]
    axis_grids = torch.meshgrid(*axis_coordinates, indexing="ij")
    squared_distance = torch.stack([grid ** 2 for grid in axis_grids]).sum(dim=0)
    importance_map = torch.exp(-0.5 * squared_distance)
    return (importance_map / importance_map.max()).clamp_(min=MINIMUM_IMPORTANCE)


class SlidingWindowPredictor:
    def __init__(self, model: SegmentationModel, patch_shape: tuple[int, ...], device: torch.device,
                 autocast_dtype: torch.dtype | None, window_batch_size: int = 2, is_test_time_augmentation_enabled: bool = False):
        self.model = model
        self.patch_shape = tuple(patch_shape)
        self.device = device
        self.autocast_dtype = autocast_dtype
        self.window_batch_size = max(1, window_batch_size)
        self.flip_axes_list = TEST_TIME_FLIP_AXES if is_test_time_augmentation_enabled else ((),)
        self.importance_map = get_gaussian_importance_map(self.patch_shape, device)

    def get_flipped_window_logits(self, window_images: torch.Tensor, flip_axes: tuple[int, ...]) -> torch.Tensor:
        flipped_images = window_images.flip(flip_axes) if flip_axes else window_images
        with get_autocast_context(self.device, self.autocast_dtype):
            window_logits = get_logits(self.model(flipped_images)).float()
        return window_logits.flip(flip_axes) if flip_axes else window_logits

    def get_window_logits(self, window_images: torch.Tensor) -> torch.Tensor:
        flipped_window_logits = [self.get_flipped_window_logits(window_images, flip_axes) for flip_axes in self.flip_axes_list]
        return torch.stack(flipped_window_logits).mean(dim=0)

    def get_window_slices_list(self, padded_shape: tuple[int, ...]) -> list[tuple[slice, ...]]:
        axis_starts = [get_window_starts(size, patch) for size, patch in zip(padded_shape, self.patch_shape)]
        return [
            tuple(slice(start, start + patch) for start, patch in zip((x_start, y_start, z_start), self.patch_shape))
            for x_start in axis_starts[0] for y_start in axis_starts[1] for z_start in axis_starts[2]
        ]

    @torch.no_grad()
    def get_logit_volume(self, image: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        volume_shape = tuple(image.shape[1:])
        padded_shape = tuple(max(volume_size, patch_size) for volume_size, patch_size in zip(volume_shape, self.patch_shape))
        volume_slices = tuple(slice(0, size) for size in volume_shape)
        padded_image = torch.zeros((image.shape[0], *padded_shape), dtype=torch.float32, device=self.device)
        padded_image[(slice(None), *volume_slices)] = image.to(self.device, torch.float32)

        logit_sum = torch.zeros(padded_shape, dtype=torch.float32, device=self.device)
        weight_sum = torch.zeros(padded_shape, dtype=torch.float32, device=self.device)
        window_slices_list = self.get_window_slices_list(padded_shape)
        for batch_start in range(0, len(window_slices_list), self.window_batch_size):
            batch_window_slices = window_slices_list[batch_start:batch_start + self.window_batch_size]
            window_images = torch.stack([padded_image[(slice(None), *window_slices)] for window_slices in batch_window_slices])
            window_logits = self.get_window_logits(window_images)
            for window_index, window_slices in enumerate(batch_window_slices):
                logit_sum[window_slices] += window_logits[window_index, 0] * self.importance_map
                weight_sum[window_slices] += self.importance_map
        return (logit_sum / weight_sum)[volume_slices]
