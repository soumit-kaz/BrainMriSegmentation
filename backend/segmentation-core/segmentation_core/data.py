import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from nibabel.funcs import as_closest_canonical
from nibabel.loadsave import load as load_image
from nibabel.loadsave import save as save_image
from nibabel.nifti1 import Nifti1Image
import torch
from torch.utils.data import Dataset

from segmentation_core.settings import MASK_SUFFIX, MODALITY_SUFFIXES, NIFTI_EXTENSIONS, SegmentationSettings

MINIMUM_STANDARD_DEVIATION = 1e-6
MAXIMUM_FOREGROUND_COORDINATE_COUNT = 20_000
CACHE_IMAGE_FILE_NAME = "image.npy"
CACHE_MASK_FILE_NAME = "mask.npy"
CACHE_FOREGROUND_FILE_NAME = "foreground.npy"


def load_nifti_volume(file_path: Path) -> tuple[np.ndarray, Nifti1Image]:
    loaded_image = load_image(str(file_path))
    if not isinstance(loaded_image, Nifti1Image):
        raise ValueError(f"'{file_path.name}' is not a NIfTI-1 image.")
    canonical_image = as_closest_canonical(loaded_image)
    return np.asarray(canonical_image.get_fdata(dtype=np.float32)), canonical_image


def save_nifti_mask(mask_volume: np.ndarray, reference_image: Nifti1Image, file_path: Path) -> None:
    mask_header = reference_image.header.copy()
    mask_header.set_data_dtype(np.uint8)
    save_image(Nifti1Image(mask_volume.astype(np.uint8), reference_image.affine, mask_header), str(file_path))


def get_center_crop_slices(source_shape: tuple[int, ...], target_shape: tuple[int, ...]) -> tuple[tuple[slice, ...], tuple[slice, ...]]:
    source_slices: list[slice] = []
    target_slices: list[slice] = []
    for source_size, target_size in zip(source_shape, target_shape):
        if source_size >= target_size:
            source_start = (source_size - target_size) // 2
            source_slices.append(slice(source_start, source_start + target_size))
            target_slices.append(slice(0, target_size))
        else:
            target_start = (target_size - source_size) // 2
            source_slices.append(slice(0, source_size))
            target_slices.append(slice(target_start, target_start + source_size))
    return tuple(source_slices), tuple(target_slices)


def get_center_cropped_volume(volume: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    source_slices, target_slices = get_center_crop_slices(volume.shape[-3:], target_shape)
    cropped_volume = np.zeros(volume.shape[:-3] + tuple(target_shape), dtype=volume.dtype)
    cropped_volume[(..., *target_slices)] = volume[(..., *source_slices)]
    return cropped_volume


def get_restored_volume(cropped_volume: np.ndarray, original_shape: tuple[int, ...]) -> np.ndarray:
    source_slices, target_slices = get_center_crop_slices(original_shape, cropped_volume.shape[-3:])
    restored_volume = np.zeros(cropped_volume.shape[:-3] + tuple(original_shape), dtype=cropped_volume.dtype)
    restored_volume[(..., *source_slices)] = cropped_volume[(..., *target_slices)]
    return restored_volume


def get_slice_normalized_volume(volume: np.ndarray) -> np.ndarray:
    brain_mask = volume > 0
    brain_voxel_counts = np.maximum(brain_mask.sum(axis=(0, 1)), 1)
    slice_means = (volume * brain_mask).sum(axis=(0, 1)) / brain_voxel_counts
    slice_variances = (((volume - slice_means) ** 2) * brain_mask).sum(axis=(0, 1)) / brain_voxel_counts
    slice_standard_deviations = np.maximum(np.sqrt(slice_variances), MINIMUM_STANDARD_DEVIATION)
    normalized_volume = (volume - slice_means) / slice_standard_deviations
    return np.where(brain_mask, normalized_volume, 0.0).astype(np.float32)


def get_stacked_image(modality_volumes: list[np.ndarray], cropped_volume_shape: tuple[int, ...]) -> np.ndarray:
    if len({volume.shape for volume in modality_volumes}) != 1:
        raise ValueError("All MRI modalities must have the same shape.")
    stacked_image = np.stack([get_slice_normalized_volume(volume) for volume in modality_volumes])
    return get_center_cropped_volume(stacked_image, cropped_volume_shape)


def get_whole_tumor_mask(segmentation_volume: np.ndarray) -> np.ndarray:
    return (segmentation_volume > 0).astype(np.uint8)


def get_display_volume(volume: np.ndarray) -> np.ndarray:
    brain_values = volume[volume > 0]
    if brain_values.size == 0:
        return np.zeros(volume.shape, dtype=np.uint8)
    lower_value, upper_value = np.percentile(brain_values, (0.5, 99.5))
    scaled_volume = (volume - lower_value) / max(float(upper_value - lower_value), MINIMUM_STANDARD_DEVIATION)
    return (np.clip(scaled_volume, 0.0, 1.0) * 255).astype(np.uint8)


class BratsCaseRepository:
    def __init__(self, dataset_directory: Path):
        self.dataset_directory = dataset_directory

    @staticmethod
    def get_file_path(case_directory: Path, suffix: str) -> Path | None:
        for extension in NIFTI_EXTENSIONS:
            matching_paths = sorted(case_directory.glob(f"*-{suffix}{extension}"))
            if matching_paths:
                return matching_paths[0]
        return None

    def get_required_file_path(self, case_name: str, suffix: str) -> Path:
        file_path = self.get_file_path(self.dataset_directory / case_name, suffix)
        if file_path is None:
            raise FileNotFoundError(f"Case '{case_name}' has no '{suffix}' file.")
        return file_path

    def is_complete_case(self, case_directory: Path) -> bool:
        return all(self.get_file_path(case_directory, suffix) is not None for suffix in (*MODALITY_SUFFIXES, MASK_SUFFIX))

    def get_case_names(self) -> list[str]:
        if not self.dataset_directory.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {self.dataset_directory}")
        case_directories = (path for path in self.dataset_directory.iterdir() if path.is_dir())
        return sorted(directory.name for directory in case_directories if self.is_complete_case(directory))

    def get_modality_paths(self, case_name: str) -> list[Path]:
        return [self.get_required_file_path(case_name, suffix) for suffix in MODALITY_SUFFIXES]

    def get_mask_path(self, case_name: str) -> Path:
        return self.get_required_file_path(case_name, MASK_SUFFIX)


@dataclass
class PreprocessedCase:
    image: np.ndarray
    mask: np.ndarray
    foreground_coordinates: np.ndarray


class PreprocessedCaseProvider:
    def __init__(self, case_repository: BratsCaseRepository, settings: SegmentationSettings):
        self.case_repository = case_repository
        self.is_cache_enabled = settings.is_case_cache_enabled
        self.cropped_volume_shape = settings.cropped_volume_shape
        shape_key = "x".join(str(size) for size in settings.cropped_volume_shape)
        self.cache_directory = settings.cache_directory / f"cropped_{shape_key}"

    def get_cache_case_directory(self, case_name: str) -> Path:
        return self.cache_directory / case_name

    def is_cached(self, case_name: str) -> bool:
        return (self.get_cache_case_directory(case_name) / CACHE_FOREGROUND_FILE_NAME).exists()

    def get_computed_case(self, case_name: str) -> PreprocessedCase:
        modality_volumes = [load_nifti_volume(path)[0] for path in self.case_repository.get_modality_paths(case_name)]
        image = get_stacked_image(modality_volumes, self.cropped_volume_shape)
        segmentation_volume, _ = load_nifti_volume(self.case_repository.get_mask_path(case_name))
        mask = get_center_cropped_volume(get_whole_tumor_mask(segmentation_volume), self.cropped_volume_shape)
        foreground_coordinates = np.argwhere(mask > 0)
        if len(foreground_coordinates) > MAXIMUM_FOREGROUND_COORDINATE_COUNT:
            random_generator = np.random.default_rng(0)
            selected_indices = random_generator.choice(len(foreground_coordinates), MAXIMUM_FOREGROUND_COORDINATE_COUNT, replace=False)
            foreground_coordinates = foreground_coordinates[selected_indices]
        return PreprocessedCase(image, mask, foreground_coordinates.astype(np.int16))

    def add_case_to_cache(self, case_name: str) -> None:
        if self.is_cached(case_name):
            return
        computed_case = self.get_computed_case(case_name)
        case_directory = self.get_cache_case_directory(case_name)
        case_directory.mkdir(parents=True, exist_ok=True)
        cache_arrays = {
            CACHE_IMAGE_FILE_NAME: computed_case.image.astype(np.float16),
            CACHE_MASK_FILE_NAME: computed_case.mask,
            CACHE_FOREGROUND_FILE_NAME: computed_case.foreground_coordinates,
        }
        for file_name, array in cache_arrays.items():
            temporary_path = case_directory / f"{file_name}.{os.getpid()}.tmp"
            with temporary_path.open("wb") as temporary_file:
                np.save(temporary_file, array)
            temporary_path.replace(case_directory / file_name)

    def get_case(self, case_name: str) -> PreprocessedCase:
        if not self.is_cache_enabled:
            return self.get_computed_case(case_name)
        self.add_case_to_cache(case_name)
        case_directory = self.get_cache_case_directory(case_name)
        return PreprocessedCase(
            image=np.load(case_directory / CACHE_IMAGE_FILE_NAME, mmap_mode="r"),
            mask=np.load(case_directory / CACHE_MASK_FILE_NAME, mmap_mode="r"),
            foreground_coordinates=np.load(case_directory / CACHE_FOREGROUND_FILE_NAME),
        )


def get_case_split(case_names: list[str], settings: SegmentationSettings) -> dict[str, list[str]]:
    shuffled_case_names = list(case_names)
    random.Random(settings.random_seed).shuffle(shuffled_case_names)
    requested_counts = (settings.training_case_count, settings.validation_case_count, settings.test_case_count)
    if len(shuffled_case_names) < sum(requested_counts):
        scale = len(shuffled_case_names) / sum(requested_counts)
        validation_count = max(1, round(settings.validation_case_count * scale))
        test_count = max(1, round(settings.test_case_count * scale))
        training_count = len(shuffled_case_names) - validation_count - test_count
    else:
        training_count, validation_count, test_count = requested_counts
    if training_count < 1:
        raise ValueError(f"Not enough cases to split: found {len(shuffled_case_names)}.")
    validation_end = training_count + validation_count
    return {
        "training": shuffled_case_names[:training_count],
        "validation": shuffled_case_names[training_count:validation_end],
        "test": shuffled_case_names[validation_end:validation_end + test_count],
    }


def get_random_patch_slices(volume_shape: tuple[int, ...], patch_shape: tuple[int, ...]) -> tuple[slice, ...]:
    starts = [random.randint(0, volume_size - patch_size) for volume_size, patch_size in zip(volume_shape, patch_shape)]
    return tuple(slice(start, start + patch_size) for start, patch_size in zip(starts, patch_shape))


def get_foreground_patch_slices(volume_shape: tuple[int, ...], patch_shape: tuple[int, ...],
                                foreground_coordinates: np.ndarray) -> tuple[slice, ...]:
    tumor_coordinate = foreground_coordinates[random.randrange(len(foreground_coordinates))]
    starts: list[int] = []
    for coordinate, volume_size, patch_size in zip(tumor_coordinate, volume_shape, patch_shape):
        lowest_start = max(0, int(coordinate) - patch_size + 1)
        highest_start = min(volume_size - patch_size, int(coordinate))
        starts.append(random.randint(lowest_start, highest_start))
    return tuple(slice(start, start + patch_size) for start, patch_size in zip(starts, patch_shape))


class CaseCacheDataset(Dataset[int]):
    def __init__(self, case_provider: PreprocessedCaseProvider, case_names: list[str]):
        self.case_provider = case_provider
        self.case_names = case_names

    def __len__(self) -> int:
        return len(self.case_names)

    def __getitem__(self, index: int) -> int:
        self.case_provider.add_case_to_cache(self.case_names[index])
        return index


class BratsSegmentationDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    def __init__(self, case_provider: PreprocessedCaseProvider, case_names: list[str], settings: SegmentationSettings,
                 is_training: bool, sample_count: int | None = None):
        self.case_provider = case_provider
        self.case_names = case_names
        self.settings = settings
        self.is_training = is_training
        self.sample_count = sample_count or len(case_names)

    def __len__(self) -> int:
        return self.sample_count

    def get_training_patch_slices(self, preprocessed_case: PreprocessedCase) -> tuple[slice, ...]:
        has_foreground = len(preprocessed_case.foreground_coordinates) > 0
        if has_foreground and random.random() < self.settings.foreground_sampling_probability:
            return get_foreground_patch_slices(preprocessed_case.mask.shape, self.settings.patch_shape,
                                               preprocessed_case.foreground_coordinates)
        return get_random_patch_slices(preprocessed_case.mask.shape, self.settings.patch_shape)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        case_name = self.case_names[index % len(self.case_names)]
        preprocessed_case = self.case_provider.get_case(case_name)
        image, mask = preprocessed_case.image, preprocessed_case.mask
        if self.is_training:
            patch_slices = self.get_training_patch_slices(preprocessed_case)
            image = image[(slice(None), *patch_slices)]
            mask = mask[patch_slices]
        image_tensor = torch.from_numpy(np.asarray(image, dtype=np.float32))
        mask_tensor = torch.from_numpy(np.asarray(mask, dtype=np.float32)).unsqueeze(0)
        return image_tensor, mask_tensor, case_name


def get_modality_dropped_image(image: torch.Tensor, modality_dropout_probability: float) -> torch.Tensor:
    if random.random() >= modality_dropout_probability:
        return image
    kept_modalities = torch.ones((image.shape[0], 1, 1, 1), device=image.device)
    kept_modalities[random.randrange(image.shape[0])] = 0.0
    return image * kept_modalities


def get_augmented_batch(images: torch.Tensor, masks: torch.Tensor, modality_dropout_probability: float) -> tuple[torch.Tensor, torch.Tensor]:
    augmented_images: list[torch.Tensor] = []
    augmented_masks: list[torch.Tensor] = []
    for image, mask in zip(images, masks):
        flip_axes = [axis for axis in (1, 2, 3) if random.random() < 0.5]
        if flip_axes:
            image, mask = image.flip(flip_axes), mask.flip(flip_axes)
        if image.shape[1] == image.shape[2] and random.random() < 0.5:
            rotation_count = random.randint(1, 3)
            image, mask = image.rot90(rotation_count, dims=(1, 2)), mask.rot90(rotation_count, dims=(1, 2))
        channel_count = image.shape[0]
        intensity_scale = torch.empty((channel_count, 1, 1, 1), device=image.device).uniform_(0.9, 1.1)
        intensity_shift = torch.empty((channel_count, 1, 1, 1), device=image.device).uniform_(-0.1, 0.1)
        brain_mask = (image != 0).any(dim=0, keepdim=True)
        image = torch.where(brain_mask, image * intensity_scale + intensity_shift, image)
        augmented_images.append(get_modality_dropped_image(image, modality_dropout_probability))
        augmented_masks.append(mask)
    return torch.stack(augmented_images), torch.stack(augmented_masks)
