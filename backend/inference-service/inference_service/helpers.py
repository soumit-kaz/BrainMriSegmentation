import logging
import re
import sys
from pathlib import Path

import numpy as np
from nibabel.loadsave import load as load_image
from segmentation_core.settings import MODALITY_SUFFIXES

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
NIFTI_COMPRESSED_EXTENSION = ".nii.gz"
NIFTI_EXTENSION = ".nii"
DEFAULT_CASE_NAME = "case"


def configure_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)


def get_nifti_extension(file_name: str) -> str:
    return NIFTI_COMPRESSED_EXTENSION if file_name.lower().endswith(NIFTI_COMPRESSED_EXTENSION) else NIFTI_EXTENSION


def get_case_name(file_name: str) -> str:
    base_name = Path(file_name or DEFAULT_CASE_NAME).name
    extension = get_nifti_extension(base_name)
    if base_name.lower().endswith(extension):
        base_name = base_name[: -len(extension)]
    return re.sub(rf"[-_]?{MODALITY_SUFFIXES[0]}$", "", base_name, flags=re.IGNORECASE) or DEFAULT_CASE_NAME


def get_nifti_shape(file_path: Path) -> tuple[int, ...]:
    return tuple(int(size) for size in getattr(load_image(str(file_path)), "shape"))


def get_tumor_slice_range(tumor_area_per_slice: list[int]) -> list[int] | None:
    tumor_slice_indices = [index for index, area in enumerate(tumor_area_per_slice) if area > 0]
    return [tumor_slice_indices[0], tumor_slice_indices[-1]] if tumor_slice_indices else None


def get_largest_tumor_slice_index(tumor_area_per_slice: list[int]) -> int:
    if not any(tumor_area_per_slice):
        return len(tumor_area_per_slice) // 2
    return int(np.argmax(tumor_area_per_slice))


def get_volume_cubic_centimeters(voxel_count: int, voxel_spacing_millimeters: list[float]) -> float:
    return round(voxel_count * float(np.prod(voxel_spacing_millimeters)) / 1000, 2)


def get_job_event_channel(channel_prefix: str, job_id: str) -> str:
    return f"{channel_prefix}:{job_id}"
