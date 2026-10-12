import re
import shutil
from pathlib import Path

import numpy as np
from nibabel.loadsave import save as save_image
from nibabel.nifti1 import Nifti1Image
from segmentation_core.settings import NIFTI_EXTENSIONS

from inference_service.configuration import InferenceServiceSettings
from inference_service.domain.segmentation_job import SegmentationNotFoundError
from inference_service.utils import get_new_identifier

IDENTIFIER_PATTERN = re.compile(r"^[0-9a-f]{32}$")
INPUTS_DIRECTORY_NAME = "inputs"
RESULTS_DIRECTORY_NAME = "results"
MASK_FILE_NAME = "mask.nii.gz"
PREVIEW_FILE_EXTENSION = ".npy"


class FileScanStorage:
    def __init__(self, settings: InferenceServiceSettings):
        self.staging_directory = settings.staging_directory
        self.jobs_directory = settings.jobs_directory
        self.staging_directory.mkdir(parents=True, exist_ok=True)
        self.jobs_directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def ensure_identifier(identifier: str) -> str:
        if not IDENTIFIER_PATTERN.match(identifier):
            raise SegmentationNotFoundError(identifier)
        return identifier

    def get_job_directory(self, job_id: str) -> Path:
        return self.jobs_directory / self.ensure_identifier(job_id)

    def create_staging_directory(self) -> str:
        staging_directory_name = get_new_identifier()
        (self.staging_directory / staging_directory_name).mkdir(parents=True)
        return staging_directory_name

    def get_staging_path(self, staging_directory_name: str, file_name: str) -> Path:
        return self.staging_directory / self.ensure_identifier(staging_directory_name) / file_name

    def remove_staging_directory(self, staging_directory_name: str) -> None:
        shutil.rmtree(self.staging_directory / self.ensure_identifier(staging_directory_name), ignore_errors=True)

    def promote_staging_directory(self, staging_directory_name: str, job_id: str) -> None:
        job_directory = self.get_job_directory(job_id)
        job_directory.mkdir(parents=True)
        (self.staging_directory / self.ensure_identifier(staging_directory_name)).rename(job_directory / INPUTS_DIRECTORY_NAME)

    def get_input_path(self, job_id: str, suffix: str) -> Path:
        inputs_directory = self.get_job_directory(job_id) / INPUTS_DIRECTORY_NAME
        for extension in NIFTI_EXTENSIONS:
            input_path = inputs_directory / f"{suffix}{extension}"
            if input_path.exists():
                return input_path
        raise SegmentationNotFoundError(f"{job_id}/{suffix}")

    def save_results(self, job_id: str, preview_volumes: dict[str, np.ndarray], predicted_mask: np.ndarray,
                     reference_image: Nifti1Image) -> None:
        results_directory = self.get_job_directory(job_id) / RESULTS_DIRECTORY_NAME
        results_directory.mkdir(parents=True, exist_ok=True)
        for preview_name, preview_volume in preview_volumes.items():
            np.save(results_directory / f"{preview_name}{PREVIEW_FILE_EXTENSION}", preview_volume)
        mask_header = reference_image.header.copy()
        mask_header.set_data_dtype(np.uint8)
        mask_image = Nifti1Image(predicted_mask.astype(np.uint8), reference_image.affine, mask_header)
        save_image(mask_image, str(results_directory / MASK_FILE_NAME))

    def get_preview_volume(self, job_id: str, preview_name: str) -> np.ndarray:
        preview_path = self.get_job_directory(job_id) / RESULTS_DIRECTORY_NAME / f"{preview_name}{PREVIEW_FILE_EXTENSION}"
        if not preview_path.exists():
            raise SegmentationNotFoundError(f"{job_id}/{preview_name}")
        return np.load(preview_path, mmap_mode="r")

    def get_mask_path(self, job_id: str) -> Path:
        mask_path = self.get_job_directory(job_id) / RESULTS_DIRECTORY_NAME / MASK_FILE_NAME
        if not mask_path.exists():
            raise SegmentationNotFoundError(f"{job_id}/mask")
        return mask_path

    def delete_job_files(self, job_id: str) -> None:
        shutil.rmtree(self.get_job_directory(job_id), ignore_errors=True)
