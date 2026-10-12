import io

import numpy as np
from PIL import Image

OPAQUE_ALPHA = 255
RGBA_CHANNEL_COUNT = 4


class PngSliceRenderer:
    @staticmethod
    def get_display_slice(volume: np.ndarray, slice_index: int) -> np.ndarray:
        if not 0 <= slice_index < volume.shape[2]:
            raise IndexError(f"Slice index must be between 0 and {volume.shape[2] - 1}.")
        return np.ascontiguousarray(np.rot90(volume[:, :, slice_index]))

    @staticmethod
    def get_png_bytes(image: Image.Image) -> bytes:
        image_buffer = io.BytesIO()
        image.save(image_buffer, format="PNG")
        return image_buffer.getvalue()

    def get_image_png(self, volume: np.ndarray, slice_index: int) -> bytes:
        return self.get_png_bytes(Image.fromarray(self.get_display_slice(volume, slice_index).astype(np.uint8)))

    def get_mask_png(self, mask_volume: np.ndarray, slice_index: int, color: tuple[int, int, int]) -> bytes:
        mask_slice = self.get_display_slice(mask_volume, slice_index) > 0
        rgba_slice = np.zeros((*mask_slice.shape, RGBA_CHANNEL_COUNT), dtype=np.uint8)
        rgba_slice[mask_slice] = (*color, OPAQUE_ALPHA)
        return self.get_png_bytes(Image.fromarray(rgba_slice))
