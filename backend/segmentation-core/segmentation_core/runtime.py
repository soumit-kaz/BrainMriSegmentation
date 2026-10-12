import contextlib
import random
from typing import Any

import numpy as np
import torch


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_device_name(device: torch.device) -> str:
    return torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"


def get_autocast_dtype(device: torch.device, is_mixed_precision_enabled: bool) -> torch.dtype | None:
    if not is_mixed_precision_enabled or device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def get_precision_name(autocast_dtype: torch.dtype | None) -> str:
    if autocast_dtype == torch.bfloat16:
        return "bfloat16"
    if autocast_dtype == torch.float16:
        return "float16"
    return "float32"


def get_autocast_context(device: torch.device, autocast_dtype: torch.dtype | None) -> contextlib.AbstractContextManager[Any]:
    if autocast_dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=autocast_dtype)


def set_random_seed(random_seed: int) -> None:
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
