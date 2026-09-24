"""Select a usable PyTorch device and precision on Mac or NVIDIA hardware."""
import re

import torch


CUDA_DEVICE = re.compile(r"cuda(?::(\d+))?$")


def _mps_available():
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def select_device(requested):
    requested = requested.strip().lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        return "mps" if _mps_available() else "cpu"
    if requested == "cpu":
        return "cpu"
    if requested == "mps":
        if not _mps_available():
            raise ValueError("MPS is unavailable. Use --device cpu instead.")
        return "mps"

    match = CUDA_DEVICE.fullmatch(requested)
    if not match:
        raise ValueError("--device must be auto, cpu, mps, cuda, or cuda:N.")
    if not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Install CUDA-enabled PyTorch.")
    index = int(match.group(1) or 0)
    count = torch.cuda.device_count()
    if index >= count:
        raise ValueError(f"CUDA device {index} does not exist; found {count} device(s).")
    return f"cuda:{index}"


def select_precision(requested, device):
    precision = (
        "float16" if device.startswith(("cuda", "mps")) else "float32"
    ) if requested == "auto" else requested
    if device == "cpu" and precision != "float32":
        raise ValueError("CPU inference supports --dtype float32 in this project.")
    if device == "mps" and precision == "bfloat16":
        raise ValueError("Use --dtype float16 or float32 with MPS.")
    if device.startswith("cuda") and precision == "bfloat16":
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                raise ValueError(f"{device} does not support bfloat16.")
    return precision
