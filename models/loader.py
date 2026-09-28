"""Read checkpoint configs and load pretrained weights into the local models."""
import json
from types import SimpleNamespace

import torch

from models.config import MODEL_DIR, MODEL_FILES


def check_model_files():
    missing = [name for name in MODEL_FILES if not (MODEL_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Model files missing. Run `python -m scripts.download_model` first.\n" + "\n".join(missing)
        )


def read_config(path):
    """Load a JSON config with attribute access: ``config.in_channels``."""
    return json.loads(path.read_text(), object_hook=lambda fields: SimpleNamespace(**fields))


def load_model(model_class, config, checkpoint, device, dtype):
    """Build ``model_class(config)`` and fill it with the checkpoint's weights.

    The model is created on the "meta" device (shapes only, no memory), then
    the checkpoint tensors are assigned directly. ``strict=True`` fails loudly
    if any checkpoint key and model attribute do not match one-to-one.
    """
    with torch.device("meta"):
        model = model_class(config)
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(state_dict, strict=True, assign=True)
    return model.eval().requires_grad_(False).to(device=device, dtype=dtype)
