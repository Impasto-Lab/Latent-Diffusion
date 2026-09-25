"""Read checkpoint files and bind pretrained weights to a local model."""
import json

import torch

from models.config import MODEL_FILES


def check_model_files(model_dir):
    missing = [name for name in MODEL_FILES if not (model_dir / name).is_file()]
    if missing:
        message = "Model files missing. Run python -m scripts.download_model first.\n"
        raise FileNotFoundError(message + "\n".join(missing))


def read_config(path):
    return json.loads(path.read_text())


def load_pretrained_model(model_class, config, checkpoint, device, dtype):
    """Build on ``meta`` and assign checkpoint tensors without double allocation."""
    with torch.device("meta"):
        model = model_class(config)
    state_dict = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    model.load_state_dict(state_dict, strict=True, assign=True)
    return model.eval().requires_grad_(False).to(device=device, dtype=dtype)
