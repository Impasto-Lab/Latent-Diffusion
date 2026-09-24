"""Instantiate local model definitions and attach the published weights."""
import json

import torch
from transformers import BertTokenizer

from models.bert import LDMBertModel
from models.components import LDMComponents
from models.config import LATENT_SCALING_FACTOR, MODEL_FILES
from models.scheduler import DDIMScheduler
from models.unet import ConditionalUNet
from models.vqvae import AutoencoderKL


def _read_config(path):
    return json.loads(path.read_text())


def _load_weights(model_class, config, checkpoint):
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
    return model


def load_components(model_dir, device, dtype):
    missing = [name for name in MODEL_FILES if not (model_dir / name).is_file()]
    if missing:
        message = "Model files missing. Run python -m scripts.download_model first.\n"
        raise FileNotFoundError(message + "\n".join(missing))

    scheduler_config = _read_config(model_dir / "scheduler" / "scheduler_config.json")
    scheduler = DDIMScheduler(scheduler_config)
    tokenizer = BertTokenizer.from_pretrained(
        model_dir / "tokenizer",
        local_files_only=True,
    )
    bert = _load_weights(
        LDMBertModel,
        _read_config(model_dir / "bert" / "config.json"),
        model_dir / "bert" / "pytorch_model.bin",
    )
    unet_config = _read_config(model_dir / "unet" / "config.json")
    # The converted U-Net config omits its cross-attention width; it equals
    # the output width of this checkpoint's learned text encoder.
    unet_config["cross_attention_dim"] = bert.config.d_model
    unet = _load_weights(
        ConditionalUNet,
        unet_config,
        model_dir / "unet" / "diffusion_pytorch_model.bin",
    )
    vae_config = _read_config(model_dir / "vqvae" / "config.json")
    # The VAE config omits the latent amplitude used by the text-to-image model.
    vae_config["scaling_factor"] = LATENT_SCALING_FACTOR
    autoencoder = _load_weights(
        AutoencoderKL,
        vae_config,
        model_dir / "vqvae" / "diffusion_pytorch_model.bin",
    )
    components = LDMComponents(
        tokenizer=tokenizer,
        text_encoder=bert,
        unet=unet,
        autoencoder=autoencoder,
        scheduler=scheduler,
    )
    for module in (
        components.text_encoder,
        components.unet,
        components.autoencoder,
    ):
        module.eval().requires_grad_(False)

    # The U-Net runs at every DDIM step and always uses the selected device.
    # On CUDA, keep all components together. On MPS, leave the one-shot text
    # encoder and image decoder on CPU to fit the 16 GiB Mac comfortably.
    components.unet.to(dtype=dtype).to(device)
    if device.startswith("cuda"):
        components.text_encoder.to(dtype=dtype).to(device)
        components.autoencoder.to(dtype=dtype).to(device)
    return components
