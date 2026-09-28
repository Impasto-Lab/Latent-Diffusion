"""Paths and facts about the pinned checkpoint."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "CompVis/ldm-text2im-large-256"
REVISION = "30de525ca11a880baea4962827fb6cb0bb268955"
MODEL_DIR = ROOT / "models" / "ldm-text2im-large-256"
MANIFEST_PATH = ROOT / "models" / "model_manifest.json"
MODEL_FILES = (
    "model_index.json",
    "bert/config.json",
    "bert/pytorch_model.bin",
    "unet/config.json",
    "unet/diffusion_pytorch_model.bin",
    "vqvae/config.json",
    "vqvae/diffusion_pytorch_model.bin",
    "scheduler/scheduler_config.json",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.txt",
)

# The U-Net was trained on autoencoder latents multiplied by this constant,
# which scales them to roughly unit variance. Not stored in the JSON configs.
LATENT_SCALING_FACTOR = 0.18215
