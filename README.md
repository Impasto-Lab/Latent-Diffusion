# Latent Diffusion Models

An inference-only, educational implementation of [Latent Diffusion Models](https://arxiv.org/abs/2112.10752). It uses the authors’ [pretrained 256×256 text-to-image checkpoint](https://huggingface.co/CompVis/ldm-text2im-large-256). The text encoder, U-Net, autoencoder, and DDIM sampler are defined locally in PyTorch.

[Detailed documentation in Chinese](README.zh-CN.md)

## Setup

Use the `diffusion` conda environment with an appropriate [PyTorch build](https://pytorch.org/get-started/locally/) for your Mac or NVIDIA GPU. Then run:

```bash
conda activate diffusion
python -m pip install -r requirements.txt
python -m scripts.download_model
```

The checkpoint download is approximately 5.73 GiB. Generation runs offline after the download.

## Generate

```bash
python generate.py --prompt "A red fox in a snowy forest, oil painting"
```

The image is saved to `outputs/`. Use `--output outputs/fox.png` to choose a path, or `--device cuda` on an NVIDIA GPU. The default device selection supports CUDA, MPS, and CPU.

## Project layout

- `generate.py` — complete text-to-image inference flow
- `models/` — local model definitions, weight loading, and DDIM sampler
- `scripts/` — pretrained checkpoint download
- `utils/` — device selection, command-line options, output, and shape tracing

This repository reproduces inference with pretrained weights; it does not reproduce training or the paper’s benchmark metrics.
