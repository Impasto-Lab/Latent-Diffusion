"""Text-to-image with the Latent Diffusion Model (Rombach et al., 2022).

    prompt ──tokenizer──▶ token ids ──LDMBert──▶ text features ───────────────┐
                                                                               ▼
    random latent ──▶ [ U-Net predicts noise ─▶ guidance ─▶ DDIM update ] × steps
                                                                               │
    PNG ◀── pixels ◀── autoencoder.decode ◀── clean latent ◀───────────────────┘
"""
import argparse
import os
from datetime import datetime
from pathlib import Path

# Let PyTorch run ops that Apple's MPS backend lacks on the CPU instead of failing.
# Must be set before torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402
from transformers import BertTokenizer  # noqa: E402

from models.autoencoder import AutoencoderKL  # noqa: E402
from models.bert import LDMBertModel  # noqa: E402
from models.config import LATENT_SCALING_FACTOR, MODEL_DIR, ROOT  # noqa: E402
from models.loader import check_model_files, load_model, read_config  # noqa: E402
from models.scheduler import DDIMScheduler  # noqa: E402
from models.unet import ConditionalUNet  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt", default="A red fox in a snowy forest, oil painting")
    parser.add_argument("--steps", type=int, default=50, help="DDIM steps, 1-999")
    parser.add_argument("--guidance", type=float, default=5.0, help="classifier-free guidance scale")
    parser.add_argument("--eta", type=float, default=0.0, help="DDIM noise: 0 = deterministic, 1 = DDPM-like")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=256, help="multiple of 64")
    parser.add_argument("--height", type=int, default=256, help="multiple of 64")
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, cuda, cuda:1, ...")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--output", type=Path, help="PNG path (default: outputs/<time>-seed<seed>.png)")
    args = parser.parse_args()
    # The autoencoder shrinks images 8x, then the U-Net halves the latent 3 more
    # times. Its skip connections only line up if every halving is exact: 8 * 2**3 = 64.
    if args.width % 64 or args.height % 64:
        parser.error("--width and --height must be multiples of 64.")
    return args


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_models(device, dtype):
    check_model_files()
    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR / "tokenizer", local_files_only=True)

    text_encoder = load_model(
        LDMBertModel, read_config(MODEL_DIR / "bert" / "config.json"),
        MODEL_DIR / "bert" / "pytorch_model.bin", device, dtype,
    )

    unet_config = read_config(MODEL_DIR / "unet" / "config.json")
    # Cross-attention reads the text features, so its input width must equal the
    # text encoder's output width (1280). The converted config leaves it out.
    unet_config.cross_attention_dim = text_encoder.config.d_model
    unet = load_model(
        ConditionalUNet, unet_config, MODEL_DIR / "unet" / "diffusion_pytorch_model.bin", device, dtype,
    )

    # The published folder is named "vqvae", but it holds a KL autoencoder.
    autoencoder = load_model(
        AutoencoderKL, read_config(MODEL_DIR / "vqvae" / "config.json"),
        MODEL_DIR / "vqvae" / "diffusion_pytorch_model.bin", device, dtype,
    )

    scheduler = DDIMScheduler(read_config(MODEL_DIR / "scheduler" / "scheduler_config.json"))
    return tokenizer, text_encoder, unet, autoencoder, scheduler


def main():
    args = parse_args()
    device = pick_device(args.device)
    if args.dtype == "auto":
        dtype = torch.float32 if device.type == "cpu" else torch.float16
    else:
        dtype = getattr(torch, args.dtype)
    output = args.output or ROOT / "outputs" / f"{datetime.now():%Y%m%d-%H%M%S}-seed{args.seed}.png"
    print(f"Device: {device}; dtype: {dtype}")

    tokenizer, text_encoder, unet, autoencoder, scheduler = load_models(device, dtype)

    with torch.inference_mode():
        # 1. Encode the text. Classifier-free guidance compares the prediction
        #    *with* the prompt against one *without* it (the empty prompt), so we
        #    encode both and run them through the U-Net as a batch of 2.
        #    guidance = 1 would just return the prompt's prediction, so skip the extra work.
        use_guidance = args.guidance != 1
        prompts = ["", args.prompt] if use_guidance else [args.prompt]
        tokens = tokenizer(
            prompts,
            padding="max_length",  # always 77 tokens: [CLS] text [SEP] [PAD] ...
            max_length=text_encoder.config.max_position_embeddings,
            truncation=True,
            return_tensors="pt",
        )
        # LDMBert was trained without a padding mask, so none is passed here.
        context = text_encoder(tokens.input_ids.to(device))  # [len(prompts), 77, 1280]

        # 2. Start from pure Gaussian noise in latent space: [1, 4, height/8, width/8].
        #    Sample on the CPU so a seed gives the same image on CPU, CUDA, and MPS.
        downsample = 2 ** (len(autoencoder.config.block_out_channels) - 1)  # 8
        shape = (1, unet.config.in_channels, args.height // downsample, args.width // downsample)
        generator = torch.Generator().manual_seed(args.seed)
        latents = torch.randn(shape, generator=generator, dtype=dtype).to(device)

        # 3. Denoise: at each timestep the U-Net predicts the noise in the latent,
        #    and DDIM uses that prediction to step to a slightly cleaner latent.
        scheduler.set_timesteps(args.steps)
        for t in tqdm(scheduler.timesteps, desc="DDIM"):
            noise = unet(torch.cat([latents] * len(prompts)), t, context)
            if use_guidance:
                noise_empty, noise_prompt = noise.chunk(2)
                # Start from the prompt-free prediction and push further in the
                # direction the prompt changes it.
                noise = noise_empty + args.guidance * (noise_prompt - noise_empty)
            latents = scheduler.step(noise, t, latents, eta=args.eta, generator=generator)

        # 4. Decode the latent to pixels: undo the training-time latent scaling,
        #    decode to [-1, 1], then map to [0, 1].
        image = autoencoder.decode(latents / LATENT_SCALING_FACTOR)  # [1, 3, height, width]
        image = (image / 2 + 0.5).clamp(0, 1)

    # 5. Save: [1, 3, H, W] in [0, 1] -> [H, W, 3] bytes.
    pixels = (image[0].permute(1, 2, 0).float().cpu().numpy() * 255).round().astype("uint8")
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(output)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
