"""Generate an image with the paper authors' latent diffusion checkpoint."""
from utils.runtime import configure_inference_environment


# MPS fallback must be configured before importing PyTorch.
configure_inference_environment()

import torch
from tqdm.auto import tqdm
from transformers import BertTokenizer

from models.bert import LDMBertModel
from models.config import LATENT_SCALING_FACTOR, MODEL_DIR
from models.loader import check_model_files, load_pretrained_model, read_config
from models.scheduler import DDIMScheduler
from models.unet import ConditionalUNet
from models.vqvae import AutoencoderKL
from utils.cli import build_parser, validate_generation_args
from utils.debug import install_shape_tracing
from utils.environment import select_device, select_precision
from utils.output import resolve_output_path, save_image


def main():
    # 1. Choose the prompt, DDIM settings, and hardware.
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_generation_args(args)
        device = select_device(args.device)
        precision = select_precision(args.dtype, device)
        dtype = getattr(torch, precision)
        output = resolve_output_path(args.output, args.seed)
    except ValueError as error:
        parser.error(str(error))

    print(f"Device: {device}; dtype: {precision}", flush=True)
    print(
        f"{args.width}x{args.height}, {args.steps} DDIM steps, "
        f"CFG={args.guidance}, eta={args.eta}, seed={args.seed}",
        flush=True,
    )

    # 2. Create each model.
    check_model_files(MODEL_DIR)
    tokenizer = BertTokenizer.from_pretrained(
        MODEL_DIR / "tokenizer", local_files_only=True
    )

    text_config = read_config(MODEL_DIR / "bert" / "config.json")
    text_encoder = load_pretrained_model(
        LDMBertModel,
        text_config,
        MODEL_DIR / "bert" / "pytorch_model.bin",
        device,
        dtype,
    )

    unet_config = read_config(MODEL_DIR / "unet" / "config.json")
    # The converted config omits this width; it must match the text encoder.
    unet_config["cross_attention_dim"] = text_encoder.config.d_model
    unet = load_pretrained_model(
        ConditionalUNet,
        unet_config,
        MODEL_DIR / "unet" / "diffusion_pytorch_model.bin",
        device,
        dtype,
    )

    autoencoder_config = read_config(MODEL_DIR / "vqvae" / "config.json")
    autoencoder_config["scaling_factor"] = LATENT_SCALING_FACTOR
    autoencoder = load_pretrained_model(
        AutoencoderKL,
        autoencoder_config,
        MODEL_DIR / "vqvae" / "diffusion_pytorch_model.bin",
        device,
        dtype,
    )

    scheduler_config = read_config(MODEL_DIR / "scheduler" / "scheduler_config.json")
    scheduler = DDIMScheduler(scheduler_config)
    if args.trace_shapes:
        install_shape_tracing(text_encoder, unet, autoencoder)

    with torch.inference_mode():
        # 3. Encode the prompt and the empty prompt for classifier-free guidance.
        def encode(text):
            tokens = tokenizer(
                text,
                padding="max_length",
                max_length=text_encoder.max_sequence_length,
                truncation=True,
                return_tensors="pt",
            )
            # The published text encoder does not use a padding mask.
            return text_encoder(tokens.input_ids.to(device))

        context = encode(args.prompt)
        unconditional = encode("") if args.guidance != 1.0 else None
        model_context = (
            torch.cat([unconditional, context]) if unconditional is not None else context
        )

        # 4. Start from Gaussian noise in the autoencoder's latent space.
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        factor = 2 ** (len(autoencoder.config.block_out_channels) - 1)
        shape = (
            1,
            unet.config.in_channels,
            args.height // factor,
            args.width // factor,
        )
        latents = torch.randn(
            shape, generator=generator, dtype=dtype, device="cpu"
        ).to(device)
        scheduler.set_timesteps(args.steps)

        # 5. Predict noise with the U-Net, then update the latent with DDIM.
        for timestep in tqdm(scheduler.timesteps, desc="DDIM"):
            if unconditional is None:
                noise = unet(
                    latents, timestep, encoder_hidden_states=context
                )
            else:
                # The first half is the empty prompt; the second is the prompt.
                model_latents = torch.cat([latents, latents])
                noise_unconditional, noise_conditional = unet(
                    model_latents,
                    timestep,
                    encoder_hidden_states=model_context,
                ).chunk(2)
                noise = noise_unconditional + args.guidance * (
                    noise_conditional - noise_unconditional
                )

            latents = scheduler.step(
                noise, timestep, latents, eta=args.eta, generator=generator
            )

        # 6. Decode the final latent and map pixels from [-1, 1] to [0, 1].
        scaling_factor = autoencoder.config.scaling_factor
        decoded = autoencoder.decode(latents / scaling_factor)
        pixels = (decoded / 2 + 0.5).clamp(0, 1)
        pixels = pixels.cpu().permute(0, 2, 3, 1).float().numpy()

    save_image(pixels, output)
    print(f"Saved: {output}", flush=True)


if __name__ == "__main__":
    main()
