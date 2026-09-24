"""Generate an image with the paper authors' latent diffusion checkpoint."""
from utils.runtime import configure_inference_environment


# MPS fallback must be configured before importing PyTorch.
configure_inference_environment()

import torch
from tqdm.auto import tqdm

from models.config import MODEL_DIR
from models.loader import load_components
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
        output = resolve_output_path(args.output, args.seed)
    except ValueError as error:
        parser.error(str(error))

    print(f"Device: {device}; dtype: {precision}", flush=True)
    print(
        f"{args.width}x{args.height}, {args.steps} DDIM steps, "
        f"CFG={args.guidance}, eta={args.eta}, seed={args.seed}",
        flush=True,
    )

    # 2. Build the local BERT, U-Net, autoencoder, and DDIM scheduler;
    #    then attach the pretrained weights.
    components = load_components(MODEL_DIR, device, getattr(torch, precision))
    print(
        "Components: "
        f"text_encoder={next(components.text_encoder.parameters()).device}, "
        f"unet={next(components.unet.parameters()).device}, "
        f"autoencoder={next(components.autoencoder.parameters()).device}",
        flush=True,
    )
    if args.trace_shapes:
        install_shape_tracing(components)

    with torch.inference_mode():
        # 3. Encode the prompt and the empty prompt for classifier-free guidance.
        unet_parameter = next(components.unet.parameters())
        unet_device, unet_dtype = unet_parameter.device, unet_parameter.dtype
        text_device = next(components.text_encoder.parameters()).device
        max_length = components.text_encoder.max_sequence_length

        def encode(text):
            tokens = components.tokenizer(
                text,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            # The published text encoder does not use a padding mask.
            embeddings = components.text_encoder(tokens.input_ids.to(text_device))
            return embeddings.to(device=unet_device, dtype=unet_dtype)

        context = encode(args.prompt)
        unconditional = encode("") if args.guidance != 1.0 else None
        model_context = (
            torch.cat([unconditional, context]) if unconditional is not None else context
        )

        # 4. Start from Gaussian noise in the autoencoder's latent space.
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        factor = components.latent_scale_factor
        shape = (
            1,
            components.unet.config.in_channels,
            args.height // factor,
            args.width // factor,
        )
        latents = torch.randn(
            shape, generator=generator, dtype=unet_dtype, device="cpu"
        ).to(unet_device)
        components.scheduler.set_timesteps(args.steps)

        # 5. Predict noise with the U-Net, then update the latent with DDIM.
        for timestep in tqdm(components.scheduler.timesteps, desc="DDIM"):
            if unconditional is None:
                noise = components.unet(
                    latents, timestep, encoder_hidden_states=context
                )
            else:
                # The first half is the empty prompt; the second is the prompt.
                model_latents = torch.cat([latents, latents])
                noise_unconditional, noise_conditional = components.unet(
                    model_latents,
                    timestep,
                    encoder_hidden_states=model_context,
                ).chunk(2)
                noise = noise_unconditional + args.guidance * (
                    noise_conditional - noise_unconditional
                )

            latents = components.scheduler.step(
                noise, timestep, latents, eta=args.eta, generator=generator
            )

        # 6. Decode the final latent and map pixels from [-1, 1] to [0, 1].
        decoder_parameter = next(components.autoencoder.parameters())
        latents = latents.to(
            device=decoder_parameter.device, dtype=decoder_parameter.dtype
        )
        scaling_factor = components.autoencoder.config.scaling_factor
        decoded = components.autoencoder.decode(latents / scaling_factor)
        pixels = (decoded / 2 + 0.5).clamp(0, 1)
        pixels = pixels.cpu().permute(0, 2, 3, 1).float().numpy()

    save_image(pixels, output)
    print(f"Saved: {output}", flush=True)


if __name__ == "__main__":
    main()
