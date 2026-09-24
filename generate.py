"""Generate an image with the paper authors' latent diffusion checkpoint."""
from utils.runtime import configure_inference_environment


# MPS fallback must be configured before importing PyTorch.
configure_inference_environment()

import torch  # noqa: E402

from inference.sampler import decode_latents, sample_latents  # noqa: E402
from models.config import MODEL_DIR  # noqa: E402
from models.loader import load_components  # noqa: E402
from utils.cli import build_parser, validate_generation_args  # noqa: E402
from utils.debug import install_shape_tracing  # noqa: E402
from utils.environment import select_device, select_precision  # noqa: E402
from utils.output import resolve_output_path, save_image  # noqa: E402


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

    # 3. Encode the prompt, start from Gaussian noise, and repeatedly predict
    #    noise with classifier-free guidance and update the latent using DDIM.
    latents = sample_latents(
        components,
        args.prompt,
        height=args.height,
        width=args.width,
        steps=args.steps,
        guidance=args.guidance,
        eta=args.eta,
        seed=args.seed,
    )

    # 4. Decode the final latent to RGB pixels and save the image.
    pixels = decode_latents(components, latents)
    save_image(pixels, output)
    print(f"Saved: {output}", flush=True)


if __name__ == "__main__":
    main()
