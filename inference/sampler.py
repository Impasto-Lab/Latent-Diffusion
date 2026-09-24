"""Paper-aligned LDM sampling primitives, independent of CLI and file output."""
import torch
from tqdm.auto import tqdm


@torch.inference_mode()
def sample_latents(
    components,
    prompt,
    *,
    height,
    width,
    steps,
    guidance,
    eta,
    seed,
    progress=True,
):
    device = next(components.unet.parameters()).device
    dtype = next(components.unet.parameters()).dtype
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
        # The learned LDM text encoder intentionally matches the upstream
        # published implementation and does not receive a padding mask.
        embeddings = components.text_encoder(tokens.input_ids.to(text_device))
        return embeddings.to(device=device, dtype=dtype)

    context = encode(prompt)
    unconditional = encode("") if guidance != 1.0 else None

    generator = torch.Generator(device="cpu").manual_seed(seed)
    factor = components.latent_scale_factor
    shape = (1, components.unet.config.in_channels, height // factor, width // factor)
    latents = torch.randn(shape, generator=generator, dtype=dtype, device="cpu").to(device)
    components.scheduler.set_timesteps(steps)

    timesteps = tqdm(components.scheduler.timesteps, desc="DDIM", disable=not progress)
    for timestep in timesteps:
        if unconditional is None:
            noise = components.unet(
                latents, timestep, encoder_hidden_states=context
            )
        else:
            # Predict both CFG branches in one U-Net forward with batch size 2.
            noise_unconditional, noise_conditional = components.unet(
                torch.cat([latents, latents]),
                timestep,
                encoder_hidden_states=torch.cat([unconditional, context]),
            ).chunk(2)
            noise = noise_unconditional + guidance * (
                noise_conditional - noise_unconditional
            )

        latents = components.scheduler.step(
            noise, timestep, latents, eta=eta, generator=generator
        )
    return latents


@torch.inference_mode()
def decode_latents(components, latents):
    decoder_parameter = next(components.autoencoder.parameters())
    latents = latents.to(
        device=decoder_parameter.device, dtype=decoder_parameter.dtype
    )
    scaling_factor = components.autoencoder.config.scaling_factor
    decoded = components.autoencoder.decode(latents / scaling_factor)
    pixels = (decoded / 2 + 0.5).clamp(0, 1)
    return pixels.cpu().permute(0, 2, 3, 1).float().numpy()
