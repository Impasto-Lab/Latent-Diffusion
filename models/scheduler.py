"""Explicit local DDIM timestep and latent-update equations."""
import torch

from models.layers import ConfigDict


class DDIMScheduler:
    def __init__(self, config):
        # The converted JSON says "linear"; the authors' actual LDM schedule
        # linearly interpolates sqrt(beta), then squares it. Only that schedule
        # is implemented here because this project loads one checkpoint.
        self.config = ConfigDict({
            "beta_start": config["beta_start"],
            "beta_end": config["beta_end"],
            "num_train_timesteps": config["num_train_timesteps"],
            "beta_schedule": "scaled_linear",
            "steps_offset": 1,
        })
        config = self.config
        self.betas = torch.linspace(
            config.beta_start**0.5,
            config.beta_end**0.5,
            config.num_train_timesteps,
            dtype=torch.float32,
        ).square()

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.num_inference_steps = None
        self.timesteps = None

    def set_timesteps(self, steps):
        if not 1 <= steps < self.config.num_train_timesteps:
            raise ValueError(
                f"Inference steps must be in [1, {self.config.num_train_timesteps - 1}]."
            )
        self.num_inference_steps = steps
        ratio = self.config.num_train_timesteps // steps
        self.timesteps = torch.arange(steps, dtype=torch.int64).mul(ratio).flip(0)
        self.timesteps = self.timesteps + self.config.steps_offset

    def _variance(self, timestep, previous_timestep):
        alpha_t = self.alphas_cumprod[timestep]
        alpha_previous = (
            self.alphas_cumprod[previous_timestep]
            if previous_timestep >= 0
            else self.alphas_cumprod[0]
        )
        beta_t = 1 - alpha_t
        beta_previous = 1 - alpha_previous
        return (beta_previous / beta_t) * (1 - alpha_t / alpha_previous)

    def step(self, predicted_noise, timestep, sample, *, eta=0.0, generator=None):
        """Apply equations 12 and 16 from DDIM to obtain the previous latent."""
        if self.num_inference_steps is None:
            raise ValueError("Call set_timesteps() before step().")
        timestep = int(timestep)
        previous_timestep = (
            timestep - self.config.num_train_timesteps // self.num_inference_steps
        )
        alpha_t = self.alphas_cumprod[timestep]
        alpha_previous = (
            self.alphas_cumprod[previous_timestep]
            if previous_timestep >= 0
            else self.alphas_cumprod[0]
        )
        beta_t = 1 - alpha_t

        predicted_original = (
            sample - beta_t.sqrt() * predicted_noise
        ) / alpha_t.sqrt()

        sigma = eta * self._variance(timestep, previous_timestep).sqrt()
        direction = (1 - alpha_previous - sigma.square()).sqrt() * predicted_noise
        previous_sample = alpha_previous.sqrt() * predicted_original + direction

        if eta > 0:
            noise_device = sample.device if generator is None else generator.device
            noise = torch.randn(
                sample.shape,
                generator=generator,
                device=noise_device,
                dtype=sample.dtype,
            ).to(sample.device)
            previous_sample = previous_sample + sigma * noise
        return previous_sample
