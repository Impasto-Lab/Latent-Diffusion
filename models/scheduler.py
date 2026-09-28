"""DDIM sampling (Song et al., 2020) with the original LDM noise schedule.

Notation: alpha_bar[t] is the fraction of signal left at training step t,
    z_t = sqrt(alpha_bar[t]) * z_0 + sqrt(1 - alpha_bar[t]) * noise.
It starts near 1 (t = 0, almost clean) and falls toward 0 (t = 999, almost pure noise).
"""
import torch


class DDIMScheduler:
    def __init__(self, config):
        self.num_train_timesteps = config.num_train_timesteps  # 1000
        # The JSON says "linear", which in the original LDM code means: space
        # sqrt(beta) linearly, then square it. (Diffusers calls this "scaled_linear".)
        betas = torch.linspace(
            config.beta_start**0.5, config.beta_end**0.5, self.num_train_timesteps, dtype=torch.float32
        ) ** 2
        self.alpha_bar = torch.cumprod(1 - betas, dim=0)
        self.timesteps = None

    def set_timesteps(self, steps):
        """Pick ``steps`` evenly spaced training timesteps, from noisiest to cleanest."""
        if not 1 <= steps < self.num_train_timesteps:
            raise ValueError(f"steps must be in [1, {self.num_train_timesteps - 1}].")
        self.stride = self.num_train_timesteps // steps
        # e.g. 50 steps -> [981, 961, ..., 21, 1]; the +1 follows the original LDM sampler.
        self.timesteps = torch.arange(steps).flip(0) * self.stride + 1

    def step(self, noise_pred, timestep, latents, eta=0.0, generator=None):
        """Move ``latents`` from ``timestep`` to the next (less noisy) timestep.

        Equations 12 and 16 of the DDIM paper.
        """
        t = int(timestep)
        alpha_bar = self.alpha_bar[t]
        # The last step (t = 1) would land below 0; the original sampler uses alpha_bar[0] there.
        alpha_bar_prev = self.alpha_bar[max(t - self.stride, 0)]

        # 1. Remove the predicted noise to estimate the clean latent z_0.
        z0 = (latents - (1 - alpha_bar).sqrt() * noise_pred) / alpha_bar.sqrt()

        # 2. How much fresh noise to add: none for eta = 0 (deterministic DDIM),
        #    as much as DDPM for eta = 1.
        sigma = eta * ((1 - alpha_bar_prev) / (1 - alpha_bar) * (1 - alpha_bar / alpha_bar_prev)).sqrt()

        # 3. Re-noise z_0 to the previous timestep, pointing along the predicted noise.
        latents = alpha_bar_prev.sqrt() * z0 + (1 - alpha_bar_prev - sigma**2).sqrt() * noise_pred
        if eta > 0:
            # Draw on the generator's device (CPU) so results are the same on every backend.
            noise = torch.randn(latents.shape, generator=generator, dtype=latents.dtype)
            latents = latents + sigma * noise.to(latents.device)
        return latents
