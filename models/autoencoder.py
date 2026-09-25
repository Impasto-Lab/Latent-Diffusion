"""Local KL-regularized image autoencoder used by latent diffusion."""
import torch
from torch import nn

from models.config import ConfigDict
from models.layers import (
    Downsample2D,
    ResnetBlock2D,
    Upsample2D,
    VAESpatialAttention,
    group_norm,
)


class VAEMidBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attentions = nn.ModuleList([VAESpatialAttention(channels)])
        self.resnets = nn.ModuleList(
            [
                ResnetBlock2D(channels, channels, eps=1e-6),
                ResnetBlock2D(channels, channels, eps=1e-6),
            ]
        )

    def forward(self, hidden):
        hidden = self.resnets[0](hidden)
        hidden = self.attentions[0](hidden)
        return self.resnets[1](hidden)


class DownEncoderBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, layers, add_downsample):
        super().__init__()
        self.resnets = nn.ModuleList(
            [
                ResnetBlock2D(
                    in_channels if index == 0 else out_channels,
                    out_channels,
                    eps=1e-6,
                )
                for index in range(layers)
            ]
        )
        self.downsamplers = (
            nn.ModuleList([Downsample2D(out_channels, padding=0)])
            if add_downsample
            else None
        )

    def forward(self, hidden):
        for resnet in self.resnets:
            hidden = resnet(hidden)
        if self.downsamplers is not None:
            hidden = self.downsamplers[0](hidden)
        return hidden


class UpDecoderBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, layers, add_upsample):
        super().__init__()
        self.resnets = nn.ModuleList(
            [
                ResnetBlock2D(
                    in_channels if index == 0 else out_channels,
                    out_channels,
                    eps=1e-6,
                )
                for index in range(layers)
            ]
        )
        self.upsamplers = (
            nn.ModuleList([Upsample2D(out_channels)]) if add_upsample else None
        )

    def forward(self, hidden):
        for resnet in self.resnets:
            hidden = resnet(hidden)
        if self.upsamplers is not None:
            hidden = self.upsamplers[0](hidden)
        return hidden


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        channels = list(config.block_out_channels)
        self.conv_in = nn.Conv2d(config.in_channels, channels[0], 3, padding=1)
        blocks = []
        input_channels = channels[0]
        for index, output_channels in enumerate(channels):
            blocks.append(
                DownEncoderBlock2D(
                    input_channels,
                    output_channels,
                    config.layers_per_block,
                    add_downsample=index < len(channels) - 1,
                )
            )
            input_channels = output_channels
        self.down_blocks = nn.ModuleList(blocks)
        self.mid_block = VAEMidBlock(channels[-1])
        self.conv_norm_out = group_norm(channels[-1], 1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(channels[-1], 2 * config.latent_channels, 3, padding=1)

    def forward(self, image):
        hidden = self.conv_in(image)
        for block in self.down_blocks:
            hidden = block(hidden)
        hidden = self.mid_block(hidden)
        return self.conv_out(self.conv_act(self.conv_norm_out(hidden)))


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        channels = list(reversed(config.block_out_channels))
        self.conv_in = nn.Conv2d(config.latent_channels, channels[0], 3, padding=1)
        self.mid_block = VAEMidBlock(channels[0])
        blocks = []
        input_channels = channels[0]
        for index, output_channels in enumerate(channels):
            blocks.append(
                UpDecoderBlock2D(
                    input_channels,
                    output_channels,
                    config.layers_per_block + 1,
                    add_upsample=index < len(channels) - 1,
                )
            )
            input_channels = output_channels
        self.up_blocks = nn.ModuleList(blocks)
        self.conv_norm_out = group_norm(channels[-1], 1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(channels[-1], config.out_channels, 3, padding=1)

    def forward(self, latents):
        hidden = self.mid_block(self.conv_in(latents))
        for block in self.up_blocks:
            hidden = block(hidden)
        return self.conv_out(self.conv_act(self.conv_norm_out(hidden)))


class DiagonalGaussianDistribution:
    """Mean/log-variance parameterization produced by the KL encoder."""

    def __init__(self, parameters):
        self.mean, self.logvar = parameters.chunk(2, dim=1)
        self.logvar = self.logvar.clamp(-30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)

    def sample(self, generator=None):
        noise = torch.randn(
            self.mean.shape,
            generator=generator,
            device=self.mean.device,
            dtype=self.mean.dtype,
        )
        return self.mean + self.std * noise

    def mode(self):
        return self.mean


class AutoencoderKL(nn.Module):
    """Image encoder/decoder; inference uses ``decode`` after DDIM sampling."""

    def __init__(self, config):
        super().__init__()
        config = ConfigDict({name: config[name] for name in (
            "in_channels", "out_channels", "latent_channels", "block_out_channels",
            "layers_per_block", "scaling_factor",
        )})
        self.config = config
        # Retain the encoder for reconstruction examples and checkpoint compatibility.
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.quant_conv = nn.Conv2d(2 * config.latent_channels, 2 * config.latent_channels, 1)
        self.post_quant_conv = nn.Conv2d(config.latent_channels, config.latent_channels, 1)

    def encode(self, image):
        # The KL encoder predicts a Gaussian posterior, not discrete VQ codes.
        moments = self.quant_conv(self.encoder(image))
        return DiagonalGaussianDistribution(moments)

    def decode(self, latents):
        latents = self.post_quant_conv(latents)
        return self.decoder(latents)

    def forward(self, image, sample_posterior=False, generator=None):
        posterior = self.encode(image)
        latents = posterior.sample(generator) if sample_posterior else posterior.mode()
        return self.decode(latents)
