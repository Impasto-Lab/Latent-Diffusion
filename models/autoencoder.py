"""KL-regularized autoencoder: image [B, 3, H, W] <-> latent [B, 4, H/8, W/8].

Text-to-image only needs ``decode``. The encoder is included so the model is
complete (and its checkpoint loads strictly); it is what produced the
training latents that the U-Net learned to denoise.
"""
from torch import nn
from torch.nn import functional as F

from models.layers import Downsample2D, ResnetBlock2D, Upsample2D, attention, group_norm

EPS = 1e-6  # GroupNorm epsilon throughout the autoencoder


class VAESpatialAttention(nn.Module):
    """Single-head self-attention over all pixels."""

    def __init__(self, channels):
        super().__init__()
        self.group_norm = group_norm(channels, EPS)
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        self.proj_attn = nn.Linear(channels, channels)

    def forward(self, x):
        batch, channels, height, width = x.shape
        tokens = self.group_norm(x).view(batch, channels, height * width).transpose(1, 2)  # [B, h*w, C]
        hidden = attention(self.query(tokens), self.key(tokens), self.value(tokens))
        hidden = self.proj_attn(hidden).transpose(1, 2).reshape(batch, channels, height, width)
        return x + hidden


class VAEMidBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.resnets = nn.ModuleList(ResnetBlock2D(channels, channels, eps=EPS) for _ in range(2))
        self.attentions = nn.ModuleList([VAESpatialAttention(channels)])

    def forward(self, x):
        x = self.resnets[0](x)
        x = self.attentions[0](x)
        return self.resnets[1](x)


class DownEncoderBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, layers, add_downsample):
        super().__init__()
        self.resnets = nn.ModuleList(
            ResnetBlock2D(in_channels if i == 0 else out_channels, out_channels, eps=EPS) for i in range(layers)
        )
        self.downsamplers = nn.ModuleList([Downsample2D(out_channels, padding=0)]) if add_downsample else None

    def forward(self, x):
        for resnet in self.resnets:
            x = resnet(x)
        if self.downsamplers is not None:
            x = self.downsamplers[0](x)
        return x


class UpDecoderBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, layers, add_upsample):
        super().__init__()
        self.resnets = nn.ModuleList(
            ResnetBlock2D(in_channels if i == 0 else out_channels, out_channels, eps=EPS) for i in range(layers)
        )
        self.upsamplers = nn.ModuleList([Upsample2D(out_channels)]) if add_upsample else None

    def forward(self, x):
        for resnet in self.resnets:
            x = resnet(x)
        if self.upsamplers is not None:
            x = self.upsamplers[0](x)
        return x


class Encoder(nn.Module):
    """Image -> mean and log-variance of the latent (stacked along channels)."""

    def __init__(self, config):
        super().__init__()
        channels = config.block_out_channels  # [128, 256, 512, 512]
        self.conv_in = nn.Conv2d(config.in_channels, channels[0], 3, padding=1)
        inputs = [channels[0]] + channels[:-1]  # each block takes the previous block's width
        self.down_blocks = nn.ModuleList(
            DownEncoderBlock2D(inputs[i], channels[i], config.layers_per_block, add_downsample=i < len(channels) - 1)
            for i in range(len(channels))
        )
        self.mid_block = VAEMidBlock(channels[-1])
        self.conv_norm_out = group_norm(channels[-1], EPS)
        self.conv_out = nn.Conv2d(channels[-1], 2 * config.latent_channels, 3, padding=1)

    def forward(self, image):
        x = self.conv_in(image)
        for block in self.down_blocks:
            x = block(x)
        x = self.mid_block(x)
        return self.conv_out(F.silu(self.conv_norm_out(x)))


class Decoder(nn.Module):
    """Latent -> image; the encoder's layers in reverse, one extra ResNet per level."""

    def __init__(self, config):
        super().__init__()
        channels = config.block_out_channels[::-1]  # [512, 512, 256, 128]
        self.conv_in = nn.Conv2d(config.latent_channels, channels[0], 3, padding=1)
        self.mid_block = VAEMidBlock(channels[0])
        inputs = [channels[0]] + channels[:-1]
        self.up_blocks = nn.ModuleList(
            UpDecoderBlock2D(inputs[i], channels[i], config.layers_per_block + 1, add_upsample=i < len(channels) - 1)
            for i in range(len(channels))
        )
        self.conv_norm_out = group_norm(channels[-1], EPS)
        self.conv_out = nn.Conv2d(channels[-1], config.out_channels, 3, padding=1)

    def forward(self, latents):
        x = self.mid_block(self.conv_in(latents))
        for block in self.up_blocks:
            x = block(x)
        return self.conv_out(F.silu(self.conv_norm_out(x)))


class AutoencoderKL(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.quant_conv = nn.Conv2d(2 * config.latent_channels, 2 * config.latent_channels, 1)
        self.post_quant_conv = nn.Conv2d(config.latent_channels, config.latent_channels, 1)

    def encode(self, image):
        """Image in [-1, 1] -> latent. Returns the mean of the predicted Gaussian.

        The encoder also predicts a log-variance; during training a KL penalty
        keeps this Gaussian close to N(0, I) ("KL-regularized"). Using the mean
        is the usual deterministic choice at inference time.
        """
        mean, _log_variance = self.quant_conv(self.encoder(image)).chunk(2, dim=1)
        return mean

    def decode(self, latents):
        """Latent -> image in roughly [-1, 1]."""
        return self.decoder(self.post_quant_conv(latents))
