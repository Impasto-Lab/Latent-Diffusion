"""Conditional U-Net: predicts the noise in a latent, given the timestep and text.

    latents [B, 4, h, w] ─ conv_in ─ 4 down blocks ─ mid block ─ 4 up blocks ─ conv_out ─ noise [B, 4, h, w]
                                         │  skip connections (concatenated)  ▲
                                         └───────────────────────────────────┘
The file reads bottom-up: the timestep embedding, the text-conditioning
transformer, the down/mid/up blocks, and finally ``ConditionalUNet``.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from models.layers import (
    Downsample2D,
    ResnetBlock2D,
    Upsample2D,
    attention,
    group_norm,
    merge_heads,
    split_heads,
)

EPS = 1e-5  # GroupNorm epsilon of the U-Net's ResNets (the autoencoder uses 1e-6)


# ---------------------------------------------------------------------------
# Timestep: integer t -> vector added inside every ResNet
# ---------------------------------------------------------------------------

def timestep_embedding(timesteps, dim, max_period=10000):
    """Sinusoidal features of t, as in Transformer position encodings: [B] -> [B, dim]."""
    half = dim // 2
    frequencies = torch.exp(-math.log(max_period) * torch.arange(half, device=timesteps.device) / half)
    angles = timesteps.float()[:, None] * frequencies[None]
    return torch.cat([angles.cos(), angles.sin()], dim=-1)


class TimestepEmbedding(nn.Module):
    """A small MLP on top of the sinusoidal features."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, out_dim)
        self.linear_2 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        return self.linear_2(F.silu(self.linear_1(x)))


# ---------------------------------------------------------------------------
# Text conditioning: image tokens attend to themselves, then to the text
# ---------------------------------------------------------------------------

class CrossAttention(nn.Module):
    """Queries come from the image; keys and values from ``context``.

    With ``context=None`` it attends to the image itself (self-attention).
    """

    def __init__(self, dim, context_dim, heads):
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(context_dim, dim, bias=False)
        self.to_v = nn.Linear(context_dim, dim, bias=False)
        # A list because the checkpoint key is "to_out.0" (index 1 was dropout).
        self.to_out = nn.ModuleList([nn.Linear(dim, dim)])

    def forward(self, x, context=None):
        context = x if context is None else context
        query = split_heads(self.to_q(x), self.heads)
        key = split_heads(self.to_k(context), self.heads)
        value = split_heads(self.to_v(context), self.heads)
        return self.to_out[0](merge_heads(attention(query, key, value)))


class GEGLU(nn.Module):
    """Gated GELU: one projection gives a value and a gate; output = value * GELU(gate)."""

    def __init__(self, dim, inner_dim):
        super().__init__()
        self.proj = nn.Linear(dim, 2 * inner_dim)

    def forward(self, x):
        value, gate = self.proj(x).chunk(2, dim=-1)
        return value * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inner_dim = 4 * dim
        # Dropout(0) does nothing; it keeps checkpoint keys at "net.0" and "net.2".
        self.net = nn.Sequential(GEGLU(dim, inner_dim), nn.Dropout(0.0), nn.Linear(inner_dim, dim))

    def forward(self, x):
        return self.net(x)


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, context_dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = CrossAttention(dim, dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = CrossAttention(dim, context_dim, heads)
        self.norm3 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim)

    def forward(self, x, context):
        x = x + self.attn1(self.norm1(x))  # image <-> image
        x = x + self.attn2(self.norm2(x), context)  # image <- text
        return x + self.ff(self.norm3(x))


class SpatialTransformer(nn.Module):
    """Treat each pixel as a token, run a transformer block, then restore the image shape."""

    def __init__(self, channels, context_dim, heads):
        super().__init__()
        self.norm = group_norm(channels, 1e-6)
        self.proj_in = nn.Conv2d(channels, channels, 1)
        self.transformer_blocks = nn.ModuleList([BasicTransformerBlock(channels, context_dim, heads)])
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x, context):
        batch, channels, height, width = x.shape
        hidden = self.proj_in(self.norm(x))
        hidden = hidden.permute(0, 2, 3, 1).reshape(batch, height * width, channels)  # [B, h*w, C]
        for block in self.transformer_blocks:
            hidden = block(hidden, context)
        hidden = hidden.reshape(batch, height, width, channels).permute(0, 3, 1, 2)  # [B, C, h, w]
        return x + self.proj_out(hidden)


# ---------------------------------------------------------------------------
# U-Net blocks
# ---------------------------------------------------------------------------

class DownBlock2D(nn.Module):
    """ResNet (+ attention) layers, then an optional 2x downsample.

    Returns the output and every intermediate feature map, which the up path
    later uses as skip connections.
    """

    def __init__(self, in_channels, out_channels, temb_channels, layers, add_downsample, context_dim, heads):
        super().__init__()
        self.resnets = nn.ModuleList(
            ResnetBlock2D(in_channels if i == 0 else out_channels, out_channels, eps=EPS, temb_channels=temb_channels)
            for i in range(layers)
        )
        self.attentions = (
            nn.ModuleList(SpatialTransformer(out_channels, context_dim, heads) for _ in range(layers))
            if context_dim else None
        )
        self.downsamplers = nn.ModuleList([Downsample2D(out_channels, padding=1)]) if add_downsample else None

    def forward(self, x, temb, context):
        skips = []
        for i, resnet in enumerate(self.resnets):
            x = resnet(x, temb)
            if self.attentions is not None:
                x = self.attentions[i](x, context)
            skips.append(x)
        if self.downsamplers is not None:
            x = self.downsamplers[0](x)
            skips.append(x)
        return x, skips


class UpBlock2D(nn.Module):
    """Each ResNet first concatenates one skip feature map along channels.

    ``skip_channels`` lists the width of each skip this block will receive,
    in the order it receives them.
    """

    def __init__(self, in_channels, out_channels, skip_channels, temb_channels, add_upsample, context_dim, heads):
        super().__init__()
        self.resnets = nn.ModuleList(
            ResnetBlock2D(
                (in_channels if i == 0 else out_channels) + skip,
                out_channels,
                eps=EPS,
                temb_channels=temb_channels,
            )
            for i, skip in enumerate(skip_channels)
        )
        self.attentions = (
            nn.ModuleList(SpatialTransformer(out_channels, context_dim, heads) for _ in skip_channels)
            if context_dim else None
        )
        self.upsamplers = nn.ModuleList([Upsample2D(out_channels)]) if add_upsample else None

    def forward(self, x, skips, temb, context):
        for i, resnet in enumerate(self.resnets):
            x = resnet(torch.cat([x, skips.pop()], dim=1), temb)  # most recent skip first
            if self.attentions is not None:
                x = self.attentions[i](x, context)
        if self.upsamplers is not None:
            x = self.upsamplers[0](x)
        return x


class MidBlock(nn.Module):
    """ResNet -> transformer -> ResNet at the lowest resolution."""

    def __init__(self, channels, temb_channels, context_dim, heads):
        super().__init__()
        self.resnets = nn.ModuleList(
            ResnetBlock2D(channels, channels, eps=EPS, temb_channels=temb_channels) for _ in range(2)
        )
        self.attentions = nn.ModuleList([SpatialTransformer(channels, context_dim, heads)])

    def forward(self, x, temb, context):
        x = self.resnets[0](x, temb)
        x = self.attentions[0](x, context)
        return self.resnets[1](x, temb)


# ---------------------------------------------------------------------------
# The full U-Net
# ---------------------------------------------------------------------------

class ConditionalUNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        channels = config.block_out_channels  # [320, 640, 1280, 1280], one per resolution level
        levels = len(channels)
        layers = config.layers_per_block  # ResNets per down block
        temb_channels = 4 * channels[0]
        heads = config.attention_head_dim  # legacy name: this checkpoint stores the head *count* here
        context_dim = config.cross_attention_dim

        self.time_embedding = TimestepEmbedding(channels[0], temb_channels)
        self.conv_in = nn.Conv2d(config.in_channels, channels[0], 3, padding=1)

        # Down path. Track the width of every feature map ``forward`` will save
        # as a skip, so the up path can be built to consume exactly those.
        skip_channels = [channels[0]]  # conv_in output
        self.down_blocks = nn.ModuleList()
        width = channels[0]
        for level, block_type in enumerate(config.down_block_types):
            add_downsample = level < levels - 1
            self.down_blocks.append(DownBlock2D(
                width, channels[level], temb_channels, layers, add_downsample,
                context_dim if block_type.startswith("CrossAttn") else None, heads,
            ))
            width = channels[level]
            skip_channels += [width] * layers  # one per ResNet
            if add_downsample:
                skip_channels.append(width)  # and one after downsampling

        self.mid_block = MidBlock(width, temb_channels, context_dim, heads)

        # Up path: mirror image of the down path. Each up block has one more
        # ResNet than a down block so that all skips (including conv_in's) are used.
        self.up_blocks = nn.ModuleList()
        for level, block_type in enumerate(config.up_block_types):
            out_channels = channels[levels - 1 - level]
            block_skips = [skip_channels.pop() for _ in range(layers + 1)]
            self.up_blocks.append(UpBlock2D(
                width, out_channels, block_skips, temb_channels, level < levels - 1,
                context_dim if block_type.startswith("CrossAttn") else None, heads,
            ))
            width = out_channels

        self.conv_norm_out = group_norm(channels[0], EPS)
        self.conv_out = nn.Conv2d(channels[0], config.out_channels, 3, padding=1)

    def forward(self, latents, timestep, context):
        # The same timestep applies to every sample in the batch.
        timesteps = timestep.to(latents.device).expand(latents.shape[0])
        temb = timestep_embedding(timesteps, self.config.block_out_channels[0])
        temb = self.time_embedding(temb.to(latents.dtype))

        x = self.conv_in(latents)
        skips = [x]
        for block in self.down_blocks:
            x, block_skips = block(x, temb, context)
            skips += block_skips

        x = self.mid_block(x, temb, context)

        for block in self.up_blocks:
            x = block(x, skips, temb, context)  # pops the skips it uses

        return self.conv_out(F.silu(self.conv_norm_out(x)))
