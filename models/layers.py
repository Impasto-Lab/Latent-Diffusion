"""Small PyTorch building blocks shared by the local U-Net and KL autoencoder.

The module names intentionally match the published checkpoint keys, so each
saved tensor can be traced to the layer that consumes it.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def group_norm(channels, eps):
    return nn.GroupNorm(32, channels, eps=eps, affine=True)


class ResnetBlock2D(nn.Module):
    """Pre-normalized residual block shared by the U-Net and KL autoencoder."""

    def __init__(self, in_channels, out_channels, *, temb_channels=None, eps=1e-6):
        super().__init__()
        self.norm1 = group_norm(in_channels, eps)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        if temb_channels is not None:
            self.time_emb_proj = nn.Linear(temb_channels, out_channels)
        else:
            self.time_emb_proj = None
        self.norm2 = group_norm(out_channels, eps)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.nonlinearity = nn.SiLU()
        if in_channels != out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.conv_shortcut = None

    def forward(self, input_tensor, temb=None):
        hidden = self.conv1(self.nonlinearity(self.norm1(input_tensor)))
        if self.time_emb_proj is not None:
            # Add the time vector at every spatial position of each channel.
            hidden = hidden + self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]
        hidden = self.conv2(self.nonlinearity(self.norm2(hidden)))
        residual = (
            self.conv_shortcut(input_tensor.contiguous())
            if self.conv_shortcut is not None
            else input_tensor
        )
        return residual + hidden


class Downsample2D(nn.Module):
    def __init__(self, channels, *, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=padding)

    def forward(self, hidden):
        if self.padding == 0:
            hidden = F.pad(hidden, (0, 1, 0, 1))
        return self.conv(hidden)


class Upsample2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, hidden):
        hidden = F.interpolate(hidden, scale_factor=2.0, mode="nearest")
        return self.conv(hidden)


class CrossAttention(nn.Module):
    """Multi-head self/cross attention with explicit head reshaping."""

    def __init__(self, query_dim, context_dim, heads):
        super().__init__()
        if query_dim % heads:
            raise ValueError(f"{query_dim=} must be divisible by {heads=}")
        self.heads = heads
        self.head_dim = query_dim // heads
        self.scale = self.head_dim**-0.5
        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k = nn.Linear(context_dim, query_dim, bias=False)
        self.to_v = nn.Linear(context_dim, query_dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(query_dim, query_dim)])

    def _to_heads(self, tensor):
        batch, length, channels = tensor.shape
        tensor = tensor.view(batch, length, self.heads, channels // self.heads)
        return tensor.permute(0, 2, 1, 3).reshape(batch * self.heads, length, -1)

    def _attention(self, query, key, value):
        scores = torch.baddbmm(
            torch.empty(
                query.shape[0], query.shape[1], key.shape[1],
                device=query.device, dtype=query.dtype,
            ),
            query,
            key.transpose(1, 2),
            beta=0,
            alpha=self.scale,
        )
        probabilities = scores.softmax(dim=-1)
        return torch.bmm(probabilities, value)

    def forward(self, hidden, context=None):
        context = hidden if context is None else context
        batch, length, channels = hidden.shape
        query = self._to_heads(self.to_q(hidden))
        key = self._to_heads(self.to_k(context))
        value = self._to_heads(self.to_v(context))
        hidden = self._attention(query, key, value)
        hidden = hidden.view(batch, self.heads, length, self.head_dim)
        hidden = hidden.permute(0, 2, 1, 3).reshape(batch, length, channels)
        return self.to_out[0](hidden)


class GEGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim * 8)

    def forward(self, hidden):
        value, gate = self.proj(hidden).chunk(2, dim=-1)
        return value * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.ModuleList([GEGLU(dim), nn.Dropout(0.0), nn.Linear(dim * 4, dim)])

    def forward(self, hidden):
        for layer in self.net:
            hidden = layer(hidden)
        return hidden


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, context_dim, heads):
        super().__init__()
        self.attn1 = CrossAttention(dim, dim, heads)
        self.ff = FeedForward(dim)
        self.attn2 = CrossAttention(dim, context_dim, heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, hidden, context):
        # First relate image tokens to each other, then read the text tokens.
        hidden = hidden + self.attn1(self.norm1(hidden))
        hidden = hidden + self.attn2(self.norm2(hidden), context)
        return hidden + self.ff(self.norm3(hidden))


class SpatialTransformer(nn.Module):
    """Flatten an image to tokens, apply self/cross attention, then restore it."""

    def __init__(self, channels, context_dim, heads):
        super().__init__()
        self.norm = group_norm(channels, 1e-6)
        self.proj_in = nn.Conv2d(channels, channels, 1)
        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(channels, context_dim, heads)]
        )
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, hidden, context):
        residual = hidden
        hidden = self.proj_in(self.norm(hidden))
        batch, channels, height, width = hidden.shape
        hidden = hidden.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        for block in self.transformer_blocks:
            hidden = block(hidden, context)
        hidden = hidden.reshape(batch, height, width, channels).permute(0, 3, 1, 2)
        return residual + self.proj_out(hidden)


class VAESpatialAttention(nn.Module):
    """Single-head spatial attention used in the KL autoencoder bottleneck."""

    def __init__(self, channels):
        super().__init__()
        self.group_norm = group_norm(channels, 1e-6)
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        self.proj_attn = nn.Linear(channels, channels)
        self.scale = channels**-0.5

    def forward(self, hidden):
        residual = hidden
        batch, channels, height, width = hidden.shape
        hidden = self.group_norm(hidden).view(batch, channels, height * width).transpose(1, 2)
        query = self.query(hidden)
        key = self.key(hidden)
        value = self.value(hidden)
        scores = torch.baddbmm(
            torch.empty(batch, height * width, height * width, device=hidden.device, dtype=hidden.dtype),
            query,
            key.transpose(1, 2),
            beta=0,
            alpha=self.scale,
        )
        hidden = torch.bmm(scores.softmax(dim=-1), value)
        hidden = self.proj_attn(hidden).transpose(1, 2).reshape(batch, channels, height, width)
        return residual + hidden


def timestep_embedding(timesteps, channels, max_period=10000):
    """Sinusoidal embedding used to turn a scalar diffusion step into a vector."""
    if timesteps.ndim == 0:
        timesteps = timesteps[None]
    half = channels // 2
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    phases = timesteps.float()[:, None] * frequencies[None]
    embedding = torch.cat([phases.cos(), phases.sin()], dim=-1)
    if channels % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding
