"""Building blocks shared by the text encoder, U-Net, and autoencoder.

Attribute names (``norm1``, ``conv_shortcut``, ...) match the published
checkpoint keys, so every saved tensor maps to exactly one layer here.
"""
from torch import nn
from torch.nn import functional as F


def group_norm(channels, eps):
    return nn.GroupNorm(32, channels, eps=eps)


class ResnetBlock2D(nn.Module):
    """Two pre-normalized 3x3 convolutions plus a residual connection.

    In the U-Net, the timestep vector ``temb`` is added between the two convs;
    the autoencoder has no timestep, so it builds the block without one.
    """

    def __init__(self, in_channels, out_channels, *, eps, temb_channels=None):
        super().__init__()
        self.norm1 = group_norm(in_channels, eps)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_emb_proj = nn.Linear(temb_channels, out_channels) if temb_channels else None
        self.norm2 = group_norm(out_channels, eps)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        # A 1x1 conv matches channel counts when the residual changes width.
        self.conv_shortcut = (
            nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x, temb=None):
        hidden = self.conv1(F.silu(self.norm1(x)))
        if self.time_emb_proj is not None:
            # One value per channel, broadcast over every pixel: [B, C] -> [B, C, 1, 1].
            hidden = hidden + self.time_emb_proj(F.silu(temb))[:, :, None, None]
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return self.conv_shortcut(x) + hidden


class Downsample2D(nn.Module):
    """Halve height and width with a stride-2 3x3 conv."""

    def __init__(self, channels, *, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=padding)

    def forward(self, x):
        # The autoencoder was trained with padding only on the right and bottom.
        if self.padding == 0:
            x = F.pad(x, (0, 1, 0, 1))
        return self.conv(x)


class Upsample2D(nn.Module):
    """Double height and width with nearest-neighbor resize, then a 3x3 conv."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


def split_heads(x, heads):
    """[batch, tokens, heads * dim] -> [batch, heads, tokens, dim]"""
    batch, tokens, _ = x.shape
    return x.view(batch, tokens, heads, -1).transpose(1, 2)


def merge_heads(x):
    """[batch, heads, tokens, dim] -> [batch, tokens, heads * dim]"""
    batch, heads, tokens, dim = x.shape
    return x.transpose(1, 2).reshape(batch, tokens, heads * dim)


def attention(query, key, value):
    """softmax(Q K^T / sqrt(d)) V over the last two dimensions: [..., tokens, dim].

    Every attention layer in this project (text encoder, U-Net, autoencoder)
    calls this function; they differ only in where Q, K, and V come from.
    """
    scale = query.shape[-1] ** -0.5
    scores = (query * scale) @ key.transpose(-1, -2)  # [..., query tokens, key tokens]
    return scores.softmax(dim=-1) @ value
