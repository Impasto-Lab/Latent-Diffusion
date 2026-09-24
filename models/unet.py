"""Local conditional U-Net used for latent-space noise prediction.

Read ``ConditionalUNet.forward`` from top to bottom to follow the feature maps:
input convolution -> four down blocks -> bottleneck -> four up blocks -> noise.
"""
import torch
from torch import nn

from models.layers import (
    ConfigDict,
    Downsample2D,
    ResnetBlock2D,
    SpatialTransformer,
    Upsample2D,
    group_norm,
    timestep_embedding,
)


class TimestepEmbedding(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.linear_1 = nn.Linear(input_channels, output_channels)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(output_channels, output_channels)

    def forward(self, sample):
        return self.linear_2(self.act(self.linear_1(sample)))


class DownBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, temb_channels, layers, add_downsample):
        super().__init__()
        self.resnets = nn.ModuleList(
            [
                ResnetBlock2D(
                    in_channels if index == 0 else out_channels,
                    out_channels,
                    temb_channels=temb_channels,
                    eps=1e-5,
                )
                for index in range(layers)
            ]
        )
        self.downsamplers = (
            nn.ModuleList([Downsample2D(out_channels, padding=1)])
            if add_downsample
            else None
        )

    def forward(self, hidden, temb):
        new_skips = []
        for resnet in self.resnets:
            hidden = resnet(hidden, temb)
            new_skips.append(hidden)
        if self.downsamplers is not None:
            hidden = self.downsamplers[0](hidden)
            new_skips.append(hidden)
        return hidden, new_skips


class CrossAttnDownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        temb_channels,
        layers,
        add_downsample,
        context_dim,
        heads,
    ):
        super().__init__()
        self.resnets = nn.ModuleList(
            [
                ResnetBlock2D(
                    in_channels if index == 0 else out_channels,
                    out_channels,
                    temb_channels=temb_channels,
                    eps=1e-5,
                )
                for index in range(layers)
            ]
        )
        self.attentions = nn.ModuleList(
            [SpatialTransformer(out_channels, context_dim, heads) for _ in range(layers)]
        )
        self.downsamplers = (
            nn.ModuleList([Downsample2D(out_channels, padding=1)])
            if add_downsample
            else None
        )

    def forward(self, hidden, temb, context):
        new_skips = []
        for layer_index, resnet in enumerate(self.resnets):
            hidden = resnet(hidden, temb)
            hidden = self.attentions[layer_index](hidden, context)
            new_skips.append(hidden)
        if self.downsamplers is not None:
            hidden = self.downsamplers[0](hidden)
            new_skips.append(hidden)
        return hidden, new_skips


class UpBlock2D(nn.Module):
    def __init__(
        self,
        skip_channels,
        out_channels,
        previous_channels,
        temb_channels,
        layers,
        add_upsample,
    ):
        super().__init__()
        resnets = []
        for index in range(layers):
            current_skip_channels = skip_channels if index == layers - 1 else out_channels
            current_channels = previous_channels if index == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    current_channels + current_skip_channels,
                    out_channels,
                    temb_channels=temb_channels,
                    eps=1e-5,
                )
            )
        self.resnets = nn.ModuleList(resnets)
        self.upsamplers = (
            nn.ModuleList([Upsample2D(out_channels)]) if add_upsample else None
        )

    def forward(self, hidden, skip_features, temb):
        for resnet in self.resnets:
            skip = skip_features.pop()
            hidden = resnet(torch.cat([hidden, skip], dim=1), temb)
        if self.upsamplers is not None:
            hidden = self.upsamplers[0](hidden)
        return hidden


class CrossAttnUpBlock2D(nn.Module):
    def __init__(
        self,
        skip_channels,
        out_channels,
        previous_channels,
        temb_channels,
        layers,
        add_upsample,
        context_dim,
        heads,
    ):
        super().__init__()
        resnets = []
        for index in range(layers):
            current_skip_channels = skip_channels if index == layers - 1 else out_channels
            current_channels = previous_channels if index == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    current_channels + current_skip_channels,
                    out_channels,
                    temb_channels=temb_channels,
                    eps=1e-5,
                )
            )
        self.resnets = nn.ModuleList(resnets)
        self.upsamplers = (
            nn.ModuleList([Upsample2D(out_channels)]) if add_upsample else None
        )
        self.attentions = nn.ModuleList(
            [SpatialTransformer(out_channels, context_dim, heads) for _ in range(layers)]
        )

    def forward(self, hidden, skip_features, temb, context):
        for layer_index, resnet in enumerate(self.resnets):
            skip = skip_features.pop()
            hidden = resnet(torch.cat([hidden, skip], dim=1), temb)
            hidden = self.attentions[layer_index](hidden, context)
        if self.upsamplers is not None:
            hidden = self.upsamplers[0](hidden)
        return hidden


class UNetMidBlock2DCrossAttn(nn.Module):
    def __init__(self, channels, temb_channels, context_dim, heads):
        super().__init__()
        self.resnets = nn.ModuleList(
            [
                ResnetBlock2D(channels, channels, temb_channels=temb_channels, eps=1e-5),
                ResnetBlock2D(channels, channels, temb_channels=temb_channels, eps=1e-5),
            ]
        )
        self.attentions = nn.ModuleList(
            [SpatialTransformer(channels, context_dim, heads)]
        )

    def forward(self, hidden, temb, context):
        hidden = self.resnets[0](hidden, temb)
        hidden = self.attentions[0](hidden, context)
        return self.resnets[1](hidden, temb)


class ConditionalUNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        config = ConfigDict({name: config[name] for name in (
            "in_channels", "out_channels", "block_out_channels", "down_block_types",
            "up_block_types", "layers_per_block", "attention_head_dim",
            "cross_attention_dim", "norm_eps",
        )})
        self.config = config

        channels = list(config.block_out_channels)
        time_channels = channels[0] * 4
        heads = config.attention_head_dim
        self.conv_in = nn.Conv2d(config.in_channels, channels[0], 3, padding=1)
        self.time_embedding = TimestepEmbedding(channels[0], time_channels)

        if len(config.down_block_types) != len(channels):
            raise ValueError("Each U-Net channel level needs one down block type.")
        if len(config.up_block_types) != len(channels):
            raise ValueError("Each U-Net channel level needs one up block type.")

        self.down_blocks = nn.ModuleList()
        input_channels = channels[0]
        for level in range(len(channels)):
            block_type = config.down_block_types[level]
            output_channels = channels[level]
            add_downsample = level < len(channels) - 1

            if block_type == "CrossAttnDownBlock2D":
                down_block = CrossAttnDownBlock2D(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    temb_channels=time_channels,
                    layers=config.layers_per_block,
                    add_downsample=add_downsample,
                    context_dim=config.cross_attention_dim,
                    heads=heads,
                )
            elif block_type == "DownBlock2D":
                down_block = DownBlock2D(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    temb_channels=time_channels,
                    layers=config.layers_per_block,
                    add_downsample=add_downsample,
                )
            else:
                raise ValueError(f"Unknown down block type: {block_type}")

            self.down_blocks.append(down_block)
            input_channels = output_channels

        self.mid_block = UNetMidBlock2DCrossAttn(
            channels[-1], time_channels, config.cross_attention_dim, heads
        )

        self.up_blocks = nn.ModuleList()
        reversed_channels = list(reversed(channels))
        previous_channels = reversed_channels[0]
        for level in range(len(channels)):
            block_type = config.up_block_types[level]
            output_channels = reversed_channels[level]
            skip_channels = reversed_channels[min(level + 1, len(channels) - 1)]
            add_upsample = level < len(channels) - 1

            if block_type == "CrossAttnUpBlock2D":
                up_block = CrossAttnUpBlock2D(
                    skip_channels=skip_channels,
                    out_channels=output_channels,
                    previous_channels=previous_channels,
                    temb_channels=time_channels,
                    layers=config.layers_per_block + 1,
                    add_upsample=add_upsample,
                    context_dim=config.cross_attention_dim,
                    heads=heads,
                )
            elif block_type == "UpBlock2D":
                up_block = UpBlock2D(
                    skip_channels=skip_channels,
                    out_channels=output_channels,
                    previous_channels=previous_channels,
                    temb_channels=time_channels,
                    layers=config.layers_per_block + 1,
                    add_upsample=add_upsample,
                )
            else:
                raise ValueError(f"Unknown up block type: {block_type}")

            self.up_blocks.append(up_block)
            previous_channels = output_channels
        self.conv_norm_out = group_norm(channels[0], config.norm_eps)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(channels[0], config.out_channels, 3, padding=1)

    def forward(self, sample, timestep, encoder_hidden_states):
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], device=sample.device)
        elif timesteps.ndim == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])
        temb = timestep_embedding(timesteps, self.config.block_out_channels[0])
        temb = self.time_embedding(temb.to(dtype=sample.dtype))

        # Save the input feature map and every down-block feature map. The up
        # blocks consume this list from the end, one skip per ResNet layer.
        hidden = self.conv_in(sample)
        skip_features = [hidden]
        for down_block in self.down_blocks:
            if isinstance(down_block, CrossAttnDownBlock2D):
                hidden, new_skips = down_block(hidden, temb, encoder_hidden_states)
            else:
                hidden, new_skips = down_block(hidden, temb)
            skip_features.extend(new_skips)

        hidden = self.mid_block(hidden, temb, encoder_hidden_states)

        for up_block in self.up_blocks:
            if isinstance(up_block, CrossAttnUpBlock2D):
                hidden = up_block(hidden, skip_features, temb, encoder_hidden_states)
            else:
                hidden = up_block(hidden, skip_features, temb)

        if skip_features:
            raise RuntimeError("U-Net up blocks did not consume all skip features.")

        hidden = self.conv_out(self.conv_act(self.conv_norm_out(hidden)))
        return hidden
