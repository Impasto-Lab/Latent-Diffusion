"""Local definition of the learned text transformer shipped with the LDM."""
import torch
from torch import nn
from torch.nn import functional as F

from models.config import ConfigDict


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.encoder_attention_heads
        self.head_dim = config.head_dim
        self.inner_dim = self.num_heads * self.head_dim
        self.scale = self.head_dim**-0.5
        self.k_proj = nn.Linear(config.d_model, self.inner_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, self.inner_dim, bias=False)
        self.q_proj = nn.Linear(config.d_model, self.inner_dim, bias=False)
        self.out_proj = nn.Linear(self.inner_dim, config.d_model)

    def _heads(self, tensor):
        batch, length, _ = tensor.shape
        return tensor.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, hidden):
        batch, length, _ = hidden.shape
        query = self._heads(self.q_proj(hidden)) * self.scale
        key = self._heads(self.k_proj(hidden))
        value = self._heads(self.v_proj(hidden))
        query = query.reshape(batch * self.num_heads, length, self.head_dim)
        key = key.reshape(batch * self.num_heads, length, self.head_dim)
        value = value.reshape(batch * self.num_heads, length, self.head_dim)
        scores = torch.bmm(query, key.transpose(1, 2))
        hidden = torch.bmm(scores.softmax(dim=-1), value)
        hidden = hidden.view(batch, self.num_heads, length, self.head_dim)
        hidden = hidden.transpose(1, 2).reshape(batch, length, self.inner_dim)
        return self.out_proj(hidden)


class BertEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = BertSelfAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.fc1 = nn.Linear(config.d_model, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, config.d_model)
        self.final_layer_norm = nn.LayerNorm(config.d_model)

    def forward(self, hidden):
        residual = hidden
        hidden = self.self_attn(self.self_attn_layer_norm(hidden))
        hidden = residual + hidden
        residual = hidden
        hidden = F.gelu(self.fc1(self.final_layer_norm(hidden)))
        hidden = self.fc2(hidden)
        hidden = residual + hidden
        if hidden.dtype == torch.float16 and not torch.isfinite(hidden).all():
            limit = torch.finfo(hidden.dtype).max - 1000
            hidden = hidden.clamp(-limit, limit)
        return hidden


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.embed_positions = nn.Embedding(config.max_position_embeddings, config.d_model)
        self.layers = nn.ModuleList(
            [BertEncoderLayer(config) for _ in range(config.encoder_layers)]
        )
        self.layer_norm = nn.LayerNorm(config.d_model)

    def forward(self, input_ids):
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)[None]
        hidden = self.embed_tokens(input_ids) + self.embed_positions(positions)
        for layer in self.layers:
            hidden = layer(hidden)
        return self.layer_norm(hidden)


class LDMBertModel(nn.Module):
    """32-layer prompt encoder; output shape is ``[batch, 77, 1280]``."""

    def __init__(self, config):
        super().__init__()
        config = ConfigDict({name: config[name] for name in (
            "vocab_size", "max_position_embeddings", "d_model", "encoder_layers",
            "encoder_ffn_dim", "encoder_attention_heads", "head_dim",
        )})
        self.config = config
        self.model = BertEncoder(config)
        # Unused during text encoding, but required for strict checkpoint loading.
        self.to_logits = nn.Linear(config.d_model, config.vocab_size)

    @property
    def max_sequence_length(self) -> int:
        """Maximum token count accepted by this checkpoint's position table."""
        return int(self.config.max_position_embeddings)

    def forward(self, input_ids):
        return self.model(input_ids)
