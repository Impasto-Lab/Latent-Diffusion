"""LDMBert: the 32-layer text transformer trained together with the LDM.

Token IDs [batch, 77] -> text features [batch, 77, 1280]. Despite the name,
it is a plain pre-LayerNorm transformer encoder, not a pretrained BERT; only
the tokenizer (vocabulary) comes from BERT.
"""
import torch
from torch import nn
from torch.nn import functional as F

from models.layers import attention, merge_heads, split_heads


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.encoder_attention_heads
        inner_dim = self.heads * config.head_dim  # 8 * 64 = 512, narrower than d_model
        self.q_proj = nn.Linear(config.d_model, inner_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, inner_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, inner_dim, bias=False)
        self.out_proj = nn.Linear(inner_dim, config.d_model)

    def forward(self, x):
        query = split_heads(self.q_proj(x), self.heads)
        key = split_heads(self.k_proj(x), self.heads)
        value = split_heads(self.v_proj(x), self.heads)
        return self.out_proj(merge_heads(attention(query, key, value)))


class BertEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.self_attn = BertSelfAttention(config)
        self.final_layer_norm = nn.LayerNorm(config.d_model)
        self.fc1 = nn.Linear(config.d_model, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, config.d_model)

    def forward(self, x):
        x = x + self.self_attn(self.self_attn_layer_norm(x))
        return x + self.fc2(F.gelu(self.fc1(self.final_layer_norm(x))))


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.embed_positions = nn.Embedding(config.max_position_embeddings, config.d_model)
        self.layers = nn.ModuleList(BertEncoderLayer(config) for _ in range(config.encoder_layers))
        self.layer_norm = nn.LayerNorm(config.d_model)

    def forward(self, input_ids):
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        x = self.embed_tokens(input_ids) + self.embed_positions(positions)
        for layer in self.layers:
            x = layer(x)
        return self.layer_norm(x)


class LDMBertModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = BertEncoder(config)
        # Unused for text-to-image; kept so every checkpoint tensor has a home.
        self.to_logits = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, input_ids):
        return self.model(input_ids)
