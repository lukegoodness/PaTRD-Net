"""
iTransformer
========================================================================
Reference:
    Liu, Y., Hu, T., Zhang, H., Wu, H., Wang, S., Ma, L., & Long, M. (2024).
    "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting."
    ICLR 2024.
    Code: https://github.com/thuml/iTransformer

Core idea (inverted):
    - Treat each variate as a TOKEN (instead of each time step).
    - Token dimension: time steps (seq_len).
    - Sequence length (for attention): number of variates.
    - This learns cross-variate dependencies via self-attention,
      and intra-variate dependencies via FFN.

Adapted for the PaTRD-Net pipeline (Config-based interface).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + self.dropout(attn_out)
        h = self.norm2(x)
        ff_out = self.linear2(self.dropout(self.act(self.linear1(h))))
        x = x + self.dropout(ff_out)
        return x


class Model(nn.Module):
    """iTransformer: invert variates as tokens."""

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.d_model = getattr(configs, "d_model", 512)
        self.n_heads = getattr(configs, "n_heads", 8)
        self.d_ff = getattr(configs, "d_ff", 512)
        self.e_layers = getattr(configs, "e_layers", 3)
        self.dropout_p = getattr(configs, "dropout", 0.1)
        self.use_norm = getattr(configs, "use_norm", True)

        # Embed each variate (its full time series) into d_model
        # input to embed: [B, N, L] -> embed last dim L -> [B, N, d_model]
        self.embedding = nn.Linear(self.seq_len, self.d_model)
        self.embed_dropout = nn.Dropout(self.dropout_p)

        # Transformer encoder over N tokens
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(self.d_model, self.n_heads, self.d_ff, self.dropout_p)
            for _ in range(self.e_layers)
        ])
        self.encoder_norm = nn.LayerNorm(self.d_model)

        # Project each variate token back to pred_len
        self.projector = nn.Linear(self.d_model, self.pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, N]
        if self.use_norm:
            mean = x.mean(dim=1, keepdim=True).detach()
            std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            x = (x - mean) / std

        # to [B, N, L] then embed L -> d_model
        x = x.permute(0, 2, 1)
        x = self.embedding(x)                      # [B, N, d_model]
        x = self.embed_dropout(x)

        # encoder over N variate tokens
        for layer in self.encoder_layers:
            x = layer(x)
        x = self.encoder_norm(x)

        # project each token to pred_len
        x = self.projector(x)                      # [B, N, P]

        # to [B, P, N]
        x = x.permute(0, 2, 1)

        if self.use_norm:
            x = x * std + mean

        return x

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def regularization_loss(self) -> torch.Tensor:
        return torch.tensor(0.0)
