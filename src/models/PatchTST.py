"""
PatchTST
========================================================================
Reference:
    Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023).
    "A Time Series is Worth 64 Words: Long-Term Forecasting with Transformers."
    ICLR 2023.
    Code: https://github.com/yuqinie98/PatchTST

Core ideas:
    (1) Channel Independence: each channel processed independently with
        shared Transformer weights.
    (2) Patching: split L into N_patch overlapping/non-overlapping patches.
    (3) Standard Transformer encoder on patches.
    (4) Flatten + Linear head for forecasting.

Adapted for the PaTRD-Net pipeline (Config-based interface).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------
# RevIN (Reversible Instance Normalization, Kim et al. ICLR 2022)
# Reversible instance normalization used by this baseline.
# -----------------------------------------------------------------
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor, mode: str):
        if mode == "norm":
            self.mean = x.mean(dim=1, keepdim=True).detach()
            self.std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
            x = (x - self.mean) / self.std
            if self.affine:
                x = x * self.weight + self.bias
        elif mode == "denorm":
            if self.affine:
                x = (x - self.bias) / (self.weight + self.eps * self.eps)
            x = x * self.std + self.mean
        return x


# -----------------------------------------------------------------
# Standard MHA block
# -----------------------------------------------------------------
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
        # Pre-LN per PatchTST official
        h = self.norm1(x)
        attn_out, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + self.dropout(attn_out)
        h = self.norm2(x)
        ff_out = self.linear2(self.dropout(self.act(self.linear1(h))))
        x = x + self.dropout(ff_out)
        return x


# -----------------------------------------------------------------
# Model
# -----------------------------------------------------------------
class Model(nn.Module):
    """PatchTST backbone + Flatten head."""

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in

        # Patching hyperparameters per official defaults
        self.patch_len = getattr(configs, "patch_size", 16)
        self.stride = getattr(configs, "stride", 8)
        self.padding_patch = getattr(configs, "padding_patch", "end")
        self.d_model = getattr(configs, "d_model", 128)
        self.n_heads = getattr(configs, "n_heads", 16)
        self.d_ff = getattr(configs, "d_ff", 256)
        self.e_layers = getattr(configs, "e_layers", 3)
        self.dropout_p = getattr(configs, "dropout", 0.2)
        self.individual = getattr(configs, "individual", False)
        self.use_revin = getattr(configs, "use_revin", True)

        # padding to make patches uniform
        if self.padding_patch == "end":
            self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
            self.patch_num = int((self.seq_len - self.patch_len) / self.stride + 2)
        else:
            self.patch_num = int((self.seq_len - self.patch_len) / self.stride + 1)

        # RevIN
        if self.use_revin:
            self.revin = RevIN(self.channels)

        # Patch embedding
        self.W_P = nn.Linear(self.patch_len, self.d_model)
        self.W_pos = nn.Parameter(torch.zeros(self.patch_num, self.d_model))
        nn.init.uniform_(self.W_pos, -0.02, 0.02)
        self.input_dropout = nn.Dropout(self.dropout_p)

        # Transformer encoder stack
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(self.d_model, self.n_heads, self.d_ff, self.dropout_p)
            for _ in range(self.e_layers)
        ])

        # Flatten + Linear head per official
        head_dim = self.d_model * self.patch_num
        if self.individual:
            self.head = nn.ModuleList([nn.Linear(head_dim, self.pred_len)
                                       for _ in range(self.channels)])
        else:
            self.head = nn.Linear(head_dim, self.pred_len)
        self.head_dropout = nn.Dropout(self.dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, N]
        if self.use_revin:
            x = self.revin(x, "norm")

        # to [B, N, L]
        x = x.permute(0, 2, 1)
        B, N, L = x.shape

        # padding
        if self.padding_patch == "end":
            x = self.padding_patch_layer(x)

        # patching: [B, N, patch_num, patch_len]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # merge B and N for channel-independent processing
        x = x.reshape(B * N, self.patch_num, self.patch_len)

        # patch embedding + pos + dropout
        x = self.W_P(x) + self.W_pos                          # [B*N, patch_num, d_model]
        x = self.input_dropout(x)

        # transformer
        for layer in self.encoder_layers:
            x = layer(x)

        # flatten head
        x = x.reshape(B, N, self.patch_num * self.d_model)    # [B, N, d_model*patch_num]
        if self.individual:
            outs = []
            for i in range(self.channels):
                outs.append(self.head[i](x[:, i, :]))
            x_out = torch.stack(outs, dim=1)                   # [B, N, P]
        else:
            x_out = self.head(x)                               # [B, N, P]
        x_out = self.head_dropout(x_out)

        # to [B, P, N]
        x_out = x_out.permute(0, 2, 1)

        if self.use_revin:
            x_out = self.revin(x_out, "denorm")
        return x_out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def regularization_loss(self) -> torch.Tensor:
        return torch.tensor(0.0)
