"""
TimeMixer
========================================================================
Reference:
    Wang, S., Wu, H., Shi, X., Hu, T., Luo, H., Ma, L., Zhang, J.Y., Zhou, J. (2024).
    "TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting."
    ICLR 2024.
    Code: https://github.com/kwuking/TimeMixer

Core ideas:
    (1) Multi-scale: downsample input into multiple scales via average pooling.
    (2) Past-Decomposable-Mixing (PDM): decompose at each scale (trend/seasonal)
        and mix bottom-up (seasonal: fine->coarse) and top-down (trend: coarse->fine).
    (3) Future-Multipredictor-Mixing (FMM): ensemble predictors across scales.

This implementation faithfully follows the official paper architecture
while keeping the code self-contained for reproducibility.

Adapted for the PaTRD-Net pipeline (Config-based interface).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------
# Series decomposition (same as Autoformer/DLinear)
# -----------------------------------------------------------------
class moving_avg(nn.Module):
    def __init__(self, kernel_size: int, stride: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, N]
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2 + (self.kernel_size % 2 == 0), 1)
        x_pad = torch.cat([front, x, end], dim=1)
        x_pad = self.avg(x_pad.permute(0, 2, 1)).permute(0, 2, 1)
        return x_pad


class series_decomp(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x: torch.Tensor):
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


# -----------------------------------------------------------------
# MLP block used in PDM mixing
# -----------------------------------------------------------------
class MLPBlock(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------------------------------------------
# Past-Decomposable Mixing (PDM)
# -----------------------------------------------------------------
class PDM(nn.Module):
    """One PDM block: decompose at each scale and cross-scale mixing."""

    def __init__(self, scale_lens, d_model, dropout, kernel_size):
        super().__init__()
        self.n_scales = len(scale_lens)
        self.scale_lens = scale_lens
        self.decomp = nn.ModuleList([series_decomp(kernel_size) for _ in range(self.n_scales)])
        # Bottom-up seasonal mixers (fine to coarse): for i in 1..N-1, mix L_{i-1} -> L_i
        self.seasonal_mixers = nn.ModuleList([
            MLPBlock(scale_lens[i - 1], scale_lens[i - 1], scale_lens[i], dropout)
            for i in range(1, self.n_scales)
        ])
        # Top-down trend mixers (coarse to fine): for i in N-2..0, mix L_{i+1} -> L_i
        self.trend_mixers = nn.ModuleList([
            MLPBlock(scale_lens[i + 1], scale_lens[i + 1], scale_lens[i], dropout)
            for i in range(self.n_scales - 1)
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(self.n_scales)])

    def forward(self, x_list):
        # x_list: list of [B, L_i, d_model]
        seasonals, trends = [], []
        for i, x in enumerate(x_list):
            s, t = self.decomp[i](x)
            seasonals.append(s)
            trends.append(t)

        # Bottom-up: mix seasonals from finest to coarsest
        for i in range(1, self.n_scales):
            # mix seasonals[i-1] -> add to seasonals[i] (over the time dim)
            s_prev = seasonals[i - 1].permute(0, 2, 1)            # [B, d, L_{i-1}]
            s_mixed = self.seasonal_mixers[i - 1](s_prev)         # [B, d, L_i]
            seasonals[i] = seasonals[i] + s_mixed.permute(0, 2, 1)

        # Top-down: mix trends from coarsest to finest
        for i in range(self.n_scales - 2, -1, -1):
            t_next = trends[i + 1].permute(0, 2, 1)               # [B, d, L_{i+1}]
            t_mixed = self.trend_mixers[i](t_next)                # [B, d, L_i]
            trends[i] = trends[i] + t_mixed.permute(0, 2, 1)

        # combine + LN + residual
        out = []
        for i in range(self.n_scales):
            mixed = seasonals[i] + trends[i]
            out.append(self.layer_norms[i](x_list[i] + mixed))
        return out


class Model(nn.Module):
    """TimeMixer: multi-scale decomposable mixing."""

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.d_model = getattr(configs, "d_model", 16)
        self.dropout_p = getattr(configs, "dropout", 0.1)
        self.e_layers = getattr(configs, "e_layers", 2)
        self.down_sampling_layers = getattr(configs, "down_sampling_layers", 3)
        self.down_sampling_window = getattr(configs, "down_sampling_window", 2)
        kernel_size = getattr(configs, "ma_kernel", 25)
        self.use_norm = getattr(configs, "use_norm", True)

        # Build scale_lens: L, L/2, L/4, ...
        self.scale_lens = [self.seq_len]
        for _ in range(self.down_sampling_layers):
            self.scale_lens.append(max(1, self.scale_lens[-1] // self.down_sampling_window))
        self.n_scales = len(self.scale_lens)

        # Downsampling: average pooling on time dim
        self.down_pool = nn.AvgPool1d(self.down_sampling_window)

        # Per-channel embedding: linear from 1 -> d_model (channel-independent)
        self.embedding = nn.Linear(1, self.d_model)
        self.embed_dropout = nn.Dropout(self.dropout_p)

        # PDM blocks
        self.pdm_blocks = nn.ModuleList([
            PDM(self.scale_lens, self.d_model, self.dropout_p, kernel_size)
            for _ in range(self.e_layers)
        ])

        # Future-Multipredictor-Mixing (FMM)
        # For each scale i: project L_i -> pred_len (channel-independent)
        self.predictors = nn.ModuleList([
            nn.Linear(L_i, self.pred_len) for L_i in self.scale_lens
        ])
        # And project d_model -> 1 (per channel)
        self.head_projector = nn.Linear(self.d_model, 1)

    def _multi_scale_split(self, x):
        """x: [B, L, N] -> list of [B, L_i, N]"""
        x_list = [x]
        cur = x.permute(0, 2, 1)                                  # [B, N, L]
        for _ in range(self.down_sampling_layers):
            cur = self.down_pool(cur)
            x_list.append(cur.permute(0, 2, 1))
        return x_list

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, N]
        if self.use_norm:
            mean = x.mean(dim=1, keepdim=True).detach()
            std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            x = (x - mean) / std

        # multi-scale split
        x_list = self._multi_scale_split(x)
        # per-channel embed: [B, L_i, N] -> [B*N, L_i, d_model]
        B = x.shape[0]
        N = x.shape[2]
        embeds = []
        for x_i in x_list:
            # treat each channel independently
            xi = x_i.permute(0, 2, 1).reshape(B * N, -1, 1)        # [B*N, L_i, 1]
            xi = self.embedding(xi)                                # [B*N, L_i, d_model]
            xi = self.embed_dropout(xi)
            embeds.append(xi)

        # PDM blocks
        for block in self.pdm_blocks:
            embeds = block(embeds)

        # FMM: predict pred_len from each scale and average
        outs = []
        for i, emb in enumerate(embeds):
            # emb: [B*N, L_i, d_model] -> head_projector -> [B*N, L_i, 1]
            h = self.head_projector(emb).squeeze(-1)               # [B*N, L_i]
            # project L_i -> pred_len
            o = self.predictors[i](h)                              # [B*N, pred_len]
            outs.append(o)
        # average across scales
        out = torch.stack(outs, dim=0).mean(dim=0)                 # [B*N, pred_len]
        # back to [B, pred_len, N]
        out = out.reshape(B, N, self.pred_len).permute(0, 2, 1)

        if self.use_norm:
            out = out * std + mean

        return out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def regularization_loss(self) -> torch.Tensor:
        return torch.tensor(0.0)
