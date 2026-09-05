"""
DLinear
========================================================================
Reference:
    Zeng, A., Chen, M., Zhang, L., & Xu, Q. (2023).
    "Are Transformers Effective for Time Series Forecasting?"
    AAAI 2023.
    Code: https://github.com/cure-lab/LTSF-Linear

A simple but strong linear baseline that:
    (1) Decomposes the input series into trend (moving average) and
        seasonal (residual) components.
    (2) Applies one independent Linear layer to each component.
    (3) Sums the two branches as the final prediction.

Adapted for the PaTRD-Net pipeline (Config-based interface).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class moving_avg(nn.Module):
    """Moving average kernel for trend extraction (DLinear §3.2)."""
    def __init__(self, kernel_size: int, stride: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, N]
        # pad on the front and back so output length equals input length
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2 + (self.kernel_size % 2 == 0), 1)
        x_pad = torch.cat([front, x, end], dim=1)
        x_pad = self.avg(x_pad.permute(0, 2, 1)).permute(0, 2, 1)
        return x_pad


class series_decomp(nn.Module):
    """Series decomposition: trend + residual."""
    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x: torch.Tensor):
        trend = self.moving_avg(x)
        residual = x - trend
        return residual, trend


class Model(nn.Module):
    """DLinear with channel-independent linear projection."""

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in

        # DLinear-Individual: per-channel linear (论文中也叫 DLinear-I)
        # 这里采用论文主推的 "shared" 版本：所有通道共享 Linear
        # 若需要 Individual 版本，请把 self.individual = True 并改成 ModuleList
        self.individual = getattr(configs, "individual", False)
        kernel_size = getattr(configs, "ma_kernel", 25)
        self.decomposition = series_decomp(kernel_size)

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            for _ in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, N]
        seasonal_init, trend_init = self.decomposition(x)
        # to [B, N, L]
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)

        if self.individual:
            seasonal_output = torch.zeros([x.size(0), self.channels, self.pred_len],
                                          dtype=seasonal_init.dtype, device=x.device)
            trend_output = torch.zeros([x.size(0), self.channels, self.pred_len],
                                       dtype=trend_init.dtype, device=x.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x_out = seasonal_output + trend_output
        # back to [B, P, N]
        return x_out.permute(0, 2, 1)

    # -------------------------------------------------------------
    # compatibility interface
    # -------------------------------------------------------------
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def regularization_loss(self) -> torch.Tensor:
        # DLinear 不需要额外正则项
        return torch.tensor(0.0)
