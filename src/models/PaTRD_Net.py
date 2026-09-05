"""
PaTRD-Net — Patch-wise Trend–Residual Decomposition Network.

============================================================
10 项优化对应关系
============================================================
Opt 1 (50 epochs + cosine)    : 由外部 run script 控制（见 experiments/*）
Opt 2 (AdaptiveRevIN 凸组合)  : AdaptiveRevIN 类重写，x_n = α·RevIN(x) + (1-α)·x
Opt 3 (Patch-Mixer)           : PatchMixer 模块，跨 patch 维线性混合
Opt 4 (MA 分解趋势分支)        : SeriesDecomp + 双 Linear 趋势分支
Opt 5 (层级输出投影)           : patch_agg (d_model → d_model/4) + head
Opt 6 (α_c ADF 初始化)        : init_alpha_from_data() 接口，train 前调用
Opt 7 (长 L for 长 horizon)    : 由外部 config 提供 seq_len，模型接受 variable L
Opt 8 (Channel-Mixer)         : 可选通道混合，eye 初始化，默认开启
Opt 9 (融合权重 init=0)       : branch_weight 初始化 0.0 → sigmoid(0)=0.5 中性
Opt 10 (多处 Dropout)         : input_dropout / trend_dropout / fusion_dropout
============================================================
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from models.KAN import KAN
except Exception:
    # allow running this file standalone (src/models as a package)
    from KAN import KAN


# -------------------------------------------------------------
# Opt 2 : AdaptiveRevIN — 输出空间凸组合版
# -------------------------------------------------------------
class AdaptiveRevIN(nn.Module):
    """
    逐通道可学习 α_c ∈ (0, 1) 在 "完全 RevIN" 与 "恒等映射" 之间做 **真正的输出空间凸组合**：

        x_norm  = α_c · (x − μ)/σ  +  (1 − α_c) · x                (归一化)
        y_out   = α_c · (y_norm · σ + μ)  +  (1 − α_c) · y_norm    (反归一化)

    与"统计量线性插值"做法相比，上述定义在端点严格退化为 RevIN / identity，
    中间值具有明确的概率解释：α_c 即 RevIN 路径权重。
    """

    def __init__(self, num_features: int, eps: float = 1e-5, init_logit: float = 5.0):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        # α_c = sigmoid(logit_c); logit=5 → α≈0.993 (近乎 full RevIN)
        self.alpha_logit = nn.Parameter(torch.full((num_features,), float(init_logit)))
        self.affine_weight = nn.Parameter(torch.ones(num_features))
        self.affine_bias = nn.Parameter(torch.zeros(num_features))
        # forward 之间保存的中间量（denorm 用）
        self._saved_mean: Optional[torch.Tensor] = None
        self._saved_std: Optional[torch.Tensor] = None
        self._saved_alpha: Optional[torch.Tensor] = None

    # --------------------------------------------------------
    # Opt 6 : ADF / 方差比驱动的 α_c 预初始化
    # --------------------------------------------------------
    @torch.no_grad()
    def init_alpha_from_data(self, train_x: torch.Tensor) -> None:
        """按通道 ADF p-value 决定初始 α_c。

        p > 0.05  (非平稳) → logit=+5   (α≈0.99，几乎 full RevIN)
        p ≤ 0.05 (平稳)    → logit=-2   (α≈0.12，几乎 identity)

        Parameters
        ----------
        train_x : [N_sample, L, N] 训练集原始输入
        """
        try:
            from statsmodels.tsa.stattools import adfuller
        except ImportError:
            print("[AdaptiveRevIN] statsmodels not installed; skip ADF init.")
            return
        x = train_x.detach().cpu().numpy().reshape(-1, self.num_features)
        N_max_samples = min(5000, x.shape[0])
        logits = []
        for c in range(self.num_features):
            series = x[:N_max_samples, c]
            try:
                _stat, p, *_ = adfuller(series, regression="ct", autolag="AIC")
                logit = 5.0 if p > 0.05 else -2.0
            except Exception:
                logit = 5.0
            logits.append(logit)
        self.alpha_logit.copy_(torch.tensor(logits, dtype=self.alpha_logit.dtype, device=self.alpha_logit.device))

    def _stats(self, x: torch.Tensor):
        mean = x.mean(dim=1, keepdim=True)                              # [B, 1, N]
        std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps)
        return mean, std

    def forward(self, x: torch.Tensor, mode: str,
                disable_adaptive: bool = False) -> torch.Tensor:
        """
        disable_adaptive=True  → 强制 α=1（等价标准 RevIN；用于 w_adaptive_revin 消融）
        """
        if disable_adaptive:
            alpha = torch.ones(1, 1, self.num_features,
                               device=x.device, dtype=x.dtype)
        else:
            alpha = torch.sigmoid(self.alpha_logit).view(1, 1, -1)      # [1,1,N]
        if mode == "norm":
            mean, std = self._stats(x)
            self._saved_mean = mean
            self._saved_std = std
            self._saved_alpha = alpha
            # 输出空间凸组合：x_n = α·(x-μ)/σ + (1-α)·x
            x_revin = (x - mean) / std
            x_n = alpha * x_revin + (1.0 - alpha) * x
            x_n = x_n * self.affine_weight + self.affine_bias
            return x_n
        elif mode == "denorm":
            # 先撤掉 affine
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)
            # 正确代数逆： x = (x_n + α·μ/σ) / (α/σ + 1-α)
            # 端点验证 α=0 → x_n；α=1 → x_n·σ + μ  √
            mu = self._saved_mean
            sigma = self._saved_std
            a = self._saved_alpha
            numer = x + a * mu / sigma
            denom = a / sigma + (1.0 - a)
            return numer / denom
        else:
            raise ValueError(mode)


# -------------------------------------------------------------
# Opt 4 : 移动平均分解
# -------------------------------------------------------------
class SeriesDecomp(nn.Module):
    """DLinear 风格的 MA 分解。  x  =  trend  +  seasonal"""

    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.kernel_size = kernel_size
        assert kernel_size % 2 == 1, "kernel_size 需为奇数"

    def forward(self, x: torch.Tensor):
        # x: [B, N, L]
        pad = (self.kernel_size - 1) // 2
        front = x[..., :1].repeat(1, 1, pad)
        back = x[..., -1:].repeat(1, 1, pad)
        padded = torch.cat([front, x, back], dim=-1)                     # [B, N, L+2*pad]
        trend = F.avg_pool1d(padded, self.kernel_size, stride=1)
        seasonal = x - trend
        return trend, seasonal


# -------------------------------------------------------------
# Opt 4 : 双 Linear 趋势分支
# -------------------------------------------------------------
class DecompTrendBranch(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, kernel_size: int = 25,
                 dropout: float = 0.1):
        super().__init__()
        self.decomp = SeriesDecomp(kernel_size)
        self.trend_linear = nn.Linear(seq_len, pred_len)
        self.seasonal_linear = nn.Linear(seq_len, pred_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, L]
        trend_x, seasonal_x = self.decomp(x)
        trend_out = self.trend_linear(trend_x)
        season_out = self.seasonal_linear(seasonal_x)
        return self.dropout(trend_out + season_out)                      # [B, N, P]


# -------------------------------------------------------------
# Opt 3 : 跨 patch 混合
# -------------------------------------------------------------
class PatchMixer(nn.Module):
    """Patch 维残差混合。  输入 [B, N, P_count, d_model]。

    仅在 P_count 维做 Linear(P→P) + GELU + Linear，参数量 O(P²)，
    对典型 P=11 仅 ~242 参数；对长 L 时 P 最多 40，参数量仍 <4K。
    """

    def __init__(self, num_patches: int, expansion: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(num_patches, num_patches * expansion)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(num_patches * expansion, num_patches)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, P, D] → 在 P 维 MLP → 残差回加
        h = x.transpose(-1, -2)                                           # [B, N, D, P]
        h = self.fc2(self.act(self.fc1(h)))
        h = h.transpose(-1, -2)                                           # [B, N, P, D]
        return x + self.drop(h)


# -------------------------------------------------------------
# Opt 5 : 层级输出头
# -------------------------------------------------------------
class HierarchicalHead(nn.Module):
    """
    朴素做法: Linear(num_patches * d_model → pred_len)  参数 ~33.8K
    本版本: Linear(d → d/4)  + Linear(num_patches * d/4 → pred_len)  参数 ~12.6K
    """

    def __init__(self, num_patches: int, d_model: int, pred_len: int,
                 reduce_ratio: int = 4):
        super().__init__()
        self.d_reduced = d_model // reduce_ratio
        self.patch_agg = nn.Linear(d_model, self.d_reduced)
        self.head = nn.Linear(num_patches * self.d_reduced, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, P, D]
        B, N, P, D = x.shape
        h = self.patch_agg(x)                                             # [B, N, P, D/4]
        h = h.reshape(B, N, P * self.d_reduced)
        return self.head(h)                                               # [B, N, pred_len]


# -------------------------------------------------------------
# Opt 8 : 轻量通道混合
# -------------------------------------------------------------
class ChannelMixer(nn.Module):
    """Identity-initialized Linear(N, N)，配可学习门控 g。

    forward 输出 = x + g · (W x − x)，g 初始 0 → 初始训练完全保留 CI baseline。
    """

    def __init__(self, num_channels: int):
        super().__init__()
        self.linear = nn.Linear(num_channels, num_channels)
        nn.init.eye_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.gate = nn.Parameter(torch.tensor(0.0))                       # g = 0 at init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, pred_len, N]
        return x + torch.sigmoid(self.gate) * (self.linear(x) - x)


# -------------------------------------------------------------
# _PatchEmbedding
# -------------------------------------------------------------
class _PatchEmbedding(nn.Module):
    def __init__(self, seq_len: int, patch_size: int, d_model: int, stride: int):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.num_patches = (seq_len - patch_size) // stride + 1
        self.proj = nn.Linear(patch_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, L] → [B, N, P_count, d_model]
        patches = x.unfold(dimension=2, size=self.patch_size, step=self.stride)
        return self.proj(patches)


# =============================================================
# 主模型
# =============================================================
class Model(nn.Module):
    """PaTRD-Net

    核心前向流：
        x → AdaptiveRevIN.norm
          → permute [B,N,L]
          → { DecompTrendBranch (Opt 4) ; (Patch-KAN + PatchMixer (Opt 3) + HierHead (Opt 5)) }
          → Fusion(sigmoid-gated, init=0)
          → permute [B,P,N]
          → ChannelMixer (Opt 8)
          → AdaptiveRevIN.denorm
    """

    # 消融开关 — 由 experiments/02_ablation_study.py 通过 configs 设置
    ABLATION_FLAGS = (
        "w_adaptive_revin",         # False → 退化为标准 RevIN
        "w_trend",                  # False → trend_out = 0
        "w_residual",               # False → residual_out = 0
        "w_patching",               # False → patch=1 退化为无 patching
        "w_kan",                    # False → KAN 替为同参数 MLP
        "w_fusion",                 # False → 均匀加法
        "w_patch_mixer",            # Opt 3 单独消融
        "w_ma_decomp",              # Opt 4 单独消融
        "w_channel_mixer",          # Opt 8 单独消融
    )

    def __init__(self, configs):
        super().__init__()
        # 基本超参
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = getattr(configs, "d_model", 128)
        self.patch_size = getattr(configs, "patch_size", 16)
        self.stride = getattr(configs, "stride", 8)
        self.dropout_rate = getattr(configs, "dropout", 0.1)
        self.ma_kernel = getattr(configs, "ma_kernel", 25)

        # 消融开关（默认全开）
        for flag in self.ABLATION_FLAGS:
            setattr(self, flag, bool(getattr(configs, flag, True)))

        # --------- Opt 2 : AdaptiveRevIN ---------
        self.revin = AdaptiveRevIN(self.enc_in)

        # --------- Opt 10 : Input dropout ---------
        self.input_dropout = nn.Dropout(self.dropout_rate * 0.5)         # 0.05

        # --------- Opt 4 : MA-decomp 趋势分支 ---------
        if self.w_ma_decomp:
            self.trend_branch = DecompTrendBranch(
                self.seq_len, self.pred_len,
                kernel_size=self.ma_kernel,
                dropout=self.dropout_rate,
            )
        else:
            # 消融：退化为单 Linear 趋势
            self.trend_branch = nn.Sequential(
                nn.Linear(self.seq_len, self.pred_len),
                nn.Dropout(self.dropout_rate),
            )

        # --------- Patch 分支 ---------
        if self.w_patching:
            self.patch_embed = _PatchEmbedding(
                self.seq_len, self.patch_size, self.d_model, self.stride
            )
            self.num_patches = self.patch_embed.num_patches
        else:
            # 无 patching：等价 patch_size=1, stride=1（消融对照）
            self.patch_embed = _PatchEmbedding(
                self.seq_len, 1, self.d_model, 1
            )
            self.num_patches = self.patch_embed.num_patches

        # --------- KAN or MLP (消融) ---------
        if self.w_kan:
            self.mixer_core = KAN(
                layers_hidden=[self.d_model, self.d_model * 2, self.d_model],
                grid_size=5,
                spline_order=3,
                scale_noise=0.1, scale_base=1.0, scale_spline=1.0,
                base_activation=torch.nn.SiLU,
                grid_eps=0.02, grid_range=[-1, 1],
                regularize_activation=1.0, regularize_entropy=1.0,
                update_grid=False,
            )
        else:
            # 等参数 MLP baseline
            self.mixer_core = nn.Sequential(
                nn.Linear(self.d_model, self.d_model * 2),
                nn.SiLU(),
                nn.Linear(self.d_model * 2, self.d_model),
            )

        # --------- Opt 3 : Patch Mixer ---------
        if self.w_patch_mixer:
            self.patch_mixer = PatchMixer(self.num_patches,
                                          expansion=2,
                                          dropout=self.dropout_rate)
        else:
            self.patch_mixer = nn.Identity()

        self.norm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(self.dropout_rate)

        # --------- Opt 5 : 层级输出头 ---------
        self.residual_head = HierarchicalHead(
            self.num_patches, self.d_model, self.pred_len, reduce_ratio=4
        )

        # --------- Opt 9 : Fusion 权重初始化 0.0 ---------
        if self.w_fusion:
            self.branch_weight = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer("branch_weight_fixed", torch.tensor(0.0))

        self.fusion_dropout = nn.Dropout(self.dropout_rate)

        # --------- Opt 8 : Channel Mixer ---------
        if self.w_channel_mixer:
            self.channel_mixer = ChannelMixer(self.enc_in)
        else:
            self.channel_mixer = nn.Identity()

    # ---------- 前向 ----------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, N]
        # Opt 2 / 10 : AdaptiveRevIN + input dropout
        # w_adaptive_revin = False 时，强制 α = 1 → 退化为标准 RevIN
        x = self.revin(x, "norm", disable_adaptive=not self.w_adaptive_revin)
        x = self.input_dropout(x)

        # [B, L, N] → [B, N, L]
        x = x.permute(0, 2, 1)

        # ---------- 趋势分支 ----------
        if self.w_trend:
            trend_out = self.trend_branch(x)                              # [B, N, P]
        else:
            trend_out = torch.zeros(x.size(0), x.size(1), self.pred_len,
                                    device=x.device, dtype=x.dtype)

        # ---------- 残差（Patch-KAN）分支 ----------
        if self.w_residual:
            patches = self.patch_embed(x)                                 # [B, N, Pc, D]
            patches = self.mixer_core(patches)                            # Patch-wise KAN/MLP
            patches = self.patch_mixer(patches)                           # Opt 3 cross-patch
            patches = self.norm(patches)
            patches = self.dropout(patches)
            residual_out = self.residual_head(patches)                    # [B, N, P]
        else:
            residual_out = torch.zeros_like(trend_out)

        # ---------- 融合 ----------
        if self.w_fusion:
            w = torch.sigmoid(self.branch_weight)                         # Opt 9 init 0 → 0.5
            out = w * trend_out + (1.0 - w) * residual_out
        else:
            out = 0.5 * trend_out + 0.5 * residual_out
        out = self.fusion_dropout(out)

        # [B, N, P] → [B, P, N]
        out = out.permute(0, 2, 1)

        # ---------- Opt 8 : Channel Mixer ----------
        out = self.channel_mixer(out)

        # ---------- Opt 2 : AdaptiveRevIN denorm ----------
        out = self.revin(out, "denorm", disable_adaptive=not self.w_adaptive_revin)
        return out

    # ---------- KAN 正则 ----------
    def regularization_loss(self) -> torch.Tensor:
        try:
            if self.w_kan and hasattr(self.mixer_core, "regularization_loss"):
                reg = self.mixer_core.regularization_loss()
                if isinstance(reg, torch.Tensor) and not torch.isnan(reg):
                    return reg * 0.001
        except Exception:
            pass
        return torch.tensor(0.0)

    # ---------- 参数计数 ----------
    def count_parameters(self, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters()
                   if (p.requires_grad or not trainable_only))


# -------------------------------------------------------------
# 辅助：用于 ADF 初始化的简易训练前调用接口
# -------------------------------------------------------------
def warm_start_model(model: Model, train_x: torch.Tensor) -> Model:
    """训练开始前调用一次，按 ADF 检验初始化 alpha_logit。"""
    model.revin.init_alpha_from_data(train_x)
    return model
