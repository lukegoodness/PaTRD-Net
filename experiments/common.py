"""
./experiments/common.py

所有 4 个实验脚本共用的工具：
    * 数据加载
    * Config 构造
    * 训练循环（50 epochs + warmup + cosine schedule）
    * 评估指标（MAE/MSE/RMSE/MAPE）
    * 结果保存

使用方式（从 ./ 目录运行）:
    $ python3 experiments/01_main_benchmark.py

"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler

# -----------------------------------------------------------------
# 目录定位
# -----------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
MODEL_DIR = REPO_ROOT / "src" / "models"
REPORT_DIR = REPO_ROOT / "report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def ensure_model_on_path():
    """将 src/models 挂到 sys.path，以便 `from PaTRD_Net import Model`"""
    code_str = str(MODEL_DIR)
    if code_str not in sys.path:
        sys.path.insert(0, code_str)


def ensure_src_on_path(src_root: str):
    """Put src/ on sys.path so that data_provider / utils / models can be imported."""
    src_abs = str(Path(src_root).resolve())
    if src_abs not in sys.path:
        sys.path.insert(0, src_abs)


# -----------------------------------------------------------------
# 数据集 / horizon 定义
# -----------------------------------------------------------------
DATASET_SPECS = {
    # name        : (enc_in, freq, data_file,                pred_lens,        base_seq_len)
    "ETTh1":        (7,  "h",   "ETTh1.csv",                [96, 192, 336, 720], 96),
    "ETTh2":        (7,  "h",   "ETTh2.csv",                [96, 192, 336, 720], 96),
    "ETTm1":        (7,  "min", "ETTm1.csv",                [96, 192, 336, 720], 96),
    "ETTm2":        (7,  "min", "ETTm2.csv",                [96, 192, 336, 720], 96),
    "Weather":      (21, "h",   "weather.csv",              [96, 192, 336, 720], 96),
    "ILI":          (7,  "h",   "ILI.csv",                  [24, 36, 48, 60],    36),
    # --- 扩展数据集 ---
    "Electricity":  (321,"h",   "electricity.csv",          [96, 192, 336, 720], 96),
    "Traffic":      (862,"h",   "traffic.csv",              [96, 192, 336, 720], 96),
}

CORE_DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "ILI"]
EXTENDED_DATASETS = CORE_DATASETS + ["Electricity", "Traffic"]


# -----------------------------------------------------------------
# Opt 7 : 长 horizon 自适应 seq_len
# -----------------------------------------------------------------
def adaptive_seq_len(dataset: str, pred_len: int, enable: bool = True) -> int:
    """返回按 Opt 7 策略调整的回看长度 L。"""
    if not enable or dataset == "ILI":
        return DATASET_SPECS[dataset][4]
    if pred_len <= 192:
        return 96
    elif pred_len <= 336:
        return 192
    else:
        return 336


# -----------------------------------------------------------------
# Experiment configuration
# -----------------------------------------------------------------
@dataclass
class Config:
    # 基本
    model: str = "PaTRD_Net"
    data: str = "ETTh1"
    data_path: str = "ETTh1.csv"
    root_path: str = "./dataset/"
    features: str = "M"
    target: str = "OT"
    freq: str = "h"
    embed: str = "timeF"

    # 输入 / 预测
    seq_len: int = 96
    label_len: int = 48
    pred_len: int = 96
    enc_in: int = 7
    dec_in: int = 7
    c_out: int = 7

    # 模型超参
    d_model: int = 128
    patch_size: int = 16
    stride: int = 8
    dropout: float = 0.1
    ma_kernel: int = 25

    # 训练
    batch_size: int = 32
    learning_rate: float = 1e-3
    train_epochs: int = 50                  # Opt 1 : 10 → 50
    warmup_epochs: int = 3                  # Opt 1 : warmup
    patience: int = 8                       # Opt 1 : 3 → 8
    lradj: str = "cosine"                   # Opt 1 : cosine schedule
    weight_decay: float = 1e-5

    # 其它
    num_workers: int = 0
    itr: int = 1
    seed: int = 2024
    use_gpu: bool = True
    use_adf_init: bool = False              # Optional ADF initialization is disabled by default.

    # Ablation 开关（全开 = 完整 PaTRD-Net）
    w_adaptive_revin: bool = True
    w_trend: bool = True
    w_residual: bool = True
    w_patching: bool = True
    w_kan: bool = True
    w_fusion: bool = True
    w_patch_mixer: bool = True              # Opt 3
    w_ma_decomp: bool = True                # Opt 4
    w_channel_mixer: bool = True            # Opt 8

    # 附加字段（供 data_provider 参考）
    num_patches: int = 0                    # computed
    scale: bool = True
    percent: int = 100
    train_only: bool = False                # Required by data_factory.


def build_config(dataset: str, pred_len: int, seed: int = 2024,
                 seq_len: Optional[int] = None, **overrides) -> Config:
    enc_in, freq, data_file, _, base_L = DATASET_SPECS[dataset]
    seq_len = seq_len if seq_len is not None else adaptive_seq_len(dataset, pred_len)
    cfg = Config(
        data=dataset,
        data_path=data_file,
        freq=freq,
        seq_len=seq_len,
        label_len=seq_len // 2,
        pred_len=pred_len,
        enc_in=enc_in,
        dec_in=enc_in,
        c_out=enc_in,
        seed=seed,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    # Reduce batch size for memory-intensive configurations unless explicitly overridden.
    if "batch_size" not in overrides:
        if pred_len >= 720:
            cfg.batch_size = 4  # 针对长序列强制使用最小 batch
        elif dataset == "Traffic":
            cfg.batch_size = 8
        elif dataset == "Electricity":
            cfg.batch_size = 8  # 从 16 改为 8
        elif dataset == "ILI":
            cfg.batch_size = 16
    return cfg


# -----------------------------------------------------------------
# 随机种子
# -----------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------
# Opt 1 : Warmup + Cosine 学习率
# -----------------------------------------------------------------
class WarmupCosine(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int,
                 base_lr: float, min_lr_ratio: float = 0.01):
        self.warmup_epochs = max(1, warmup_epochs)
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer)

    def get_lr(self):
        e = self.last_epoch
        if e < self.warmup_epochs:
            ratio = (e + 1) / self.warmup_epochs
            return [self.base_lr * ratio for _ in self.base_lrs]
        # cosine
        progress = (e - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
        progress = min(1.0, progress)
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = self.base_lr * (self.min_lr_ratio + (1 - self.min_lr_ratio) * cos)
        return [lr for _ in self.base_lrs]


# -----------------------------------------------------------------
# Data loaders
# -----------------------------------------------------------------
def get_data_loaders(cfg: Config):
    """Load data through data_provider.data_factory.data_provider.

    The caller must invoke ensure_src_on_path first.
    """
    from data_provider.data_factory import data_provider                     # type: ignore
    train_data, train_loader = data_provider(cfg, "train")
    val_data, val_loader     = data_provider(cfg, "val")
    test_data, test_loader   = data_provider(cfg, "test")
    return (train_data, train_loader), (val_data, val_loader), (test_data, test_loader)


# -----------------------------------------------------------------
# 指标
# -----------------------------------------------------------------
def metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    err = pred - true
    mae = np.mean(np.abs(err))
    mse = np.mean(err ** 2)
    rmse = math.sqrt(mse)
    # MAPE 防零
    denom = np.maximum(np.abs(true), 1e-8)
    mape = np.mean(np.abs(err) / denom)
    return {"MAE": float(mae), "MSE": float(mse),
            "RMSE": float(rmse), "MAPE": float(mape)}


# -----------------------------------------------------------------
# 预测缓存路径构造
# -----------------------------------------------------------------
def build_preds_path(model: str, ds: str, pl: int, seed: int,
                     cache_dir: Optional[Path] = None) -> Path:
    """统一的 preds_cache 文件命名。
    格式: {model}_{ds}_P{pl}_s{seed}.npz   (model 中的 / 替换为 _)
    """
    if cache_dir is None:
        cache_dir = REPORT_DIR / "preds_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_model = model.replace("/", "_").replace("\\", "_")
    return cache_dir / f"{safe_model}_{ds}_P{pl}_s{seed}.npz"


# -----------------------------------------------------------------
# 训练一个 (dataset, pred_len, seed) 配置
# -----------------------------------------------------------------
def train_one_config(cfg: Config, model_kwargs: Optional[dict] = None,
                     verbose: bool = False,
                     model_class=None, model_module: str = "PaTRD_Net",
                     skip_adf: bool = False,
                     save_preds_to: Optional[Path] = None) -> Tuple[dict, float, int]:
    """训练并返回 (test_metrics, elapsed_seconds, best_epoch)。

    model_class / model_module support alternate model implementations.
      - The default model is PaTRD-Net.
      - baseline 训练时传 model_module='DLinear'/'PatchTST'/... 从 src/models 加载
      - skip_adf=True 用于不支持 warm_start_model 的 baseline

    save_preds_to optionally stores normalized predictions and targets
    for subsequent statistical analysis.
    """
    ensure_model_on_path()

    # 加载模型类
    if model_class is None:
        if model_module == "PaTRD_Net":
            from PaTRD_Net import Model, warm_start_model            # type: ignore
            warm_start_fn = warm_start_model
        else:
            # 从 src/models 加载 baseline
            import importlib
            mod = importlib.import_module(f"models.{model_module}")
            Model = getattr(mod, "Model")
            warm_start_fn = None
            skip_adf = True
    else:
        Model = model_class
        warm_start_fn = None
        skip_adf = True

    set_seed(cfg.seed)
    device = torch.device("cuda" if (cfg.use_gpu and torch.cuda.is_available())
                          else "cpu")

    # --- data ---
    # Retain test_data for optional inverse transformation.
    (_, train_loader), (_, val_loader), (test_data, test_loader) = get_data_loaders(cfg)

    # --- model ---
    model = Model(cfg).to(device)

    # ADF 初始化（仅本文模型支持）
    if cfg.use_adf_init and not skip_adf and warm_start_fn is not None:
        try:
            # 预取一个 batch 的 x
            xs = []
            for batch in train_loader:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                xs.append(x)
                if len(xs) >= 4:
                    break
            if xs:
                all_x = torch.cat(xs, dim=0)
                warm_start_fn(model, all_x)
        except Exception as e:
            if verbose:
                print(f"[ADF init skipped] {e}")

    # --- optimizer / scheduler ---
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=cfg.learning_rate,
                                 weight_decay=cfg.weight_decay)
    scheduler = WarmupCosine(optimizer, cfg.warmup_epochs,
                             cfg.train_epochs, cfg.learning_rate)
    criterion = nn.MSELoss()

    best_val = math.inf
    best_epoch = 0
    best_state = None
    patience_cnt = 0
    t0 = time.time()

    for epoch in range(cfg.train_epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            x = batch[0].float().to(device)
            y = batch[1].float().to(device)

            optimizer.zero_grad()
            out = model(x)
            # Forecast targets occupy the last pred_len steps.
            y_tgt = y[:, -cfg.pred_len:, :]
            loss = criterion(out, y_tgt) + model.regularization_loss().to(device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        scheduler.step()
        train_loss /= max(1, n_batches)

        # val
        model.eval()
        val_loss = 0.0
        vb = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].float().to(device)
                y = batch[1].float().to(device)[:, -cfg.pred_len:, :]
                loss = criterion(model(x), y)
                val_loss += loss.item()
                vb += 1
        val_loss /= max(1, vb)

        if verbose:
            print(f"  ep {epoch+1:02d}  train={train_loss:.5f}  val={val_loss:.5f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= cfg.patience:
                break

    # 载回 best
    if best_state is not None:
        model.load_state_dict(best_state)

    # --- test ---
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in test_loader:
            x = batch[0].float().to(device)
            y = batch[1].float().to(device)[:, -cfg.pred_len:, :]
            out = model(x)
            preds.append(out.cpu().numpy())
            trues.append(y.cpu().numpy())
    if not preds:
        return {"MAE": float("nan"), "MSE": float("nan"),
                "RMSE": float("nan"), "MAPE": float("nan")}, 0.0, 0
    pred = np.concatenate(preds, axis=0)
    true = np.concatenate(trues, axis=0)

    # Evaluate in normalized (z-score) space. Predictions and targets are
    # already expressed in the StandardScaler space, so no inverse transform
    # is applied before computing the metrics.

    # Optionally cache normalized prediction tensors for paired tests.
    if save_preds_to is not None:
        try:
            save_preds_to.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(save_preds_to, pred=pred, true=true)
            if verbose:
                print(f"  [save_preds] {save_preds_to}")
        except Exception as e:
            print(f"  [WARN] save_preds failed: {e}")

    elapsed = time.time() - t0
    return metrics(pred, true), elapsed, best_epoch


# -----------------------------------------------------------------
# Train without evaluation for robustness experiments.
# -----------------------------------------------------------------
def train_model_only(cfg: Config, model_module: str = "PaTRD_Net",
                     skip_adf: bool = False, verbose: bool = False):
    """只跑训练并返回训练后的模型，供调用方做自定义评估（如噪声、漂移）。

    返回:
        model       : 训练后已 load 到 best_state 的模型
        test_data   : 用于 inverse_transform
        test_loader : 用于评估
        device      : torch.device
        best_epoch  : int
    """
    ensure_model_on_path()

    if model_module == "PaTRD_Net":
        from PaTRD_Net import Model, warm_start_model  # type: ignore
        warm_start_fn = warm_start_model
    else:
        import importlib
        mod = importlib.import_module(f"models.{model_module}")
        Model = getattr(mod, "Model")
        warm_start_fn = None
        skip_adf = True

    set_seed(cfg.seed)
    device = torch.device("cuda" if (cfg.use_gpu and torch.cuda.is_available())
                          else "cpu")
    (_, train_loader), (_, val_loader), (test_data, test_loader) = get_data_loaders(cfg)
    model = Model(cfg).to(device)

    if cfg.use_adf_init and not skip_adf and warm_start_fn is not None:
        try:
            xs = []
            for batch in train_loader:
                xs.append(batch[0])
                if len(xs) >= 4: break
            if xs:
                warm_start_fn(model, torch.cat(xs, 0))
        except Exception as e:
            if verbose: print(f"  [ADF skipped] {e}")

    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate,
                           weight_decay=cfg.weight_decay)
    sched = WarmupCosine(opt, cfg.warmup_epochs, cfg.train_epochs, cfg.learning_rate)
    crit = nn.MSELoss()

    best_val, best_epoch, best_state, patience = math.inf, 0, None, 0
    for epoch in range(cfg.train_epochs):
        model.train()
        for batch in train_loader:
            x = batch[0].float().to(device); y = batch[1].float().to(device)
            opt.zero_grad()
            loss = crit(model(x), y[:, -cfg.pred_len:, :]) + \
                   model.regularization_loss().to(device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval(); vl, n = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].float().to(device); y = batch[1].float().to(device)
                vl += crit(model(x), y[:, -cfg.pred_len:, :]).item(); n += 1
        vl /= max(1, n)
        if vl < best_val - 1e-6:
            best_val, best_epoch, patience = vl, epoch + 1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= cfg.patience: break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, test_data, test_loader, device, best_epoch


def evaluate_with_input_transform(model, test_loader, test_data, device,
                                  pred_len: int,
                                  input_transform=None,
                                  do_inverse: bool = False):
    """在 test 集上评估，可选对输入施加 transform（如噪声）。

    do_inverse defaults to False for evaluation in normalized space.
    Set it to True only when values in the original scale are required.

    返回:
        metrics dict (MAE/MSE/RMSE/MAPE)
    """
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in test_loader:
            x = batch[0].float().to(device)
            y = batch[1].float().to(device)[:, -pred_len:, :]
            if input_transform is not None:
                x = input_transform(x)
            out = model(x)
            preds.append(out.cpu().numpy())
            trues.append(y.cpu().numpy())
    if not preds:
        return {"MAE": float("nan"), "MSE": float("nan"),
                "RMSE": float("nan"), "MAPE": float("nan")}
    pred = np.concatenate(preds, 0)
    true = np.concatenate(trues, 0)
    if do_inverse:
        try:
            sh = pred.shape
            pred = test_data.inverse_transform(pred.reshape(-1, sh[-1])).reshape(sh)
            true = test_data.inverse_transform(true.reshape(-1, sh[-1])).reshape(sh)
        except Exception:
            pass
    return metrics(pred, true)


# -----------------------------------------------------------------
# Prediction caching for paired statistical tests.
# -----------------------------------------------------------------
def train_and_save_preds(cfg: Config, out_npz: Path, verbose=False) -> dict:
    """除指标外另存 (pred, true) 到 out_npz，供 DM 检验使用。"""
    ensure_model_on_path()
    from PaTRD_Net import Model, warm_start_model                              # type: ignore

    set_seed(cfg.seed)
    device = torch.device("cuda" if (cfg.use_gpu and torch.cuda.is_available())
                          else "cpu")
    # Retain test_data for optional inverse transformation.
    (_, train_loader), (_, val_loader), (test_data, test_loader) = get_data_loaders(cfg)
    model = Model(cfg).to(device)
    if cfg.use_adf_init:
        try:
            xs = [next(iter(train_loader))[0] for _ in range(1)]
            warm_start_model(model, torch.cat(xs, dim=0))
        except Exception:
            pass
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate,
                           weight_decay=cfg.weight_decay)
    sched = WarmupCosine(opt, cfg.warmup_epochs, cfg.train_epochs, cfg.learning_rate)
    crit = nn.MSELoss()

    best_val, best_ep, best_state, patience = math.inf, 0, None, 0
    for epoch in range(cfg.train_epochs):
        model.train()
        for batch in train_loader:
            x = batch[0].float().to(device); y = batch[1].float().to(device)
            opt.zero_grad()
            loss = crit(model(x), y[:, -cfg.pred_len:, :]) + \
                   model.regularization_loss().to(device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        vl = 0.0; n = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].float().to(device); y = batch[1].float().to(device)
                vl += crit(model(x), y[:, -cfg.pred_len:, :]).item(); n += 1
        vl /= max(1, n)
        if vl < best_val - 1e-6:
            best_val, best_ep = vl, epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in test_loader:
            x = batch[0].float().to(device); y = batch[1].float().to(device)[:, -cfg.pred_len:, :]
            preds.append(model(x).cpu().numpy()); trues.append(y.cpu().numpy())
    pred = np.concatenate(preds, axis=0); true = np.concatenate(trues, axis=0)

    # Evaluate and store predictions in normalized space.
    np.savez_compressed(out_npz, pred=pred, true=true)
    return metrics(pred, true)


# -----------------------------------------------------------------
# 通用 CSV 保存
# -----------------------------------------------------------------
def save_csv(rows: List[dict], path: Path) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# -----------------------------------------------------------------
# CLI 辅助
# -----------------------------------------------------------------
def add_common_cli(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--src_root",
                        default=str(REPO_ROOT / "src"),
                        help="Path to src/ (provides data_provider, utils and baseline models).")
    parser.add_argument("--datasets", nargs="+", default=CORE_DATASETS,
                        help=f"Subset of {list(DATASET_SPECS.keys())}.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--dry_run", action="store_true",
                        help="Only build configs, don't train.")
    parser.add_argument("--tag", default="",
                        help="Suffix appended to output filename.")


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def get_report_path(name: str, tag: str = "") -> Path:
    suffix = f"_{tag}" if tag else ""
    return REPORT_DIR / f"{name}_{timestamp()}{suffix}.csv"
