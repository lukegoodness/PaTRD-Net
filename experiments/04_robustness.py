"""
04_robustness.py
========================================================================
实验 4：鲁棒性分析

三个子实验
----------------------------------------------------------------
    (a) 多种子稳定性
        - 5 个 seed × 每个 (ds, pl)
        - 计算 MAE/MSE 的 mean ± std + CV
        - 目标：CV < 5% 在 ≥ 88% 配置上达标

    (b) 噪声鲁棒性 (改进：复用 train_one_config 的训练逻辑 + inverse_transform)
        - 训练阶段不加噪
        - 仅在 test 阶段对输入注入高斯白噪声，SNR ∈ {10, 20, 30, 40} dB
        - 报告每个 SNR 下的 MAE / MSE 与干净基线的退化率

    (c) 分布漂移 (改进：真正实现时间窗对比，不再 placeholder)
        - 把 test 集按时间分前/后两半
        - 同一模型在两个时间窗上的指标差异即为漂移敏感度
        - 数据集本身已按时间顺序划分，这给出一个保守但真实的漂移信号

对应论文：Robustness Analysis 章节

可选：通过 --include_baselines 让 4 个 baseline 也跑噪声/漂移，便于对比本文模型是否更鲁棒。

输出
----------------------------------------------------------------
report/robust_seed_<timestamp>.csv     多种子统计量
report/robust_noise_<timestamp>.csv    各 SNR 下的退化率
report/robust_shift_<timestamp>.csv    分布漂移（time-window 对比）
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

import common  # noqa: E402
from common import (
    DATASET_SPECS, CORE_DATASETS,
    build_config, train_one_config, save_csv, add_common_cli,
    get_report_path, ensure_src_on_path, ensure_model_on_path,
    set_seed, metrics, train_model_only, evaluate_with_input_transform,
)


SUPPORTED_MODELS = ["PaTRD_Net", "DLinear", "PatchTST", "iTransformer", "TimeMixer"]


# ----------------------------------------------------------------
# (a) 多种子稳定性
# ----------------------------------------------------------------
def multi_seed(args: argparse.Namespace,
               model_module: str = "PaTRD_Net") -> List[dict]:
    seeds = args.seeds
    datasets = [d for d in args.datasets if d in DATASET_SPECS] or CORE_DATASETS

    rows = []
    for ds in datasets:
        for pl in DATASET_SPECS[ds][3]:
            per_seed = []
            for sd in seeds:
                cfg = build_config(ds, pl, seed=sd, train_epochs=args.epochs)
                set_seed(cfg.seed)
                m, dt, be = train_one_config(cfg, verbose=args.verbose,
                                             model_module=model_module)
                per_seed.append(m)
                print(f"  [{model_module}][seed={sd}] {ds}/P{pl}  MAE={m['MAE']:.4f}  best_ep={be}")

            maes = [m["MAE"] for m in per_seed]
            mses = [m["MSE"] for m in per_seed]
            rows.append({
                "model":    model_module if model_module != "PaTRD_Net" else "PaTRD-Net",
                "dataset":  ds,
                "pred_len": pl,
                "n_seed":   len(seeds),
                "seeds":    ",".join(map(str, seeds)),
                "MAE_mean": round(float(np.mean(maes)), 5),
                "MAE_std":  round(float(np.std(maes, ddof=1)) if len(seeds) > 1 else 0.0, 5),
                "MSE_mean": round(float(np.mean(mses)), 5),
                "MSE_std":  round(float(np.std(mses, ddof=1)) if len(seeds) > 1 else 0.0, 5),
                "MAE_CV":   round(float(np.std(maes, ddof=1) / np.mean(maes)) if len(seeds) > 1 and np.mean(maes) > 0 else 0.0, 5),
                "MSE_CV":   round(float(np.std(mses, ddof=1) / np.mean(mses)) if len(seeds) > 1 and np.mean(mses) > 0 else 0.0, 5),
            })
    return rows


# ----------------------------------------------------------------
# (b) 噪声鲁棒性 (改进版：复用 train_model_only)
# ----------------------------------------------------------------
def make_noise_transform(snr_db: float, base_seed: int):
    """构造一个 transform 函数：对输入注入高斯白噪声。
    noise_power = signal_power × 10^(-SNR/10)
    """
    def transform(x: torch.Tensor) -> torch.Tensor:
        # 用 base_seed 保证不同 SNR 下噪声样本一致
        g = torch.Generator(device=x.device)
        g.manual_seed(base_seed)
        sig_p = x.pow(2).mean()
        noise_p = sig_p * (10.0 ** (-snr_db / 10.0))
        noise = torch.randn(x.shape, generator=g, device=x.device, dtype=x.dtype)
        return x + noise * noise_p.sqrt()
    return transform


def noise_test(args: argparse.Namespace,
               model_module: str = "PaTRD_Net") -> List[dict]:
    datasets = [d for d in args.datasets if d in DATASET_SPECS] or CORE_DATASETS
    snr_list = args.snrs
    rows: List[dict] = []

    for ds in datasets:
        for pl in DATASET_SPECS[ds][3]:
            cfg = build_config(ds, pl, seed=args.seed, train_epochs=args.epochs)
            set_seed(cfg.seed)
            print(f"\n  [{model_module}] training on {ds}/P{pl} (clean)...")
            try:
                model, test_data, test_loader, device, _ = train_model_only(
                    cfg, model_module=model_module, verbose=args.verbose)
            except Exception as e:
                print(f"    !! training failed: {e}")
                continue

            # 干净 baseline（归一化空间评估，与文献口径一致）
            m_clean = evaluate_with_input_transform(
                model, test_loader, test_data, device, cfg.pred_len,
                input_transform=None, do_inverse=False)

            # 每个 SNR
            snr_records = {}
            for snr in snr_list:
                tr = make_noise_transform(snr, base_seed=cfg.seed)
                m_n = evaluate_with_input_transform(
                    model, test_loader, test_data, device, cfg.pred_len,
                    input_transform=tr, do_inverse=False)
                snr_records[snr] = m_n

            row = {
                "model":     model_module if model_module != "PaTRD_Net" else "PaTRD-Net",
                "dataset":   ds,
                "pred_len":  pl,
                "MAE_clean": round(m_clean["MAE"], 5),
                "MSE_clean": round(m_clean["MSE"], 5),
            }
            for snr in snr_list:
                r = snr_records[snr]
                row[f"MAE_snr{int(snr)}"] = round(r["MAE"], 5)
                row[f"MSE_snr{int(snr)}"] = round(r["MSE"], 5)
                if m_clean["MAE"] > 0:
                    row[f"MAE_deg%_snr{int(snr)}"] = round(
                        (r["MAE"] - m_clean["MAE"]) / m_clean["MAE"] * 100, 3)
                else:
                    row[f"MAE_deg%_snr{int(snr)}"] = ""
            rows.append(row)
            print(f"  {ds}/P{pl}  clean MAE={m_clean['MAE']:.4f}  " +
                  "  ".join(f"SNR{int(snr)}={snr_records[snr]['MAE']:.4f}"
                           for snr in snr_list))
    return rows


# ----------------------------------------------------------------
# (c) 分布漂移：时间窗对比 (改进版：真正实现)
# ----------------------------------------------------------------
def evaluate_on_window(model, test_loader, test_data, device,
                       pred_len: int, window: str = "first_half"):
    """在 test 集的前半或后半评估。"""
    model.eval()
    preds, trues = [], []
    # 收集所有 batch
    all_batches = []
    with torch.no_grad():
        for batch in test_loader:
            all_batches.append(batch)
    n_batches = len(all_batches)
    if window == "first_half":
        sel = all_batches[: n_batches // 2]
    elif window == "second_half":
        sel = all_batches[n_batches // 2 :]
    else:
        sel = all_batches

    with torch.no_grad():
        for batch in sel:
            x = batch[0].float().to(device)
            y = batch[1].float().to(device)[:, -pred_len:, :]
            preds.append(model(x).cpu().numpy())
            trues.append(y.cpu().numpy())
    if not preds:
        return {"MAE": float("nan"), "MSE": float("nan"),
                "RMSE": float("nan"), "MAPE": float("nan")}
    pred = np.concatenate(preds, 0); true = np.concatenate(trues, 0)
    # Evaluate in normalized space without inverse transformation.
    return metrics(pred, true)


def shift_test(args: argparse.Namespace,
               model_module: str = "PaTRD_Net") -> List[dict]:
    """分布漂移测试：把 test 集分前/后两半，对比模型在不同时间窗的退化。

    数据集本身已按时间顺序划分 (train 在前，test 在后)，
    我们进一步把 test 一分为二，模拟「短期未来」vs「长期未来」的分布差异。
    """
    datasets = (args.shift_datasets
                if args.shift_datasets
                else [d for d in args.datasets if d in DATASET_SPECS])
    if not datasets:
        datasets = ["ETTh1", "ETTm1", "Weather"]

    rows: List[dict] = []
    for ds in datasets:
        if ds not in DATASET_SPECS:
            continue
        for pl in [96]:  # 漂移只在最短 horizon 上测，节省时间
            cfg = build_config(ds, pl, seed=args.seed, train_epochs=args.epochs)
            set_seed(cfg.seed)
            print(f"\n  [{model_module}] training on {ds}/P{pl}...")
            try:
                model, test_data, test_loader, device, _ = train_model_only(
                    cfg, model_module=model_module, verbose=args.verbose)
            except Exception as e:
                print(f"    !! training failed: {e}")
                rows.append({"model": model_module, "dataset": ds, "pred_len": pl,
                             "status": f"failed: {e}"})
                continue

            m_full = evaluate_on_window(model, test_loader, test_data, device,
                                        cfg.pred_len, "full")
            m_first = evaluate_on_window(model, test_loader, test_data, device,
                                         cfg.pred_len, "first_half")
            m_second = evaluate_on_window(model, test_loader, test_data, device,
                                          cfg.pred_len, "second_half")

            shift_ratio = (
                (m_second["MAE"] - m_first["MAE"]) / m_first["MAE"]
                if m_first["MAE"] > 0 else 0.0
            )

            rows.append({
                "model":               model_module if model_module != "PaTRD_Net" else "PaTRD-Net",
                "dataset":             ds,
                "pred_len":            pl,
                "MAE_full":            round(m_full["MAE"], 5),
                "MAE_first_half":      round(m_first["MAE"], 5),
                "MAE_second_half":     round(m_second["MAE"], 5),
                "MSE_full":            round(m_full["MSE"], 5),
                "MSE_first_half":      round(m_first["MSE"], 5),
                "MSE_second_half":     round(m_second["MSE"], 5),
                "shift_drift_pct":     round(shift_ratio * 100, 3),
                "interpretation":      "stable" if abs(shift_ratio) < 0.1 else "drift",
            })
            print(f"  {ds}/P{pl}  full={m_full['MAE']:.4f}  "
                  f"first={m_first['MAE']:.4f}  second={m_second['MAE']:.4f}  "
                  f"drift={shift_ratio*100:+.2f}%")
    return rows


# ----------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PaTRD-Net Robustness Analysis (revised)")
    add_common_cli(p)
    p.add_argument("--mode", choices=["seed", "noise", "shift", "all"],
                   default="all")
    p.add_argument("--seeds", type=int, nargs="+",
                   default=[2024, 2025, 2026, 2027, 2028])
    p.add_argument("--snrs", type=float, nargs="+",
                   default=[10.0, 20.0, 30.0, 40.0])
    p.add_argument("--shift_datasets", nargs="*", default=None)
    p.add_argument("--include_baselines", action="store_true",
                   help="让 4 个 baseline 也跑噪声/漂移测试（不含 multi_seed），便于公平对比本文模型")
    p.add_argument("--baselines", nargs="+",
                   default=["DLinear", "PatchTST", "iTransformer", "TimeMixer"])
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_src_on_path(args.src_root)
    ensure_model_on_path()

    # 模型列表
    models = ["PaTRD_Net"]  # 本文模型总是运行
    if args.include_baselines:
        models += args.baselines

    if args.dry_run:
        datasets = [d for d in args.datasets if d in DATASET_SPECS] or CORE_DATASETS
        print(f"\n[DRY RUN] mode={args.mode}")
        print(f"  models    : {models}")
        print(f"  datasets  : {datasets}")
        print(f"  seeds     : {args.seeds}")
        print(f"  snrs      : {args.snrs}")
        n_seed = sum(len(args.seeds) * len(DATASET_SPECS[d][3]) for d in datasets) if args.mode in ("seed","all") else 0
        n_noise = sum(len(DATASET_SPECS[d][3]) for d in datasets) * len(models) if args.mode in ("noise","all") else 0
        n_shift = len(datasets) * len(models) if args.mode in ("shift","all") else 0
        print(f"  trainings : seed={n_seed}  noise={n_noise}  shift={n_shift}  TOTAL={n_seed+n_noise+n_shift}")
        return

    if args.mode in ("seed", "all"):
        print("\n===== (a) Multi-seed stability (PaTRD-Net only) =====")
        r1 = multi_seed(args, model_module="PaTRD_Net")
        p1 = get_report_path("robust_seed", args.tag)
        save_csv(r1, p1)
        print(f"  → {p1}")

    if args.mode in ("noise", "all"):
        print(f"\n===== (b) Noise injection (models: {models}) =====")
        all_rows = []
        for m in models:
            print(f"\n  --- model: {m} ---")
            all_rows.extend(noise_test(args, model_module=m))
        p2 = get_report_path("robust_noise", args.tag)
        save_csv(all_rows, p2)
        print(f"  → {p2}")

    if args.mode in ("shift", "all"):
        print(f"\n===== (c) Distribution shift (models: {models}) =====")
        all_rows = []
        for m in models:
            print(f"\n  --- model: {m} ---")
            all_rows.extend(shift_test(args, model_module=m))
        p3 = get_report_path("robust_shift", args.tag)
        save_csv(all_rows, p3)
        print(f"  → {p3}")


if __name__ == "__main__":
    main()
