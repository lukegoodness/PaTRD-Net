"""
06_hyperparameter_sensitivity.py
========================================================================
实验 6：超参数敏感性分析（PaTRD-Net）

策略：One-at-a-Time (OAT)
    - 固定其他超参为默认值
    - 每次只扫描一个超参的所有候选值
    - 全面版：每个超参 × 4 horizon × 2 datasets (ETTh1 + ETTm1)

Supports one-at-a-time hyperparameter sensitivity analysis.

扫描的超参（8 个）
----------------------------------------------------------------
    d_model        : {64, 128, 256, 512}              default 128
    patch_size     : {8, 16, 32}                      default 16
    stride         : {4, 8, 16}                       default 8
    ma_kernel      : {5, 11, 25, 35, 45}              default 25
    learning_rate  : {1e-4, 5e-4, 1e-3, 5e-3}         default 1e-3
    dropout        : {0.05, 0.1, 0.2, 0.3}            default 0.1
    train_epochs   : {20, 50, 100}                    default 50
    weight_decay   : {1e-6, 1e-5, 1e-4, 1e-3}         default 1e-5

总训练数：
    (4+3+3+5+4+4+3+4) - 8 (defaults) + 1 (baseline)
    = 30 - 8 + 1 = 23 configs × 4 horizon × 2 datasets
    = 184 训练
输出
----------------------------------------------------------------
report/hyperparam_sensitivity_<timestamp>.csv  全部结果
report/hyperparam_best_<timestamp>.csv         每超参最佳值
report/hyperparam_curves_<timestamp>.png       8 子图敏感度曲线
"""
from __future__ import annotations

import argparse
import traceback
from pathlib import Path
from typing import Dict, List

import common  # noqa: E402
from common import (
    DATASET_SPECS, build_config, train_one_config, save_csv,
    add_common_cli, get_report_path, ensure_src_on_path,
    ensure_model_on_path, set_seed, REPORT_DIR, timestamp,
)


# --------- 超参网格 ---------
# 每个超参的默认值与候选值
HYPERPARAM_GRID = {
    "d_model":       {"default": 128,   "values": [64, 128, 256, 512]},
    "patch_size":    {"default": 16,    "values": [8, 16, 32]},
    "stride":        {"default": 8,     "values": [4, 8, 16]},
    "ma_kernel":     {"default": 25,    "values": [5, 11, 25, 35, 45]},
    "learning_rate": {"default": 1e-3,  "values": [1e-4, 5e-4, 1e-3, 5e-3]},
    "dropout":       {"default": 0.1,   "values": [0.05, 0.1, 0.2, 0.3]},
    "train_epochs":  {"default": 50,    "values": [20, 50, 100]},
    "weight_decay":  {"default": 1e-5,  "values": [1e-6, 1e-5, 1e-4, 1e-3]},
}

# 默认数据集 + horizon 集合
DEFAULT_DATASETS = ["ETTh1", "ETTm1"]
DEFAULT_HORIZONS = [96, 192, 336, 720]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PaTRD-Net Hyperparameter Sensitivity Analysis (OAT)")
    add_common_cli(p)
    p.add_argument("--hp_datasets", nargs="+", default=DEFAULT_DATASETS,
                   help="敏感性分析专用数据集（默认 ETTh1 + ETTm1）")
    p.add_argument("--hp_horizons", type=int, nargs="+", default=DEFAULT_HORIZONS,
                   help="敏感性分析专用 horizon（默认 96/192/336/720）")
    p.add_argument("--params", nargs="*", default=None,
                   help=f"仅扫描指定超参（默认全部）：{list(HYPERPARAM_GRID)}")
    p.add_argument("--plot", action="store_true", default=True,
                   help="跑完后画 sensitivity curves（默认 True）")
    p.add_argument("--no_plot", dest="plot", action="store_false")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def make_plan(params: List[str], datasets: List[str],
              horizons: List[int], seed: int) -> List[dict]:
    """构造 OAT 训练计划。

    每个超参的每个非默认值 × 每个 horizon × 每个 dataset 跑一次。
    再加上 baseline (所有超参都为默认值) × 每个 horizon × dataset。
    """
    plan = []
    # baseline (all defaults)
    for ds in datasets:
        for pl in horizons:
            plan.append({
                "scan_param": "baseline",
                "param_value": "default",
                "dataset": ds,
                "pred_len": pl,
                "seed": seed,
                "overrides": {},  # all defaults
            })
    # OAT scan
    for param in params:
        grid = HYPERPARAM_GRID[param]
        default = grid["default"]
        for v in grid["values"]:
            if v == default:
                continue  # 默认值已在 baseline 里
            for ds in datasets:
                for pl in horizons:
                    plan.append({
                        "scan_param": param,
                        "param_value": v,
                        "dataset": ds,
                        "pred_len": pl,
                        "seed": seed,
                        "overrides": {param: v},
                    })
    return plan


def run_one_config(item: dict, epochs_default: int, verbose: bool) -> dict:
    overrides = dict(item["overrides"])
    # train_epochs 单独处理（既是被扫描的超参，也是 build_config 参数）
    epochs = overrides.pop("train_epochs", epochs_default)

    cfg = build_config(item["dataset"], item["pred_len"],
                       seed=item["seed"], train_epochs=epochs,
                       **overrides)
    set_seed(cfg.seed)
    try:
        m, dt, be = train_one_config(cfg, verbose=verbose)
    except Exception as e:
        print(f"  !! failed: {e}")
        traceback.print_exc()
        m = {"MAE": float("nan"), "MSE": float("nan"),
             "RMSE": float("nan"), "MAPE": float("nan")}
        dt, be = 0.0, 0

    return {
        "scan_param":  item["scan_param"],
        "param_value": item["param_value"],
        "dataset":     item["dataset"],
        "pred_len":    item["pred_len"],
        "seq_len":     cfg.seq_len,
        "seed":        item["seed"],
        "MAE":         m["MAE"],
        "MSE":         m["MSE"],
        "RMSE":        m["RMSE"],
        "MAPE":        m["MAPE"],
        "best_epoch":  be,
        "elapsed_sec": round(dt, 2),
    }


def summarize_best(rows: List[dict]) -> List[dict]:
    """每个超参找出 MAE 最小的取值（按 (ds, pl) 聚合后再取最佳）。"""
    # baseline MAE per (ds, pl)
    base_map = {}
    for r in rows:
        if r["scan_param"] == "baseline" and r["MAE"] == r["MAE"]:
            base_map[(r["dataset"], r["pred_len"])] = r["MAE"]

    summaries = []
    # group by (scan_param, param_value) and compute mean MAE
    from collections import defaultdict
    agg = defaultdict(list)
    for r in rows:
        if r["scan_param"] == "baseline":
            continue
        if r["MAE"] != r["MAE"]:
            continue
        agg[(r["scan_param"], r["param_value"])].append({
            "MAE": r["MAE"],
            "ds": r["dataset"],
            "pl": r["pred_len"],
        })

    # baseline aggregated MAE (across same cells)
    base_rows = [r for r in rows if r["scan_param"] == "baseline" and r["MAE"] == r["MAE"]]
    base_mae = sum(r["MAE"] for r in base_rows) / max(1, len(base_rows))

    # per-param best
    per_param = defaultdict(list)
    for (param, val), records in agg.items():
        mean_mae = sum(r["MAE"] for r in records) / len(records)
        per_param[param].append({
            "param_value": val,
            "mean_MAE": round(mean_mae, 5),
            "n_cells": len(records),
            "delta_vs_default_pct": round((mean_mae - base_mae) / base_mae * 100, 3),
        })

    for param, items in per_param.items():
        # default 的 mean MAE = baseline
        default = HYPERPARAM_GRID[param]["default"]
        items.append({"param_value": f"{default} (default)", "mean_MAE": round(base_mae, 5),
                      "n_cells": len(base_rows), "delta_vs_default_pct": 0.0})
        # sort by MAE ascending
        items.sort(key=lambda x: x["mean_MAE"])
        # best
        summaries.append({
            "hyperparam":     param,
            "default_value":  default,
            "best_value":     items[0]["param_value"],
            "best_MAE":       items[0]["mean_MAE"],
            "best_delta_pct": items[0]["delta_vs_default_pct"],
            "n_candidates":   len(items),
            "all_values_sorted": " < ".join(f"{x['param_value']}({x['mean_MAE']:.4f})"
                                            for x in items),
        })
    return summaries


def plot_sensitivity(rows: List[dict], out_path: Path) -> None:
    """画 8 子图的敏感性曲线。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  [plot] matplotlib not installed; skipping")
        return

    from collections import defaultdict
    agg = defaultdict(list)
    for r in rows:
        if r["scan_param"] == "baseline":
            continue
        if r["MAE"] != r["MAE"]:
            continue
        agg[r["scan_param"]].append((r["param_value"], r["MAE"]))

    base_rows = [r for r in rows if r["scan_param"] == "baseline" and r["MAE"] == r["MAE"]]
    base_mae = sum(r["MAE"] for r in base_rows) / max(1, len(base_rows))

    n = len(agg)
    if n == 0:
        return
    cols = 4
    rows_n = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(4 * cols, 3 * rows_n))
    axes = np.atleast_1d(axes).ravel()

    for ax, (param, pairs) in zip(axes, agg.items()):
        # mean MAE per param_value
        pv = defaultdict(list)
        for val, mae in pairs:
            pv[val].append(mae)
        xs = sorted(pv.keys(), key=lambda x: float(x) if isinstance(x, (int, float))
                    else 0)
        ys = [sum(pv[x]) / len(pv[x]) for x in xs]
        # add default
        default = HYPERPARAM_GRID[param]["default"]
        if default not in pv:
            xs.append(default)
            ys.append(base_mae)
            order = sorted(range(len(xs)), key=lambda i: float(xs[i]))
            xs = [xs[i] for i in order]
            ys = [ys[i] for i in order]

        ax.plot(xs, ys, marker="o", linewidth=1.5)
        ax.axvline(default, color="red", linestyle="--", alpha=0.5,
                   label=f"default={default}")
        ax.set_xlabel(param)
        ax.set_ylabel("MAE (mean)")
        ax.set_title(param)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
        if param in ("learning_rate", "weight_decay"):
            ax.set_xscale("log")

    for ax in axes[n:]:
        ax.axis("off")

    plt.suptitle("PaTRD-Net Hyperparameter Sensitivity (One-at-a-Time)", y=1.0, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    ensure_src_on_path(args.src_root)
    ensure_model_on_path()

    params = args.params or list(HYPERPARAM_GRID.keys())
    params = [p for p in params if p in HYPERPARAM_GRID]
    datasets = [d for d in args.hp_datasets if d in DATASET_SPECS]
    if not datasets:
        print(f"[ERROR] No valid hp_datasets. Available: {list(DATASET_SPECS.keys())}")
        return
    horizons = args.hp_horizons

    plan = make_plan(params, datasets, horizons, args.seed)

    print(f"[06_hp_sensitivity] params : {params}")
    print(f"[06_hp_sensitivity] datasets: {datasets}")
    print(f"[06_hp_sensitivity] horizons: {horizons}")
    print(f"[06_hp_sensitivity] total configs: {len(plan)} "
          f"({sum(1 for x in plan if x['scan_param'] == 'baseline')} baseline "
          f"+ {sum(1 for x in plan if x['scan_param'] != 'baseline')} OAT)")
    print(f"[06_hp_sensitivity] default epochs: {args.epochs}")

    if args.dry_run:
        for item in plan[:20]:
            print(f"  {item['scan_param']:15s}  v={item['param_value']!s:10s}  "
                  f"{item['dataset']:10s}  P={item['pred_len']:3d}")
        if len(plan) > 20:
            print(f"  ... ({len(plan) - 20} more)")
        return

    rows: List[dict] = []
    out_path = REPORT_DIR / f"hyperparam_sensitivity_{timestamp()}{('_' + args.tag) if args.tag else ''}.csv"
    for i, item in enumerate(plan, 1):
        print(f"\n[{i}/{len(plan)}] {item['scan_param']:15s}  "
              f"v={item['param_value']!s:10s}  {item['dataset']}  P={item['pred_len']}")
        row = run_one_config(item, args.epochs, args.verbose)
        print(f"  MAE={row['MAE']:.4f}  best_ep={row['best_epoch']}  "
              f"dt={row['elapsed_sec']}s")
        rows.append(row)
        save_csv(rows, out_path)

    # 汇总最佳值
    summaries = summarize_best(rows)
    best_path = REPORT_DIR / f"hyperparam_best_{timestamp()}{('_' + args.tag) if args.tag else ''}.csv"
    save_csv(summaries, best_path)
    print(f"\n[06_hp_sensitivity] best values → {best_path}")

    # 绘图
    if args.plot:
        png_path = REPORT_DIR / f"hyperparam_curves_{timestamp()}{('_' + args.tag) if args.tag else ''}.png"
        plot_sensitivity(rows, png_path)
        print(f"[06_hp_sensitivity] curves → {png_path}")

    print(f"\n[06_hp_sensitivity] done — {len(rows)} rows → {out_path}")


if __name__ == "__main__":
    main()
