"""
01_main_benchmark.py
========================================================================
实验 1：主基准 — PaTRD-Net 在多数据集 × 多 horizon 上的对比结果。

Runs the main benchmark across datasets and prediction horizons.

主实验配置说明
----------------------------------------------------------------
(a) Opt 1  : train_epochs 10 → 50；warmup_epochs = 3；patience = 8；
             lradj = cosine，与 PatchTST / DLinear 的标准设置对齐。
(b) Opt 7  : pred_len ≤ 192 → seq_len = 96 ；
             ≤ 336 → 192 ；≤ 720 → 336（ILI 沿用 36）。
(c) Opt 6  : 在训练前通过 ADF p-value 初始化 α_c。
(d) 数据集 : 默认运行核心数据集；可通过 --include_extended
             追加 Electricity 与 Traffic。

输出
----------------------------------------------------------------
report/main_benchmark_<timestamp>.csv，列：
    dataset, pred_len, seq_len, seed, MAE, MSE, RMSE, MAPE,
    best_epoch, elapsed_sec, total_params
"""
from __future__ import annotations

import argparse
import sys
import os
import traceback
from pathlib import Path

# --- 自动修正路径，确保能找到 models 目录 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

import common as common  # noqa: E402
from common import (
    DATASET_SPECS, CORE_DATASETS, EXTENDED_DATASETS,
    build_config, train_one_config, save_csv, add_common_cli,
    get_report_path, ensure_src_on_path, ensure_model_on_path,
    set_seed, build_preds_path,
)

# 确保能找到模型
from models.PaTRD_Net import Model  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PaTRD-Net Main Benchmark")
    add_common_cli(p)
    # --- 新增参数定义 ---
    p.add_argument("--seq_len", type=int, default=None, help="手动覆盖 seq_len")
    p.add_argument("--batch_size", type=int, default=None, help="手动覆盖 batch_size")
    # ------------------
    p.add_argument("--include_extended", action="store_true",
                   help="加入 Electricity 与 Traffic 大型数据集。")
    p.add_argument("--pred_lens", "--pred_len", type=int, nargs="*", default=None,
                   dest="pred_lens",
                   help="手动指定预测长度（亦可用单数别名 --pred_len；默认使用 DATASET_SPECS）。")
    p.add_argument("--save_preds", action="store_true", default=True,
                   help="save PaTRD-Net predictions to preds_cache/PaTRD-Net_<ds>_P<pl>_s<seed>.npz")
    p.add_argument("--no_save_preds", dest="save_preds", action="store_false")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_src_on_path(args.src_root)
    ensure_model_on_path()

    # --- 统一构造 overrides 字典 ---
    overrides = {"train_epochs": args.epochs}
    if args.seq_len is not None:
        overrides["seq_len"] = args.seq_len
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    # ----------------------------

    # 选择数据集
    if args.include_extended:
        datasets = EXTENDED_DATASETS
    else:
        datasets = [d for d in args.datasets if d in DATASET_SPECS]
    if not datasets:
        datasets = CORE_DATASETS

    # 汇总实验配置
    plan = []
    for ds in datasets:
        enc_in, freq, data_file, default_plens, _ = DATASET_SPECS[ds]
        plens = args.pred_lens or default_plens
        for pl in plens:
            plan.append((ds, pl))

    print(f"[01_main_benchmark] total configs: {len(plan)}")
    print(f"[01_main_benchmark] datasets: {datasets}")
    print(f"[01_main_benchmark] epochs={args.epochs}  seed={args.seed}")

    if args.dry_run:
        for ds, pl in plan:
            # 使用 **overrides
            cfg = build_config(ds, pl, seed=args.seed, **overrides)
            print(f"  - {ds:12s}  L={cfg.seq_len}  P={pl}  "
                  f"bs={cfg.batch_size}  enc_in={cfg.enc_in}")
        return

    rows = []
    for i, (ds, pl) in enumerate(plan, 1):
        tag = f"[{i}/{len(plan)}] {ds} P={pl}"
        print(f"\n{tag}  ------------------------")
        # 使用 **overrides
        cfg = build_config(ds, pl, seed=args.seed, **overrides)
        set_seed(cfg.seed)
        try:
            n_params = Model(cfg).count_parameters()
        except Exception:
            n_params = -1
        sp = build_preds_path("PaTRD-Net", ds, pl, cfg.seed) if args.save_preds else None
        try:
            m, dt, be = train_one_config(cfg, verbose=args.verbose, save_preds_to=sp)
        except Exception as e:
            print(f"  !! failed: {e}")
            traceback.print_exc()
            m = {"MAE": float("nan"), "MSE": float("nan"),
                 "RMSE": float("nan"), "MAPE": float("nan")}
            dt, be = 0.0, 0

        row = {
            "dataset":      ds,
            "pred_len":     pl,
            "seq_len":      cfg.seq_len,
            "seed":         cfg.seed,
            "MAE":          m["MAE"],
            "MSE":          m["MSE"],
            "RMSE":         m["RMSE"],
            "MAPE":         m["MAPE"],
            "best_epoch":   be,
            "elapsed_sec":  round(dt, 2),
            "total_params": n_params,
        }
        rows.append(row)
        print(f"  MAE={m['MAE']:.4f}  MSE={m['MSE']:.4f}  "
              f"best_ep={be}  dt={dt:.1f}s")

        out_path = get_report_path("main_benchmark", args.tag)
        save_csv(rows, out_path)

    # Return a non-zero exit code when any configuration fails.
    def _is_nan(x):
        return isinstance(x, float) and x != x
    n_fail = sum(1 for r in rows if _is_nan(r["MAE"]))
    print(f"\n[01_main_benchmark] done — {len(rows)} rows ({n_fail} failed) → {out_path}")
    if n_fail:
        print(f"[01_main_benchmark] WARNING: {n_fail} config(s) produced NaN — 见上方 traceback。")
        sys.exit(1)


if __name__ == "__main__":
    main()
