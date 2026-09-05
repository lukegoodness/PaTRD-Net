"""
05_baseline_benchmark.py
========================================================================
Baseline benchmark on the SAME pipeline as PaTRD-Net:
    - Same data loader and StandardScaler preprocessing
    - Same train/val/test split
    - Same evaluation in NORMALIZED (z-score) space (no inverse_transform),
      matching the LTSF literature convention (PatchTST/DLinear/iTransformer)
    - Same train epochs / warmup+cosine schedule / patience
    - Same set of (dataset, pred_len, seed) cells as PaTRD-Net

Models: DLinear, PatchTST, iTransformer, TimeMixer
(all under src/models/)

Output
----------------------------------------------------------------
report/baseline_benchmark_<MODEL>_<timestamp>.csv

Usage
----------------------------------------------------------------
# 单个 baseline
python 05_baseline_benchmark.py --baseline PatchTST --epochs 50 --seed 2024

# 全套 baseline 串行
python 05_baseline_benchmark.py --baseline all --epochs 50 --seed 2024

# 多 seed (建议 3 seeds)
for seed in 2024 2025 2026; do
    python 05_baseline_benchmark.py --baseline all --epochs 50 --seed $seed
done
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import common
from common import (
    DATASET_SPECS, CORE_DATASETS, EXTENDED_DATASETS,
    build_config, train_one_config, save_csv, add_common_cli,
    get_report_path, ensure_src_on_path, ensure_model_on_path,
    set_seed, timestamp, build_preds_path,
)


SUPPORTED_BASELINES = ["DLinear", "PatchTST", "iTransformer", "TimeMixer"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline Benchmark on the PaTRD-Net pipeline")
    add_common_cli(p)
    p.add_argument("--baseline", default="all",
                   help=f"Which baseline to run: {SUPPORTED_BASELINES} or 'all'.")
    p.add_argument("--include_extended", action="store_true",
                   help="加入 Electricity 与 Traffic 大型数据集。")
    p.add_argument("--pred_lens", type=int, nargs="*", default=None,
                   help="手动指定预测长度（默认使用 DATASET_SPECS）。")
    # Save baseline predictions for paired cross-model tests.
    p.add_argument("--save_preds", action="store_true", default=True,
                   help="保存 baseline 预测到 preds_cache/<model>_<ds>_P<pl>_s<seed>.npz（默认开）")
    p.add_argument("--no_save_preds", dest="save_preds", action="store_false")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def run_one_baseline(baseline_name: str, plan, args):
    """Run one baseline across the full (ds, pl) plan."""
    import importlib
    ensure_model_on_path()
    mod = importlib.import_module(f"models.{baseline_name}")
    Model = getattr(mod, "Model")

    print(f"\n{'='*70}\n>>> Baseline: {baseline_name}\n{'='*70}")

    rows = []
    for i, (ds, pl) in enumerate(plan, 1):
        tag = f"[{baseline_name}][{i}/{len(plan)}] {ds} P={pl}"
        print(f"\n{tag}  ------------------------")
        cfg = build_config(ds, pl, seed=args.seed, train_epochs=args.epochs)
        set_seed(cfg.seed)
        try:
            n_params = Model(cfg).count_parameters()
        except Exception:
            n_params = -1
        sp = (build_preds_path(baseline_name, ds, pl, cfg.seed)
              if getattr(args, "save_preds", True) else None)
        try:
            m, dt, be = train_one_config(
                cfg, verbose=args.verbose,
                model_module=baseline_name,
                skip_adf=True,
                save_preds_to=sp,
            )
        except Exception as e:
            print(f"  !! failed: {e}")
            traceback.print_exc()
            m = {"MAE": float("nan"), "MSE": float("nan"),
                 "RMSE": float("nan"), "MAPE": float("nan")}
            dt = 0.0
            be = 0

        row = {
            "model":       baseline_name,
            "dataset":     ds,
            "pred_len":    pl,
            "seq_len":     cfg.seq_len,
            "seed":        args.seed,
            "MAE":         float(m["MAE"]),
            "MSE":         float(m["MSE"]),
            "RMSE":        float(m["RMSE"]),
            "MAPE":        float(m["MAPE"]),
            "best_epoch":  be,
            "elapsed_sec": round(dt, 2),
            "total_params": n_params,
        }
        rows.append(row)
        print(f"  → MAE={m['MAE']:.4f}  MSE={m['MSE']:.4f}  "
              f"best_ep={be}  dt={dt:.1f}s  params={n_params}")

        # 渐进式落盘（避免长跑失败丢数据）
        out = common.REPORT_DIR / f"baseline_benchmark_{baseline_name}_{timestamp()}_seed{args.seed}.csv"
        save_csv(rows, out)

    return rows


def main() -> None:
    args = parse_args()
    ensure_src_on_path(args.src_root)
    ensure_model_on_path()

    # 选择数据集
    if args.include_extended:
        datasets = EXTENDED_DATASETS
    else:
        datasets = [d for d in args.datasets if d in DATASET_SPECS]
    if not datasets:
        datasets = CORE_DATASETS

    # 装配 plan
    plan = []
    for ds in datasets:
        enc_in, freq, data_file, default_plens, _ = DATASET_SPECS[ds]
        plens = args.pred_lens or default_plens
        for pl in plens:
            plan.append((ds, pl))

    # 选择 baseline
    if args.baseline.lower() == "all":
        baselines = SUPPORTED_BASELINES
    else:
        if args.baseline not in SUPPORTED_BASELINES:
            print(f"Unknown baseline '{args.baseline}'. Choose from {SUPPORTED_BASELINES} or 'all'.")
            sys.exit(1)
        baselines = [args.baseline]

    print(f"[baseline_benchmark] datasets : {datasets}")
    print(f"[baseline_benchmark] configs  : {len(plan)} per baseline")
    print(f"[baseline_benchmark] baselines: {baselines}")
    print(f"[baseline_benchmark] epochs={args.epochs}  seed={args.seed}")
    print(f"[baseline_benchmark] total runs: {len(plan) * len(baselines)}")

    if args.dry_run:
        for bl in baselines:
            for ds, pl in plan:
                cfg = build_config(ds, pl, seed=args.seed, train_epochs=args.epochs)
                print(f"  DRY [{bl}] {ds:12s}  L={cfg.seq_len}  P={pl}")
        return

    # 串行跑每个 baseline
    all_rows = []
    for bl in baselines:
        all_rows.extend(run_one_baseline(bl, plan, args))

    # 总输出
    if all_rows:
        out = common.REPORT_DIR / f"baseline_benchmark_ALL_{timestamp()}_seed{args.seed}.csv"
        save_csv(all_rows, out)
        print(f"\n[baseline_benchmark] all-baseline aggregate → {out}")


if __name__ == "__main__":
    main()
