"""
08_merge_results.py
========================================================================
Merge result files from independent runs into summary tables.

输入
----------------------------------------------------------------
一个目录（--report_dir），里面汇集了独立运行产生的 CSV：
    main_benchmark_*.csv               (PaTRD-Net，来自 01)
    baseline_benchmark_*.csv           (DLinear/PatchTST/iTransformer/TimeMixer，来自 05)

功能
----------------------------------------------------------------
1. 扫描并合并所有 CSV → 统一长表 (model, dataset, pred_len, seed, MAE, MSE, ...)
2. 按 (model, dataset, pred_len) 聚合 3 seeds → mean ± std
3. 生成主表：行 = dataset×pred_len，列 = 各模型 MSE/MAE，标注每行最优
4. 统计本文模型取得最优结果的配置占比
5. 输出 CSV + LaTeX 表格

输出
----------------------------------------------------------------
report_dir/merged_long.csv            合并长表（每个 model×cell×seed 一行）
report_dir/main_table_mean_std.csv    主表 (mean±std)
report_dir/main_table.tex             LaTeX 主表
report_dir/best_rate_summary.txt      最优结果占比统计

Usage
----------------------------------------------------------------
python 08_merge_results.py --report_dir report
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd


CANON_MODELS = ["PaTRD-Net", "DLinear", "PatchTST", "iTransformer", "TimeMixer"]


def load_all_csvs(report_dir: Path) -> pd.DataFrame:
    """扫描 main_benchmark_*.csv 与 baseline_benchmark_*.csv，合并为长表。"""
    frames = []

    # 本文模型主基准 (main_benchmark_*.csv，无 model 列 → 补上)
    for f in sorted(glob.glob(str(report_dir / "main_benchmark_*.csv"))):
        try:
            df = pd.read_csv(f)
            if "model" not in df.columns:
                df["model"] = "PaTRD-Net"
            frames.append(df)
            print(f"  [PaTRD-Net] {os.path.basename(f)}: {len(df)} rows")
        except Exception as e:
            print(f"  [WARN] skip {f}: {e}")

    # baseline (baseline_benchmark_*.csv，含 model 列)
    for f in sorted(glob.glob(str(report_dir / "baseline_benchmark_*.csv"))):
        # 跳过 ALL 聚合文件，避免重复计入
        if "_ALL_" in os.path.basename(f):
            continue
        try:
            df = pd.read_csv(f)
            frames.append(df)
            print(f"  [baseline] {os.path.basename(f)}: {len(df)} rows")
        except Exception as e:
            print(f"  [WARN] skip {f}: {e}")

    if not frames:
        raise SystemExit(f"[ERROR] No CSV found under {report_dir}")

    long = pd.concat(frames, ignore_index=True)
    # 关键列存在性检查
    need = {"model", "dataset", "pred_len", "seed", "MAE", "MSE"}
    miss = need - set(long.columns)
    if miss:
        raise SystemExit(f"[ERROR] merged table missing columns: {miss}")

    # 去重（同一 model×dataset×pred_len×seed 可能被多个 CSV 重复记录，取最后一条）
    long = long.drop_duplicates(
        subset=["model", "dataset", "pred_len", "seed"], keep="last"
    ).reset_index(drop=True)
    return long


def aggregate_mean_std(long: pd.DataFrame) -> pd.DataFrame:
    """按 (model, dataset, pred_len) 聚合 seeds → mean/std。"""
    agg = (long.groupby(["model", "dataset", "pred_len"])
                .agg(MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"),
                     MSE_mean=("MSE", "mean"), MSE_std=("MSE", "std"),
                     n_seed=("seed", "nunique"))
                .reset_index())
    # std 在单 seed 时为 NaN → 填 0
    agg["MAE_std"] = agg["MAE_std"].fillna(0.0)
    agg["MSE_std"] = agg["MSE_std"].fillna(0.0)
    return agg


def build_main_table(agg: pd.DataFrame, metric: str = "MSE") -> pd.DataFrame:
    """生成主表：行=dataset×pred_len，列=各模型 metric_mean，并标注最优。"""
    mean_col = f"{metric}_mean"
    pivot = agg.pivot_table(index=["dataset", "pred_len"],
                            columns="model", values=mean_col)
    # 列顺序按 CANON_MODELS
    cols = [m for m in CANON_MODELS if m in pivot.columns]
    pivot = pivot[cols]
    # 每行最优模型（metric 越小越好）
    pivot["best_model"] = pivot[cols].idxmin(axis=1)
    pivot["is_best"] = (pivot["best_model"] == "PaTRD-Net")
    return pivot.reset_index()


def best_rate_summary(main_tbl: pd.DataFrame, metric: str) -> str:
    """统计本文模型取得最优结果的配置占比。"""
    total = len(main_tbl)
    best_cnt = int(main_tbl["is_best"].sum())
    lines = []
    lines.append("=" * 60)
    lines.append(f"BEST-RESULT SUMMARY (metric = {metric}, lower=better)")
    lines.append("=" * 60)
    lines.append(f"Total cells (dataset × pred_len) : {total}")
    lines.append(f"PaTRD-Net is best in             : {best_cnt}/{total} "
                 f"= {best_cnt/total*100:.1f}%")
    lines.append("")
    # 按数据集分解
    lines.append("Per-dataset best count:")
    for ds, g in main_tbl.groupby("dataset"):
        nb = int(g["is_best"].sum())
        lines.append(f"  {ds:10s}: {nb}/{len(g)}")
    lines.append("")
    # Descriptive category for the observed win rate.
    pct = best_cnt / total * 100
    if pct >= 75:
        verdict = "HIGH WIN RATE"
    elif pct >= 50:
        verdict = "MODERATE WIN RATE"
    else:
        verdict = "LOW WIN RATE"
    lines.append(f"VERDICT: {pct:.0f}% -> {verdict}")
    lines.append("=" * 60)
    return "\n".join(lines)


def to_latex(main_tbl: pd.DataFrame, agg: pd.DataFrame, metric: str) -> str:
    """生成 LaTeX 主表（mean±std，加粗最优）。"""
    mean_col, std_col = f"{metric}_mean", f"{metric}_std"
    cols = [m for m in CANON_MODELS if m in main_tbl.columns]
    # 建立 (ds,pl,model)->(mean,std) 查询
    lut = {(r.dataset, r.pred_len, r.model): (getattr(r, mean_col), getattr(r, std_col))
           for r in agg.itertuples()}

    out = []
    out.append("\\begin{tabular}{ll" + "c" * len(cols) + "}")
    out.append("\\toprule")
    out.append("Dataset & Horizon & " + " & ".join(cols) + " \\\\")
    out.append("\\midrule")
    for r in main_tbl.itertuples():
        ds, pl = r.dataset, r.pred_len
        best = r.best_model
        cells = []
        for m in cols:
            mean, std = lut.get((ds, pl, m), (float("nan"), 0.0))
            s = f"{mean:.3f}$\\pm${std:.3f}"
            if m == best:
                s = "\\textbf{" + s + "}"
            cells.append(s)
        out.append(f"{ds} & {pl} & " + " & ".join(cells) + " \\\\")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser(description="Merge experiment outputs into summary tables")
    p.add_argument("--report_dir", required=True,
                   help="Directory containing locally generated CSV files.")
    p.add_argument("--metric", default="MSE", choices=["MSE", "MAE"])
    args = p.parse_args()

    report_dir = Path(args.report_dir)
    print(f"[08_merge] scanning {report_dir} ...")
    long = load_all_csvs(report_dir)
    print(f"[08_merge] merged long table: {len(long)} rows, "
          f"models={sorted(long['model'].unique())}, "
          f"datasets={sorted(long['dataset'].unique())}")
    long.to_csv(report_dir / "merged_long.csv", index=False)

    agg = aggregate_mean_std(long)
    agg.to_csv(report_dir / "agg_mean_std.csv", index=False)

    main_tbl = build_main_table(agg, metric=args.metric)
    main_tbl.to_csv(report_dir / "main_table_mean_std.csv", index=False)

    tex = to_latex(main_tbl, agg, metric=args.metric)
    (report_dir / "main_table.tex").write_text(tex, encoding="utf-8")

    summary = best_rate_summary(main_tbl, args.metric)
    (report_dir / "best_rate_summary.txt").write_text(summary, encoding="utf-8")
    print()
    print(summary)
    print()
    print(f"[08_merge] outputs written to {report_dir}:")
    print("  merged_long.csv / agg_mean_std.csv / main_table_mean_std.csv")
    print("  main_table.tex / best_rate_summary.txt")
    print()
    print("NEXT: 对汇集后的 preds_cache 跑全局 DM 检验:")


if __name__ == "__main__":
    main()
