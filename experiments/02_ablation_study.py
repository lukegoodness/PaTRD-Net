"""
02_ablation_study.py
========================================================================
实验 2：消融研究 — 逐一关闭 9 个关键组件，量化其边际贡献。

Prediction caching for paired statistical analysis:
    - 默认运行 3 个随机种子
    - 每次训练后自动保存 (pred, true) 到 preds_cache/
      命名格式: abl_{variant}_{ds}_P{pl}_s{seed}.npz
      缓存数组可用于脚本外的配对统计检验

消融维度
----------------------------------------------------------------
    full                : 完整 PaTRD-Net（所有开关 True）
    w/o_adaptive_revin  : α_c 固定 = 1（退化为标准 RevIN）
    w/o_trend           : 去趋势分支
    w/o_residual        : 去 Patch-KAN 残差分支
    w/o_patching        : patch_size=1 （退化为无 patching）
    w/o_kan             : KAN → 等参数 MLP
    w/o_fusion          : 均匀加权（0.5 / 0.5）
    w/o_patch_mixer     : Opt 3 移除
    w/o_ma_decomp       : Opt 4 移除（退化为单 Linear 趋势）
    w/o_channel_mixer   : Opt 8 移除

总计：10 个变体 × (6 数据集 × 4 horizon = 24 cell) × 3 seeds = 720 次训练

输出：
    report/ablation_study_<timestamp>.csv
        列：variant, dataset, pred_len, seq_len, seed,
             MAE, MSE, RMSE, MAPE, best_epoch, elapsed_sec,
             delta_vs_full_MAE, delta_vs_full_MAE_pct
    report/preds_cache/abl_{variant}_{ds}_P{pl}_s{seed}.npz
"""
from __future__ import annotations

import argparse
import traceback
from typing import Dict, List

import common  # noqa: E402
from common import (
    DATASET_SPECS, CORE_DATASETS,
    build_config, train_one_config, save_csv, add_common_cli,
    get_report_path, ensure_src_on_path, ensure_model_on_path,
    set_seed, build_preds_path,
)


# --------- 消融变体定义 ---------
# 每个变体对应一组 flag override。"full" 为基线。
VARIANTS: Dict[str, Dict[str, bool]] = {
    "full":                   {},
    "w/o_adaptive_revin":     {"w_adaptive_revin": False},
    "w/o_trend":              {"w_trend": False},
    "w/o_residual":           {"w_residual": False},
    "w/o_patching":           {"w_patching": False},
    "w/o_kan":                {"w_kan": False},
    "w/o_fusion":             {"w_fusion": False},
    "w/o_patch_mixer":        {"w_patch_mixer": False},
    "w/o_ma_decomp":          {"w_ma_decomp": False},
    "w/o_channel_mixer":      {"w_channel_mixer": False},
}


# variant 名转 safe label（用于 preds_cache 文件名）
def variant_to_model_name(variant: str) -> str:
    """'full' -> 'PaTRD-Net'; 'w/o_kan' -> 'abl_w_o_kan'"""
    if variant == "full":
        return "PaTRD-Net"
    return "abl_" + variant.replace("/", "_")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PaTRD-Net Ablation Study")
    add_common_cli(p)
    p.add_argument("--variants", nargs="*", default=None,
                   help=f"仅运行指定变体（默认全部）：{list(VARIANTS)}")
    p.add_argument("--seeds", type=int, nargs="+", default=[2024, 2025, 2026],
                   help="种子列表（默认 3 seeds，方便 DM 检验配对）")
    p.add_argument("--save_preds", action="store_true", default=True,
                   help="保存预测张量到 preds_cache/（默认开）")
    p.add_argument("--no_save_preds", dest="save_preds", action="store_false",
                   help="不保存预测（节省磁盘）")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def run_one(variant: str, overrides: Dict[str, bool], ds: str, pl: int,
            seed: int, epochs: int, verbose: bool, save_preds: bool) -> dict:
    cfg = build_config(ds, pl, seed=seed, train_epochs=epochs, **overrides)
    set_seed(cfg.seed)

    save_path = None
    if save_preds:
        model_name = variant_to_model_name(variant)
        save_path = build_preds_path(model_name, ds, pl, seed)

    try:
        m, dt, be = train_one_config(cfg, verbose=verbose, save_preds_to=save_path)
    except Exception as e:
        print(f"  !! failed: {e}")
        traceback.print_exc()
        m = {"MAE": float("nan"), "MSE": float("nan"),
             "RMSE": float("nan"), "MAPE": float("nan")}
        dt, be = 0.0, 0
    return {
        "variant":     variant,
        "dataset":     ds,
        "pred_len":    pl,
        "seq_len":     cfg.seq_len,
        "seed":        cfg.seed,
        "MAE":         m["MAE"],
        "MSE":         m["MSE"],
        "RMSE":        m["RMSE"],
        "MAPE":        m["MAPE"],
        "best_epoch":  be,
        "elapsed_sec": round(dt, 2),
    }


def attach_deltas(rows: List[dict]) -> List[dict]:
    """按 (dataset, pred_len, seed) 计算各变体相对同 seed full 的 ΔMAE / 百分比。"""
    full_map = {}
    for r in rows:
        if r["variant"] == "full":
            full_map[(r["dataset"], r["pred_len"], r["seed"])] = r["MAE"]
    for r in rows:
        base = full_map.get((r["dataset"], r["pred_len"], r["seed"]))
        if base is None or base == 0 or r["MAE"] != r["MAE"]:
            r["delta_vs_full_MAE"] = ""
            r["delta_vs_full_MAE_pct"] = ""
        else:
            d = r["MAE"] - base
            r["delta_vs_full_MAE"] = round(d, 6)
            r["delta_vs_full_MAE_pct"] = round(d / base * 100, 3)
    return rows


def main() -> None:
    args = parse_args()
    ensure_src_on_path(args.src_root)
    ensure_model_on_path()

    variants = args.variants or list(VARIANTS.keys())
    # 确保 full 排最前，便于后续 Δ 比对
    variants = ["full"] + [v for v in variants if v != "full"]
    variants = [v for v in variants if v in VARIANTS]

    datasets = [d for d in args.datasets if d in DATASET_SPECS] or CORE_DATASETS

    # 装配 plan: (variant, ds, pl, seed)
    plan = []
    for v in variants:
        for ds in datasets:
            for pl in DATASET_SPECS[ds][3]:
                for sd in args.seeds:
                    plan.append((v, ds, pl, sd))

    print(f"[02_ablation_study] variants : {variants}")
    print(f"[02_ablation_study] datasets : {datasets}")
    print(f"[02_ablation_study] seeds    : {args.seeds}")
    print(f"[02_ablation_study] save_preds: {args.save_preds}")
    print(f"[02_ablation_study] total configs: {len(plan)} "
          f"(= {len(variants)} variants × {len(datasets)}×4 cells × {len(args.seeds)} seeds)")

    if args.dry_run:
        for v, ds, pl, sd in plan[:30]:  # 仅打印前 30 个
            print(f"  {v:20s}  {ds:10s}  P={pl:3d}  seed={sd}")
        if len(plan) > 30:
            print(f"  ... ({len(plan) - 30} more)")
        return

    rows: List[dict] = []
    out_path = get_report_path("ablation_study", args.tag)
    for i, (v, ds, pl, sd) in enumerate(plan, 1):
        print(f"\n[{i}/{len(plan)}] {v}  {ds}  P={pl}  seed={sd}  ---------------")
        if v == "w/o_patching":
            print("  [warn] w/o_patching 会放大输出头参数；如显存不足请减小 batch_size")
        row = run_one(v, VARIANTS[v], ds, pl, sd, args.epochs,
                      verbose=args.verbose, save_preds=args.save_preds)
        print(f"  MAE={row['MAE']:.4f}  best_ep={row['best_epoch']}  "
              f"dt={row['elapsed_sec']}s")
        rows.append(row)
        # 渐进式写盘
        save_csv(attach_deltas(rows), out_path)

    # 最终汇总
    rows = attach_deltas(rows)
    save_csv(rows, out_path)
    print(f"\n[02_ablation_study] done — {len(rows)} rows → {out_path}")
    if args.save_preds:
        print(f"[02_ablation_study] preds_cache populated under report/preds_cache/")


if __name__ == "__main__":
    main()
