#!/usr/bin/env python3
"""
批量生成所有 policy geometry 图。

对 outputs/ 下每个 {kl}/{run} 目录，生成:
  - pca_reward.png, pca_combo_hit.png, pca_item_freq_pct.png
  - tsne_reward.png, tsne_combo_hit.png, tsne_item_freq_pct.png

Usage:
    # 先生成所有图（含 augment）
    python3 generate_all_plots.py --augment

    # 只生成图（假设已 augment）
    python3 generate_all_plots.py

    # 强制覆盖已有图
    python3 generate_all_plots.py --force

    # 只生成特定 KL 组
    python3 generate_all_plots.py --kl kl005

    # 只生成 PCA 图
    python3 generate_all_plots.py --methods pca
"""

import sys, os, subprocess
from pathlib import Path
from argparse import ArgumentParser

SCRIPT_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = SCRIPT_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures" / "action"

# 所有 (kl, run) 组合 — 与 extract_all_experiments.py 保持一致
ALL_RUNS = [
    # === KL=1.0 ===
    ("kl1", "mix_b0"),
    ("kl1", "mix_b2"),
    ("kl1", "mix_b5"),
    ("kl1", "mix_b8"),
    ("kl1", "mix_b10"),
    ("kl1", "topdown_b0"),
    ("kl1", "topdown_b2"),
    ("kl1", "topdown_b8"),
    ("kl1", "td_b5"),
    ("kl1", "td_b10"),
    # === KL=0.05 ideal_init ===
    ("kl005", "mix_b0"),
    ("kl005", "mix_b2"),
    ("kl005", "mix_b5"),
    ("kl005", "mix_b8"),
    ("kl005", "mix_b10"),
    ("kl005", "topdown_b0"),
    ("kl005", "topdown_b2"),
    ("kl005", "topdown_b5"),
    ("kl005", "topdown_b8"),
    ("kl005", "topdown_b10"),
    # === KL=0.05 mf_init ===
    ("kl005_mf", "mix_b0"),
    ("kl005_mf", "mix_b10"),
    ("kl005_mf", "topdown_b0"),
    ("kl005_mf", "topdown_b2"),
    ("kl005_mf", "topdown_b5"),
]

METHODS = ["pca", "tsne"]
COLOR_BYS = ["reward", "combo_hit", "item_freq_pct"]


PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent

def run_augment():
    """运行 augment_slate_metrics.py"""
    script = SCRIPT_DIR / "action_cloud" / "augment_slate_metrics.py"
    if not script.exists():
        print(f"[SKIP] {script} not found, skipping augment")
        return
    print("=" * 60)
    print("[AUGMENT] Adding combo_hit/item_freq_pct to .npz files...")
    print("=" * 60)
    result = subprocess.run(
        [sys.executable, str(script), "--env_name", "mix_divpen", "--dataset_quality", "b5"],
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode != 0:
        print("[WARN] augment failed, some color modes may not work")
    print()


def generate_plots(args):
    """批量生成所有图"""
    plot_script = SCRIPT_DIR / "action_cloud" / "plot_policy_geometry.py"

    total = 0
    generated = 0
    skipped = 0
    errors = 0

    for kl, run in ALL_RUNS:
        if args.kl and kl != args.kl:
            continue

        data_dir = OUTPUTS_DIR / kl / run
        best_npz = data_dir / "best_geometry.npz"
        final_npz = data_dir / "final_geometry.npz"

        if not best_npz.exists() or not final_npz.exists():
            print(f"[SKIP] {kl}/{run}: missing .npz files")
            continue

        fig_dir = FIGURES_DIR / kl / run
        fig_dir.mkdir(parents=True, exist_ok=True)

        for method in METHODS:
            if args.methods and method not in args.methods:
                continue

            for color_by in COLOR_BYS:
                out_path = fig_dir / f"{method}_{color_by}.png"
                total += 1

                if out_path.exists() and not args.force:
                    print(f"[SKIP] {kl}/{run}/{method}_{color_by}.png (exists)")
                    skipped += 1
                    continue

                print(f"[GEN]  {kl}/{run}/{method}_{color_by}.png ...", end=" ", flush=True)
                result = subprocess.run(
                    [sys.executable, str(plot_script),
                     "--best", str(best_npz),
                     "--final", str(final_npz),
                     "--method", method,
                     "--color_by", color_by,
                     "--out", str(out_path)],
                    capture_output=True, text=True,
                    cwd=str(SCRIPT_DIR.parent.parent.parent.parent),
                )
                if result.returncode == 0:
                    print("OK")
                    generated += 1
                else:
                    print("FAIL")
                    print(f"    stderr: {result.stderr.strip()}")
                    errors += 1

    print()
    print("=" * 60)
    print(f"Done: {generated} generated, {skipped} skipped, {errors} errors (of {total} total)")
    print("=" * 60)


def main():
    parser = ArgumentParser()
    parser.add_argument("--augment", action="store_true",
                        help="Run augment_slate_metrics.py first")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing figures")
    parser.add_argument("--kl", type=str, default="",
                        help="Only process specific KL group (kl005 or kl1)")
    parser.add_argument("--methods", type=str, nargs="+", default=[],
                        help="Only generate specific methods (pca, tsne)")
    args = parser.parse_args()

    if args.augment:
        run_augment()

    generate_plots(args)


if __name__ == "__main__":
    main()
