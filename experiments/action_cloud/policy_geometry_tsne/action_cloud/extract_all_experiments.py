#!/usr/bin/env python3
"""
批量提取所有实验的 policy geometry 数据，然后 augment + 生成所有图。

Experiment → GeMS 映射：
  beta_ablation_repreduce  → KL=1.0, ideal_init
  kl005_ideal_init         → KL=0.05, ideal_init
  kl005_mf_init            → KL=0.05, mf_init

Usage:
    /data/liyuefeng/miniconda3/envs/gems/bin/python extract_all_experiments.py --extract --plot
    /data/liyuefeng/miniconda3/envs/gems/bin/python extract_all_experiments.py --plot_only  # 只画已有数据
"""

import sys, os, subprocess, json, time
from pathlib import Path
from argparse import ArgumentParser
from collections import OrderedDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = SCRIPT_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures" / "action"
EXTRACT_SCRIPT = SCRIPT_DIR / "action_cloud" / "extract_policy_geometry.py"
AUGMENT_SCRIPT = SCRIPT_DIR / "action_cloud" / "augment_slate_metrics.py"
PLOT_SCRIPT = SCRIPT_DIR / "action_cloud" / "plot_policy_geometry.py"

GEMS_DIR = PROJECT_ROOT / "checkpoints/gems"
AGENT_DIR = PROJECT_ROOT / "checkpoints/agents"

PYTHON = "/data/liyuefeng/miniconda3/envs/gems/bin/python"

# ============================================================================
# Experiment Registry: (exp_group, kl_label, gems_mode)
# ============================================================================

EXPERIMENTS = OrderedDict({
    # === KL=1.0, ideal_init (beta ablation) ===
    "beta_ablation_repreduce": {
        "kl_label": "kl1",
        "gems_mode": "ideal_init",
        "kl_value": 1.0,
        "runs": {
            "mix_b0": "mix_divpen",
            "mix_b2": "mix_divpen",
            "mix_b5": "mix_divpen",
            "mix_b8": "mix_divpen",
            "mix_b10": "mix_divpen",
            "topdown_b0": "topdown_divpen",
            "topdown_b2": "topdown_divpen",
            "topdown_b8": "topdown_divpen",
            "td_b5": "topdown_divpen",
            "td_b10": "topdown_divpen",
        },
    },
    # === KL=0.05, ideal_init ===
    "kl005_ideal_init": {
        "kl_label": "kl005",
        "gems_mode": "ideal_init",
        "kl_value": 0.05,
        "runs": {
            "mix_b0": "mix_divpen",
            "mix_b2": "mix_divpen",
            "mix_b5": "mix_divpen",
            "mix_b8": "mix_divpen",
            "mix_b10": "mix_divpen",
            "topdown_b0": "topdown_divpen",
            "topdown_b2": "topdown_divpen",
            "topdown_b5": "topdown_divpen",
            "topdown_b8": "topdown_divpen",
            "topdown_b10": "topdown_divpen",
        },
    },
    # === KL=0.05, mf_init ===
    "kl005_mf_init": {
        "kl_label": "kl005_mf",
        "gems_mode": "mf_init",
        "kl_value": 0.05,
        "runs": {
            "mix_b0": "mix_divpen",
            "mix_b10": "mix_divpen",
            "topdown_b0": "topdown_divpen",
            "topdown_b2": "topdown_divpen",
            "topdown_b5": "topdown_divpen",
        },
    },
})


def resolve_gems_checkpoint(env_name: str, kl_value: float, gems_mode: str) -> Path:
    """根据 env, KL, gems_mode 找到对应的 GeMS checkpoint。"""
    if kl_value == 1.0 and gems_mode == "ideal_init":
        # Old naming: GeMS_{env}_b5_ideal_init_latent32_beta1.0_click1.0_seed58407201.ckpt
        name = f"GeMS_{env_name}_b5_ideal_init_latent32_beta1.0_click1.0_seed58407201.ckpt"
    elif gems_mode == "ideal_init":
        name = f"GeMS_{env_name}_b5_pretrained_latent32_beta{kl_value}_click1.0_seed58407201_ideal_init.ckpt"
    elif gems_mode == "mf_init":
        name = f"GeMS_{env_name}_b5_pretrained_latent32_beta{kl_value}_click1.0_seed58407201_mf_init.ckpt"
    else:
        raise ValueError(f"Unknown gems_mode: {gems_mode}")

    ckpt = GEMS_DIR / name
    if not ckpt.exists():
        raise FileNotFoundError(f"GeMS checkpoint not found: {ckpt}")
    return ckpt


def run_cmd(cmd: list, desc: str = "") -> bool:
    """Run a command, return True on success."""
    tag = f"[{desc}] " if desc else ""
    print(f"{tag}Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True,
                            cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"  FAIL: {result.stderr.strip()[-200:]}")
        return False
    print(f"  OK")
    return True


def main():
    parser = ArgumentParser()
    parser.add_argument("--extract", action="store_true",
                        help="Run extraction for missing .npz files")
    parser.add_argument("--plot", action="store_true",
                        help="Generate plots (after extraction)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-extraction even if .npz exists")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Eval episodes per extraction")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done, don't execute")
    parser.add_argument("--exp_group", type=str, default="",
                        help="Only process specific experiment group")
    parser.add_argument("--run", type=str, default="",
                        help="Only process specific run name")
    args = parser.parse_args()

    if not args.extract and not args.plot:
        print("Specify --extract, --plot, or both")
        return

    # ====================================================================
    # Phase 1: Extract
    # ====================================================================
    if args.extract:
        total = 0
        existing = 0
        extracted = 0
        failed = []

        for exp_group, exp_config in EXPERIMENTS.items():
            if args.exp_group and args.exp_group not in exp_group:
                continue

            kl_label = exp_config["kl_label"]
            gems_mode = exp_config["gems_mode"]
            kl_value = exp_config["kl_value"]

            for run_name, env_name in exp_config["runs"].items():
                if args.run and args.run not in run_name:
                    continue

                for tag in ["best", "final"]:
                    total += 1
                    out_dir = OUTPUTS_DIR / kl_label / run_name
                    out_path = out_dir / f"{tag}_geometry.npz"

                    if out_path.exists() and not args.force:
                        existing += 1
                        print(f"[SKIP] {kl_label}/{run_name}/{tag} (exists)")
                        continue

                    ckpt_path = AGENT_DIR / exp_group / run_name / f"iql_{tag}.pt"
                    if not ckpt_path.exists():
                        print(f"[MISS] {kl_label}/{run_name}/{tag} — checkpoint not found: {ckpt_path}")
                        failed.append(f"{kl_label}/{run_name}/{tag}")
                        continue

                    try:
                        gems_ckpt = resolve_gems_checkpoint(env_name, kl_value, gems_mode)
                    except FileNotFoundError as e:
                        print(f"[MISS] {kl_label}/{run_name}/{tag} — {e}")
                        failed.append(f"{kl_label}/{run_name}/{tag}")
                        continue

                    out_dir.mkdir(parents=True, exist_ok=True)

                    cmd = [
                        PYTHON, str(EXTRACT_SCRIPT),
                        "--env_name", env_name,
                        "--run", run_name,
                        "--checkpoint_tag", tag,
                        "--checkpoint", str(ckpt_path),
                        "--gems_checkpoint", str(gems_ckpt),
                        "--episodes", str(args.episodes),
                        "--dataset_quality", "b5",
                        "--device", "cuda:0",
                    ]

                    if args.dry_run:
                        print(f"[DRY] {kl_label}/{run_name}/{tag}")
                        print(f"  agent: {ckpt_path}")
                        print(f"  gems:  {gems_ckpt}")
                        continue

                    print(f"[EXTRACT] {kl_label}/{run_name}/{tag} ", end="", flush=True)
                    if run_cmd(cmd, desc=f"{kl_label}/{run_name}/{tag}"):
                        extracted += 1
                        # Move output from default path to organized path
                        default_out = OUTPUTS_DIR / f"{env_name}_{run_name}_{tag}_geometry.npz"
                        if default_out.exists() and default_out != out_path:
                            default_out.rename(out_path)
                            print(f"  Moved: {default_out.name} → {out_path}")
                    else:
                        failed.append(f"{kl_label}/{run_name}/{tag}")

        print()
        print(f"Extraction: {extracted} done, {existing} skipped, {len(failed)} failed (of {total} total)")
        if failed:
            print("Failed:")
            for f in failed:
                print(f"  - {f}")

    # ====================================================================
    # Phase 2: Augment
    # ====================================================================
    if args.extract or args.plot:
        print()
        print("=" * 60)
        print("[AUGMENT] Adding combo_hit/item_freq_pct to all .npz files")
        print("=" * 60)
        # Run augment for each env that has data
        for env_name in ["mix_divpen", "topdown_divpen"]:
            run_cmd(
                [PYTHON, str(AUGMENT_SCRIPT),
                 "--env_name", env_name,
                 "--dataset_quality", "b5",
                 "--outputs_dir", str(OUTPUTS_DIR)],
                desc=f"augment_{env_name}"
            )

    # ====================================================================
    # Phase 3: Generate plots
    # ====================================================================
    if args.plot:
        print()
        print("=" * 60)
        print("[PLOT] Generating all figures")
        print("=" * 60)

        METHODS = ["pca", "tsne"]
        COLOR_BYS = ["reward", "combo_hit", "item_freq_pct"]

        total_p = 0
        gen_p = 0
        err_p = 0

        for exp_group, exp_config in EXPERIMENTS.items():
            if args.exp_group and args.exp_group not in exp_group:
                continue
            kl_label = exp_config["kl_label"]

            for run_name in exp_config["runs"]:
                if args.run and args.run not in run_name:
                    continue

                data_dir = OUTPUTS_DIR / kl_label / run_name
                best_npz = data_dir / "best_geometry.npz"
                final_npz = data_dir / "final_geometry.npz"

                if not best_npz.exists() or not final_npz.exists():
                    continue

                fig_dir = FIGURES_DIR / kl_label / run_name
                fig_dir.mkdir(parents=True, exist_ok=True)

                for method in METHODS:
                    for color_by in COLOR_BYS:
                        out_path = fig_dir / f"{method}_{color_by}.png"
                        total_p += 1

                        if out_path.exists() and not args.force:
                            continue

                        cmd = [
                            PYTHON, str(PLOT_SCRIPT),
                            "--best", str(best_npz),
                            "--final", str(final_npz),
                            "--method", method,
                            "--color_by", color_by,
                            "--out", str(out_path),
                        ]

                        if args.dry_run:
                            continue

                        if run_cmd(cmd, desc=f"plot_{kl_label}/{run_name}/{method}_{color_by}"):
                            gen_p += 1
                        else:
                            err_p += 1

        print()
        print(f"Plots: {gen_p} generated, {err_p} errors (of {total_p} total)")

    # ====================================================================
    # Summary
    # ====================================================================
    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)
    npz_count = len(list(OUTPUTS_DIR.rglob("*_geometry.npz")))
    png_count = len(list(FIGURES_DIR.rglob("*.png")))
    print(f"  .npz files: {npz_count}")
    print(f"  .png files: {png_count}")


if __name__ == "__main__":
    main()
