#!/usr/bin/env python3
"""Plot offline dataset action cloud colored by quality (reward / combo_hit / item_freq).

The gray background in trajectory plots only shows *where* dataset actions lie;
this script shows *which regions are good vs bad* in the same PCA space.

Usage:
    python experiments/rl_effectiveness_validation/scripts/plot_dataset_action_quality.py \
        --run_label M1_iql_b2_dense

    # overlay one policy checkpoint for comparison
    python ... --run_label M1_iql_b2_dense --policy_tag peak1700
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parents[3]

COLOR_SPECS = {
    "reward": {
        "key": "dataset_reward",
        "policy_key": "policy_reward",
        "label": "Step Reward",
        "cmap": "RdYlGn",
        "discrete": False,
    },
    "item_freq_pct": {
        "key": "dataset_item_freq_pct_mean",
        "policy_key": "policy_item_freq_pct_mean",
        "label": "Item Freq Percentile (mean)",
        "cmap": "viridis",
        "discrete": False,
    },
    "combo_hit": {
        "key": "dataset_combo_hit",
        "policy_key": "policy_combo_hit",
        "label": "Combo Hit (top-1000)",
        "cmap": "coolwarm",
        "discrete": True,
    },
}


def load_cloud(path: Path, n_max: int = 12000, seed: int = 42):
    data = np.load(str(path), allow_pickle=True)
    meta = json.loads(str(data.get("metadata", "{}")))
    latent = np.asarray(data["dataset_latent_raw"])
    idx = None
    if len(latent) > n_max:
        idx = np.random.default_rng(seed).choice(len(latent), n_max, replace=False)
        latent = latent[idx]

    def _take(key):
        if key not in data:
            return None
        arr = np.asarray(data[key])
        if idx is not None and len(arr) == len(data["dataset_latent_raw"]):
            arr = arr[idx]
        return arr

    quality = {name: _take(spec["key"]) for name, spec in COLOR_SPECS.items()}
    return latent, quality, meta


def load_policy_overlay(path: Path, n_max: int = 4000, seed: int = 42):
    data = np.load(str(path), allow_pickle=True)
    latent = np.asarray(data["policy_latent_raw"])
    idx = None
    if len(latent) > n_max:
        idx = np.random.default_rng(seed + 1).choice(len(latent), n_max, replace=False)
        latent = latent[idx]

    def _take(key, default=0.0):
        if key not in data:
            return np.full(len(latent), default, dtype=np.float32)
        arr = np.asarray(data[key])
        if idx is not None and len(arr) == len(data["policy_latent_raw"]):
            arr = arr[idx]
        return arr[: len(latent)]

    colors = {
        "reward": _take("policy_reward", 0.0),
        "item_freq_pct": _take("policy_item_freq_pct_mean", 0.0),
        "combo_hit": _take("policy_combo_hit", 0),
    }
    return latent, colors


def scatter_quality(ax, xy, values, spec, alpha=0.45, s=3, title_suffix=""):
    if spec["discrete"]:
        miss = values <= 0
        hit = values > 0
        ax.scatter(xy[miss, 0], xy[miss, 1], c="#4575b4", s=s, alpha=alpha * 0.7,
                   edgecolors="none", label=f"Miss ({miss.sum()})")
        ax.scatter(xy[hit, 0], xy[hit, 1], c="#d73027", s=s * 1.4, alpha=min(alpha + 0.25, 0.95),
                   edgecolors="black", linewidth=0.05, label=f"Hit ({hit.sum()})")
        ax.legend(markerscale=3, fontsize=8, loc="best")
        stat = f"hit rate={hit.mean():.1%}"
    else:
        vmin, vmax = np.percentile(values, [2, 98])
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=values, s=s, alpha=alpha, cmap=spec["cmap"],
                        vmin=vmin, vmax=vmax, edgecolors="none")
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label=spec["label"])
        stat = f"mean={values.mean():.2f}"
    ax.set_title(f"{spec['label']}{title_suffix}\n{stat}")


def plot_dataset_only(run_label: str, latent, quality, meta, out_dir: Path):
    pca = PCA(n_components=2, random_state=42)
    xy = pca.fit_transform(latent)
    ev = pca.explained_variance_ratio_ * 100

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    env = meta.get("env_name", "mix_divpen")
    dq = meta.get("dataset_quality", "b5")
    fig.suptitle(
        f"{run_label} — Dataset Action Cloud Quality ({env}/{dq}, n={len(latent)})",
        fontsize=13,
    )

    for ax, (name, spec) in zip(axes, COLOR_SPECS.items()):
        vals = quality[name]
        if vals is None:
            ax.set_title(f"{spec['label']} (missing)")
            ax.axis("off")
            continue
        scatter_quality(ax, xy, vals, spec, alpha=0.5, s=4)
        ax.set_xlabel(f"PC1 ({ev[0]:.1f}%)")
        ax.set_ylabel(f"PC2 ({ev[1]:.1f}%)")

    fig.tight_layout()
    out_path = out_dir / "dataset_action_quality_panels.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")

    for name, spec in COLOR_SPECS.items():
        vals = quality[name]
        if vals is None:
            continue
        fig, ax = plt.subplots(figsize=(9, 8))
        scatter_quality(ax, xy, vals, spec, alpha=0.55, s=5)
        ax.set_xlabel(f"PC1 ({ev[0]:.1f}%)")
        ax.set_ylabel(f"PC2 ({ev[1]:.1f}%)")
        ax.set_title(f"{run_label} — Dataset cloud colored by {spec['label']}")
        fig.tight_layout()
        single = out_dir / f"dataset_action_quality_{name}.png"
        fig.savefig(single, dpi=150)
        plt.close(fig)
        print(f"Saved {single}")


def plot_with_policy_overlay(
    run_label: str,
    latent,
    quality,
    policy_latent,
    policy_colors,
    policy_tag: str,
    out_dir: Path,
):
    pca = PCA(n_components=2, random_state=42)
    combined = np.vstack([latent, policy_latent])
    reduced = pca.fit_transform(combined)
    d_xy = reduced[: len(latent)]
    p_xy = reduced[len(latent) :]
    ev = pca.explained_variance_ratio_ * 100

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle(
        f"{run_label} — Dataset quality + policy overlay ({policy_tag})",
        fontsize=13,
    )

    for ax, (name, spec) in zip(axes, COLOR_SPECS.items()):
        vals = quality[name]
        if vals is None:
            ax.axis("off")
            continue
        scatter_quality(ax, d_xy, vals, spec, alpha=0.35, s=3, title_suffix=" (dataset)")
        pvals = policy_colors[name]
        if spec["discrete"]:
            hit = pvals > 0
            ax.scatter(
                p_xy[~hit, 0], p_xy[~hit, 1], s=28, c="none",
                edgecolors="#2166ac", linewidth=0.8, marker="o", label="Policy miss",
            )
            ax.scatter(
                p_xy[hit, 0], p_xy[hit, 1], s=32, c="#ffd92f",
                edgecolors="black", linewidth=0.4, marker="*", label="Policy hit",
            )
        else:
            ax.scatter(
                p_xy[:, 0], p_xy[:, 1], s=18, c="#111111", alpha=0.75,
                edgecolors="white", linewidth=0.2, marker="o",
                label=f"Policy ({policy_tag})",
            )
        ax.legend(markerscale=1.5, fontsize=8)
        ax.set_xlabel(f"PC1 ({ev[0]:.1f}%)")
        ax.set_ylabel(f"PC2 ({ev[1]:.1f}%)")

    fig.tight_layout()
    out_path = out_dir / f"dataset_quality_with_policy_{policy_tag}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_label", type=str, default="M1_iql_b2_dense")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="experiments/rl_effectiveness_validation/geometry_exports",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/rl_effectiveness_validation/analysis/figures",
    )
    parser.add_argument("--policy_tag", type=str, default=None,
                        help="Optional checkpoint tag to overlay as black stars")
    args = parser.parse_args()

    in_root = PROJECT_ROOT / args.input_dir / args.run_label
    out_dir = PROJECT_ROOT / args.output_dir / args.run_label
    out_dir.mkdir(parents=True, exist_ok=True)

    cloud_path = in_root / "dataset_cloud.npz"
    if not cloud_path.exists():
        raise SystemExit(f"Missing {cloud_path}. Run extract_checkpoint_geometry.py first.")

    latent, quality, meta = load_cloud(cloud_path)
    if quality["reward"] is None:
        raise SystemExit(
            "dataset_cloud.npz has no quality labels. "
            "Run augment_dataset_cloud_quality.py first."
        )

    plot_dataset_only(args.run_label, latent, quality, meta, out_dir)

    if args.policy_tag:
        policy_path = in_root / f"{args.policy_tag}_geometry.npz"
        if not policy_path.exists():
            raise FileNotFoundError(f"Missing {policy_path}")
        p_latent, p_colors = load_policy_overlay(policy_path)
        plot_with_policy_overlay(
            args.run_label, latent, quality, p_latent, p_colors, args.policy_tag, out_dir,
        )


if __name__ == "__main__":
    main()
