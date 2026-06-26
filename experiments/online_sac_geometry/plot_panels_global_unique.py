#!/usr/bin/env python3
"""PCA panels colored by cumulative global_unique (slate item coverage during eval).

Usage:
    python experiments/online_sac_geometry/plot_panels_global_unique.py
    python experiments/online_sac_geometry/plot_panels_global_unique.py --target action
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from plot_common import (
    BASELINE_STEP,
    CATALOG_SIZE,
    DEFAULT_RUN,
    discover_steps,
    npz_path,
    panel_figsize,
    panel_point_size,
    step_label,
)


def cumulative_global_unique(slates: np.ndarray) -> np.ndarray:
    seen: set[int] = set()
    counts: list[int] = []
    for slate in slates:
        seen.update(int(x) for x in slate.tolist())
        counts.append(len(seen))
    return np.asarray(counts, dtype=np.float32)


def load_milestone(run_dir: Path, step: int, vector_key: str) -> tuple[np.ndarray, np.ndarray, int]:
    path = npz_path(run_dir, step)
    if path is None:
        raise FileNotFoundError(f"missing step {step}")
    data = np.load(path, allow_pickle=True)
    slates = np.asarray(data["slates"], dtype=np.int64)
    cum_unique = cumulative_global_unique(slates)
    vectors = np.asarray(data[vector_key], dtype=np.float32)
    if len(vectors) != len(cum_unique):
        raise ValueError(f"step {step}: len({vector_key})={len(vectors)} != len(slates)={len(cum_unique)}")
    return vectors, cum_unique, int(cum_unique[-1])


def plot_panels(
    run_dir: Path,
    out_path: Path,
    vector_key: str,
    space_label: str,
    steps: list[int],
    run_label: str,
) -> None:
    if BASELINE_STEP not in steps:
        raise SystemExit(f"--steps must include {BASELINE_STEP}")

    cloud999, _, _ = load_milestone(run_dir, BASELINE_STEP, vector_key)
    entries = []
    for step in steps:
        vectors, cum_unique, final_unique = load_milestone(run_dir, step, vector_key)
        entries.append({
            "step": step,
            "vectors": vectors,
            "cum_unique": cum_unique,
            "global_unique": final_unique,
        })

    pca = PCA(n_components=2, random_state=42)
    pca.fit(cloud999)
    cloud_xy = pca.transform(cloud999)

    n = len(entries)
    pt = panel_point_size(n)
    fig, axes = plt.subplots(1, n, figsize=panel_figsize(n), squeeze=False)

    for ax, entry in zip(axes[0], entries):
        ax.scatter(
            cloud_xy[:, 0], cloud_xy[:, 1],
            c="#bdbdbd", s=1.2, alpha=0.4, zorder=1,
        )
        xy = pca.transform(entry["vectors"])
        sc = ax.scatter(
            xy[:, 0], xy[:, 1],
            c=entry["cum_unique"], s=pt, cmap="viridis",
            vmin=0, vmax=CATALOG_SIZE,
            edgecolors="black", linewidth=0.08, zorder=2,
        )
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="cum unique")
        gu = entry["global_unique"]
        ax.set_title(f"{step_label(entry['step'])}\nglobal_unique={gu}", fontsize=9 if n > 8 else 10)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")

    steps_str = ", ".join(step_label(s) for s in steps)
    fig.suptitle(
        f"{run_label} — {space_label} checkpoint panels (global_unique)\n"
        f"grey = online step-999 baseline cloud | color = cumulative unique items (of {CATALOG_SIZE}) | {steps_str}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path} ({n} panels: {steps})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default=DEFAULT_RUN)
    parser.add_argument("--run_label", type=str, default="Online SAC+GeMS diffuse_mix")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--steps", type=int, nargs="+", default=None,
                        help="Milestone steps (default: auto-discover all npz)")
    parser.add_argument(
        "--target", type=str, default="all",
        choices=["all", "action", "belief"],
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    steps = args.steps or discover_steps(run_dir)
    if not steps:
        raise SystemExit(f"No milestone npz under {run_dir}")

    out_dir = Path(args.out_dir) if args.out_dir else (
        Path(__file__).resolve().parent / "outputs" / "trajectory_panels"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    if args.target in ("all", "action"):
        jobs.append(("policy_latent_raw", "policy latent (action z)", "trajectory_panels_action_global_unique.png"))
    if args.target in ("all", "belief"):
        jobs.append(("belief_actor", "GRU belief actor", "trajectory_panels_belief_actor_global_unique.png"))
        jobs.append(("belief_critic", "GRU belief critic", "trajectory_panels_belief_critic_global_unique.png"))

    for vector_key, space_label, fname in jobs:
        plot_panels(run_dir, out_dir / fname, vector_key, space_label, steps, args.run_label)


if __name__ == "__main__":
    main()
