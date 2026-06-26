#!/usr/bin/env python3
"""GRU belief PCA panels: grey = online step-999 baseline, color = reward.

Usage:
    python experiments/online_sac_geometry/plot_gru_panels_reward.py
    python experiments/online_sac_geometry/plot_gru_panels_reward.py --stream critic
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
    DEFAULT_RUN,
    discover_steps,
    npz_path,
    panel_figsize,
    panel_point_size,
    step_label,
)


def load_belief_reward(run_dir: Path, step: int, stream: str) -> tuple[np.ndarray, np.ndarray]:
    path = npz_path(run_dir, step)
    if path is None:
        raise FileNotFoundError(f"missing step {step}")
    data = np.load(path, allow_pickle=True)
    return (
        np.asarray(data[f"belief_{stream}"], dtype=np.float32),
        np.asarray(data["rewards"], dtype=np.float32),
    )


def plot_panels(
    run_dir: Path,
    out_path: Path,
    stream: str,
    steps: list[int],
    run_label: str,
) -> None:
    if BASELINE_STEP not in steps:
        raise SystemExit(f"--steps must include {BASELINE_STEP}")

    cloud999, _ = load_belief_reward(run_dir, BASELINE_STEP, stream)
    entries = []
    for step in steps:
        belief, reward = load_belief_reward(run_dir, step, stream)
        entries.append({"step": step, "belief": belief, "reward": reward})

    pca = PCA(n_components=2, random_state=42)
    pca.fit(cloud999)
    cloud_xy = pca.transform(cloud999)

    all_r = np.concatenate([e["reward"] for e in entries])
    vmin = float(np.percentile(all_r, 2))
    vmax = float(np.percentile(all_r, 98))
    if vmax <= vmin:
        vmin, vmax = float(all_r.min()), float(all_r.max() + 1e-6)

    n = len(entries)
    pt = panel_point_size(n)
    fig, axes = plt.subplots(1, n, figsize=panel_figsize(n), squeeze=False)

    for ax, entry in zip(axes[0], entries):
        ax.scatter(
            cloud_xy[:, 0], cloud_xy[:, 1],
            c="#bdbdbd", s=1.2, alpha=0.4, zorder=1,
        )
        xy = pca.transform(entry["belief"])
        sc = ax.scatter(
            xy[:, 0], xy[:, 1],
            c=entry["reward"], s=pt, cmap="RdYlGn",
            vmin=vmin, vmax=vmax,
            edgecolors="black", linewidth=0.08, zorder=2,
        )
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        r = entry["reward"]
        ax.set_title(f"{step_label(entry['step'])}\nreward={r.mean():.2f}", fontsize=9 if n > 8 else 10)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")

    steps_str = ", ".join(step_label(s) for s in steps)
    fig.suptitle(
        f"{run_label} — GRU belief {stream} checkpoint panels (reward)\n"
        f"grey = online step-999 baseline cloud | colored = milestone eval | {steps_str}",
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
    parser.add_argument("--stream", type=str, default="both", choices=["actor", "critic", "both"])
    parser.add_argument("--steps", type=int, nargs="+", default=None,
                        help="Milestone steps (default: auto-discover all npz)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    steps = args.steps or discover_steps(run_dir)
    if not steps:
        raise SystemExit(f"No milestone npz under {run_dir}")

    out_dir = Path(args.out_dir) if args.out_dir else (
        Path(__file__).resolve().parent / "outputs" / "trajectory_panels"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    streams = ["actor", "critic"] if args.stream == "both" else [args.stream]
    for stream in streams:
        plot_panels(
            run_dir,
            out_dir / f"trajectory_panels_belief_{stream}_reward.png",
            stream,
            steps,
            args.run_label,
        )


if __name__ == "__main__":
    main()
