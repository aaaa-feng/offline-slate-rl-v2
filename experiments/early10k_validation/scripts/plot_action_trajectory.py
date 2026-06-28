#!/usr/bin/env python3
"""Rebuild early10k_validation action-latent panels from geometry_exports."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
EXP_ROOT = PROJECT_ROOT / "experiments" / "early10k_validation"
sys.path.insert(0, str(SCRIPT_DIR))

from plot_style_common import (  # noqa: E402
    add_panel_colorbar,
    draw_baseline_cloud,
    draw_continuous_scatter,
    draw_discrete_combo,
    fit_pca_baseline,
    metric_limits,
    metric_mean,
    panel_figsize,
    panel_point_size,
    panel_title_fontsize,
    style_panel,
    tag_label,
)
from run_registry import iter_labels  # noqa: E402


DATASET_MAX_POINTS = 8000
POLICY_MAX_POINTS = 5000
BASELINE_TAG = "step0"
METRICS = {
    "reward": {
        "key": "policy_reward",
        "label": "Step Reward",
        "cmap": "RdYlGn",
        "discrete": False,
    },
    "combo_hit": {
        "key": "policy_combo_hit",
        "label": "Combo Hit (top-1000)",
        "cmap": "coolwarm",
        "discrete": True,
    },
    "item_freq_pct": {
        "key": "policy_item_freq_pct_mean",
        "label": "Item Freq Percentile (mean)",
        "cmap": "viridis",
        "discrete": False,
    },
}
REQUIRED_KEYS = {
    "dataset_latent_raw",
    "policy_latent_raw",
    "policy_reward",
    "policy_combo_hit",
    "policy_item_freq_pct_mean",
}


@dataclass
class PlotSummary:
    saved: int = 0
    missing_tags: list[str] = field(default_factory=list)
    key_errors: list[str] = field(default_factory=list)
    skipped_modes: list[str] = field(default_factory=list)


@dataclass
class StepData:
    tag: str
    dataset_vectors: np.ndarray
    policy_vectors: np.ndarray
    metrics: dict[str, np.ndarray]
    unique_slates: int | None


def load_manifest_tags() -> list[str]:
    manifest = json.loads((EXP_ROOT / "plot_tags_manifest.json").read_text())
    return list(manifest["tags"])


def stable_seed(*parts: str) -> int:
    text = "::".join(parts)
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(text)) % 1_000_000


def finite_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        return np.isfinite(x)
    return np.isfinite(x).all(axis=1)


def sampled_indices(length: int, max_points: int, seed: int) -> np.ndarray:
    if length <= max_points:
        return np.arange(length)
    return np.random.default_rng(seed).choice(length, max_points, replace=False)


def require_keys(path: Path, keys: set[str]) -> None:
    with np.load(str(path), allow_pickle=True) as data:
        missing = sorted(keys.difference(data.files))
    if missing:
        raise KeyError(f"{path}: missing keys {missing}")


def load_step(path: Path, tag: str, seed: int) -> StepData:
    require_keys(path, REQUIRED_KEYS)
    with np.load(str(path), allow_pickle=True) as data:
        dataset = np.asarray(data["dataset_latent_raw"], dtype=np.float32)
        policy = np.asarray(data["policy_latent_raw"], dtype=np.float32)

        d_mask = finite_rows(dataset)
        p_mask = finite_rows(policy)
        for spec in METRICS.values():
            p_mask &= np.isfinite(np.asarray(data[spec["key"]]))

        d_idx_all = np.flatnonzero(d_mask)
        p_idx_all = np.flatnonzero(p_mask)
        d_idx = d_idx_all[sampled_indices(len(d_idx_all), DATASET_MAX_POINTS, seed)]
        p_idx = p_idx_all[sampled_indices(len(p_idx_all), POLICY_MAX_POINTS, seed + 1)]

        unique_slates = None
        if "policy_slate" in data.files:
            slates = np.asarray(data["policy_slate"])[p_idx]
            if len(slates):
                unique_slates = len({tuple(row.tolist()) for row in slates})

        metrics = {
            spec["key"]: np.asarray(data[spec["key"]], dtype=np.float64)[p_idx]
            for spec in METRICS.values()
        }
        return StepData(
            tag=tag,
            dataset_vectors=dataset[d_idx],
            policy_vectors=policy[p_idx],
            metrics=metrics,
            unique_slates=unique_slates,
        )


def load_available_steps(run_label: str, mode: str, tags: list[str], summary: PlotSummary) -> list[StepData]:
    root = EXP_ROOT / "geometry_exports" / "action" / run_label / mode
    steps: list[StepData] = []
    for tag in tags:
        path = root / f"{tag}_geometry.npz"
        if not path.exists():
            msg = f"action/{run_label}/{mode}/{tag}"
            print(f"[warning missing] {msg}")
            summary.missing_tags.append(msg)
            continue
        try:
            steps.append(load_step(path, tag, seed=stable_seed(run_label, mode, tag)))
        except KeyError as exc:
            msg = str(exc)
            print(f"[error keys] {msg}")
            summary.key_errors.append(msg)
    return steps


def pick_baseline_step(steps: list[StepData]) -> StepData:
    for step in steps:
        if step.tag == BASELINE_TAG:
            return step
    return steps[0]


def title_for_step(step: StepData, metric_key: str, discrete: bool) -> str:
    values = step.metrics[metric_key]
    if discrete:
        hit_rate = metric_mean(values > 0)
        return f"{tag_label(step.tag)}\nhit rate={hit_rate:.1%}"
    return f"{tag_label(step.tag)}\n{metric_key.removeprefix('policy_')}={metric_mean(values):.2f}"


def plot_run_mode(run_label: str, mode: str, tags: list[str], out_dir: Path, summary: PlotSummary) -> None:
    steps = load_available_steps(run_label, mode, tags, summary)
    if not steps:
        msg = f"action/{run_label}/{mode}"
        print(f"[skip] no usable action tags for {msg}")
        summary.skipped_modes.append(msg)
        return

    baseline = pick_baseline_step(steps)
    pca = fit_pca_baseline(baseline.dataset_vectors)
    n = len(steps)
    point_size = panel_point_size(n)
    title_fontsize = panel_title_fontsize(n)
    out_dir.mkdir(parents=True, exist_ok=True)

    for metric_name, spec in METRICS.items():
        metric_key = spec["key"]
        vmin, vmax = metric_limits([step.metrics[metric_key] for step in steps])
        fig, axes = plt.subplots(1, n, figsize=panel_figsize(n), squeeze=False)

        for ax, step in zip(axes[0], steps):
            draw_baseline_cloud(ax, pca, baseline.dataset_vectors)
            if spec["discrete"]:
                draw_discrete_combo(
                    ax,
                    pca,
                    step.policy_vectors,
                    step.metrics[metric_key],
                    point_size=point_size,
                )
            else:
                scatter = draw_continuous_scatter(
                    ax,
                    pca,
                    step.policy_vectors,
                    step.metrics[metric_key],
                    cmap=spec["cmap"],
                    vmin=vmin,
                    vmax=vmax,
                    point_size=point_size,
                )
                add_panel_colorbar(ax, scatter, spec["label"])
            style_panel(
                ax,
                pca,
                title_for_step(step, metric_key, spec["discrete"]),
                title_fontsize=title_fontsize,
            )

        steps_str = ", ".join(tag_label(step.tag) for step in steps)
        fig.suptitle(
            f"{run_label} / {mode} — policy latent (action z) checkpoint panels ({metric_name})\n"
            f"grey = {tag_label(baseline.tag)} dataset baseline cloud | colored = milestone eval | {steps_str}",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(out_dir / f"trajectory_panels_{metric_name}.png", dpi=150)
        plt.close(fig)
        summary.saved += 1
        print(f"[saved] {out_dir / f'trajectory_panels_{metric_name}.png'}")


def print_summary(summary: PlotSummary) -> None:
    print("\n=== ACTION_PLOT_SUMMARY ===")
    print(f"saved_figures={summary.saved}")
    print(f"missing_tags={len(summary.missing_tags)}")
    for item in summary.missing_tags:
        print(f"  missing {item}")
    print(f"key_errors={len(summary.key_errors)}")
    for item in summary.key_errors:
        print(f"  key_error {item}")
    print(f"skipped_modes={len(summary.skipped_modes)}")
    for item in summary.skipped_modes:
        print(f"  skipped {item}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_label", default="all")
    parser.add_argument("--tags", nargs="*", default=None)
    parser.add_argument("--eval_mode", choices=["det", "samp", "both"], default="both")
    parser.add_argument("--output_dir", default=str(EXP_ROOT / "analysis" / "figures" / "action"))
    args = parser.parse_args()

    tags = args.tags or load_manifest_tags()
    modes = ["det", "samp"] if args.eval_mode == "both" else [args.eval_mode]
    summary = PlotSummary()
    for run_label in iter_labels(args.run_label):
        for mode in modes:
            plot_run_mode(run_label, mode, tags, Path(args.output_dir) / run_label / mode, summary)
    print_summary(summary)


if __name__ == "__main__":
    main()
