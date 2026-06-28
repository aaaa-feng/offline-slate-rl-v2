#!/usr/bin/env python3
"""Rebuild early10k_validation GRU belief panels from geometry_exports."""

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
        "stem": "reward",
        "label": "Step Reward",
        "cmap": "RdYlGn",
        "discrete": False,
    },
    "combo_hit": {
        "stem": "combo_hit",
        "label": "Combo Hit (top-1000)",
        "cmap": "coolwarm",
        "discrete": True,
    },
    "item_freq_pct": {
        "stem": "item_freq_pct_mean",
        "label": "Item Freq Percentile (mean)",
        "cmap": "viridis",
        "discrete": False,
    },
}
REQUIRED_KEYS = {
    "dataset_belief_actor",
    "policy_belief_actor",
    "dataset_reward",
    "policy_reward",
    "dataset_combo_hit",
    "policy_combo_hit",
    "dataset_item_freq_pct_mean",
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
    dataset_metrics: dict[str, np.ndarray]
    policy_metrics: dict[str, np.ndarray]
    policy_unique_slates: int | None
    source_mode: str


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


def load_step(path: Path, tag: str, source_mode: str, seed: int) -> StepData:
    require_keys(path, REQUIRED_KEYS)
    with np.load(str(path), allow_pickle=True) as data:
        dataset = np.asarray(data["dataset_belief_actor"], dtype=np.float32)
        policy = np.asarray(data["policy_belief_actor"], dtype=np.float32)

        d_mask = finite_rows(dataset)
        p_mask = finite_rows(policy)
        for spec in METRICS.values():
            stem = spec["stem"]
            d_mask &= np.isfinite(np.asarray(data[f"dataset_{stem}"]))
            p_mask &= np.isfinite(np.asarray(data[f"policy_{stem}"]))

        d_idx_all = np.flatnonzero(d_mask)
        p_idx_all = np.flatnonzero(p_mask)
        d_idx = d_idx_all[sampled_indices(len(d_idx_all), DATASET_MAX_POINTS, seed)]
        p_idx = p_idx_all[sampled_indices(len(p_idx_all), POLICY_MAX_POINTS, seed + 1)]

        policy_unique_slates = None
        if "policy_slate" in data.files:
            slates = np.asarray(data["policy_slate"])[p_idx]
            if len(slates):
                policy_unique_slates = len({tuple(row.tolist()) for row in slates})

        dataset_metrics = {
            f"dataset_{stem}": np.asarray(data[f"dataset_{stem}"], dtype=np.float64)[d_idx]
            for stem in [spec["stem"] for spec in METRICS.values()]
        }
        policy_metrics = {
            f"policy_{stem}": np.asarray(data[f"policy_{stem}"], dtype=np.float64)[p_idx]
            for stem in [spec["stem"] for spec in METRICS.values()]
        }
        return StepData(
            tag=tag,
            dataset_vectors=dataset[d_idx],
            policy_vectors=policy[p_idx],
            dataset_metrics=dataset_metrics,
            policy_metrics=policy_metrics,
            policy_unique_slates=policy_unique_slates,
            source_mode=source_mode,
        )


def path_for(run_label: str, mode: str, tag: str) -> Path:
    return EXP_ROOT / "geometry_exports" / "belief" / run_label / mode / f"{tag}_belief.npz"


def load_policy_steps(run_label: str, mode: str, tags: list[str], summary: PlotSummary) -> list[StepData]:
    steps: list[StepData] = []
    for tag in tags:
        path = path_for(run_label, mode, tag)
        if not path.exists():
            msg = f"belief/{run_label}/{mode}/{tag}"
            print(f"[warning missing] {msg}")
            summary.missing_tags.append(msg)
            continue
        try:
            steps.append(load_step(path, tag, mode, stable_seed(run_label, mode, tag)))
        except KeyError as exc:
            msg = str(exc)
            print(f"[error keys] {msg}")
            summary.key_errors.append(msg)
    return steps


def load_dataset_steps(run_label: str, tags: list[str], summary: PlotSummary) -> list[StepData]:
    steps: list[StepData] = []
    for tag in tags:
        chosen_mode = None
        path = None
        for mode in ("det", "samp"):
            candidate = path_for(run_label, mode, tag)
            if candidate.exists():
                chosen_mode = mode
                path = candidate
                break
        if path is None or chosen_mode is None:
            msg = f"belief/{run_label}/dataset/{tag}"
            print(f"[warning missing] {msg}")
            summary.missing_tags.append(msg)
            continue
        try:
            steps.append(load_step(path, tag, chosen_mode, stable_seed(run_label, "dataset", tag)))
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


def policy_title(step: StepData, metric_key: str, discrete: bool) -> str:
    values = step.policy_metrics[metric_key]
    if discrete:
        return f"{tag_label(step.tag)}\npolicy hit rate={metric_mean(values > 0):.1%}"
    mean = metric_mean(values)
    uniq = "na" if step.policy_unique_slates is None else str(step.policy_unique_slates)
    return f"{tag_label(step.tag)}\npolicy={mean:.2f} unique={uniq}"


def dataset_title(step: StepData, metric_key: str, discrete: bool) -> str:
    values = step.dataset_metrics[metric_key]
    if discrete:
        return f"{tag_label(step.tag)}\ndataset hit rate={metric_mean(values > 0):.1%}"
    return f"{tag_label(step.tag)}\ndataset={metric_mean(values):.2f}"


def render_colored_panel(
    ax: plt.Axes,
    pca,
    vectors: np.ndarray,
    values: np.ndarray,
    spec: dict,
    *,
    point_size: float,
    vmin: float,
    vmax: float,
    with_baseline: bool,
    baseline_vectors: np.ndarray,
) -> None:
    if with_baseline:
        draw_baseline_cloud(ax, pca, baseline_vectors)
    if spec["discrete"]:
        draw_discrete_combo(ax, pca, vectors, values, point_size=point_size)
        return
    scatter = draw_continuous_scatter(
        ax,
        pca,
        vectors,
        values,
        cmap=spec["cmap"],
        vmin=vmin,
        vmax=vmax,
        point_size=point_size,
    )
    add_panel_colorbar(ax, scatter, spec["label"])


def plot_dataset_panels(run_label: str, tags: list[str], out_dir: Path, summary: PlotSummary) -> None:
    steps = load_dataset_steps(run_label, tags, summary)
    if not steps:
        msg = f"belief/{run_label}/dataset"
        print(f"[skip] no usable dataset belief tags for {msg}")
        summary.skipped_modes.append(msg)
        return

    baseline = pick_baseline_step(steps)
    pca = fit_pca_baseline(baseline.dataset_vectors)
    n = len(steps)
    point_size = panel_point_size(n)
    title_fontsize = panel_title_fontsize(n)
    out_dir.mkdir(parents=True, exist_ok=True)

    for metric_name, spec in METRICS.items():
        key = f"dataset_{spec['stem']}"
        vmin, vmax = metric_limits([step.dataset_metrics[key] for step in steps])
        fig, axes = plt.subplots(1, n, figsize=panel_figsize(n), squeeze=False)

        for ax, step in zip(axes[0], steps):
            render_colored_panel(
                ax,
                pca,
                step.dataset_vectors,
                step.dataset_metrics[key],
                spec,
                point_size=point_size,
                vmin=vmin,
                vmax=vmax,
                with_baseline=False,
                baseline_vectors=baseline.dataset_vectors,
            )
            style_panel(
                ax,
                pca,
                dataset_title(step, key, spec["discrete"]),
                title_fontsize=title_fontsize,
            )

        steps_str = ", ".join(tag_label(step.tag) for step in steps)
        fig.suptitle(
            f"{run_label} — GRU belief dataset checkpoint panels ({metric_name})\n"
            f"colored = dataset milestone eval | PCA fit on {tag_label(baseline.tag)} | {steps_str}",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(out_dir / f"dataset_panels_{metric_name}.png", dpi=150)
        plt.close(fig)
        summary.saved += 1
        print(f"[saved] {out_dir / f'dataset_panels_{metric_name}.png'}")


def plot_policy_vs_dataset(
    run_label: str,
    mode: str,
    tags: list[str],
    out_dir: Path,
    summary: PlotSummary,
) -> None:
    steps = load_policy_steps(run_label, mode, tags, summary)
    if not steps:
        msg = f"belief/{run_label}/{mode}"
        print(f"[skip] no usable policy belief tags for {msg}")
        summary.skipped_modes.append(msg)
        return

    baseline = pick_baseline_step(steps)
    pca = fit_pca_baseline(baseline.dataset_vectors)
    n = len(steps)
    point_size = panel_point_size(n)
    title_fontsize = panel_title_fontsize(n)
    out_dir.mkdir(parents=True, exist_ok=True)

    for metric_name, spec in METRICS.items():
        dataset_key = f"dataset_{spec['stem']}"
        policy_key = f"policy_{spec['stem']}"
        vmin, vmax = metric_limits(
            [step.dataset_metrics[dataset_key] for step in steps]
            + [step.policy_metrics[policy_key] for step in steps]
        )
        fig, axes = plt.subplots(2, n, figsize=panel_figsize(n, rows=2), squeeze=False)

        for idx, step in enumerate(steps):
            top_ax = axes[0, idx]
            bottom_ax = axes[1, idx]
            # Top row: dataset replay GRU cloud at this checkpoint (colored only).
            render_colored_panel(
                top_ax,
                pca,
                step.dataset_vectors,
                step.dataset_metrics[dataset_key],
                spec,
                point_size=point_size,
                vmin=vmin,
                vmax=vmax,
                with_baseline=False,
                baseline_vectors=step.dataset_vectors,
            )
            # Bottom row: grey = same-step dataset cloud; colored = policy eval GRU.
            render_colored_panel(
                bottom_ax,
                pca,
                step.policy_vectors,
                step.policy_metrics[policy_key],
                spec,
                point_size=point_size,
                vmin=vmin,
                vmax=vmax,
                with_baseline=True,
                baseline_vectors=step.dataset_vectors,
            )
            style_panel(
                top_ax,
                pca,
                dataset_title(step, dataset_key, spec["discrete"]),
                title_fontsize=title_fontsize,
            )
            style_panel(
                bottom_ax,
                pca,
                policy_title(step, policy_key, spec["discrete"]),
                title_fontsize=title_fontsize,
            )

        steps_str = ", ".join(tag_label(step.tag) for step in steps)
        fig.suptitle(
            f"{run_label} / {mode} — GRU belief dataset vs policy ({metric_name})\n"
            f"top: dataset replay only | bottom: grey = same-step dataset, colored = policy | "
            f"PCA fit on {tag_label(baseline.tag)} | {steps_str}",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(out_dir / f"policy_vs_dataset_panels_{metric_name}.png", dpi=150)
        plt.close(fig)
        summary.saved += 1
        print(f"[saved] {out_dir / f'policy_vs_dataset_panels_{metric_name}.png'}")


def print_summary(summary: PlotSummary) -> None:
    print("\n=== BELIEF_PLOT_SUMMARY ===")
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
    parser.add_argument("--output_dir", default=str(EXP_ROOT / "analysis" / "figures" / "belief"))
    args = parser.parse_args()

    tags = args.tags or load_manifest_tags()
    modes = ["det", "samp"] if args.eval_mode == "both" else [args.eval_mode]
    summary = PlotSummary()
    for run_label in iter_labels(args.run_label):
        run_out = Path(args.output_dir) / run_label
        plot_dataset_panels(run_label, tags, run_out, summary)
        for mode in modes:
            plot_policy_vs_dataset(run_label, mode, tags, run_out / mode, summary)
    print_summary(summary)


if __name__ == "__main__":
    main()
