"""Panel styling aligned with experiments/online_sac_geometry/plot_common.py."""

from __future__ import annotations

import re

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def tag_label(tag: str) -> str:
    match = re.match(r"step(\d+)$", tag)
    if not match:
        return tag
    step = int(match.group(1))
    if step == 0:
        return "step 0"
    if step % 1000 == 0:
        return f"step {step // 1000}k"
    return f"step {step}"


def panel_figsize(n: int, rows: int = 1) -> tuple[float, float]:
    width = min(4.6, max(2.8, 52 / max(n, 1)))
    return width * n, 4.8 * rows


def panel_point_size(n: int) -> float:
    return 14.0 if n <= 8 else 10.0


def panel_title_fontsize(n: int) -> int:
    return 9 if n > 8 else 10


def fit_pca_baseline(baseline_vectors: np.ndarray) -> PCA:
    pca = PCA(n_components=2, random_state=42)
    pca.fit(np.asarray(baseline_vectors, dtype=np.float64))
    return pca


def pca_axis_labels(pca: PCA) -> tuple[str, str]:
    ev = pca.explained_variance_ratio_ * 100.0
    return f"PC1 ({ev[0]:.1f}%)", f"PC2 ({ev[1]:.1f}%)"


def draw_baseline_cloud(ax: plt.Axes, pca: PCA, baseline_vectors: np.ndarray) -> None:
    cloud_xy = pca.transform(np.asarray(baseline_vectors, dtype=np.float64))
    ax.scatter(
        cloud_xy[:, 0],
        cloud_xy[:, 1],
        c="#bdbdbd",
        s=1.2,
        alpha=0.4,
        zorder=1,
    )


def draw_continuous_scatter(
    ax: plt.Axes,
    pca: PCA,
    vectors: np.ndarray,
    values: np.ndarray,
    *,
    cmap: str,
    vmin: float,
    vmax: float,
    point_size: float,
):
    xy = pca.transform(np.asarray(vectors, dtype=np.float64))
    return ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=values,
        s=point_size,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolors="black",
        linewidth=0.08,
        zorder=2,
    )


def draw_discrete_combo(
    ax: plt.Axes,
    pca: PCA,
    vectors: np.ndarray,
    values: np.ndarray,
    *,
    point_size: float,
) -> None:
    xy = pca.transform(np.asarray(vectors, dtype=np.float64))
    miss = values <= 0
    hit = values > 0
    ax.scatter(
        xy[miss, 0],
        xy[miss, 1],
        c="#4575b4",
        s=point_size,
        alpha=0.55,
        edgecolors="none",
        zorder=2,
        label=f"Miss ({miss.sum()})",
    )
    ax.scatter(
        xy[hit, 0],
        xy[hit, 1],
        c="#d73027",
        s=point_size * 1.2,
        alpha=0.9,
        edgecolors="black",
        linewidth=0.05,
        zorder=3,
        label=f"Hit ({hit.sum()})",
    )
    ax.legend(markerscale=2, fontsize=7, loc="best")


def add_panel_colorbar(ax: plt.Axes, scatter, label: str) -> None:
    plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label=label)


def style_panel(ax: plt.Axes, pca: PCA, title: str, *, title_fontsize: int) -> None:
    ax.set_title(title, fontsize=title_fontsize)
    xlabel, ylabel = pca_axis_labels(pca)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def metric_limits(values_list: list[np.ndarray]) -> tuple[float, float]:
    merged = np.concatenate([np.ravel(v) for v in values_list if len(v)])
    if len(merged) == 0:
        return 0.0, 1.0
    vmin = float(np.percentile(merged, 2))
    vmax = float(np.percentile(merged, 98))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.min(merged))
        vmax = float(np.max(merged) + 1e-6)
    return vmin, vmax


def metric_mean(values: np.ndarray) -> float:
    clean = values[np.isfinite(values)]
    return float(np.mean(clean)) if len(clean) else float("nan")
