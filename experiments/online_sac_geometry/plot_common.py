"""Shared helpers for online SAC geometry panel plots."""

from __future__ import annotations

import re
from pathlib import Path

DEFAULT_RUN = (
    "/data/liyuefeng/offline-slate-rl/experiments/online_geometry/diffuse_mix/"
    "SAC_GeMS_diffuse_mix_geometry_seed58407201"
)

BASELINE_STEP = 999
CATALOG_SIZE = 1000


def discover_steps(run_dir: Path) -> list[int]:
    steps: set[int] = set()
    for sub in ("baselines", "probes"):
        root = run_dir / sub
        if not root.is_dir():
            continue
        for d in root.iterdir():
            m = re.match(r"step_(\d+)$", d.name)
            if m and (d / "eval_trajectory.npz").is_file():
                steps.add(int(m.group(1)))
    return sorted(steps)


def npz_path(run_dir: Path, step: int) -> Path | None:
    for sub in ("baselines", "probes"):
        p = run_dir / sub / f"step_{step:05d}" / "eval_trajectory.npz"
        if p.is_file():
            return p
    return None


def step_label(step: int) -> str:
    if step == 999:
        return "step 999"
    if step == 2000:
        return "step 2k"
    if step % 1000 == 0:
        return f"step {step // 1000}k"
    return f"step {step}"


def panel_figsize(n: int) -> tuple[float, float]:
    w = min(4.6, max(2.8, 52 / max(n, 1)))
    return w * n, 4.8


def panel_point_size(n: int) -> float:
    return 14.0 if n <= 8 else 10.0
