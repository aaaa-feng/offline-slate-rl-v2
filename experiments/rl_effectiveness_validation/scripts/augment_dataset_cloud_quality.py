#!/usr/bin/env python3
"""为已有 dataset_cloud.npz 补充 reward / combo_hit / item_freq_pct 质量标签。

无需重跑 GeMS inference，仅根据 sample_indices（或同 seed 重采样）对齐离线数据集字段。

Usage:
    python experiments/rl_effectiveness_validation/scripts/augment_dataset_cloud_quality.py \
        --run_label M1_iql_b2_dense
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from config import ExperimentConfig


def build_dataset_frequency(cfg: ExperimentConfig):
    data = np.load(str(cfg.dataset_path), allow_pickle=True)
    item_freq = Counter(data["slates"].reshape(-1).tolist())
    item_total = int(sum(item_freq.values()))
    combo_freq = {}
    if "combo_freq_keys" in data and "combo_freq_vals" in data:
        combo_freq = {
            tuple(k): int(v)
            for k, v in zip(data["combo_freq_keys"], data["combo_freq_vals"])
        }
    else:
        combo_counter = Counter()
        for slate in data["slates"][: min(100000, len(data["slates"]))]:
            combo_counter[tuple(slate.tolist())] += 1
        combo_freq = dict(combo_counter.most_common(1000))
    return item_freq, item_total, combo_freq


def compute_slate_metrics(slates: np.ndarray, item_freq, item_total: int, combo_freq: dict):
    combo_hit = np.zeros(len(slates), dtype=np.int32)
    item_freq_pct = np.zeros(len(slates), dtype=np.float32)
    for i, slate in enumerate(slates):
        key = tuple(int(x) for x in slate.tolist())
        combo_hit[i] = 1 if key in combo_freq else 0
        pcts = [item_freq.get(int(x), 0) / item_total * 100 for x in slate]
        item_freq_pct[i] = float(np.mean(pcts)) if pcts else 0.0
    return combo_hit, item_freq_pct


def resolve_indices(data, meta: dict, cfg: ExperimentConfig) -> np.ndarray:
    if "sample_indices" in data:
        return np.asarray(data["sample_indices"], dtype=np.int64)
    n = len(data["dataset_latent_raw"])
    seed = int(meta.get("seed", cfg.seed))
    n_total = len(np.load(str(cfg.dataset_path), allow_pickle=True)["slates"])
    rng = np.random.default_rng(seed)
    return rng.choice(n_total, min(n, n_total), replace=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="experiments/rl_effectiveness_validation/config.yaml")
    parser.add_argument("--run_label", type=str, required=True)
    parser.add_argument(
        "--input_dir",
        type=str,
        default="experiments/rl_effectiveness_validation/geometry_exports",
    )
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(str(PROJECT_ROOT / args.config))
    cloud_path = PROJECT_ROOT / args.input_dir / args.run_label / "dataset_cloud.npz"
    if not cloud_path.exists():
        raise FileNotFoundError(f"Missing {cloud_path}")

    data = dict(np.load(str(cloud_path), allow_pickle=True))
    if all(k in data for k in ("dataset_reward", "dataset_combo_hit", "dataset_item_freq_pct_mean")):
        print(f"Already augmented: {cloud_path}")
        return

    meta = json.loads(str(data.get("metadata", "{}")))
    raw = np.load(str(cfg.dataset_path), allow_pickle=True)
    idx = resolve_indices(data, meta, cfg)
    slates = raw["slates"][idx]
    rewards = np.asarray(raw["rewards"][idx], dtype=np.float32)

    item_freq, item_total, combo_freq = build_dataset_frequency(cfg)
    combo_hit, item_freq_pct = compute_slate_metrics(slates, item_freq, item_total, combo_freq)

    data["dataset_reward"] = rewards
    data["dataset_combo_hit"] = combo_hit
    data["dataset_item_freq_pct_mean"] = item_freq_pct
    data["sample_indices"] = idx

    np.savez(str(cloud_path), **data)
    print(f"Augmented {cloud_path}")
    print(
        f"  reward mean={rewards.mean():.2f}  "
        f"combo_hit={combo_hit.mean():.2%}  "
        f"item_freq_pct={item_freq_pct.mean():.2f}"
    )


if __name__ == "__main__":
    main()
