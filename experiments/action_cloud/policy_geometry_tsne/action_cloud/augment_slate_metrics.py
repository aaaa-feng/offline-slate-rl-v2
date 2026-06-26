#!/usr/bin/env python3
"""
后处理：为已有的 geometry .npz 文件补充 combo_hit / item_freq_pct 指标。

无需重新跑 extract_policy_geometry.py —— 直接从 .npz 中的 policy_slate
计算每个策略点的 slate 质量指标，然后写回 .npz。

Usage:
    python augment_slate_metrics.py --dataset_quality b5 --env_name mix_divpen

输入: outputs/ 下所有 *_geometry.npz
输出: 原地覆盖，新增 policy_combo_hit, policy_item_freq_pct_mean 字段
"""

import sys
from pathlib import Path
from argparse import ArgumentParser
from collections import Counter

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def build_dataset_frequency(env_name: str, dataset_quality: str):
    """构建 item 频率和 top-1000 combo 频率字典。"""
    ds_path = PROJECT_ROOT / f"data/datasets/offline/{env_name}/{env_name}_{dataset_quality}_data_d4rl.npz"
    print(f"Loading dataset: {ds_path}")
    data = np.load(str(ds_path), allow_pickle=True)

    # Item freq
    all_items = data['slates'].flatten()
    item_freq = Counter(all_items)
    item_total = sum(item_freq.values())
    print(f"  Items: {len(item_freq)} unique, {item_total} total")

    # Combo freq (top-1000)
    combo_counter = Counter()
    n_slates = len(data['slates'])
    for slate in data['slates'][:min(100000, n_slates)]:
        combo_key = tuple(slate.tolist())
        combo_counter[combo_key] += 1
    combo_freq = dict(combo_counter.most_common(1000))
    print(f"  Combos: {len(combo_freq)} top combos cached")

    return item_freq, item_total, combo_freq


def compute_slate_metrics(slates: np.ndarray, item_freq: Counter, item_total: int,
                          combo_freq: dict):
    """为每个 slate 计算 per-step 指标。

    Args:
        slates: [N, rec_size] int array of decoded slates

    Returns:
        combo_hit: [N] bool (0/1) — slate 是否在 top-1000 combo 中
        item_freq_pct_mean: [N] float — slate 内 item 的平均频率百分位
    """
    N = len(slates)
    combo_hit = np.zeros(N, dtype=np.int32)
    item_freq_pct_mean = np.zeros(N, dtype=np.float32)

    for i in range(N):
        slate = slates[i]
        # combo_hit
        combo_key = tuple(slate.tolist())
        combo_hit[i] = 1 if combo_key in combo_freq else 0

        # item_freq_pct_mean
        pcts = []
        for item_id in slate:
            freq = item_freq.get(int(item_id), 0)
            pcts.append(freq / item_total * 100)
        item_freq_pct_mean[i] = float(np.mean(pcts)) if pcts else 0.0

    return combo_hit, item_freq_pct_mean


def main():
    parser = ArgumentParser()
    parser.add_argument("--env_name", type=str, default="mix_divpen")
    parser.add_argument("--dataset_quality", type=str, default="b5")
    parser.add_argument("--outputs_dir", type=str,
                        default=str(Path(__file__).resolve().parent.parent / "outputs"))
    args = parser.parse_args()

    # Build frequency dicts once
    item_freq, item_total, combo_freq = build_dataset_frequency(
        args.env_name, args.dataset_quality)

    # Find all .npz files
    outputs_dir = Path(args.outputs_dir)
    npz_files = sorted(outputs_dir.rglob("*_geometry.npz"))
    print(f"\nFound {len(npz_files)} .npz files to augment")

    for npz_path in npz_files:
        print(f"\n--- {npz_path.relative_to(outputs_dir)} ---")
        data = dict(np.load(str(npz_path), allow_pickle=True))

        if 'policy_combo_hit' in data and 'policy_item_freq_pct_mean' in data:
            print("  Already has slate metrics, skipping")
            continue

        slates = data['policy_slate']
        print(f"  slates: {slates.shape}")

        combo_hit, item_freq_pct_mean = compute_slate_metrics(
            slates, item_freq, item_total, combo_freq)

        combo_hit_rate = combo_hit.mean()
        item_freq_pct_avg = item_freq_pct_mean.mean()
        print(f"  combo_hit_rate: {combo_hit_rate:.2%} ({combo_hit.sum()}/{len(combo_hit)})")
        print(f"  item_freq_pct_mean: {item_freq_pct_avg:.2f}")

        # Add new fields
        data['policy_combo_hit'] = combo_hit
        data['policy_item_freq_pct_mean'] = item_freq_pct_mean

        # Save (overwrite)
        np.savez(str(npz_path), **data)
        print(f"  Saved (augmented)")

    print("\nDone!")


if __name__ == "__main__":
    main()
