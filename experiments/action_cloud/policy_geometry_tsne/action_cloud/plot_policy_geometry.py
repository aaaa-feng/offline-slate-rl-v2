#!/usr/bin/env python3
"""
从 extract_policy_geometry.py 产出的 .npz 画 PCA/t-SNE 对比图。

支持多种着色方式：
  --color_by reward        : per-step reward（默认）
  --color_by combo_hit     : slate 是否命中 top-1000 combo（蓝=miss, 红=hit）
  --color_by item_freq_pct : slate 内 item 的平均频率百分位

Usage:
    python plot_policy_geometry.py \
        --best outputs/mix_divpen_mix_b0_best_geometry.npz \
        --final outputs/mix_divpen_mix_b0_final_geometry.npz \
        --method pca \
        --color_by combo_hit \
        --out outputs/figures/mix_b0_action_pca_combo.png
"""

import json, sys
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_and_sample(npz_path: str, n_dataset: int = 8000, n_policy: int = 4000,
                    color_by: str = "reward"):
    """加载 npz 并采样以加速 t-SNE。返回 dataset 和 policy 数据。"""
    data = np.load(npz_path, allow_pickle=True)
    d_raw = data['dataset_latent_raw']
    p_raw = data['policy_latent_raw']

    # Determine color array
    if color_by == "reward":
        p_color = data['policy_reward'].copy()
        color_label = "Step Reward"
        cmap = 'RdYlGn'
        vmin, vmax = 0, 200
        is_discrete = False
    elif color_by == "combo_hit":
        if 'policy_combo_hit' not in data:
            raise KeyError(f"{npz_path} does not have 'policy_combo_hit'. Run augment_slate_metrics.py first.")
        p_color = data['policy_combo_hit'].copy()
        color_label = "Combo Hit"
        cmap = 'coolwarm'  # blue=miss, red=hit
        vmin, vmax = 0, 1
        is_discrete = True
    elif color_by == "item_freq_pct":
        if 'policy_item_freq_pct_mean' not in data:
            raise KeyError(f"{npz_path} does not have 'policy_item_freq_pct_mean'. Run augment_slate_metrics.py first.")
        p_color = data['policy_item_freq_pct_mean'].copy()
        color_label = "Item Freq Pct Mean"
        cmap = 'viridis'
        vmin, vmax = 0, max(p_color.max(), 1.0)
        is_discrete = False
    else:
        raise ValueError(f"Unknown color_by: {color_by}")

    # Downsample
    if len(d_raw) > n_dataset:
        d_idx = np.random.choice(len(d_raw), n_dataset, replace=False)
        d_raw = d_raw[d_idx]
    if len(p_raw) > n_policy:
        p_idx = np.random.choice(len(p_raw), n_policy, replace=False)
        p_raw = p_raw[p_idx]
        p_color = p_color[p_idx]

    metadata = json.loads(str(data['metadata']))
    p_reward = data['policy_reward'][:len(p_color)] if len(data['policy_reward']) > n_policy else data['policy_reward']
    if len(p_reward) > n_policy:
        p_reward = p_reward[p_idx]

    return d_raw, p_raw, p_color, p_reward, metadata, color_label, cmap, vmin, vmax, is_discrete


def reduce(method: str, data: np.ndarray):
    """降维"""
    if method == 'pca':
        model = PCA(n_components=2, random_state=42)
        return model.fit_transform(data)
    elif method == 'tsne':
        model = TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=-1)
        return model.fit_transform(data)
    else:
        raise ValueError(f"Unknown method: {method}")


def plot_action_compare(best_npz: str, final_npz: str, method: str, color_by: str,
                        out_path: str):
    """画 dataset action cloud + best/final policy latent action，按 color_by 着色"""
    d_b, p_b, c_b, r_b, meta_b, c_label, cmap, vmin, vmax, discrete = \
        load_and_sample(best_npz, color_by=color_by)
    d_f, p_f, c_f, r_f, meta_f, _, _, _, _, _ = \
        load_and_sample(final_npz, color_by=color_by)

    # Combine all for joint dimensionality reduction
    combined = np.vstack([d_b, p_b, d_f, p_f])
    n_d = len(d_b)
    n_pb = len(p_b)
    n_df = len(d_f)

    print(f"Reducing {combined.shape[0]} points with {method}...")
    reduced = reduce(method, combined)

    d_b_2d = reduced[:n_d]
    p_b_2d = reduced[n_d:n_d + n_pb]
    d_f_2d = reduced[n_d + n_pb:n_d + n_pb + n_df]
    p_f_2d = reduced[n_d + n_pb + n_df:]

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # --- Best ---
    ax = axes[0]
    ax.scatter(d_b_2d[:, 0], d_b_2d[:, 1], c='lightgray', s=1, alpha=0.4, label='Dataset cloud')

    if discrete:
        # Binary: draw miss and hit separately for a clean legend
        miss_idx = c_b == 0
        hit_idx = c_b == 1
        ax.scatter(p_b_2d[miss_idx, 0], p_b_2d[miss_idx, 1],
                   c='#2166ac', s=8, edgecolors='black', linewidth=0.2,
                   label=f'Miss ({miss_idx.sum()})', alpha=0.8)
        ax.scatter(p_b_2d[hit_idx, 0], p_b_2d[hit_idx, 1],
                   c='#b2182b', s=8, edgecolors='black', linewidth=0.2,
                   label=f'Hit ({hit_idx.sum()})', alpha=0.8)
    else:
        sc = ax.scatter(p_b_2d[:, 0], p_b_2d[:, 1], c=c_b, s=8,
                        cmap=cmap, edgecolors='black', linewidth=0.2,
                        vmin=vmin, vmax=vmax)
        plt.colorbar(sc, ax=ax, label=c_label)

    ax.set_title(f"BEST ({meta_b['checkpoint_tag']})\n"
                 f"{c_label} mean={c_b.mean():.2f}, reward mean={r_b.mean():.1f}")
    ax.legend(markerscale=5)

    # --- Final ---
    ax = axes[1]
    ax.scatter(d_f_2d[:, 0], d_f_2d[:, 1], c='lightgray', s=1, alpha=0.4, label='Dataset cloud')

    if discrete:
        miss_idx = c_f == 0
        hit_idx = c_f == 1
        ax.scatter(p_f_2d[miss_idx, 0], p_f_2d[miss_idx, 1],
                   c='#2166ac', s=8, edgecolors='black', linewidth=0.2,
                   label=f'Miss ({miss_idx.sum()})', alpha=0.8)
        ax.scatter(p_f_2d[hit_idx, 0], p_f_2d[hit_idx, 1],
                   c='#b2182b', s=8, edgecolors='black', linewidth=0.2,
                   label=f'Hit ({hit_idx.sum()})', alpha=0.8)
    else:
        sc = ax.scatter(p_f_2d[:, 0], p_f_2d[:, 1], c=c_f, s=8,
                        cmap=cmap, edgecolors='black', linewidth=0.2,
                        vmin=vmin, vmax=vmax)
        plt.colorbar(sc, ax=ax, label=c_label)

    ax.set_title(f"FINAL ({meta_f['checkpoint_tag']})\n"
                 f"{c_label} mean={c_f.mean():.2f}, reward mean={r_f.mean():.1f}")
    ax.legend(markerscale=5)

    fig.suptitle(f"{meta_b['env_name']} / {meta_b['run']} — Policy Latent Action ({method.upper()}, color={color_by})",
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


def main():
    parser = ArgumentParser()
    parser.add_argument("--best", type=str, required=True, help="Best checkpoint .npz")
    parser.add_argument("--final", type=str, required=True, help="Final checkpoint .npz")
    parser.add_argument("--method", type=str, default="pca", choices=["tsne", "pca"])
    parser.add_argument("--color_by", type=str, default="reward",
                        choices=["reward", "combo_hit", "item_freq_pct"],
                        help="Metric to color policy points by")
    parser.add_argument("--out", type=str, required=True, help="Output .png path")
    args = parser.parse_args()

    plot_action_compare(args.best, args.final, args.method, args.color_by, args.out)


if __name__ == "__main__":
    main()
