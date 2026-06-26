"""Shared helpers: extract dataset latent cloud + quality labels, plot PCA panels."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from src.rankers.gems.embeddings import ItemEmbeddings
from src.rankers.gems.ranker import GeMS

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

COLOR_SPECS = {
    "reward": {
        "key": "dataset_reward",
        "label": "Step Reward",
        "cmap": "RdYlGn",
        "discrete": False,
    },
    "item_freq_pct": {
        "key": "dataset_item_freq_pct_mean",
        "label": "Item Freq Percentile (mean)",
        "cmap": "viridis",
        "discrete": False,
    },
    "combo_hit": {
        "key": "dataset_combo_hit",
        "label": "Combo Hit (top-1000)",
        "cmap": "coolwarm",
        "discrete": True,
    },
}


def ckpt_slug(ckpt_path: Path) -> str:
    name = ckpt_path.stem
    env = "mix_divpen" if "mix_divpen" in name else "topdown_divpen"
    quality = "b5" if "_b5_" in name else "b3"
    beta_m = re.search(r"beta([\d.]+)", name)
    beta = beta_m.group(1) if beta_m else "na"
    if "ideal_init" in name:
        embed = "ideal_init"
    elif "mf_init" in name or "mf_fixed" in name:
        embed = "mf_init"
    else:
        embed = "scratch"
    return f"{env}_{quality}_beta{beta}_{embed}"


def slug_to_ckpt(slug: str, gems_dir: Path | None = None) -> Path:
    """mix_divpen_b5_beta0.05_ideal_init -> GeMS ckpt path."""
    # {env}_{b5|b3}_beta{kl}_{ideal_init|mf_init}
    m = re.match(r"^(mix_divpen|topdown_divpen)_(b[35])_beta([\d.]+)_(ideal_init|mf_init)$", slug)
    if not m:
        raise ValueError(f"Invalid slug: {slug}")
    env, quality, beta, embed = m.group(1), m.group(2), m.group(3), m.group(4)
    gems_dir = gems_dir or (PROJECT_ROOT / "checkpoints/gems")
    candidates = [
        gems_dir / f"GeMS_{env}_{quality}_pretrained_latent32_beta{beta}_click1.0_seed58407201_{embed}.ckpt",
        gems_dir / f"GeMS_{env}_{quality}_ideal_init_latent32_beta{beta}_click1.0_seed58407201.ckpt",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No checkpoint for slug {slug}: tried {[str(p) for p in candidates]}")


def parse_ckpt_meta(ckpt_path: Path) -> dict:
    name = ckpt_path.name
    env = "mix_divpen" if "mix_divpen" in name else "topdown_divpen"
    quality = "b5" if "_b5_" in name else ("b3" if "_b3_" in name else "b5")
    beta_m = re.search(r"beta([\d.]+)", name)
    lambda_kl = float(beta_m.group(1)) if beta_m else 0.05
    if "ideal_init" in name:
        embed = "ideal_init"
    elif "mf_init" in name:
        embed = "mf_init"
    else:
        embed = "scratch"
    return {
        "env": env,
        "quality": quality,
        "lambda_kl": lambda_kl,
        "embed": embed,
        "ckpt": name,
        "slug": ckpt_slug(ckpt_path),
    }


def load_gems_from_ckpt(ckpt_path: Path, device: torch.device, lambda_kl: float = 0.05):
    temp = ItemEmbeddings.from_pretrained(
        str(PROJECT_ROOT / "data/embeddings/item_embeddings_diffuse.pt"), device)
    ranker = GeMS.load_from_checkpoint(
        str(ckpt_path),
        map_location=device,
        item_embeddings=temp,
        item_embedd_dim=20,
        device=device,
        rec_size=10,
        latent_dim=32,
        lambda_click=1.0,
        lambda_KL=lambda_kl,
        lambda_prior=1.0,
        ranker_lr=3e-3,
        fixed_embedds="scratch",
        ranker_sample=False,
        hidden_layers_infer=[512, 256],
        hidden_layers_decoder=[256, 512],
    )
    ranker.freeze()
    return ranker.to(device)


def build_dataset_frequency(env: str, quality: str):
    ds_path = PROJECT_ROOT / f"data/datasets/offline/{env}/{env}_{quality}_data_d4rl.npz"
    data = np.load(str(ds_path), allow_pickle=True)
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
    return ds_path, item_freq, item_total, combo_freq


def compute_slate_metrics(slates: np.ndarray, item_freq, item_total: int, combo_freq: dict):
    combo_hit = np.zeros(len(slates), dtype=np.int32)
    item_freq_pct = np.zeros(len(slates), dtype=np.float32)
    for i, slate in enumerate(slates):
        key = tuple(int(x) for x in slate.tolist())
        combo_hit[i] = 1 if key in combo_freq else 0
        pcts = [item_freq.get(int(x), 0) / item_total * 100 for x in slate]
        item_freq_pct[i] = float(np.mean(pcts)) if pcts else 0.0
    return combo_hit, item_freq_pct


def extract_cloud_quality(
    ckpt_path: Path,
    device: torch.device,
    n_samples: int = 20000,
    seed: int = 58407201,
) -> dict:
    meta = parse_ckpt_meta(ckpt_path)
    env, quality = meta["env"], meta["quality"]
    ranker = load_gems_from_ckpt(ckpt_path, device, meta["lambda_kl"])

    ds_path, item_freq, item_total, combo_freq = build_dataset_frequency(env, quality)
    raw = np.load(str(ds_path), allow_pickle=True)
    n_total = len(raw["slates"])
    n = min(n_samples, n_total)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_total, n, replace=False)
    slates_np = raw["slates"][idx]
    slates = torch.tensor(slates_np, dtype=torch.long, device=device)
    clicks = torch.zeros_like(slates, dtype=torch.float32)
    rewards = np.asarray(raw["rewards"][idx], dtype=np.float32)

    parts = []
    bs = 5000
    with torch.no_grad():
        for start in range(0, n, bs):
            end = min(start + bs, n)
            mu, _ = ranker.run_inference(slates[start:end], clicks[start:end])
            parts.append(mu.cpu())
    latent = torch.cat(parts, dim=0).float().numpy()
    center = latent.mean(axis=0)
    combo_hit, item_freq_pct = compute_slate_metrics(slates_np, item_freq, item_total, combo_freq)

    return {
        "dataset_latent_raw": latent,
        "dataset_center": center,
        "dataset_reward": rewards,
        "dataset_combo_hit": combo_hit,
        "dataset_item_freq_pct_mean": item_freq_pct,
        "sample_indices": idx,
        "metadata": json.dumps({
            **meta,
            "dataset_path": str(ds_path),
            "gems_checkpoint": str(ckpt_path),
            "dataset_samples": n,
            "seed": seed,
        }),
    }


def _subsample(latent, arrays: dict, n_max: int, seed: int):
    if len(latent) <= n_max:
        return latent, {k: v for k, v in arrays.items()}
    idx = np.random.default_rng(seed).choice(len(latent), n_max, replace=False)
    out = {k: v[idx] for k, v in arrays.items()}
    return latent[idx], out


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


def plot_cloud_quality_panels(export_path: Path, out_dir: Path, n_display: int = 12000):
    data = np.load(str(export_path), allow_pickle=True)
    meta = json.loads(str(data["metadata"]))
    latent = np.asarray(data["dataset_latent_raw"])
    quality = {
        name: np.asarray(data[spec["key"]])
        for name, spec in COLOR_SPECS.items()
        if spec["key"] in data
    }
    latent, quality = _subsample(latent, quality, n_display, seed=42)

    pca = PCA(n_components=2, random_state=42)
    xy = pca.fit_transform(latent)
    ev = pca.explained_variance_ratio_ * 100
    label = meta.get("slug", export_path.parent.name)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle(
        f"{label} — Dataset Action Cloud Quality "
        f"({meta['env']}/{meta['quality']}, GeMS β={meta['lambda_kl']}, n={len(latent)})",
        fontsize=12,
    )
    for ax, (name, spec) in zip(axes, COLOR_SPECS.items()):
        scatter_quality(ax, xy, quality[name], spec, alpha=0.5, s=4)
        ax.set_xlabel(f"PC1 ({ev[0]:.1f}%)")
        ax.set_ylabel(f"PC2 ({ev[1]:.1f}%)")
    fig.tight_layout()
    panels = out_dir / "dataset_action_quality_panels.png"
    fig.savefig(panels, dpi=150)
    plt.close(fig)

    reward_fig = out_dir / "dataset_action_quality_reward.png"
    fig, ax = plt.subplots(figsize=(9, 8))
    scatter_quality(ax, xy, quality["reward"], COLOR_SPECS["reward"], alpha=0.55, s=5)
    ax.set_xlabel(f"PC1 ({ev[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({ev[1]:.1f}%)")
    ax.set_title(f"{label} — Dataset cloud colored by Step Reward")
    fig.tight_layout()
    fig.savefig(reward_fig, dpi=150)
    plt.close(fig)

    for name, spec in COLOR_SPECS.items():
        if name == "reward":
            continue
        fig, ax = plt.subplots(figsize=(9, 8))
        scatter_quality(ax, xy, quality[name], spec, alpha=0.55, s=5)
        ax.set_xlabel(f"PC1 ({ev[0]:.1f}%)")
        ax.set_ylabel(f"PC2 ({ev[1]:.1f}%)")
        ax.set_title(f"{label} — Dataset cloud colored by {spec['label']}")
        fig.tight_layout()
        single = out_dir / f"dataset_action_quality_{name}.png"
        fig.savefig(single, dpi=150)
        plt.close(fig)

    return panels, reward_fig
