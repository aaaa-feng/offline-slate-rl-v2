#!/usr/bin/env python3
"""Extract dataset action cloud + quality labels for one GeMS checkpoint.

Usage:
    python experiments/action_cloud/static_analysis/extract_cloud_quality.py \
        --ckpt checkpoints/gems/GeMS_mix_divpen_b5_pretrained_latent32_beta0.05_click1.0_seed58407201_ideal_init.ckpt

    python experiments/action_cloud/static_analysis/extract_cloud_quality.py --all_b5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
STATIC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(STATIC_DIR))

from cloud_quality_common import ckpt_slug, extract_cloud_quality, slug_to_ckpt

EXPORT_ROOT = Path(__file__).resolve().parent / "exports"


def save_export(ckpt_path: Path, payload: dict, out_root: Path) -> Path:
    slug = ckpt_slug(ckpt_path)
    out_dir = out_root / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dataset_cloud.npz"
    import numpy as np
    np.savez(str(out_path), **payload)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None, help="Path to GeMS .ckpt")
    parser.add_argument("--slug", type=str, default=None, help="e.g. mix_divpen_b5_beta0.05_ideal_init")
    parser.add_argument("--all_b5", action="store_true", help="Process all b5 GeMS ckpts")
    parser.add_argument("--n_samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=58407201)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output_dir", type=str, default=str(EXPORT_ROOT))
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_root = Path(args.output_dir)

    if args.all_b5:
        ckpt_dir = PROJECT_ROOT / "checkpoints/gems"
        ckpts = sorted(ckpt_dir.glob("GeMS_*_b5_*latent32*.ckpt"))
        ckpts = [p for p in ckpts if "_backup" not in p.name]
    elif args.slug:
        ckpts = [slug_to_ckpt(args.slug)]
    elif args.ckpt:
        ckpts = [Path(args.ckpt)]
        if not ckpts[0].is_absolute():
            ckpts[0] = PROJECT_ROOT / ckpts[0]
    else:
        parser.error("Provide --ckpt, --slug, or --all_b5")

    for ckpt_path in ckpts:
        slug = ckpt_slug(ckpt_path)
        out_path = out_root / slug / "dataset_cloud.npz"
        if out_path.exists() and not args.force:
            print(f"SKIP {slug}: {out_path} exists")
            continue
        print(f"Extracting {ckpt_path.name} -> {slug} ...")
        payload = extract_cloud_quality(ckpt_path, device, args.n_samples, args.seed)
        saved = save_export(ckpt_path, payload, out_root)
        rewards = payload["dataset_reward"]
        hits = payload["dataset_combo_hit"]
        print(
            f"  saved {saved}  reward_mean={rewards.mean():.2f}  "
            f"combo_hit={hits.mean():.2%}  n={len(rewards)}"
        )


if __name__ == "__main__":
    main()
