#!/usr/bin/env python3
"""
BPR MF Embedding 训练入口。

Usage:
    python scripts/train_mf.py \\
        --dataset mix_divpen_b5 \\
        --output mf_mix_b5.pt
"""

import sys
from pathlib import Path
from argparse import ArgumentParser

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rankers.gems.embeddings import MFEmbeddings


def main():
    parser = ArgumentParser(description="Train BPR MF embeddings")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset identifier (e.g. mix_divpen_b5)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output filename (default: mf_{dataset}.pt)")
    parser.add_argument("--num_items", type=int, default=1000)
    parser.add_argument("--item_embedd_dim", type=int, default=20)
    parser.add_argument("--train_val_split", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_neg_sample", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    device = torch.device(args.device)

    # Infer dataset path
    # e.g. mix_divpen_b5 → env_name=mix_divpen
    env_name = args.dataset.rsplit("_", 1)[0]
    dataset_path = PROJECT_ROOT / "data/datasets/offline" / env_name / f"{args.dataset}.pt"

    if not dataset_path.exists():
        print(f"ERROR: Dataset not found: {dataset_path}")
        sys.exit(1)

    # Output: mf_mix_b5.pt, mf_topdown_b3.pt, etc.
    parts = args.dataset.rsplit("_", 2)
    output_name = args.output or f"mf_{parts[0]}_{parts[-1]}.pt"
    output_path = PROJECT_ROOT / "data/embeddings/mf" / output_name

    mf = MFEmbeddings(
        num_items=args.num_items,
        item_embedd_dim=args.item_embedd_dim,
        device=device,
        train_val_split_MF=args.train_val_split,
        batch_size_MF=args.batch_size,
        lr_MF=args.lr,
        num_neg_sample_MF=args.num_neg_sample,
        weight_decay_MF=args.weight_decay,
        patience_MF=args.patience,
    )

    mf.train(dataset_path=str(dataset_path), data_dir="", output_path=str(output_path))


if __name__ == "__main__":
    main()
