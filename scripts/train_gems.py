#!/usr/bin/env python3
"""
GeMS VAE 离线预训练入口。

支持 --item_embedds scratch（随机初始化）和 pretrained（从文件加载）。

Usage:
    python scripts/train_gems.py \\
        --item_embedds scratch \\
        --dataset mix_divpen_b5 \\
        --latent_dim 32 \\
        --lambda_KL 1.0 \\
        --lambda_click 1.0 \\
        --seed 58407201 \\
        --max_epochs 50

    或使用 YAML 配置:
    python scripts/train_gems.py --config experiments/gems_scratch/config.yaml
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar, ModelCheckpoint

# 确保项目根目录在 path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rankers.gems.ranker import GeMS
from src.rankers.gems.embeddings import ItemEmbeddings
from src.data.slate_dataset import OfflineSlateDataModule
from src.utils.logger import SwanlabLogger
from config import ExperimentConfig


def create_parser():
    parser = ArgumentParser(description="Train GeMS VAE offline")

    # Config file
    parser.add_argument("--config", type=str, default=None,
                        help="YAML config file path")

    # Data
    parser.add_argument("--dataset", type=str, default="mix_divpen_b5",
                        help="Dataset identifier (e.g. mix_divpen_b5)")

    # Embedding
    parser.add_argument("--item_embedds", type=str, default="scratch",
                        choices=["scratch", "pretrained"],
                        help="Embedding init: scratch (random) or pretrained (from file)")
    parser.add_argument("--embedding_path", type=str, default=None,
                        help="Path to pretrained embeddings (.pt). Required if --item_embedds pretrained.")

    # Model
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--hidden_layers_infer", type=int, nargs="+", default=[512, 256])
    parser.add_argument("--hidden_layers_decoder", type=int, nargs="+", default=[256, 512])

    # Loss
    parser.add_argument("--lambda_KL", type=float, default=1.0)
    parser.add_argument("--lambda_click", type=float, default=0.5)
    parser.add_argument("--lambda_prior", type=float, default=0.0)
    parser.add_argument("--fixed_embedds", type=str, default="scratch",
                        choices=["scratch", "mf_fixed"])

    # Training
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_epochs", type=int, default=15)
    parser.add_argument("--ranker_lr", type=float, default=1e-3)
    parser.add_argument("--val_split", type=float, default=0.1)

    # System
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=58407201)
    parser.add_argument("--progress_bar", action="store_true")

    # Fixed params
    parser.add_argument("--num_items", type=int, default=1000)
    parser.add_argument("--item_embedd_dim", type=int, default=20)
    parser.add_argument("--rec_size", type=int, default=10)

    # Experiment tag
    parser.add_argument("--experiment_tag", type=str, default="",
                        help="SwanLab experiment name suffix (e.g. ideal_init, mf_init)")

    # Logging
    parser.add_argument("--swan_project", type=str, default="offline_slate_rl_gems_202606")
    parser.add_argument("--swan_workspace", type=str, default="Cliff")
    parser.add_argument("--swan_mode", type=str, default="cloud")

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    # Load config from YAML if provided
    if args.config:
        cfg = ExperimentConfig.from_yaml(args.config)
        # Override with CLI args
        for k, v in vars(args).items():
            if v is not None and k != "config":
                setattr(cfg, k, v)
    else:
        cfg = ExperimentConfig.from_args(args)

    print("=" * 80)
    print("=== GeMS VAE Offline Pretraining ===")
    print(f"Dataset: {args.dataset}")
    print(f"Item embedds: {args.item_embedds}")
    print(f"Latent dim: {args.latent_dim}")
    print(f"Lambda KL: {args.lambda_KL}, Lambda click: {args.lambda_click}")
    print(f"Seed: {args.seed}")
    print("=" * 80)

    pl.seed_everything(args.seed)

    # Setup logger
    exp_name = (f"gems_{args.dataset}_kl{args.lambda_KL}"
                f"_click{args.lambda_click}_lr{args.ranker_lr}")
    if args.experiment_tag:
        exp_name = f"{exp_name}_{args.experiment_tag}"
    logger = SwanlabLogger(
        project=args.swan_project,
        experiment_name=exp_name,
        workspace=args.swan_workspace,
        config=vars(args),
        mode=args.swan_mode,
    )

    # Load item embeddings
    device = torch.device(args.device)
    if args.item_embedds == "scratch":
        print("### Initializing embeddings from scratch...")
        item_embeddings = ItemEmbeddings.from_scratch(
            args.num_items, args.item_embedd_dim, device=device)
        print(f"✓ Scratch embeddings: [{args.num_items}, {args.item_embedd_dim}]")
    else:
        if args.embedding_path is None:
            parser.error("--embedding_path is required when --item_embedds pretrained")
        print(f"### Loading pretrained embeddings from {args.embedding_path}...")
        item_embeddings = ItemEmbeddings.from_pretrained(args.embedding_path, device=device)
        print(f"✓ Loaded embeddings: shape={item_embeddings.embedd.weight.shape}")

    # Parse dataset identifier to env_name and quality
    # e.g. mix_divpen_b5 → env_name=mix_divpen, quality=b5
    env_name, quality = args.dataset.rsplit("_", 1)
    if quality not in ("b3", "b5"):
        # dataset format might be "mix_divpen" (no quality suffix)
        env_name, quality = args.dataset, "b5"

    # Setup DataModule
    print("### Setting up DataModule...")
    data_path = PROJECT_ROOT / "data/datasets/offline" / env_name / f"{args.dataset}_data_d4rl.npz"
    if not data_path.exists():
        print(f"ERROR: Dataset not found at {data_path}")
        sys.exit(1)

    data_module = OfflineSlateDataModule(
        data_dir=str(PROJECT_ROOT / "data/datasets/offline"),
        env_name=env_name,
        quality=quality,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        dataset_path=str(data_path),
    )
    print(f"✓ DataModule created (dataset: {data_path})")

    # Create GeMS model
    print("### Creating GeMS model...")
    ranker = GeMS(
        item_embeddings=item_embeddings,
        item_embedd_dim=args.item_embedd_dim,
        device=device,
        rec_size=args.rec_size,
        latent_dim=args.latent_dim,
        lambda_click=args.lambda_click,
        lambda_KL=args.lambda_KL,
        lambda_prior=args.lambda_prior,
        ranker_lr=args.ranker_lr,
        fixed_embedds=args.fixed_embedds,
        ranker_sample=False,
        hidden_layers_infer=args.hidden_layers_infer,
        hidden_layers_decoder=args.hidden_layers_decoder,
        num_items=args.num_items,
    )
    print("✓ GeMS model created")

    # Checkpoint
    ckpt_dir = PROJECT_ROOT / "checkpoints/gems"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    tag_suffix = f"_{args.experiment_tag}" if args.experiment_tag else ""
    ckpt_name = (f"GeMS_{args.dataset}_{args.item_embedds}"
                 f"_latent{args.latent_dim}_beta{args.lambda_KL}"
                 f"_click{args.lambda_click}_seed{args.seed}{tag_suffix}")

    # Trainer
    print("### Starting training...")
    trainer = pl.Trainer(
        enable_progress_bar=args.progress_bar,
        logger=logger,
        callbacks=[
            RichProgressBar(),
            ModelCheckpoint(
                monitor='val_loss',
                dirpath=str(ckpt_dir),
                filename=ckpt_name,
                save_top_k=1,
                mode='min',
            ),
        ],
        accelerator="gpu" if args.device == "cuda" else "cpu",
        devices=1 if args.device == "cuda" else None,
        max_epochs=args.max_epochs,
    )

    trainer.fit(ranker, data_module)

    print(f"\n✓ Training complete. Checkpoint: {ckpt_dir}/{ckpt_name}.ckpt")


if __name__ == "__main__":
    main()
