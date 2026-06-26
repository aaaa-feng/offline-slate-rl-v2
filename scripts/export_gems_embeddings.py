#!/usr/bin/env python3
"""Export item embeddings from GeMS checkpoints.

Examples:
    python scripts/export_gems_embeddings.py \
        --ckpt checkpoints/gems/GeMS_mix_divpen_b5_ideal_init_latent32_beta0.05_click1.0_seed58407201.ckpt \
        --out data/embeddings/gems/gems_mix_divpen_b5_ideal_init_kl0.05_seed58407201.pt

    python scripts/export_gems_embeddings.py \
        --ckpt-dir checkpoints/gems \
        --out-dir data/embeddings/gems
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def find_embedding_weight(state_dict: Dict[str, torch.Tensor]) -> Tuple[str, torch.Tensor]:
    """Find the item embedding tensor in a GeMS Lightning state_dict."""
    candidates = (
        "item_embeddings.weight",
        "ranker.item_embeddings.weight",
        "item_embeddings.embedd.weight",
        "ranker.item_embeddings.embedd.weight",
    )
    for key in candidates:
        if key in state_dict:
            return key, state_dict[key]

    for key, value in state_dict.items():
        if key.endswith("item_embeddings.weight") or key.endswith("item_embeddings.embedd.weight"):
            return key, value

    available = [key for key in state_dict if "embed" in key.lower()]
    raise KeyError(f"Cannot find item embedding weight. Embedding-related keys: {available}")


def default_output_name(ckpt_path: Path) -> str:
    """Create a stable embedding filename from a checkpoint name."""
    stem = ckpt_path.stem
    name = stem
    name = name.replace("GeMS_", "gems_")
    name = name.replace("_latent", "_latent")
    name = re.sub(r"_beta([^_]+)", r"_kl\1", name)
    return f"{name}.pt"


def export_one(ckpt_path: Path, out_path: Path, overwrite: bool = False) -> Path:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists, pass --overwrite to replace: {out_path}")

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    key, weight = find_embedding_weight(state_dict)
    weight = weight.detach().cpu().clone()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(weight, str(out_path))

    meta = {
        "source_ckpt": str(ckpt_path),
        "state_dict_key": key,
        "output": str(out_path),
        "shape": list(weight.shape),
        "mean": float(weight.mean().item()),
        "std": float(weight.std().item()),
        "min": float(weight.min().item()),
        "max": float(weight.max().item()),
    }
    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(
        f"exported {ckpt_path.name} -> {out_path} "
        f"shape={tuple(weight.shape)} key={key} mean={meta['mean']:.4f} std={meta['std']:.4f}"
    )
    return out_path


def resolve_path(path: Optional[str]) -> Optional[Path]:
    if path is None:
        return None
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export GeMS item embeddings to standalone .pt files.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ckpt", type=str, help="Single GeMS checkpoint path")
    group.add_argument("--ckpt-dir", type=str, help="Directory containing GeMS .ckpt files")
    parser.add_argument("--out", type=str, help="Output .pt path for --ckpt")
    parser.add_argument("--out-dir", type=str, default="data/embeddings/gems",
                        help="Output directory for --ckpt-dir, or default location for --ckpt")
    parser.add_argument("--glob", type=str, default="*.ckpt", help="Glob used with --ckpt-dir")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .pt/.json outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = resolve_path(args.out_dir)

    if args.ckpt:
        ckpt_path = resolve_path(args.ckpt)
        out_path = resolve_path(args.out) if args.out else out_dir / default_output_name(ckpt_path)
        export_one(ckpt_path, out_path, overwrite=args.overwrite)
        return

    ckpt_dir = resolve_path(args.ckpt_dir)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint dir not found: {ckpt_dir}")

    ckpts = sorted(ckpt_dir.glob(args.glob))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints matched {args.glob!r} under {ckpt_dir}")

    for ckpt_path in ckpts:
        out_path = out_dir / default_output_name(ckpt_path)
        export_one(ckpt_path, out_path, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
