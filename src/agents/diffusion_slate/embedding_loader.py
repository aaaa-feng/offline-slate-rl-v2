"""Load item embedding table from a GeMS checkpoint or exported tensor."""

import logging
from pathlib import Path
import torch
import torch.nn as nn


def _find_embedding_weight(state_dict):
    candidates = [
        "item_embeddings.weight",
        "ranker.item_embeddings.weight",
        "item_embeddings.embedd.weight",
        "ranker.item_embeddings.embedd.weight",
    ]
    for candidate in candidates:
        if candidate in state_dict:
            return candidate, state_dict[candidate]

    for key, value in state_dict.items():
        if key.endswith("item_embeddings.weight") or key.endswith("item_embeddings.embedd.weight"):
            return key, value

    available = [k for k in state_dict.keys() if "embed" in k.lower()]
    raise KeyError(f"Cannot find item embedding weight. Embedding-related keys: {available}")


def _load_weight(path: str):
    src = Path(path)
    obj = torch.load(str(src), map_location="cpu")

    if isinstance(obj, torch.Tensor):
        return "tensor", obj

    if isinstance(obj, dict):
        if "state_dict" in obj:
            return _find_embedding_weight(obj["state_dict"])
        if "weight" in obj and isinstance(obj["weight"], torch.Tensor):
            return "weight", obj["weight"]
        return _find_embedding_weight(obj)

    raise TypeError(f"Unsupported embedding file format at {path}: {type(obj)}")


def load_embedding_table(path: str, device: torch.device) -> nn.Embedding:
    """Load item embeddings from exported .pt tensor or GeMS .ckpt.

    Args:
        path: exported embedding .pt file, or GeMS/Lightning checkpoint.
        device: torch device.
    Returns:
        Frozen nn.Embedding with shape [num_items, item_dim].
    """
    key, weight = _load_weight(path)
    if weight.ndim != 2:
        raise ValueError(f"Expected embedding weight to be 2D, got shape={tuple(weight.shape)} from {path}")

    emb_mean = weight.mean().item()
    emb_std = weight.std().item()
    logging.info(f"[embedding_loader] Loaded from {path}")
    logging.info(f"  key={key}, shape={tuple(weight.shape)}, mean={emb_mean:.4f}, std={emb_std:.4f}")

    table = nn.Embedding(weight.shape[0], weight.shape[1],
                         _weight=weight.detach().clone())
    table.requires_grad_(False)
    return table.to(device)
