"""
GeMS Checkpoint 路径解析 + Ranker 加载 + Embedding 提取

从老项目 checkpoint_utils.py + ranker_factory.py 合并简化。
仅支持 mix_divpen / topdown_divpen 环境。
"""

import logging
from pathlib import Path
from typing import Tuple
import torch

from src.rankers.gems.embeddings import ItemEmbeddings
from src.rankers.gems.ranker import GeMS


# GeMS checkpoint 命名模板
# 格式: GeMS_{env_name}_{quality}_{mode}_latent{dim}_beta{kl}_click{lc}_seed{seed}.ckpt
#
# Embedding 初始化模式:
#   scratch    — 随机初始化，VAE 联合学习（论文默认，推荐）
#   ideal_init — env ground-truth embedding (item_embeddings_diffuse.pt) 初始化后训练
#                等同原论文 ideal 模式的弱化版。仅用于对照旧实验结果，不应用于正式实验。
#   mf_fixed   — BPR MF embedding 初始化并冻结（仅用于消融）
GEMS_CKPT_DIR = Path(__file__).resolve().parent.parent.parent / "checkpoints/gems"

# lambda_click 对于 mix_divpen/topdown_divpen 统一为 1.0
LAMBDA_CLICK = 1.0


def resolve_gems_checkpoint(
    env_name: str,
    dataset_quality: str,
    gems_embedding_mode: str = "scratch",
    latent_dim: int = 32,
    lambda_KL: float = 1.0,
    seed: int = 58407201,
) -> Tuple[Path, float]:
    """
    解析 GeMS checkpoint 路径。

    Args:
        env_name: mix_divpen | topdown_divpen
        dataset_quality: b5 | b3
        gems_embedding_mode: scratch | ideal_init | mf_fixed
        latent_dim: latent 维度
        lambda_KL: KL loss 权重
        seed: 随机种子

    Returns:
        (checkpoint_path, lambda_click)
    """
    suffix = f"_{gems_embedding_mode}"

    ckpt_name = (f"GeMS_{env_name}_{dataset_quality}{suffix}"
                 f"_latent{latent_dim}_beta{lambda_KL}"
                 f"_click{LAMBDA_CLICK}_seed{seed}.ckpt")

    ckpt_path = GEMS_CKPT_DIR / ckpt_name

    if not ckpt_path.exists():
        available = list(GEMS_CKPT_DIR.glob(f"GeMS_{env_name}*.ckpt"))
        msg = f"GeMS checkpoint 不存在: {ckpt_path}\n"
        if available:
            msg += "可用:\n" + "\n".join(f"  - {c.name}" for c in available)
        else:
            msg += f"目录 {GEMS_CKPT_DIR} 中没有 {env_name} 的 checkpoint"
        raise FileNotFoundError(msg)

    return ckpt_path, LAMBDA_CLICK


def load_gems_ranker(
    env_name: str,
    dataset_quality: str,
    gems_embedding_mode: str = "scratch",
    device: torch.device = torch.device("cuda"),
    item_embedd_dim: int = 20,
    rec_size: int = 10,
    latent_dim: int = 32,
    lambda_KL: float = 1.0,
    seed: int = 58407201,
) -> Tuple[GeMS, int, ItemEmbeddings]:
    """
    加载 GeMS ranker 并提取训练后的 embedding。

    返回 (ranker, action_dim, item_embeddings)。
    """
    gems_path, lambda_click = resolve_gems_checkpoint(
        env_name=env_name,
        dataset_quality=dataset_quality,
        gems_embedding_mode=gems_embedding_mode,
        latent_dim=latent_dim,
        lambda_KL=lambda_KL,
        seed=seed,
    )

    # 临时 embedding（仅用于 PL 的 load_from_checkpoint 结构初始化，会被 ckpt 覆盖）
    temp_embeddings = ItemEmbeddings.from_pretrained(
        str(Path(__file__).resolve().parent.parent.parent / "data/embeddings/item_embeddings_diffuse.pt"),
        device,
    )

    ranker = GeMS.load_from_checkpoint(
        str(gems_path),
        map_location=device,
        item_embeddings=temp_embeddings,
        item_embedd_dim=item_embedd_dim,
        device=device,
        rec_size=rec_size,
        latent_dim=latent_dim,
        lambda_click=lambda_click,
        lambda_KL=lambda_KL,
        lambda_prior=1.0,
        ranker_lr=3e-3,
        fixed_embedds="scratch",
        ranker_sample=False,
        hidden_layers_infer=[512, 256],
        hidden_layers_decoder=[256, 512],
    )
    ranker.freeze()
    ranker = ranker.to(device)

    # 提取训练后的 embedding
    gems_embedding_weights = ranker.item_embeddings.weight.data.clone()
    item_embeddings = ItemEmbeddings(
        num_items=ranker.num_items,
        item_embedd_dim=item_embedd_dim,
        device=device,
        weights=gems_embedding_weights,
    )

    action_dim, _ = ranker.get_action_dim()

    logging.info(
        f"GeMS checkpoint: {gems_path.name}  |  "
        f"mode={gems_embedding_mode}  |  latent_dim={action_dim}  |  "
        f"embed=[{ranker.num_items},{item_embedd_dim}]  |  lambda_click={lambda_click}"
    )

    return ranker, action_dim, item_embeddings
