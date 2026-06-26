"""
Implicit Q-Learning (IQL) for GeMS datasets with Dual-Stream E2E GRU
Adapted from CORL: https://github.com/tinkoff-ai/CORL
Original paper: https://arxiv.org/pdf/2110.06169.pdf

Enhancements:
- Dual-Stream End-to-End GRU Architecture
- SwanLab logging support
- TrajectoryReplayBuffer for episode-based sampling

Key Features:
- Expectile regression for V-function
- Advantage Weighted Regression (AWR) for policy
- No explicit Q-target backup
"""
import copy
import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.common import set_seed, soft_update
from src.agents.iql.networks import (
    TanhGaussianActor,
    DeterministicActor,
    FixedGaussianActor,
    Critic,
    ValueFunction,
    LOG_STD_MIN,
    LOG_STD_MAX,
)
from src.belief.gru import GRUBelief
from src.rankers.gems.embeddings import ItemEmbeddings

# Note: TrajectoryReplayBuffer, OfflineEvalEnv are wired in scripts/train_agent.py.
# IQLConfig is now ExperimentConfig from top-level config.py.

# SwanLab Logger
try:
    from src.utils.logger import SwanlabLogger
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False
    logging.warning("SwanLab not available")

# 口径 B：归一化标签 L2 质心散布（离线复核常数，用于 z_outer_ratio）
REF_L2_SPREAD_B = 1.163


def _distribution_stats_1d(x: torch.Tensor) -> Dict[str, float]:
    """标量序列的 mean/std/分位数（用于 eval 上 Q/V/Adv 聚合）。"""
    if x.numel() == 0:
        return {k: 0.0 for k in ("mean", "std", "q10", "q50", "q90", "min", "max")}
    qs = torch.quantile(x, torch.tensor([0.1, 0.5, 0.9], device=x.device))
    return {
        "mean": x.mean().item(),
        "std": x.std().item(),
        "q10": qs[0].item(),
        "q50": qs[1].item(),
        "q90": qs[2].item(),
        "min": x.min().item(),
        "max": x.max().item(),
    }


def _belief_cloud_stats(states: torch.Tensor) -> Dict[str, float]:
    """Belief state 点云：mean_norm / std_mean / radius。"""
    norms = torch.norm(states, dim=-1)
    per_dim_std = states.std(dim=0)
    centroid = states.mean(dim=0)
    radius = torch.norm(states - centroid.unsqueeze(0), dim=-1).mean().item()
    return {
        "mean_norm": norms.mean().item(),
        "std_mean": per_dim_std.mean().item(),
        "radius": radius,
    }


def compute_train_eval_shift_metrics(
    train_m: Dict[str, float], eval_m: Dict[str, float]
) -> Dict[str, float]:
    """Category 26 (Q/V) + 27 (GRU) train–eval 偏移标量。"""
    eps = 1e-6

    def ratio(a: float, b: float) -> float:
        return a / (b + eps)

    out: Dict[str, float] = {}
    for name, train_key, eval_key in (
        ("actor_norm_ratio", "gru_belief_mean_norm", "eval_belief_mean_norm"),
        ("actor_std_ratio", "gru_belief_std_mean", "eval_belief_std_mean"),
        ("actor_radius_ratio", "gru_belief_radius", "eval_belief_radius"),
        ("critic_v_norm_ratio", "gru_critic_v_mean_norm", "eval_critic_v_mean_norm"),
        ("critic_v_std_ratio", "gru_critic_v_std_mean", "eval_critic_v_std_mean"),
        ("critic_v_radius_ratio", "gru_critic_v_radius", "eval_critic_v_radius"),
    ):
        out[name] = ratio(eval_m.get(eval_key, 0.0), train_m.get(train_key, 0.0))

    out["v_mean_gap"] = eval_m.get("eval_v_mean", 0.0) - train_m.get("v_value_mean", 0.0)
    out["q_mean_gap"] = eval_m.get("eval_q_mean", 0.0) - train_m.get("q_value_mean", 0.0)
    out["adv_mean_gap"] = eval_m.get("eval_adv_mean", 0.0) - train_m.get("advantage_mean", 0.0)
    out["adv_q90_gap"] = eval_m.get("eval_adv_q90", 0.0) - train_m.get("adv_q90", 0.0)
    out["v_std_ratio"] = ratio(eval_m.get("eval_v_std", 0.0), train_m.get("v_value_std", 0.0))
    out["q_std_ratio"] = ratio(eval_m.get("eval_q_std", 0.0), train_m.get("q_value_std", 0.0))
    out["v_ks_proxy"] = abs(eval_m.get("eval_v_q50", 0.0) - train_m.get("v_q50", 0.0)) + abs(
        eval_m.get("eval_v_q90", 0.0) - train_m.get("v_q90", 0.0)
    )
    out["q_ks_proxy"] = abs(eval_m.get("eval_q_q50", 0.0) - train_m.get("q_q50", 0.0)) + abs(
        eval_m.get("eval_q_q90", 0.0) - train_m.get("q_q90", 0.0)
    )
    return out


def build_swanlab_eval_metrics(
    eval_m: Dict[str, float],
    train_m: Optional[Dict[str, float]] = None,
    prefix: str = "00_Eval",
) -> Dict[str, float]:
    """周期 / Final eval → SwanLab（Reward 置顶，Value/GRU/Shift 分组）。"""
    p = prefix
    out: Dict[str, float] = {
        f"{p}/Reward/mean": eval_m["mean_reward"],
        f"{p}/Reward/std": eval_m["std_reward"],
        f"{p}/Reward/median": eval_m.get("median_reward", eval_m["mean_reward"]),
        f"{p}/Reward/iqm": eval_m.get("iqm_reward", eval_m["mean_reward"]),
        f"{p}/Reward/min": eval_m.get("min_reward", 0.0),
        f"{p}/Reward/max": eval_m.get("max_reward", 0.0),
        f"{p}/Task/combo_soft_hit_rate": eval_m.get("combo_soft_hit_rate", 0.0),
        f"{p}/Task/combo_top1_repeat_share": eval_m.get("combo_top1_repeat_share", 0.0),
        f"{p}/Task/global_unique_items": float(eval_m.get("global_unique_items", 0)),
        f"{p}/Task/item_freq_percentile_mean": eval_m.get("item_freq_percentile_mean", 0.0),
        f"{p}/Env/mean_episode_length": eval_m["mean_episode_length"],
        f"{p}/Env/early_termination_rate": eval_m.get("early_termination_rate", 0.0),
        f"{p}/Value/v_mean": eval_m.get("eval_v_mean", 0.0),
        f"{p}/Value/v_std": eval_m.get("eval_v_std", 0.0),
        f"{p}/Value/q_mean": eval_m.get("eval_q_mean", 0.0),
        f"{p}/Value/q_std": eval_m.get("eval_q_std", 0.0),
        f"{p}/Value/adv_mean": eval_m.get("eval_adv_mean", 0.0),
        f"{p}/Value/adv_q90": eval_m.get("eval_adv_q90", 0.0),
        f"{p}/GRU-Actor/mean_norm": eval_m.get("eval_belief_mean_norm", 0.0),
        f"{p}/GRU-Actor/std_mean": eval_m.get("eval_belief_std_mean", 0.0),
        f"{p}/GRU-Actor/radius": eval_m.get("eval_belief_radius", 0.0),
        f"{p}/GRU-CriticV/mean_norm": eval_m.get("eval_critic_v_mean_norm", 0.0),
        f"{p}/GRU-CriticV/std_mean": eval_m.get("eval_critic_v_std_mean", 0.0),
        f"{p}/GRU-CriticV/radius": eval_m.get("eval_critic_v_radius", 0.0),
    }
    if train_m is not None:
        shift = compute_train_eval_shift_metrics(train_m, eval_m)
        out.update(
            {
                f"{p}/Shift-Value/v_mean_gap": shift["v_mean_gap"],
                f"{p}/Shift-Value/q_mean_gap": shift["q_mean_gap"],
                f"{p}/Shift-Value/adv_mean_gap": shift["adv_mean_gap"],
                f"{p}/Shift-Value/adv_q90_gap": shift["adv_q90_gap"],
                f"{p}/Shift-Value/v_std_ratio": shift["v_std_ratio"],
                f"{p}/Shift-Value/q_std_ratio": shift["q_std_ratio"],
                f"{p}/Shift-Value/v_ks_proxy": shift["v_ks_proxy"],
                f"{p}/Shift-Value/q_ks_proxy": shift["q_ks_proxy"],
                f"{p}/Shift-GRU/actor_norm_ratio": shift["actor_norm_ratio"],
                f"{p}/Shift-GRU/actor_std_ratio": shift["actor_std_ratio"],
                f"{p}/Shift-GRU/actor_radius_ratio": shift["actor_radius_ratio"],
                f"{p}/Shift-GRU/critic_v_norm_ratio": shift["critic_v_norm_ratio"],
                f"{p}/Shift-GRU/critic_v_std_ratio": shift["critic_v_std_ratio"],
                f"{p}/Shift-GRU/critic_v_radius_ratio": shift["critic_v_radius_ratio"],
            }
        )
    return out


def build_swanlab_train_metrics(
    metrics: Dict[str, float], diag_metrics: Dict[str, float]
) -> Dict[str, float]:
    """训练步 SwanLab 指标（10_–91_ 前缀，条数与改前一致 + CriticV GRU 3 条）。"""
    m = metrics
    d = diag_metrics
    return {
        "10_Train-Opt/Loss/actor_loss": m["actor_loss"],
        "10_Train-Opt/Loss/critic_loss": m["critic_loss"],
        "10_Train-Opt/Loss/value_loss": m["value_loss"],
        "10_Train-Opt/Loss/awr_loss": m["awr_loss"],
        "10_Train-Opt/Loss/bc_loss": m["bc_loss"],
        "10_Train-Opt/Loss/bc_weighted": m["bc_weighted"],
        "10_Train-Opt/AWR-Weight/before_clip_max": m["weight_before_clip_max"],
        "10_Train-Opt/AWR-Weight/before_clip_min": m["weight_before_clip_min"],
        "10_Train-Opt/AWR-Weight/max": m["weight_max"],
        "10_Train-Opt/AWR-Weight/mean": m["weight_mean"],
        "10_Train-Opt/AWR-Entropy/entropy": m["weight_entropy"],
        "10_Train-Opt/AWR-Entropy/normalized": m["weight_entropy_normalized"],
        "10_Train-Opt/Gradient/actor_norm": m["actor_grad_norm"],
        "10_Train-Opt/Gradient/critic_norm": m["critic_grad_norm"],
        "10_Train-Opt/Gradient/value_norm": m["value_grad_norm"],
        "11_Train-Value/V-Summary/mean": m["v_value_mean"],
        "11_Train-Value/V-Summary/min": m["v_value_min"],
        "11_Train-Value/V-Summary/max": m["v_value_max"],
        "11_Train-Value/V-Summary/std": m["v_value_std"],
        "11_Train-Value/Q-Summary/mean": m["q_value_mean"],
        "11_Train-Value/Q-Summary/min": m["q_value_min"],
        "11_Train-Value/Q-Summary/max": m["q_value_max"],
        "11_Train-Value/Q-Summary/std": m["q_value_std"],
        "11_Train-Value/Q-Summary/target_mean": m["target_q_mean"],
        "11_Train-Value/Q-Summary/target_min": m["target_q_min"],
        "11_Train-Value/Q-Summary/target_max": m["target_q_max"],
        "11_Train-Value/TD-Summary/td_error": m["td_error"],
        "11_Train-Value/Adv-Summary/mean": m["advantage_mean"],
        "11_Train-Value/Adv-Summary/std": m["advantage_std"],
        "11_Train-Value/Adv-Summary/max": m["advantage_max"],
        "11_Train-Value/Adv-Summary/min": m["advantage_min"],
        "11_Train-Value/Expectile/v_above_q_rate": m["v_above_q_rate"],
        "11_Train-Value/Expectile/v_above_target_q_rate": m["v_above_target_q_rate"],
        "11_Train-Value/Expectile/match_error": m["expectile_match_error"],
        "11_Train-Value/V-Quantile/q10": m["v_q10"],
        "11_Train-Value/V-Quantile/q25": m["v_q25"],
        "11_Train-Value/V-Quantile/q50": m["v_q50"],
        "11_Train-Value/V-Quantile/q75": m["v_q75"],
        "11_Train-Value/V-Quantile/q90": m["v_q90"],
        "11_Train-Value/Q-Quantile/q10": m["q_q10"],
        "11_Train-Value/Q-Quantile/q25": m["q_q25"],
        "11_Train-Value/Q-Quantile/q50": m["q_q50"],
        "11_Train-Value/Q-Quantile/q75": m["q_q75"],
        "11_Train-Value/Q-Quantile/q90": m["q_q90"],
        "11_Train-Value/TargetQ-Quantile/q10": m["target_q_q10"],
        "11_Train-Value/TargetQ-Quantile/q50": m["target_q_q50"],
        "11_Train-Value/TargetQ-Quantile/q90": m["target_q_q90"],
        "11_Train-Value/NextV-Quantile/q10": m["next_v_q10"],
        "11_Train-Value/NextV-Quantile/q50": m["next_v_q50"],
        "11_Train-Value/NextV-Quantile/q90": m["next_v_q90"],
        "11_Train-Value/VQ-Gap-Quantile/q10": m["vq_gap_q10"],
        "11_Train-Value/VQ-Gap-Quantile/q25": m["vq_gap_q25"],
        "11_Train-Value/VQ-Gap-Quantile/q50": m["vq_gap_q50"],
        "11_Train-Value/VQ-Gap-Quantile/q75": m["vq_gap_q75"],
        "11_Train-Value/VQ-Gap-Quantile/q90": m["vq_gap_q90"],
        "11_Train-Value/Adv-Quantile/q10": m["adv_q10"],
        "11_Train-Value/Adv-Quantile/q25": m["adv_q25"],
        "11_Train-Value/Adv-Quantile/q50": m["adv_q50"],
        "11_Train-Value/Adv-Quantile/q75": m["adv_q75"],
        "11_Train-Value/Adv-Quantile/q90": m["adv_q90"],
        "11_Train-Value/Adv-Shape/positive_rate": m["adv_positive_rate"],
        "11_Train-Value/Adv-Shape/near_zero_rate": m["adv_near_zero_rate"],
        "11_Train-Value/Adv-Shape/effective_range": m["adv_effective_range"],
        "11_Train-Value/Adv-Shape/skewness": m["adv_skewness"],
        "11_Train-Value/TD-Quantile/q10": m["td_error_q10"],
        "11_Train-Value/TD-Quantile/q50": m["td_error_q50"],
        "11_Train-Value/TD-Quantile/q90": m["td_error_q90"],
        "11_Train-Value/TD-Quantile/overestimate_rate": m["td_error_overestimate_rate"],
        "11_Train-Value/Dynamic/v_next_vs_current_ratio": m["v_next_vs_current_ratio"],
        "11_Train-Value/Dynamic/explosion_rate": m["value_explosion_rate"],
        "12_Train-Policy/Geo/z_to_dataset_center_mean": m["z_to_dataset_center_mean"],
        "12_Train-Policy/Geo/z_to_boundary_mean": m["z_to_boundary_mean"],
        "12_Train-Policy/Geo/z_outer_ratio": m["z_outer_ratio"],
        "12_Train-Policy/Geo/ood_over_z": m["ood_over_z"],
        "12_Train-Policy/Geo/ood_distance_mean_det": m["ood_distance_mean_det"],
        "12_Train-Policy/Geo/ood_distance_max_det": m["ood_distance_max_det"],
        "12_Train-Policy/Geo/ood_distance_mean_samp": m["ood_distance_mean_samp"],
        "12_Train-Policy/Geo/ood_distance_max_samp": m["ood_distance_max_samp"],
        "12_Train-Policy/ActorMu/mu_min": m["actor_mu_min"],
        "12_Train-Policy/ActorMu/mu_max": m["actor_mu_max"],
        "12_Train-Policy/ActorMu/mu_mean": m["actor_mu_mean"],
        "12_Train-Policy/ActorMu/mu_iqr": m["mu_iqr"],
        "12_Train-Policy/ActorMu/entropy": m["policy_entropy"],
        "12_Train-Policy/ActorMu/distance_to_origin": m["policy_distance_to_origin"],
        "12_Train-Policy/ActorMu/tanh_saturation_ratio": m["tanh_saturation_ratio"],
        "12_Train-Policy/LogStd/raw_min": m["actor_log_std_raw_min"],
        "12_Train-Policy/LogStd/raw_mean": m["actor_log_std_raw_mean"],
        "12_Train-Policy/LogStd/raw_max": m["actor_log_std_raw_max"],
        "12_Train-Policy/LogStd/min": m["actor_log_std_min"],
        "12_Train-Policy/LogStd/mean": m["actor_log_std_mean"],
        "12_Train-Policy/LogStd/max": m["actor_log_std_max"],
        "12_Train-Policy/LogStd/floor_hit_rate": m["actor_log_std_floor_hit_rate"],
        "12_Train-Policy/LogStd/ceiling_hit_rate": m["actor_log_std_ceiling_hit_rate"],
        "12_Train-Policy/LogStd/max_per_dim_floor_hit": m["max_per_dim_floor_hit"],
        "12_Train-Policy/LogStd/mean_per_dim_floor_hit": m["mean_per_dim_floor_hit"],
        "12_Train-Policy/LogStd/num_dims_with_high_floor_hit": m["num_dims_with_high_floor_hit"],
        "12_Train-Policy/AWR/awr_weight_std": m["awr_weight_std"],
        "12_Train-Policy/AWR/log_prob_true_mean": m["log_prob_true_mean"],
        "13_Train-GRU/Actor/mean_norm": m["gru_belief_mean_norm"],
        "13_Train-GRU/Actor/std_mean": m["gru_belief_std_mean"],
        "13_Train-GRU/Actor/radius": m["gru_belief_radius"],
        "13_Train-GRU/Actor/transition_magnitude": m["gru_transition_magnitude"],
        "13_Train-GRU/Actor/transition_zero_rate": m["gru_transition_zero_rate"],
        "13_Train-GRU/Actor/dead_neuron_rate": m["gru_dead_neuron_rate"],
        "13_Train-GRU/Actor/transition_max": m["gru_transition_max"],
        "13_Train-GRU/CriticV/mean_norm": m["gru_critic_v_mean_norm"],
        "13_Train-GRU/CriticV/std_mean": m["gru_critic_v_std_mean"],
        "13_Train-GRU/CriticV/radius": m["gru_critic_v_radius"],
        "20_Train-Actor/Jacobian/min": m["jacobian_term_min"],
        "20_Train-Actor/Jacobian/max": m["jacobian_term_max"],
        "20_Train-Actor/Jacobian/has_negative": m["jacobian_term_has_negative"],
        "20_Train-Actor/Jacobian/has_nan": m["jacobian_term_has_nan"],
        "20_Train-Actor/LogProb/min_raw": m["log_prob_min_raw"],
        "20_Train-Actor/LogProb/max_raw": m["log_prob_max_raw"],
        "20_Train-Actor/OOB/count": m["oob_count"],
        "20_Train-Actor/OOB/rate": m["oob_rate"],
        "20_Train-Actor/OOB/atanh_protected_has_nan": m["atanh_protected_has_nan"],
        "20_Train-Actor/OOB/atanh_raw_has_nan": m["atanh_raw_has_nan"],
        "20_Train-Actor/OOB/atanh_raw_has_inf": m["atanh_raw_has_inf"],
        "20_Train-Actor/OOB/atanh_domain_violation_rate": m["atanh_domain_violation_rate"],
        "20_Train-Actor/OOB/true_action_abs_max": m["true_action_abs_max"],
        "20_Train-Actor/OOB/action_ratio_abs_max": m["action_ratio_abs_max"],
        "30_Train-Rep/actor_svd_rank": d["actor_svd_rank"],
        "30_Train-Rep/critic_svd_rank": d["critic_svd_rank"],
        "30_Train-Rep/actor_condition_number": d["actor_condition_number"],
        "30_Train-Rep/critic_condition_number": d["critic_condition_number"],
        "30_Train-Rep/consistency": d["representation_consistency"],
        "90_Train-Data/batch_reward_mean": m["batch_reward_mean"],
        "91_Train-Health/Input/s_actor_has_nan": m["s_actor_has_nan"],
        "91_Train-Health/Input/s_actor_has_inf": m["s_actor_has_inf"],
        "91_Train-Health/Input/s_actor_min": m["s_actor_min"],
        "91_Train-Health/Input/s_actor_max": m["s_actor_max"],
        "91_Train-Health/Input/true_actions_has_nan": m["true_actions_has_nan"],
        "91_Train-Health/Input/true_actions_has_inf": m["true_actions_has_inf"],
        "91_Train-Health/Input/true_actions_min": m["true_actions_min"],
        "91_Train-Health/Input/true_actions_max": m["true_actions_max"],
    }


def expectile_loss(diff: torch.Tensor, expectile: float = 0.7) -> torch.Tensor:
    """
    Expectile regression loss (核心IQL loss)
    L(u) = |tau - I(u < 0)| * u^2

    Args:
        diff: Difference tensor (V(s) - Q(s,a))
        expectile: Expectile parameter (default 0.7)

    Returns:
        Expectile loss
    """
    weight = torch.where(diff > 0, expectile, 1 - expectile)
    return weight * (diff ** 2)


def compute_svd_rank(states: torch.Tensor, eps: float = 1e-8) -> tuple:
    """
    Compute SVD-based effective rank and condition number for representation collapse detection.

    Args:
        states: State tensor [batch_size, state_dim]
        eps: Small constant for numerical stability

    Returns:
        effective_rank: Effective rank (normalized by state_dim)
        condition_number: Ratio of max to min singular value
    """
    if states.dim() != 2:
        states = states.view(-1, states.size(-1))

    # Center the data
    states_centered = states - states.mean(dim=0, keepdim=True)

    # SVD decomposition
    try:
        U, S, V = torch.svd(states_centered)

        # Effective rank: (sum of singular values)^2 / sum of squared singular values
        sum_s = S.sum()
        sum_s2 = (S ** 2).sum()
        effective_rank = (sum_s ** 2) / (sum_s2 + eps)

        # Condition number: max / min singular value
        condition_number = S[0] / (S[-1] + eps)

        return effective_rank.item(), condition_number.item()
    except:
        return 0.0, 0.0


def compute_gradient_conflict(grad1: torch.Tensor, grad2: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Compute cosine similarity between two gradient tensors.

    Args:
        grad1: First gradient tensor
        grad2: Second gradient tensor
        eps: Small constant for numerical stability

    Returns:
        cos_sim: Cosine similarity (-1 to 1)
                 -1.0 = completely opposite
                  0.0 = orthogonal
                 +1.0 = completely aligned
    """
    # Flatten gradients
    g1_flat = grad1.view(-1)
    g2_flat = grad2.view(-1)

    # Compute cosine similarity
    dot_product = torch.dot(g1_flat, g2_flat)
    norm1 = torch.norm(g1_flat)
    norm2 = torch.norm(g2_flat)

    cos_sim = dot_product / (norm1 * norm2 + eps)

    return cos_sim.item()


class IQLAgent:
    """Implicit Q-Learning Agent with Dual-Stream E2E GRU (GeMS-aligned)"""

    def __init__(
        self,
        action_dim: int,
        config,  # ExperimentConfig (duck-typed, matches IQLConfig fields)
        ranker_params: Dict,
        ranker=None,  # 🔥 Solution B: Accept ranker for real-time inference
    ):
        self.config = config
        self.device = torch.device(config.device)
        self.action_dim = action_dim
        self.max_action = 1.0
        self.total_it = 0

        # 🔥 NEW: Eval 时收集 belief state 的缓存
        self._last_belief_state = None
        self._last_belief_state_critic_v = None
        self._eval_diag_buffers = None

        # 0. Ranker for real-time action inference (Solution B)
        self.ranker = ranker

        # Extract action normalization parameters from ranker_params
        self.action_center = ranker_params['action_center'].to(self.device)
        self.action_scale = ranker_params['action_scale'].to(self.device)
        
        # 🔥 NEW: Extract dataset latent space global range (for probe)
        self.dataset_center = ranker_params.get('dataset_center', self.action_center).to(self.device)
        self.action_range = ranker_params.get('action_range', self.action_scale * 2).to(self.device)

        # Extract GeMS-trained embeddings from ranker_params
        self.item_embeddings = ranker_params['item_embeddings']

        # Calculate explicit input_dim for GRU
        input_dim = config.rec_size * (config.item_embedd_dim + 1)

        # GRU Architecture Ablation: 根据 gru_mode 配置 beliefs
        if config.gru_mode == "q_independent":
            beliefs_list = ["actor", "critic_v", "critic_q"]
        else:
            beliefs_list = ["actor", "critic_v"]

        # Initialize Dual-Stream GRU
        self.belief = GRUBelief(
            item_embeddings=self.item_embeddings,
            belief_state_dim=config.belief_hidden_dim,
            item_embedd_dim=config.item_embedd_dim,
            rec_size=config.rec_size,
            ranker=None,
            device=self.device,
            belief_lr=0.0,
            hidden_layers_reduction=[],
            beliefs=beliefs_list,  # 🔥 动态配置
            hidden_dim=config.belief_hidden_dim,
            input_dim=input_dim
        )

        # Double-freeze embeddings (after GRUBelief's deepcopy)
        for module in self.belief.item_embeddings:
            self.belief.item_embeddings[module].freeze()

        # Actor - support multiple architectures
        actor_desc = ""
        if config.actor_type == "gaussian":
            self.actor = TanhGaussianActor(
                state_dim=config.belief_hidden_dim,
                action_dim=action_dim,
                max_action=self.max_action,
                hidden_dim=config.hidden_dim,
                n_hidden=config.n_hidden,
            ).to(self.device)
            actor_desc = "TanhGaussian (learnable variance)"
        elif config.actor_type == "deterministic":
            self.actor = DeterministicActor(
                state_dim=config.belief_hidden_dim,
                action_dim=action_dim,
                max_action=self.max_action,
                hidden_dim=config.hidden_dim,
                n_hidden=config.n_hidden,
            ).to(self.device)
            actor_desc = "Deterministic (no variance)"
        elif config.actor_type == "fixed_gaussian":
            self.actor = FixedGaussianActor(
                state_dim=config.belief_hidden_dim,
                action_dim=action_dim,
                max_action=self.max_action,
                hidden_dim=config.hidden_dim,
                n_hidden=config.n_hidden,
                fixed_std=config.fixed_std,
            ).to(self.device)
            actor_desc = f"FixedGaussian (fixed_std={config.fixed_std})"
        else:
            raise ValueError(f"Unknown actor_type: {config.actor_type}")


        # Critics
        self.critic_1 = Critic(config.belief_hidden_dim, action_dim, config.hidden_dim).to(self.device)
        self.critic_2 = Critic(config.belief_hidden_dim, action_dim, config.hidden_dim).to(self.device)

        # Value function
        self.value = ValueFunction(config.belief_hidden_dim, config.hidden_dim, config.n_hidden).to(self.device)

        # Target critics
        self.critic_1_target = copy.deepcopy(self.critic_1)
        self.critic_2_target = copy.deepcopy(self.critic_2)

        # Three separate optimizers (IQL-specific)
        # 🔥 GRU Architecture Ablation: 根据 gru_mode 配置 optimizer

        # Value optimizer: always includes Value Network + critic_v GRU
        self.value_optimizer = torch.optim.Adam([
            {'params': self.belief.gru["critic_v"].parameters()},
            {'params': self.value.parameters()}
        ], lr=config.value_lr)

        # Critic optimizer: 根据 gru_mode 配置
        if config.gru_mode == "q_independent":
            # Q有独立GRU (critic_q)
            self.critic_optimizer = torch.optim.Adam([
                {'params': self.belief.gru["critic_q"].parameters()},
                {'params': self.critic_1.parameters()},
                {'params': self.critic_2.parameters()}
            ], lr=config.critic_lr)
            critic_desc = "critic_q GRU (Q independent)"
        elif config.gru_mode == "qv_shared_all_update":
            # QV共用 critic_v GRU，Q梯度也更新
            self.critic_optimizer = torch.optim.Adam([
                {'params': self.belief.gru["critic_v"].parameters()},
                {'params': self.critic_1.parameters()},
                {'params': self.critic_2.parameters()}
            ], lr=config.critic_lr)
            critic_desc = "critic_v GRU (Q gradient updates GRU)"
        else:  # qv_shared_detach (默认)
            # Q梯度detach，critic_optimizer不含GRU
            self.critic_optimizer = torch.optim.Adam([
                {'params': self.critic_1.parameters()},
                {'params': self.critic_2.parameters()}
            ], lr=config.critic_lr)
            critic_desc = "critic_v GRU detached (Q gradient not updating GRU)"

        # Actor optimizer: includes Actor + Actor GRU
        self.actor_optimizer = torch.optim.Adam([
            {'params': self.belief.gru["actor"].parameters()},
            {'params': self.actor.parameters()}
        ], lr=config.actor_lr)

        logging.info(
            f"IQLAgent: {actor_desc}  |  gru_mode={config.gru_mode} ({len(beliefs_list)} GRUs)  |  "
            f"critic_opt: {critic_desc}  |  embed=[{self.item_embeddings.num_items},{self.item_embeddings.embedd_dim}] frozen"
        )

    def train(self, batch) -> Dict[str, float]:
        """
        Train one step with IQL loss (Three-step training)

        Step 1: Value Network Update (uses s_critic_v)
        Step 2: Critic Update (uses s_critic_q or s_critic_v based on gru_mode)
        Step 3: Actor Update (uses s_actor for policy, s_critic_v for advantage)
        """
        self.total_it += 1

        # Dual-Stream GRU forward
        # 🔥 GRU Architecture Ablation: 根据 gru_mode 获取状态表征
        states, next_states = self.belief.forward_batch(batch)
        s_actor = states["actor"]
        s_critic_v = states["critic_v"]  # V网络使用的状态表征
        ns_critic_v = next_states["critic_v"]

        # 根据 gru_mode 设置 Q网络使用的状态表征
        if self.config.gru_mode == "q_independent":
            s_critic_q = states["critic_q"]  # Q有独立GRU
            ns_critic_q = next_states["critic_q"]
        else:
            # qv_shared_detach 和 qv_shared_all_update: Q和V共用 critic_v
            s_critic_q = s_critic_v
            ns_critic_q = ns_critic_v

        # ========================================================================
        # 🔥 NEW: GRU Belief Diagnostics (Category 25)
        # 监控GRU表征质量：分布范围、时序动态、表征坍缩、跨模块差异
        # 目标：诊断训练坍塌是否由GRU表征问题引起
        # ========================================================================
        with torch.no_grad():
            ns_actor = next_states["actor"]

            # --- A. Belief State 分布范围 (训练时的状态空间覆盖) ---
            # Actor belief
            gru_actor_mean = s_actor.mean().item()
            gru_actor_std = s_actor.std().item()
            gru_actor_norm = torch.norm(s_actor, dim=-1)  # [batch] 每个样本的L2范数
            gru_actor_norm_mean = gru_actor_norm.mean().item()
            gru_actor_norm_std = gru_actor_norm.std().item()
            gru_actor_norm_min = gru_actor_norm.min().item()
            gru_actor_norm_max = gru_actor_norm.max().item()
            # Per-dimension std: 哪些维度有信号、哪些是死的
            gru_actor_per_dim_std = s_actor.std(dim=0)  # [hidden_dim]
            gru_actor_per_dim_std_mean = gru_actor_per_dim_std.mean().item()
            gru_actor_dead_dim_rate = (gru_actor_per_dim_std < 0.01).float().mean().item()
            gru_actor_active_dim_count = (gru_actor_per_dim_std > 0.05).float().sum().item()
            # 分位数覆盖
            gru_actor_p10 = torch.quantile(s_actor, 0.1).item()
            gru_actor_p90 = torch.quantile(s_actor, 0.9).item()
            gru_actor_coverage = gru_actor_p90 - gru_actor_p10  # P90-P10 覆盖宽度

            # Critic_v belief
            gru_cv_mean = s_critic_v.mean().item()
            gru_cv_std = s_critic_v.std().item()
            gru_cv_norm = torch.norm(s_critic_v, dim=-1)
            gru_cv_norm_mean = gru_cv_norm.mean().item()
            gru_cv_norm_std = gru_cv_norm.std().item()
            gru_cv_norm_min = gru_cv_norm.min().item()
            gru_cv_norm_max = gru_cv_norm.max().item()
            gru_cv_per_dim_std = s_critic_v.std(dim=0)
            gru_cv_dead_dim_rate = (gru_cv_per_dim_std < 0.01).float().mean().item()
            gru_cv_active_dim_count = (gru_cv_per_dim_std > 0.05).float().sum().item()
            gru_cv_p10 = torch.quantile(s_critic_v, 0.1).item()
            gru_cv_p90 = torch.quantile(s_critic_v, 0.9).item()
            gru_cv_coverage = gru_cv_p90 - gru_cv_p10

            # --- A2. Belief State 覆盖半径 (到centroid的平均L2距离) ---
            gru_actor_centroid = s_actor.mean(dim=0)  # [hidden_dim]
            gru_actor_radius = torch.norm(s_actor - gru_actor_centroid.unsqueeze(0), dim=-1).mean().item()

            gru_cv_centroid = s_critic_v.mean(dim=0)
            gru_cv_radius = torch.norm(s_critic_v - gru_cv_centroid.unsqueeze(0), dim=-1).mean().item()

            # --- B. 时序动态 (GRU是否在更新 / 是否遗忘) ---
            # ||s_{t+1} - s_t|| 衡量GRU每步更新的幅度
            gru_actor_transition = torch.norm(ns_actor - s_actor, dim=-1)
            gru_actor_transition_mean = gru_actor_transition.mean().item()
            gru_actor_transition_max = gru_actor_transition.max().item()
            gru_actor_transition_min = gru_actor_transition.min().item()

            gru_cv_transition = torch.norm(ns_critic_v - s_critic_v, dim=-1)
            gru_cv_transition_mean = gru_cv_transition.mean().item()
            gru_cv_transition_max = gru_cv_transition.max().item()
            gru_cv_transition_min = gru_cv_transition.min().item()

            # 零更新比例：transition ≈ 0 意味着GRU停止更新（遗忘门关闭）
            gru_actor_zero_transition_rate = (gru_actor_transition < 0.001).float().mean().item()
            gru_cv_stale_rate = (gru_cv_transition < 0.01).float().mean().item()

            # --- C. 表征坍缩检测 ---
            # Cosine相似度矩阵的均值：如果所有样本彼此相似，说明坍缩
            # 用采样子集避免O(N^2)计算
            sample_size = min(64, s_actor.size(0))
            s_actor_sample = s_actor[:sample_size]
            s_actor_normed = F.normalize(s_actor_sample, dim=-1)
            actor_cos_sim_matrix = torch.mm(s_actor_normed, s_actor_normed.t())
            gru_actor_collapse_score = actor_cos_sim_matrix.mean().item()  # 越接近1越坍缩

            s_cv_sample = s_critic_v[:sample_size]
            s_cv_normed = F.normalize(s_cv_sample, dim=-1)
            cv_cos_sim_matrix = torch.mm(s_cv_normed, s_cv_normed.t())
            gru_cv_collapse_score = cv_cos_sim_matrix.mean().item()

            # --- D. 跨模块差异 (Actor vs Critic belief一致性) ---
            # 逐样本cosine相似度
            gru_actor_cv_cos_sim = F.cosine_similarity(s_actor, s_critic_v, dim=-1)
            gru_actor_cv_cos_sim_mean = gru_actor_cv_cos_sim.mean().item()
            gru_actor_cv_cos_sim_min = gru_actor_cv_cos_sim.min().item()
            # 逐样本L2距离
            gru_actor_cv_l2_dist = torch.norm(s_actor - s_critic_v, dim=-1)
            gru_actor_cv_l2_dist_mean = gru_actor_cv_l2_dist.mean().item()
            gru_actor_cv_l2_dist_max = gru_actor_cv_l2_dist.max().item()

        # Real-time action inference for label actions (A/B via CLI)
        flat_slates = torch.cat(batch.obs["slate"], dim=0)
        flat_clicks = torch.cat(batch.obs["clicks"], dim=0)

        if self.config.label_click_mode == "real":
            label_clicks = flat_clicks.float()
        else:
            label_clicks = torch.zeros_like(flat_slates, dtype=torch.float32)

        with torch.no_grad():
            true_actions, _ = self.ranker.run_inference(flat_slates, label_clicks)
            true_actions = (true_actions - self.action_center) / self.action_scale

            # 🔥 SAFETY CLAMP: Architecture Alignment
            # Reason: GeMS outputs unbounded space (-∞, +∞), but Actor expects bounded space (-1, 1)
            # This clamp prevents NaN in atanh() and log(1 - x²) computations
            # Using 0.99 instead of 0.999 to leave a larger safety margin
            true_actions = torch.clamp(true_actions, min=-0.99, max=0.99)

        # ========================================================================
        # 🔥 CONVICTION METRICS: Quantify Out-of-Bounds Behavior
        # ========================================================================
        with torch.no_grad():
            # 1. Out-of-Bounds (OOB) Counter
            # Count how many normalized actions exceed [-1, 1] range
            oob_mask = (true_actions.abs() >= 1.0)
            oob_count = oob_mask.sum().item()
            oob_rate = oob_count / true_actions.numel()

            # 2. The "Atanh Test" - Direct Evidence of NaN Source
            # Compute both protected (current) and raw atanh to prove the issue
            action_ratio = true_actions / self.max_action

            # Protected version (what we currently use with clamping)
            action_ratio_protected = torch.clamp(action_ratio, -0.999, 0.999)
            atanh_protected = torch.atanh(action_ratio_protected)
            atanh_protected_has_nan = torch.isnan(atanh_protected).any().item()

            # Raw version (what would happen without clamping)
            # This will produce NaN if |action_ratio| > 1
            # 🔥 FIX: Compute domain violation rate BEFORE clamping for proper diagnosis
            atanh_domain_violation_rate = (action_ratio.abs() > 1.0).float().mean().item()
            atanh_raw = torch.atanh(action_ratio)  # True raw computation without clamp
            atanh_raw_has_nan = torch.isnan(atanh_raw).any().item()
            atanh_raw_has_inf = torch.isinf(atanh_raw).any().item()

            # 3. Extreme Values - Maximum Deviation
            true_action_abs_max = true_actions.abs().max().item()
            action_ratio_abs_max = action_ratio.abs().max().item()

        rewards = torch.cat(batch.rewards, dim=0) if batch.rewards else None
        dones = torch.cat(batch.dones, dim=0) if batch.dones else None

        # ========================================================================
        # Step 1: Value Network Update (Expectile Regression)
        # ========================================================================
        with torch.no_grad():
            # Compute target Q-values (使用 critic_v 的状态表征)
            target_q1, target_q2 = self.critic_1_target.both(s_critic_v, true_actions)
            target_q = torch.min(target_q1, target_q2)

        # Current V-value (keep gradient flow to GRU)
        current_v = self.value(s_critic_v)

        # 🔥 Numerical Stability: Clamp V values to prevent explosion
        # ⚠️ DISABLED for experiment: testing if clamp affects Q/V learning
        # current_v = torch.clamp(current_v, min=-100.0, max=100.0)

        # Expectile loss
        value_loss = expectile_loss(target_q - current_v, self.config.tau).mean()

        # 🔥 NEW: Expectile有效性检查 - V是否真的估计Q的高分位数？
        # 理论预期：P(V > Q) 应该 ≈ τ (默认0.8)
        # 如果远低于τ，说明V在"懒惰学习"，只跟踪Q均值而非高分位数
        with torch.no_grad():
            v_above_q_rate = (current_v > target_q).float().mean().item()
            expectile_match_error = abs(v_above_q_rate - self.config.tau)

            # V值分位数分布
            v_percentiles = torch.quantile(current_v, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=current_v.device))
            v_q10 = v_percentiles[0].item()
            v_q25 = v_percentiles[1].item()
            v_q50 = v_percentiles[2].item()
            v_q75 = v_percentiles[3].item()
            v_q90 = v_percentiles[4].item()

            # Q值分位数分布（用于对比V是否覆盖Q的高值区域）
            q_percentiles = torch.quantile(target_q, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=target_q.device))
            q_q10 = q_percentiles[0].item()
            q_q25 = q_percentiles[1].item()
            q_q50 = q_percentiles[2].item()
            q_q75 = q_percentiles[3].item()
            q_q90 = q_percentiles[4].item()

            # V-Q gap分位数（观察V是否追上Q）
            vq_gap = target_q - current_v
            vq_gap_percentiles = torch.quantile(vq_gap, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=vq_gap.device))
            vq_gap_q10 = vq_gap_percentiles[0].item()
            vq_gap_q25 = vq_gap_percentiles[1].item()
            vq_gap_q50 = vq_gap_percentiles[2].item()
            vq_gap_q75 = vq_gap_percentiles[3].item()
            vq_gap_q90 = vq_gap_percentiles[4].item()

        # Optimize value network
        # 🔥 GRU Architecture Ablation: qv_shared_all_update 模式需要特殊处理
        # 当 QV 共用 GRU 且 Q 梯度也更新时，不能在 Step 1 就 step()，否则 GRU 参数被修改
        # 导致 Step 2 的 backward() 报错 (in-place modification)
        self.value_optimizer.zero_grad()
        value_loss.backward(retain_graph=True)
        value_grad_norm = torch.nn.utils.clip_grad_norm_(
            list(self.belief.gru["critic_v"].parameters()) + list(self.value.parameters()),
            10.0
        )
        # 🔥 只有在 qv_shared_detach 或 q_independent 时才立即更新
        # qv_shared_all_update 模式下，等 Step 2 完成后再一起更新
        if self.config.gru_mode != "qv_shared_all_update":
            self.value_optimizer.step()

        # ========================================================================
        # Step 2: Critic Update (Standard Bellman Backup)
        # ========================================================================
        with torch.no_grad():
            # Next state value (使用 critic_v 的下一状态)
            next_v = self.value(ns_critic_v)
            # 🔥 Numerical Stability: Clamp next V values
            # ⚠️ DISABLED for experiment: testing if clamp affects Q/V learning
            # next_v = torch.clamp(next_v, min=-100.0, max=100.0)

            if rewards is not None and dones is not None:
                target_q = rewards + (1 - dones) * self.config.gamma * next_v
            else:
                target_q = next_v * self.config.gamma

            # 🔥 Numerical Stability: Clamp target Q values to prevent explosion
            # ⚠️ DISABLED for experiment: testing if clamp affects Q/V learning
            # target_q = torch.clamp(target_q, min=-100.0, max=100.0)

        # Current Q-values - 🔥 GRU Architecture Ablation
        # 根据 gru_mode 处理状态表征的梯度流
        if self.config.gru_mode == "q_independent":
            # Q有独立GRU，不detach，梯度传回 critic_q GRU
            current_q1, current_q2 = self.critic_1.both(s_critic_q, true_actions)
        elif self.config.gru_mode == "qv_shared_all_update":
            # QV共用 critic_v GRU，Q梯度也更新（不detach）
            current_q1, current_q2 = self.critic_1.both(s_critic_q, true_actions)
        else:  # qv_shared_detach (默认)
            # Q梯度detach，避免与V的梯度冲突
            current_q1, current_q2 = self.critic_1.both(s_critic_q.detach(), true_actions)

        # 🔥 Numerical Stability: Clamp current Q values
        # ⚠️ DISABLED for experiment: testing if clamp affects Q/V learning
        # current_q1 = torch.clamp(current_q1, min=-100.0, max=100.0)
        # current_q2 = torch.clamp(current_q2, min=-100.0, max=100.0)

        # Critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        # Optimize critics
        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        # 🔥 GRU Architecture Ablation: 根据 gru_mode 配置梯度裁剪范围
        if self.config.gru_mode == "q_independent":
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                list(self.belief.gru["critic_q"].parameters()) + list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
                10.0
            )
        elif self.config.gru_mode == "qv_shared_all_update":
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                list(self.belief.gru["critic_v"].parameters()) + list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
                10.0
            )
        else:  # qv_shared_detach
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
                10.0
            )
        self.critic_optimizer.step()

        # 🔥 GRU Architecture Ablation: qv_shared_all_update 模式下，Step 2 完成后更新 value_optimizer
        if self.config.gru_mode == "qv_shared_all_update":
            self.value_optimizer.step()

        # 🔥 NEW: TD Error分布和值学习动态诊断
        with torch.no_grad():
            # TD Error分位数
            td_error = current_q1 - target_q
            td_percentiles = torch.quantile(td_error.abs(), torch.tensor([0.1, 0.5, 0.9], device=td_error.device))
            td_error_q10 = td_percentiles[0].item()
            td_error_q50 = td_percentiles[1].item()
            td_error_q90 = td_percentiles[2].item()
            td_error_overestimate_rate = (current_q1 > target_q).float().mean().item()

            # Target Q分位数
            target_q_percentiles = torch.quantile(target_q, torch.tensor([0.1, 0.5, 0.9], device=target_q.device))
            target_q_q10 = target_q_percentiles[0].item()
            target_q_q50 = target_q_percentiles[1].item()
            target_q_q90 = target_q_percentiles[2].item()

            # Next V分位数
            next_v_percentiles = torch.quantile(next_v, torch.tensor([0.1, 0.5, 0.9], device=next_v.device))
            next_v_q10 = next_v_percentiles[0].item()
            next_v_q50 = next_v_percentiles[1].item()
            next_v_q90 = next_v_percentiles[2].item()

            # 值学习动态：V(s') vs V(s)
            # 如果 V(s') 系统性高于 V(s)，说明值函数在膨胀
            v_next_vs_current_ratio = (next_v.abs().mean() / (current_v.detach().abs().mean() + 1e-6)).item()
            value_explosion_rate = ((next_v.abs() > 2 * current_v.detach().abs()).float().mean().item())

            # V > target_q的比例（Bellman backup后的expectile检查）
            v_above_target_q_rate = (current_v.detach() > target_q).float().mean().item()

        # ========================================================================
        # Step 3: Actor Update (Advantage Weighted Regression)
        # ========================================================================

        # 🔍 FORENSIC MONITOR 0A: Input Health (GRU state)
        with torch.no_grad():
            s_actor_has_nan = torch.isnan(s_actor).any().item()
            s_actor_has_inf = torch.isinf(s_actor).any().item()
            s_actor_min = s_actor.min().item()
            s_actor_max = s_actor.max().item()
            s_actor_mean = s_actor.mean().item()

        # 🔍 FORENSIC MONITOR 0B: Target Health (GeMS output)
        with torch.no_grad():
            true_actions_has_nan = torch.isnan(true_actions).any().item()
            true_actions_has_inf = torch.isinf(true_actions).any().item()
            true_actions_min = true_actions.min().item()
            true_actions_max = true_actions.max().item()
            true_actions_mean = true_actions.mean().item()

        with torch.no_grad():
            # Compute advantage using s_critic_v (V网络的状态表征)
            # 🔥 GRU Architecture Ablation: advantage计算始终使用 critic_v
            v = self.value(s_critic_v.detach())
            q1, q2 = self.critic_1.both(s_critic_v.detach(), true_actions)
            q = torch.min(q1, q2)
            advantage = q - v

            # 🔍 FORENSIC MONITOR 1: Advantage Extremes (before clipping)
            advantage_max_raw = advantage.max().item()
            advantage_min_raw = advantage.min().item()

            # 🔥 NEW: Advantage分布形态诊断
            # 分位数分布：观察是否集中在0附近
            adv_percentiles = torch.quantile(advantage, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=advantage.device))
            adv_q10 = adv_percentiles[0].item()
            adv_q25 = adv_percentiles[1].item()
            adv_q50 = adv_percentiles[2].item()
            adv_q75 = adv_percentiles[3].item()
            adv_q90 = adv_percentiles[4].item()

            # Advantage形态指标
            adv_positive_rate = (advantage > 0).float().mean().item()
            adv_near_zero_rate = (advantage.abs() < 0.1).float().mean().item()  # threshold=0.1
            adv_effective_range = adv_q90 - adv_q10  # 分布宽度

            # Advantage偏度（三阶矩）
            adv_mean = advantage.mean()
            adv_std = advantage.std() + 1e-6
            adv_skewness = ((advantage - adv_mean) ** 3).mean() / (adv_std ** 3).item()

            # Compute weights (clamp before exp to prevent overflow)
            advantage_scaled = advantage * self.config.beta

            # 🔍 FORENSIC MONITOR 2: Weight Explosion (before clipping)
            weight_before_clip_max = advantage_scaled.max().item()
            weight_before_clip_min = advantage_scaled.min().item()

            advantage_clipped = torch.clamp(advantage_scaled, min=-5.0, max=5.0)
            exp_adv = torch.exp(advantage_clipped)

            # 🔍 FORENSIC MONITOR 3: Weight Explosion (after exp)
            weight_max = exp_adv.max().item()
            weight_mean = exp_adv.mean().item()

            # 🔥 NEW: AWR权重熵（判断权重是否均匀分布）
            # 高熵(>0.9) → 权重均匀，无区分力
            # 低熵(<0.5) → 权重集中，区分力强
            weights_normalized = exp_adv / (exp_adv.sum() + 1e-6)  # 归一化为概率分布
            weight_entropy = -(weights_normalized * torch.log(weights_normalized + 1e-6)).sum().item()
            max_entropy = torch.log(torch.tensor(exp_adv.shape[0], dtype=torch.float32, device=exp_adv.device)).item()  # 最大熵（均匀分布）
            weight_entropy_normalized = weight_entropy / (max_entropy + 1e-6)

        # FIX P2: align true_actions to the valid atanh domain [-0.999, 0.999]
        # The previous clamp to [-3.0, 3.0] was dead code (true_actions already
        # clamped to [-0.99, 0.99] above) and misaligned with networks.py safe_action
        # ⚠️ DISABLED for experiment: testing if clamp affects Q/V learning
        # true_actions_clamped = torch.clamp(true_actions, min=-0.999, max=0.999)
        # Instead, use true_actions directly (already clamped to [-0.99, 0.99] above)
        true_actions_clamped = true_actions  # Direct reference

        # 🔍 FORENSIC MONITOR 4: Policy Internal Diagnostics
        # Branch by actor_type to avoid AttributeError
        with torch.no_grad():
            if self.config.actor_type == "deterministic":
                # Deterministic actor: only has mu, no log_std
                hidden = self.actor.trunk(s_actor)
                actor_mu = self.actor.mu(hidden)

                # Actor mu statistics
                actor_mu_min = actor_mu.min().item()
                actor_mu_max = actor_mu.max().item()
                actor_mu_mean = actor_mu.mean().item()

                # ⚠️ Dummy values for log_std (deterministic has no variance)
                actor_log_std_raw_min = 0.0
                actor_log_std_raw_mean = 0.0
                actor_log_std_raw_max = 0.0
                actor_log_std_min = 0.0
                actor_log_std_mean = 0.0
                actor_log_std_max = 0.0
                actor_log_std_floor_hit_rate = 0.0
                actor_log_std_ceiling_hit_rate = 0.0

                # Jacobian term (still relevant for action clamping)
                action_ratio = true_actions_clamped / self.max_action
                action_ratio_squared = action_ratio.pow(2)
                jacobian_term = 1 - action_ratio_squared
                jacobian_term_min = jacobian_term.min().item()
                jacobian_term_max = jacobian_term.max().item()
                jacobian_term_has_negative = (jacobian_term <= 0).any().item()
                jacobian_term_has_nan = torch.isnan(jacobian_term).any().item()

                # Mu distribution analysis
                mu_after_tanh = torch.tanh(actor_mu)
                tanh_saturation_ratio = (mu_after_tanh.abs() > 0.95).float().mean().item()
                mu_percentiles = torch.quantile(actor_mu, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=actor_mu.device))
                mu_p10 = mu_percentiles[0].item()
                mu_p25 = mu_percentiles[1].item()
                mu_p50 = mu_percentiles[2].item()
                mu_p75 = mu_percentiles[3].item()
                mu_p90 = mu_percentiles[4].item()
                mu_iqr = mu_p75 - mu_p10
                policy_distance_to_origin = torch.norm(actor_mu, dim=-1).mean().item()

                # ⚠️ Dummy values for per-dimension log_std analysis
                max_per_dim_floor_hit = 0.0
                mean_per_dim_floor_hit = 0.0
                num_dims_with_high_floor_hit = 0

            elif self.config.actor_type == "fixed_gaussian":
                # Fixed Gaussian: mu is learnable, log_std is fixed buffer
                hidden = self.actor.trunk(s_actor)
                actor_mu = self.actor.mu(hidden)
                actor_log_std = self.actor.log_std  # Fixed buffer, not a function call

                # Actor mu statistics
                actor_mu_min = actor_mu.min().item()
                actor_mu_max = actor_mu.max().item()
                actor_mu_mean = actor_mu.mean().item()

                # Log_std statistics (fixed, so raw = clamped)
                actor_log_std_raw_min = actor_log_std.min().item()
                actor_log_std_raw_mean = actor_log_std.mean().item()
                actor_log_std_raw_max = actor_log_std.max().item()
                actor_log_std_min = actor_log_std.min().item()
                actor_log_std_mean = actor_log_std.mean().item()
                actor_log_std_max = actor_log_std.max().item()
                # Floor/ceiling hit rate = 0 (log_std is fixed, never hits bounds)
                actor_log_std_floor_hit_rate = 0.0
                actor_log_std_ceiling_hit_rate = 0.0

                # Jacobian term
                action_ratio = true_actions_clamped / self.max_action
                action_ratio_squared = action_ratio.pow(2)
                jacobian_term = 1 - action_ratio_squared
                jacobian_term_min = jacobian_term.min().item()
                jacobian_term_max = jacobian_term.max().item()
                jacobian_term_has_negative = (jacobian_term <= 0).any().item()
                jacobian_term_has_nan = torch.isnan(jacobian_term).any().item()

                # Mu distribution analysis
                mu_after_tanh = torch.tanh(actor_mu)
                tanh_saturation_ratio = (mu_after_tanh.abs() > 0.95).float().mean().item()
                mu_percentiles = torch.quantile(actor_mu, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=actor_mu.device))
                mu_p10 = mu_percentiles[0].item()
                mu_p25 = mu_percentiles[1].item()
                mu_p50 = mu_percentiles[2].item()
                mu_p75 = mu_percentiles[3].item()
                mu_p90 = mu_percentiles[4].item()
                mu_iqr = mu_p75 - mu_p10
                policy_distance_to_origin = torch.norm(actor_mu, dim=-1).mean().item()

                # ⚠️ Dummy values for per-dimension log_std analysis (fixed variance)
                max_per_dim_floor_hit = 0.0
                mean_per_dim_floor_hit = 0.0
                num_dims_with_high_floor_hit = 0

            else:
                # Gaussian actor: learnable mu and log_std
                hidden = self.actor.trunk(s_actor)
                actor_mu = self.actor.mu(hidden)
                actor_log_std_raw = self.actor.log_std(hidden)
                actor_log_std = torch.clamp(actor_log_std_raw, min=LOG_STD_MIN, max=LOG_STD_MAX)

                # Actor output statistics
                actor_mu_min = actor_mu.min().item()
                actor_mu_max = actor_mu.max().item()
                actor_mu_mean = actor_mu.mean().item()

                actor_log_std_raw_min = actor_log_std_raw.min().item()
                actor_log_std_raw_mean = actor_log_std_raw.mean().item()
                actor_log_std_raw_max = actor_log_std_raw.max().item()

                actor_log_std_min = actor_log_std.min().item()
                actor_log_std_mean = actor_log_std.mean().item()
                actor_log_std_max = actor_log_std.max().item()
                actor_log_std_floor_hit_rate = (actor_log_std_raw <= LOG_STD_MIN).float().mean().item()
                actor_log_std_ceiling_hit_rate = (actor_log_std_raw >= LOG_STD_MAX).float().mean().item()

                # 🔍 FORENSIC MONITOR 4B: Gaussian Intermediate Terms
                # Compute the problematic term: 1 - (action / max_action)^2
                action_ratio = true_actions_clamped / self.max_action
                action_ratio_squared = action_ratio.pow(2)
                jacobian_term = 1 - action_ratio_squared

                jacobian_term_min = jacobian_term.min().item()
                jacobian_term_max = jacobian_term.max().item()
                jacobian_term_has_negative = (jacobian_term <= 0).any().item()
                jacobian_term_has_nan = torch.isnan(jacobian_term).any().item()

                # 🔍 FORENSIC MONITOR 4C: Distribution Analysis (Gemini suggestion)
                # 1. Tanh Saturation - ratio of mu values pushed to boundaries
                mu_after_tanh = torch.tanh(actor_mu)
                tanh_saturation_ratio = (mu_after_tanh.abs() > 0.95).float().mean().item()

                # 2. Mu Distribution Percentiles (Q10, Q25, Q50, Q75, Q90)
                mu_percentiles = torch.quantile(actor_mu, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=actor_mu.device))
                mu_p10 = mu_percentiles[0].item()
                mu_p25 = mu_percentiles[1].item()
                mu_p50 = mu_percentiles[2].item()
                mu_p75 = mu_percentiles[3].item()
                mu_p90 = mu_percentiles[4].item()

                # 3. Per-dimension log_std analysis (32 dimensions)
                # Compute floor hit rate per dimension
                per_dim_floor_hit = (actor_log_std_raw <= LOG_STD_MIN).float().mean(dim=0)  # [32]
                per_dim_log_std_mean = actor_log_std.mean(dim=0)  # [32]

                # Summary stats for per-dimension analysis
                max_per_dim_floor_hit = per_dim_floor_hit.max().item()
                mean_per_dim_floor_hit = per_dim_floor_hit.mean().item()
                num_dims_with_high_floor_hit = (per_dim_floor_hit > 0.5).sum().item()  # How many dims have >50% floor hit

                # Mu interquartile range (IQR) - measure of distribution spread
                mu_iqr = mu_p75 - mu_p10

                # 🔥 NEW: Policy distance to origin (data mean) - for BC gravity evidence
                policy_distance_to_origin = torch.norm(actor_mu, dim=-1).mean().item()

        # 🔥 Actor Loss - branch by actor_type
        if self.config.actor_type == "deterministic":
            # Deterministic actor: use weighted MSE instead of log_prob
            actor_mu = self.actor.get_mu(s_actor)
            actor_action = torch.tanh(actor_mu) * self.max_action
            # Weighted MSE: exp_adv weights each sample
            mse_per_sample = F.mse_loss(actor_action, true_actions_clamped, reduction='none').mean(dim=-1, keepdim=True)
            awr_loss = (exp_adv * mse_per_sample).mean()

            # BC loss (same as before)
            bc_loss = F.mse_loss(actor_action, true_actions_clamped)

            # No log_prob for deterministic actor
            policy_entropy = 0.0
            log_prob_min_raw = 0.0
            log_prob_max_raw = 0.0
        else:
            # Gaussian actors (gaussian or fixed_gaussian): use log_prob
            log_prob = self.actor.log_prob(s_actor, true_actions_clamped)

            # 🔍 FORENSIC MONITOR 5: Log Probability Stability (before clipping)
            log_prob_min_raw = log_prob.min().item()
            log_prob_max_raw = log_prob.max().item()

            # NOTE: clamp is now inside networks.py log_prob() at -100 instead of -20
            # No additional clamp here - log_prob already has gradient-safe range

            # 🔥 NEW METRIC: Policy Entropy (监控策略是否坍缩)
            policy_entropy = -log_prob.mean().item()

            # AWR loss
            awr_loss = -(exp_adv * log_prob).mean()

            # FIX P0: BC Loss - provides stable gradient independent of Advantage
            # When AWR weights are near-uniform (advantage≈0), bc_loss keeps Actor
            # tracking the data distribution and prevents gradient death
            actor_mu, _ = self.actor(s_actor, deterministic=True)
            bc_loss = F.mse_loss(actor_mu, true_actions_clamped)

        lambda_bc = getattr(self.config, 'lambda_bc', 0.5)
        actor_loss = awr_loss + lambda_bc * bc_loss

        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(
            list(self.belief.gru["actor"].parameters()) + list(self.actor.parameters()),
            1.0  # FIX Phase 3: reduced from 10.0 — smaller steps prevent overshooting optimal policy
        )
        self.actor_optimizer.step()

        # Update target networks (use iql_tau for soft update, not expectile tau)
        soft_update(self.critic_1_target, self.critic_1, self.config.iql_tau)
        soft_update(self.critic_2_target, self.critic_2, self.config.iql_tau)

        # ========================================================================
        # 🔥 Enhanced Monitoring Metrics
        # ========================================================================
        # 🔥 OOD Distance Probe: Actor 输出与数据集真实动作的距离
        with torch.no_grad():
            # 确定性动作的 OOD 距离
            pred_actions_det, _ = self.actor(s_actor, deterministic=True)
            ood_distances_det = torch.norm(pred_actions_det - true_actions_clamped, dim=-1)
            ood_distance_mean_det = ood_distances_det.mean().item()
            ood_distance_max_det = ood_distances_det.max().item()

            # 采样动作的 OOD 距离（仅 Gaussian Actor）
            if self.config.actor_type in ["gaussian", "fixed_gaussian"]:
                pred_actions_samp, _ = self.actor(s_actor, deterministic=False)
                ood_distances_samp = torch.norm(pred_actions_samp - true_actions_clamped, dim=-1)
                ood_distance_mean_samp = ood_distances_samp.mean().item()
                ood_distance_max_samp = ood_distances_samp.max().item()
            else:
                ood_distance_mean_samp = 0.0
                ood_distance_max_samp = 0.0
            
            # 🔥 NEW: 潜空间全局定位探针 (到数据集中心/边界的距离)
            # 1. 到数据集中心的距离
            dist_to_center = torch.norm(pred_actions_det - self.dataset_center.to(pred_actions_det.device), dim=-1)
            z_to_dataset_center_mean = dist_to_center.mean().item()
            
            # 2. 到数据集边界的距离 (相对值，0=在中心，1=在边界)
            pred_normalized = (pred_actions_det - self.dataset_center.to(pred_actions_det.device)) / (self.action_range.to(pred_actions_det.device) / 2 + 1e-6)
            dist_to_boundary = torch.abs(pred_normalized).max(dim=-1).values
            z_to_boundary_mean = dist_to_boundary.mean().item()

            z_outer_ratio = z_to_dataset_center_mean / REF_L2_SPREAD_B
            ood_over_z = ood_distance_mean_det / (z_to_dataset_center_mean + 1e-8)

            if self.config.actor_type in ["gaussian", "fixed_gaussian"]:
                log_prob_true_mean = self.actor.log_prob(
                    s_actor, true_actions_clamped
                ).mean().item()
            else:
                log_prob_true_mean = (
                    -F.mse_loss(
                        pred_actions_det, true_actions_clamped, reduction="none"
                    )
                    .mean(dim=-1)
                    .mean()
                    .item()
                )

        # 🔥 采样时序监控：batch 平均 reward
        if batch.rewards is not None:
            batch_reward_mean = torch.cat(batch.rewards, dim=0).mean().item()
        else:
            batch_reward_mean = 0.0

        metrics = {
            # Loss metrics
            "value_loss": value_loss.item(),
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "awr_loss": awr_loss.item(),
            "bc_loss": bc_loss.item(),
            "bc_weighted": (lambda_bc * bc_loss).item(),

            # V-value statistics (enhanced)
            "v_value_mean": current_v.mean().item(),
            "v_value_min": current_v.min().item(),
            "v_value_max": current_v.max().item(),
            "v_value_std": current_v.std().item(),

            # Q-value statistics (enhanced)
            "q_value_mean": current_q1.mean().item(),
            "q_value_min": current_q1.min().item(),
            "q_value_max": current_q1.max().item(),
            "q_value_std": current_q1.std().item(),
            "target_q_mean": target_q.mean().item(),
            "target_q_min": target_q.min().item(),
            "target_q_max": target_q.max().item(),

            # TD error
            "td_error": (current_q1 - target_q).abs().mean().item(),

            # Advantage statistics (basic)
            "advantage_mean": advantage.mean().item(),
            "advantage_std": advantage.std().item(),

            # 🔍 FORENSIC LEVEL 0: Input & Target Health
            "s_actor_has_nan": s_actor_has_nan,
            "s_actor_has_inf": s_actor_has_inf,
            "s_actor_min": s_actor_min,
            "s_actor_max": s_actor_max,
            "s_actor_mean": s_actor_mean,
            "true_actions_has_nan": true_actions_has_nan,
            "true_actions_has_inf": true_actions_has_inf,
            "true_actions_min": true_actions_min,
            "true_actions_max": true_actions_max,
            "true_actions_mean": true_actions_mean,

            # 🔍 FORENSIC LEVEL 1: Advantage Extremes
            "advantage_max": advantage_max_raw,
            "advantage_min": advantage_min_raw,

            # 🔍 FORENSIC: Weight Explosion Monitor
            "weight_before_clip_max": weight_before_clip_max,
            "weight_before_clip_min": weight_before_clip_min,
            "weight_max": weight_max,
            "weight_mean": weight_mean,

            # 🔥 NEW METRICS: AWR Weight Distribution & Policy Entropy
            "awr_weight_std": exp_adv.std().item(),
            "policy_entropy": policy_entropy,

            # 🔍 FORENSIC: Policy Internal Diagnostics
            "actor_mu_min": actor_mu_min,
            "actor_mu_max": actor_mu_max,
            "actor_mu_mean": actor_mu_mean,
            # Distribution percentiles
            "mu_p10": mu_p10,
            "mu_p25": mu_p25,
            "mu_p50": mu_p50,
            "mu_p75": mu_p75,
            "mu_p90": mu_p90,
            "mu_iqr": mu_iqr,
            # Tanh saturation
            "tanh_saturation_ratio": tanh_saturation_ratio,
            # Per-dimension analysis
            "max_per_dim_floor_hit": max_per_dim_floor_hit,
            "mean_per_dim_floor_hit": mean_per_dim_floor_hit,
            "num_dims_with_high_floor_hit": num_dims_with_high_floor_hit,
            # Log std stats
            "actor_log_std_raw_min": actor_log_std_raw_min,
            "actor_log_std_raw_mean": actor_log_std_raw_mean,
            "actor_log_std_raw_max": actor_log_std_raw_max,
            "actor_log_std_min": actor_log_std_min,
            "actor_log_std_mean": actor_log_std_mean,
            "actor_log_std_max": actor_log_std_max,
            "actor_log_std_floor_hit_rate": actor_log_std_floor_hit_rate,
            "actor_log_std_ceiling_hit_rate": actor_log_std_ceiling_hit_rate,

            # 🔍 FORENSIC: Jacobian Term (1 - (action/max_action)^2)
            "jacobian_term_min": jacobian_term_min,
            "jacobian_term_max": jacobian_term_max,
            "jacobian_term_has_negative": jacobian_term_has_negative,
            "jacobian_term_has_nan": jacobian_term_has_nan,

            # 🔍 FORENSIC: Log Probability Stability
            "log_prob_min_raw": log_prob_min_raw,
            "log_prob_max_raw": log_prob_max_raw,

            # 🔥 CONVICTION METRICS: Quantitative Proof of OOB Problem
            "oob_count": oob_count,
            "oob_rate": oob_rate,
            "atanh_protected_has_nan": atanh_protected_has_nan,
            "atanh_raw_has_nan": atanh_raw_has_nan,
            "atanh_raw_has_inf": atanh_raw_has_inf,
            "atanh_domain_violation_rate": atanh_domain_violation_rate,
            "true_action_abs_max": true_action_abs_max,
            "action_ratio_abs_max": action_ratio_abs_max,

            # Gradient norms
            "value_grad_norm": value_grad_norm.item(),
            "critic_grad_norm": critic_grad_norm.item(),
            "actor_grad_norm": actor_grad_norm.item(),

            # 🔥 NEW: Policy distance to origin (BC gravity evidence)
            "policy_distance_to_origin": policy_distance_to_origin,

            # 🔥 OOD Distance Probe: Actor 输出与数据集真实动作的距离
            "ood_distance_mean_det": ood_distance_mean_det,
            "ood_distance_max_det": ood_distance_max_det,
            "ood_distance_mean_samp": ood_distance_mean_samp,
            "ood_distance_max_samp": ood_distance_max_samp,

            # 🔥 NEW: 潜空间全局定位探针
            "z_to_dataset_center_mean": z_to_dataset_center_mean,
            "z_to_boundary_mean": z_to_boundary_mean,
            "z_outer_ratio": z_outer_ratio,
            "ood_over_z": ood_over_z,
            "log_prob_true_mean": log_prob_true_mean,

            # 🔥 采样时序监控探针
            "batch_reward_mean": batch_reward_mean,

            # ======== 🔥 NEW: 值学习诊断指标 ========
            # [14] Expectile有效性检查
            "v_above_q_rate": v_above_q_rate,
            "v_above_target_q_rate": v_above_target_q_rate,
            "expectile_match_error": expectile_match_error,

            # [15] V值分位数
            "v_q10": v_q10,
            "v_q25": v_q25,
            "v_q50": v_q50,
            "v_q75": v_q75,
            "v_q90": v_q90,

            # [16] Q值分位数
            "q_q10": q_q10,
            "q_q25": q_q25,
            "q_q50": q_q50,
            "q_q75": q_q75,
            "q_q90": q_q90,

            # [17] Target Q分位数
            "target_q_q10": target_q_q10,
            "target_q_q50": target_q_q50,
            "target_q_q90": target_q_q90,

            # [18] Next V分位数
            "next_v_q10": next_v_q10,
            "next_v_q50": next_v_q50,
            "next_v_q90": next_v_q90,

            # [19] V-Q Gap分位数
            "vq_gap_q10": vq_gap_q10,
            "vq_gap_q25": vq_gap_q25,
            "vq_gap_q50": vq_gap_q50,
            "vq_gap_q75": vq_gap_q75,
            "vq_gap_q90": vq_gap_q90,

            # [20] Advantage分位数
            "adv_q10": adv_q10,
            "adv_q25": adv_q25,
            "adv_q50": adv_q50,
            "adv_q75": adv_q75,
            "adv_q90": adv_q90,

            # [21] Advantage形态
            "adv_positive_rate": adv_positive_rate,
            "adv_near_zero_rate": adv_near_zero_rate,
            "adv_effective_range": adv_effective_range,
            "adv_skewness": adv_skewness,

            # [22] AWR权重熵
            "weight_entropy": weight_entropy,
            "weight_entropy_normalized": weight_entropy_normalized,

            # [23] TD Error分位数
            "td_error_q10": td_error_q10,
            "td_error_q50": td_error_q50,
            "td_error_q90": td_error_q90,
            "td_error_overestimate_rate": td_error_overestimate_rate,

            # [24] 值学习动态
            "v_next_vs_current_ratio": v_next_vs_current_ratio,
            "value_explosion_rate": value_explosion_rate,

            # ======== 🔥 NEW: [25] GRU Belief Diagnostics ========
            # 状态空间范围（3个）
            "gru_belief_mean_norm": gru_actor_norm_mean,
            "gru_belief_std_mean": gru_actor_per_dim_std_mean,
            "gru_belief_radius": gru_actor_radius,
            # 时序动态（4个）
            "gru_transition_magnitude": gru_actor_transition_mean,
            "gru_transition_zero_rate": gru_actor_zero_transition_rate,
            "gru_dead_neuron_rate": gru_actor_dead_dim_rate,
            "gru_transition_max": gru_actor_transition_max,
            # Critic-V GRU（与 Q/V 同流；用于 train–eval 偏移对比）
            "gru_critic_v_mean_norm": gru_cv_norm_mean,
            "gru_critic_v_std_mean": gru_cv_per_dim_std.mean().item(),
            "gru_critic_v_radius": gru_cv_radius,
        }

        return metrics

    def compute_representation_diagnostics(self, batch) -> Dict[str, float]:
        """
        Compute representation diagnostics for monitoring training health.

        Returns:
            Dictionary with SVD rank and representation consistency metrics
        """
        # Forward pass to get states
        states, _ = self.belief.forward_batch(batch)
        s_actor = states["actor"]
        s_critic_v = states["critic_v"]  # 🔥 GRU Architecture Ablation

        # Compute SVD rank for both streams
        actor_rank, actor_condition = compute_svd_rank(s_actor)
        critic_rank, critic_condition = compute_svd_rank(s_critic_v)

        # Compute representation consistency (cosine similarity between actor and critic states)
        s_actor_flat = s_actor.view(-1)
        s_critic_v_flat = s_critic_v.view(-1)  # 🔥 GRU Architecture Ablation

        dot_product = torch.dot(s_actor_flat, s_critic_v_flat)
        norm_actor = torch.norm(s_actor_flat)
        norm_critic = torch.norm(s_critic_v_flat)

        representation_consistency = (dot_product / (norm_actor * norm_critic + 1e-8)).item()

        return {
            "actor_svd_rank": actor_rank,
            "actor_condition_number": actor_condition,
            "critic_svd_rank": critic_rank,
            "critic_condition_number": critic_condition,
            "representation_consistency": representation_consistency,
        }

    @torch.no_grad()
    def act(self, obs: Dict[str, Any], deterministic: bool = True) -> np.ndarray:
        """
        Select action using Actor GRU and decode to slate.

        Returns:
            slate: numpy array of shape [rec_size] containing item IDs
        """
        # 统一转为 Tensor (无 Batch 维度)
        slate = torch.as_tensor(obs["slate"], dtype=torch.long, device=self.device)
        clicks = torch.as_tensor(obs["clicks"], dtype=torch.long, device=self.device)

        # 构造输入 (不加 unsqueeze(0)!)
        obs_tensor = {"slate": slate, "clicks": clicks}

        # Dual-stream belief（eval 诊断需 actor + critic_v）
        belief_states = self.belief.forward(obs_tensor, done=False)
        belief_state = belief_states["actor"]

        # 🔥 NEW: 缓存 belief state 供 eval 诊断使用
        self._last_belief_state = belief_state.detach()
        if "critic_v" in belief_states:
            self._last_belief_state_critic_v = belief_states["critic_v"].detach()

        # Actor prediction
        raw_action, _ = self.actor(belief_state, deterministic=deterministic, need_log_prob=False)

        # Temperature scaling (实验: 把 Actor 输出拉近数据云)
        temperature = getattr(self, '_eval_temperature', 1.0)

        # Denormalize
        latent_action = raw_action * self.action_scale * temperature + self.action_center

        # 🔥 NEW: 使用 GeMS ranker 解码 latent action 为 slate
        if self.ranker is None:
            raise RuntimeError(
                "IQLAgent.act() requires a ranker for slate decoding. "
                "Please provide ranker during initialization."
            )

        # 🔧 FIX: 确保设备一致性 - 使用 ranker 的设备而非 agent 的设备
        # 原因：ranker 可能在不同设备上（CPU/CUDA/CUDA:0/CUDA:1）
        ranker_device = next(self.ranker.parameters()).device
        latent_action = latent_action.to(ranker_device)

        # 添加 batch 维度 (ranker 期望 [batch_size, latent_dim])
        latent_action_batched = latent_action.unsqueeze(0)  # [1, latent_dim]

        # 解码为 slate
        slate_tensor = self.ranker.rank(latent_action_batched)  # [1, rec_size]

        # 移除 batch 维度并转换为 numpy
        slate_output = slate_tensor.squeeze(0).cpu().numpy()  # [rec_size]

        return slate_output

    def reset_hidden(self):
        """Reset dual-stream GRU hidden states"""
        dummy_obs = {
            "slate": torch.zeros((1, self.config.rec_size), dtype=torch.long, device=self.device),
            "clicks": torch.zeros((1, self.config.rec_size), dtype=torch.long, device=self.device)
        }
        self.belief.forward(dummy_obs, done=True)

    def reset_eval_diag_buffers(self) -> None:
        """每次 evaluate_policy 开始前清空 eval 侧 Q/V/GRU 采样缓冲。"""
        self._eval_diag_buffers = {
            "belief_actor": [],
            "belief_critic_v": [],
            "v": [],
            "q": [],
            "adv": [],
        }

    @torch.no_grad()
    def collect_eval_step_metrics(self, obs: Dict[str, Any]) -> None:
        """
        在 eval 每步 act() 之后调用：缓存 belief 并在 (s_critic_v, 数据标签动作) 上算 V/Q/Adv。
        标签动作与训练一致（GeMS + label_click_mode + 归一化 + clamp）。
        """
        if self._eval_diag_buffers is None:
            self.reset_eval_diag_buffers()
        if self._last_belief_state is None or self._last_belief_state_critic_v is None:
            return
        if self.ranker is None:
            return

        self._eval_diag_buffers["belief_actor"].append(
            self._last_belief_state.cpu().clone()
        )
        self._eval_diag_buffers["belief_critic_v"].append(
            self._last_belief_state_critic_v.cpu().clone()
        )

        slate = torch.as_tensor(obs["slate"], dtype=torch.long, device=self.device)
        if slate.dim() == 0:
            slate = slate.unsqueeze(0)
        slate_b = slate.unsqueeze(0) if slate.dim() == 1 else slate

        if self.config.label_click_mode == "real":
            clicks = torch.as_tensor(obs["clicks"], dtype=torch.long, device=self.device)
            label_clicks = clicks.float()
            if label_clicks.dim() == 1:
                label_clicks = label_clicks.unsqueeze(0)
        else:
            label_clicks = torch.zeros_like(slate_b, dtype=torch.float32, device=self.device)

        true_actions, _ = self.ranker.run_inference(slate_b, label_clicks)
        true_actions = (true_actions - self.action_center) / self.action_scale
        true_actions = torch.clamp(true_actions, min=-0.99, max=0.99)

        s_cv = self._last_belief_state_critic_v
        if s_cv.dim() == 1:
            s_cv = s_cv.unsqueeze(0)

        v = self.value(s_cv).reshape(-1)
        q1, q2 = self.critic_1.both(s_cv, true_actions)
        q = torch.min(q1, q2).reshape(-1)
        adv = q - v

        self._eval_diag_buffers["v"].append(v.detach().cpu())
        self._eval_diag_buffers["q"].append(q.detach().cpu())
        self._eval_diag_buffers["adv"].append(adv.detach().cpu())

    @torch.no_grad()
    def summarize_eval_diag_buffers(self) -> Dict[str, float]:
        """聚合 eval 缓冲 → Category 26/27 所需的 eval_* 字段。"""
        empty = {
            "eval_belief_mean_norm": 0.0,
            "eval_belief_std_mean": 0.0,
            "eval_belief_radius": 0.0,
            "eval_critic_v_mean_norm": 0.0,
            "eval_critic_v_std_mean": 0.0,
            "eval_critic_v_radius": 0.0,
            "eval_v_mean": 0.0,
            "eval_v_std": 0.0,
            "eval_v_q10": 0.0,
            "eval_v_q50": 0.0,
            "eval_v_q90": 0.0,
            "eval_q_mean": 0.0,
            "eval_q_std": 0.0,
            "eval_q_q10": 0.0,
            "eval_q_q50": 0.0,
            "eval_q_q90": 0.0,
            "eval_adv_mean": 0.0,
            "eval_adv_std": 0.0,
            "eval_adv_q10": 0.0,
            "eval_adv_q50": 0.0,
            "eval_adv_q90": 0.0,
        }
        buf = self._eval_diag_buffers
        if buf is None or not buf["belief_actor"]:
            return empty

        actor_stats = _belief_cloud_stats(torch.stack(buf["belief_actor"], dim=0))
        cv_stats = _belief_cloud_stats(torch.stack(buf["belief_critic_v"], dim=0))
        v_stats = _distribution_stats_1d(torch.cat(buf["v"], dim=0).flatten())
        q_stats = _distribution_stats_1d(torch.cat(buf["q"], dim=0).flatten())
        adv_stats = _distribution_stats_1d(torch.cat(buf["adv"], dim=0).flatten())

        return {
            "eval_belief_mean_norm": actor_stats["mean_norm"],
            "eval_belief_std_mean": actor_stats["std_mean"],
            "eval_belief_radius": actor_stats["radius"],
            "eval_critic_v_mean_norm": cv_stats["mean_norm"],
            "eval_critic_v_std_mean": cv_stats["std_mean"],
            "eval_critic_v_radius": cv_stats["radius"],
            "eval_v_mean": v_stats["mean"],
            "eval_v_std": v_stats["std"],
            "eval_v_q10": v_stats["q10"],
            "eval_v_q50": v_stats["q50"],
            "eval_v_q90": v_stats["q90"],
            "eval_q_mean": q_stats["mean"],
            "eval_q_std": q_stats["std"],
            "eval_q_q10": q_stats["q10"],
            "eval_q_q50": q_stats["q50"],
            "eval_q_q90": q_stats["q90"],
            "eval_adv_mean": adv_stats["mean"],
            "eval_adv_std": adv_stats["std"],
            "eval_adv_q10": adv_stats["q10"],
            "eval_adv_q50": adv_stats["q50"],
            "eval_adv_q90": adv_stats["q90"],
        }

    def save(self, filepath: str):
        """Save model with embeddings metadata for standalone loading"""
        torch.save({
            'belief_state_dict': self.belief.state_dict(),
            'actor_state_dict': self.actor.state_dict(),
            'critic_1_state_dict': self.critic_1.state_dict(),
            'critic_2_state_dict': self.critic_2.state_dict(),
            'value_state_dict': self.value.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'action_center': self.action_center,
            'action_scale': self.action_scale,
            'total_it': self.total_it,
            'embeddings_meta': {
                'num_items': self.item_embeddings.num_items,
                'embedd_dim': self.item_embeddings.embedd_dim,
            },
            'action_dim': self.action_dim,
            'config': self.config,
        }, filepath)
        logging.info(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.belief.load_state_dict(checkpoint['belief_state_dict'])
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic_1.load_state_dict(checkpoint['critic_1_state_dict'])
        self.critic_2.load_state_dict(checkpoint['critic_2_state_dict'])
        self.value.load_state_dict(checkpoint['value_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.action_center = checkpoint['action_center']
        self.action_scale = checkpoint['action_scale']
        self.total_it = checkpoint['total_it']
        logging.info(f"Model loaded from {filepath}")

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, embedding_path: str, device: torch.device):
        """
        Load IQLAgent from checkpoint without requiring GeMS.

        Args:
            checkpoint_path: Path to saved agent checkpoint
            embedding_path: Path to item embeddings (.pt file)
            device: Device to load model on

        Returns:
            Loaded IQLAgent instance
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Extract metadata
        config = checkpoint['config']
        action_dim = checkpoint['action_dim']
        embeddings_meta = checkpoint['embeddings_meta']

        # Load embeddings
        embedding_weights = torch.load(embedding_path, map_location=device)
        item_embeddings = ItemEmbeddings(
            num_items=embeddings_meta['num_items'],
            item_embedd_dim=embeddings_meta['embedd_dim'],
            device=device,
            weights=embedding_weights
        )

        # Freeze embeddings
        for param in item_embeddings.parameters():
            param.requires_grad = False

        # Construct ranker_params
        ranker_params = {
            'item_embeddings': item_embeddings,
            'action_center': checkpoint['action_center'],
            'action_scale': checkpoint['action_scale'],
            'num_items': embeddings_meta['num_items'],
            'item_embedd_dim': embeddings_meta['embedd_dim'],
        }

        # Create agent
        agent = cls(action_dim=action_dim, config=config, ranker_params=ranker_params)

        # Load state dicts
        agent.belief.load_state_dict(checkpoint['belief_state_dict'])
        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        agent.critic_1.load_state_dict(checkpoint['critic_1_state_dict'])
        agent.critic_2.load_state_dict(checkpoint['critic_2_state_dict'])
        agent.value.load_state_dict(checkpoint['value_state_dict'])
        agent.total_it = checkpoint['total_it']

        logging.info(f"IQLAgent loaded from {checkpoint_path}")
        return agent

