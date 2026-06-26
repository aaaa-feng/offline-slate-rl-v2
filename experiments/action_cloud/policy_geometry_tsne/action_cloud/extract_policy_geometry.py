#!/usr/bin/env python3
"""
抽取策略几何数据：数据集 action cloud + 策略 eval latent/belief/slate。

Usage:
    python extract_policy_geometry.py \
        --env_name mix_divpen \
        --run mix_b0 \
        --checkpoint_tag best \
        --checkpoint checkpoints/agents/kl005_ideal_init/mix_b0/iql_best.pt \
        --gems_checkpoint checkpoints/gems/GeMS_mix_divpen_b5_pretrained_latent32_beta0.05_click1.0_seed58407201_ideal_init.ckpt \
        --episodes 50 \
        --dataset_samples 20000 \
        --device cuda
"""

import sys, os, json
from pathlib import Path
from argparse import ArgumentParser
from collections import Counter

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import ExperimentConfig
from src.rankers.gems.ranker import GeMS
from src.rankers.gems.embeddings import ItemEmbeddings
from src.env.simulators import TopicRec


def load_geMS(ckpt_path: str, device: torch.device):
    """加载 GeMS checkpoint"""
    temp = ItemEmbeddings.from_pretrained(
        str(PROJECT_ROOT / "data/embeddings/item_embeddings_diffuse.pt"), device)
    ranker = GeMS.load_from_checkpoint(ckpt_path, map_location=device, item_embeddings=temp,
        item_embedd_dim=20, device=device, rec_size=10, latent_dim=32,
        lambda_click=1.0, lambda_KL=1.0, lambda_prior=1.0, ranker_lr=3e-3,
        fixed_embedds="scratch", ranker_sample=False,
        hidden_layers_infer=[512, 256], hidden_layers_decoder=[256, 512])
    ranker.freeze()
    ranker = ranker.to(device)
    emb = ranker.item_embeddings.weight.data.clone()
    item_emb = ItemEmbeddings(ranker.num_items, 20, device=device, weights=emb)
    return ranker, item_emb


def build_dataset_frequency(env_name: str, dataset_quality: str):
    """构建 item 频率和 top-1000 combo 频率字典（与 eval_env.py 口径一致）。"""
    ds_path = PROJECT_ROOT / f"data/datasets/offline/{env_name}/{env_name}_{dataset_quality}_data_d4rl.npz"
    data = np.load(str(ds_path), allow_pickle=True)
    all_items = data['slates'].flatten()
    item_freq = Counter(all_items)
    item_total = sum(item_freq.values())

    combo_counter = Counter()
    n_slates = len(data['slates'])
    for slate in data['slates'][:min(100000, n_slates)]:
        combo_key = tuple(slate.tolist())
        combo_counter[combo_key] += 1
    combo_freq = dict(combo_counter.most_common(1000))

    return item_freq, item_total, combo_freq


def compute_per_slate_metrics(slates: np.ndarray, item_freq: Counter, item_total: int,
                               combo_freq: dict):
    """为每个 slate 计算 per-step 指标。

    Returns:
        combo_hit: [N] int32 (0/1)
        item_freq_pct_mean: [N] float32
    """
    N = len(slates)
    combo_hit = np.zeros(N, dtype=np.int32)
    item_freq_pct_mean = np.zeros(N, dtype=np.float32)
    for i in range(N):
        combo_key = tuple(slates[i].tolist())
        combo_hit[i] = 1 if combo_key in combo_freq else 0
        pcts = [item_freq.get(int(item_id), 0) / item_total * 100 for item_id in slates[i]]
        item_freq_pct_mean[i] = float(np.mean(pcts)) if pcts else 0.0
    return combo_hit, item_freq_pct_mean


def extract_dataset_cloud(ranker, env_name: str, dataset_quality: str,
                          device: torch.device, n_samples: int = 20000):
    """抽取数据集 action cloud"""
    ds_path = PROJECT_ROOT / f"data/datasets/offline/{env_name}/{env_name}_{dataset_quality}_data_d4rl.npz"
    data = np.load(str(ds_path), allow_pickle=True)
    N = min(n_samples, len(data['slates']))
    idx = np.random.choice(len(data['slates']), N, replace=False)
    slates = torch.tensor(data['slates'][idx], dtype=torch.long, device=device)
    clicks = torch.zeros_like(slates, dtype=torch.float32)  # fake_zero

    mu_list = []
    bs = 5000
    with torch.no_grad():
        for i in range(0, N, bs):
            end = min(i + bs, N)
            mu, _ = ranker.run_inference(slates[i:end], clicks[i:end])
            mu_list.append(mu.cpu())
    latent_mu = torch.cat(mu_list, dim=0)  # [N, 32]

    # Normalize
    a_min = latent_mu.min(dim=0)[0]
    a_max = latent_mu.max(dim=0)[0]
    center = (a_max + a_min) / 2
    scale = (a_max - a_min) / 2 + 1e-6
    ta = (latent_mu - center) / scale
    ta = torch.clamp(ta, min=-0.99, max=0.99)

    return latent_mu.numpy(), ta.numpy(), center.numpy(), scale.numpy()


def extract_policy_trajectory(agent, ranker, env_name: str, dataset_quality: str,
                              device: torch.device, n_episodes: int = 50,
                              item_freq=None, item_total=None, combo_freq=None):
    """在 RecSim 上跑 eval 并记录策略的中间状态"""
    cfg = ExperimentConfig(env_name=env_name, dataset_quality=dataset_quality, device=str(device))
    env_params = cfg.get_env_params()

    env = TopicRec(
        num_items=env_params["num_items"], rec_size=env_params["rec_size"],
        dataset_name="eval", sim_seed=cfg.seed + 9999, filename=None, device=device,
        env_embedds=str(PROJECT_ROOT / "data/embeddings" / env_params["env_embedds"]),
        click_model=env_params["click_model"],
        topic_size=env_params["topic_size"], num_topics=env_params["num_topics"],
        episode_length=env_params["episode_length"],
        env_alpha=1.0, env_propensities=[], click_only_once=False,
        rel_threshold=None, prop_threshold=None,
        diversity_penalty=env_params["diversity_penalty"],
        diversity_threshold=env_params.get("diversity_threshold", 4),
        env_offset=env_params.get("env_offset", 0.28),
        env_slope=env_params.get("env_slope", 100),
        env_omega=env_params.get("env_omega", 0.9),
        recent_items_maxlen=env_params.get("recent_items_maxlen", 10),
        short_term_boost=env_params.get("short_term_boost", 1.0),
        boredom_threshold=env_params["boredom_threshold"],
        boredom_moving_window=env_params.get("boredom_moving_window", 5),
    )

    latent_raw_list = []
    action_norm_list = []
    belief_list = []
    slate_list = []
    reward_list = []
    episode_ids = []

    for ep in range(n_episodes):
        obs, _info = env.reset()
        if hasattr(agent, 'reset_hidden'):
            agent.reset_hidden()

        ep_reward = 0.0
        step_count = 0
        while True:
            slate = torch.as_tensor(obs["slate"], dtype=torch.long, device=device)
            clicks = torch.as_tensor(obs["clicks"], dtype=torch.long, device=device)

            # 手动执行 agent.act() 的中间步骤以提取 latent/belief
            with torch.no_grad():
                belief_states = agent.belief.forward({"slate": slate, "clicks": clicks}, done=False)
                belief_actor = belief_states["actor"]

                raw_action, _ = agent.actor(belief_actor, deterministic=True, need_log_prob=False)
                latent_action = raw_action * agent.action_scale + agent.action_center

                # Decode
                ranker_device = next(ranker.parameters()).device
                latent_batched = latent_action.to(ranker_device).unsqueeze(0)
                slate_tensor = ranker.rank(latent_batched).squeeze(0)

            # Record
            latent_raw_list.append(latent_action.detach().cpu().numpy())
            action_norm_list.append(raw_action.detach().cpu().numpy())
            belief_list.append(belief_actor.detach().cpu().numpy())
            slate_list.append(slate_tensor.cpu().numpy())
            episode_ids.append(ep)

            # Step env
            obs, reward, done, info = env.step(slate_tensor, return_scores=False)
            reward_list.append(reward.item() if isinstance(reward, torch.Tensor) else reward)
            step_count += 1

            if done or step_count >= 100:
                break

    # Compute per-slate quality metrics if frequency dicts are available
    if item_freq is not None and combo_freq is not None:
        combo_hit, item_freq_pct = compute_per_slate_metrics(
            np.array(slate_list), item_freq, item_total, combo_freq)
    else:
        combo_hit = np.zeros(len(slate_list), dtype=np.int32)
        item_freq_pct = np.zeros(len(slate_list), dtype=np.float32)

    return {
        'latent_raw': np.array(latent_raw_list),
        'action_norm': np.array(action_norm_list),
        'belief_actor': np.array(belief_list),
        'slate': np.array(slate_list),
        'reward': np.array(reward_list),
        'episode_id': np.array(episode_ids),
        'combo_hit': combo_hit,
        'item_freq_pct_mean': item_freq_pct,
    }


def main():
    parser = ArgumentParser()
    parser.add_argument("--env_name", type=str, required=True)
    parser.add_argument("--run", type=str, required=True)
    parser.add_argument("--checkpoint_tag", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True, help="Agent IQL checkpoint path")
    parser.add_argument("--gems_checkpoint", type=str, required=True, help="GeMS checkpoint path")
    parser.add_argument("--dataset_quality", type=str, default="b5")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--dataset_samples", type=int, default=20000)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Env: {args.env_name}, Run: {args.run}, Tag: {args.checkpoint_tag}")

    # --- Load GeMS ---
    print("\n[1/5] Loading GeMS...")
    ranker, item_emb = load_geMS(args.gems_checkpoint, device)

    # --- Dataset cloud ---
    print(f"[2/5] Extracting dataset action cloud ({args.dataset_samples} samples)...")
    d_raw, d_norm, d_center, d_scale = extract_dataset_cloud(
        ranker, args.env_name, args.dataset_quality, device, args.dataset_samples)
    print(f"  raw shape: {d_raw.shape}, norm shape: {d_norm.shape}")

    # --- Build dataset frequency dicts for slate quality metrics ---
    print(f"[3/5] Building dataset frequency dicts...")
    item_freq, item_total, combo_freq = build_dataset_frequency(
        args.env_name, args.dataset_quality)
    print(f"  {len(item_freq)} unique items, {len(combo_freq)} top combos")

    # --- Load agent ---
    print(f"[4/5] Loading agent from {args.checkpoint}...")
    from src.agents.iql.agent import IQLAgent
    cfg = ExperimentConfig(env_name=args.env_name, dataset_quality=args.dataset_quality, device=str(device))

    # Compute action norm for ranker_params
    ds_path = cfg.dataset_path
    data = np.load(str(ds_path), allow_pickle=True)
    sample_size = min(10000, len(data['slates']))
    sample_idx = np.random.choice(len(data['slates']), sample_size, replace=False)
    sample_slates = torch.tensor(data['slates'][sample_idx], device=device, dtype=torch.long)
    sample_clicks = torch.zeros_like(sample_slates, dtype=torch.float32)
    with torch.no_grad():
        sample_actions, _ = ranker.run_inference(sample_slates, sample_clicks)
    a_min = sample_actions.min(dim=0)[0]
    a_max = sample_actions.max(dim=0)[0]
    a_center = (a_max + a_min) / 2
    a_scale = (a_max - a_min) / 2 + 1e-6
    a_range = a_max - a_min

    ranker_params = {
        'action_center': a_center,
        'action_scale': a_scale,
        'dataset_center': a_center.clone(),
        'action_range': a_range,
        'item_embeddings': item_emb,
    }

    agent = IQLAgent(action_dim=32, config=cfg, ranker_params=ranker_params, ranker=ranker)
    agent.load(args.checkpoint)

    # --- Eval policy ---
    print(f"[5/5] Running {args.episodes} eval episodes...")
    policy = extract_policy_trajectory(
        agent, ranker, args.env_name, args.dataset_quality, device, args.episodes,
        item_freq=item_freq, item_total=item_total, combo_freq=combo_freq)
    print(f"  steps: {len(policy['latent_raw'])}, "
          f"reward mean: {policy['reward'].mean():.2f}, "
          f"unique slates: {len(np.unique(policy['slate'], axis=0))}, "
          f"combo_hit: {policy['combo_hit'].mean():.2%}, "
          f"item_freq_pct: {policy['item_freq_pct_mean'].mean():.2f}")

    # --- Save ---
    out_dir = Path(__file__).resolve().parent.parent / "outputs"
    out_path = out_dir / f"{args.env_name}_{args.run}_{args.checkpoint_tag}_geometry.npz"
    np.savez(
        out_path,
        dataset_latent_raw=d_raw,
        dataset_latent_norm=d_norm,
        dataset_center=d_center,
        dataset_scale=d_scale,
        policy_latent_raw=policy['latent_raw'],
        policy_action_norm=policy['action_norm'],
        policy_belief_actor=policy['belief_actor'],
        policy_slate=policy['slate'],
        policy_reward=policy['reward'],
        policy_episode_id=policy['episode_id'],
        policy_combo_hit=policy['combo_hit'],
        policy_item_freq_pct_mean=policy['item_freq_pct_mean'],
        metadata=json.dumps({
            'env_name': args.env_name, 'run': args.run,
            'checkpoint_tag': args.checkpoint_tag,
            'checkpoint': args.checkpoint,
            'gems_checkpoint': args.gems_checkpoint,
            'episodes': args.episodes,
            'dataset_samples': args.dataset_samples,
        }),
    )
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
