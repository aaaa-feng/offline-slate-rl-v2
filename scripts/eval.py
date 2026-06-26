#!/usr/bin/env python3
"""
独立评估脚本（支持 Actor 温度缩放）。

加载训练好的 IQL/BC agent 和 GeMS ranker，在 RecSim 环境上评估。

Usage:
    python scripts/eval.py \
        --algo iql \
        --env_name mix_divpen \
        --dataset_quality b5 \
        --gems_embedding_mode ideal_init \
        --checkpoint checkpoints/agents/beta_ablation_repreduce/mix_b8/iql_final.pt \
        --temperature 0.5 \
        --episodes 100
"""

import sys
from pathlib import Path
from argparse import ArgumentParser
from collections import Counter

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import ExperimentConfig
from src.utils.common import set_seed
from src.utils.checkpoint import load_gems_ranker


def main():
    parser = ArgumentParser(description="Evaluate offline RL agent with temperature scaling")
    parser.add_argument("--algo", type=str, default="iql", choices=["iql", "bc"])
    parser.add_argument("--env_name", type=str, default="mix_divpen")
    parser.add_argument("--dataset_quality", type=str, default="b5")
    parser.add_argument("--gems_embedding_mode", type=str, default="ideal_init")
    parser.add_argument("--checkpoint", type=str, required=True, help="Agent checkpoint path")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Actor output temperature (< 1.0 拉近动作云)")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=58407201)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    cfg = ExperimentConfig(
        env_name=args.env_name,
        dataset_quality=args.dataset_quality,
        gems_embedding_mode=args.gems_embedding_mode,
        algo=args.algo,
        device=args.device,
        seed=args.seed,
    )

    print(f"Loading ranker ({args.gems_embedding_mode} GeMS)...")
    ranker, action_dim, item_embeddings = load_gems_ranker(
        env_name=cfg.env_name,
        dataset_quality=cfg.dataset_quality,
        gems_embedding_mode=cfg.gems_embedding_mode,
        device=device,
        seed=cfg.seed,
    )

    # Compute action norm for ranker_params (needed for agent init)
    ds_path = cfg.dataset_path
    data = np.load(str(ds_path), allow_pickle=True)
    sample_size = min(10000, len(data['slates']))
    sample_indices = np.random.choice(len(data['slates']), sample_size, replace=False)
    sample_slates = torch.tensor(data['slates'][sample_indices], device=device, dtype=torch.long)
    sample_clicks = torch.zeros_like(sample_slates, dtype=torch.float32)
    with torch.no_grad():
        sample_actions, _ = ranker.run_inference(sample_slates, sample_clicks)
    action_min = sample_actions.min(dim=0)[0]
    action_max = sample_actions.max(dim=0)[0]
    action_center = (action_max + action_min) / 2
    action_scale = (action_max - action_min) / 2 + 1e-6
    dataset_center = action_center.clone()
    action_range = action_max - action_min

    ranker_params = {
        'action_center': action_center,
        'action_scale': action_scale,
        'dataset_center': dataset_center,
        'action_range': action_range,
        'item_embeddings': item_embeddings,
    }

    print(f"Loading agent from {args.checkpoint}...")
    if args.algo == "iql":
        from src.agents.iql.agent import IQLAgent
        agent = IQLAgent(action_dim=action_dim, config=cfg, ranker_params=ranker_params, ranker=ranker)
        agent.load(args.checkpoint)
    elif args.algo == "bc":
        from src.agents.bc import BCAgent
        agent = BCAgent(action_dim=action_dim, config=cfg, ranker_params=ranker_params, ranker=ranker)
        agent.load(args.checkpoint)

    # Apply temperature
    agent._eval_temperature = args.temperature

    # Setup eval env
    from src.env.simulators import TopicRec
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

    print(f"Evaluating: T={args.temperature}, episodes={args.episodes}")
    print(f"  env={args.env_name}, checkpoint={Path(args.checkpoint).name}")

    episode_rewards = []
    episode_lengths = []
    global_item_counter = Counter()

    for ep in range(args.episodes):
        obs, _info = env.reset()
        if hasattr(agent, 'reset_hidden'):
            agent.reset_hidden()

        ep_reward = 0.0
        ep_len = 0

        while True:
            slate = agent.act(obs, deterministic=True)
            if isinstance(slate, np.ndarray):
                slate_tensor = torch.from_numpy(slate).long()
            else:
                slate_tensor = slate.long()

            for it in slate_tensor.cpu().tolist():
                global_item_counter[it] += 1

            obs, reward, done, info = env.step(slate_tensor, return_scores=False)
            ep_reward += reward.item() if isinstance(reward, torch.Tensor) else reward
            ep_len += 1

            if done or ep_len >= 100:
                break

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_len)

        if (ep + 1) % 20 == 0:
            rewards_sofar = np.array(episode_rewards[-20:])
            print(f"  ep {ep+1}/{args.episodes}: recent mean={rewards_sofar.mean():.1f} +/- {rewards_sofar.std():.1f}")

    rewards = np.array(episode_rewards)
    sorted_r = np.sort(rewards)
    n = len(sorted_r)
    q25_idx, q75_idx = n // 4, (3 * n) // 4
    iqm = sorted_r[q25_idx:q75_idx].mean() if q75_idx > q25_idx else sorted_r.mean()

    print()
    print("=" * 60)
    print(f"Evaluation Results (T={args.temperature})")
    print("=" * 60)
    print(f"  Mean Reward:     {rewards.mean():.2f} +/- {rewards.std():.2f}")
    print(f"  Median Reward:   {np.median(rewards):.2f}")
    print(f"  IQM Reward:      {iqm:.2f}")
    print(f"  Min / Max:       {rewards.min():.2f} / {rewards.max():.2f}")
    print(f"  Ep Length:       {np.mean(episode_lengths):.1f}")
    print(f"  Unique Items:    {len(global_item_counter)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
