#!/usr/bin/env python3
"""Offline Q/V/Advantage action-ranking probe for early10k checkpoints.

This script does not train. It loads existing IQL checkpoints and asks:
for the same replay states, which action source does the critic rank higher?
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import types
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))


def install_config_shim_if_needed() -> None:
    """Allow torch.load() to unpickle old checkpoints if root config.py is absent."""
    try:
        __import__("config")
        return
    except ModuleNotFoundError:
        pass

    module = types.ModuleType("config")

    class ExperimentConfig:
        @property
        def tau(self):
            return getattr(self, "expectile", 0.8)

        @property
        def dataset_path(self) -> Path:
            return (
                PROJECT_ROOT
                / "data/datasets/offline"
                / self.env_name
                / f"{self.env_name}_{self.dataset_quality}_data_d4rl.npz"
            )

        @property
        def gems_checkpoint_dir(self) -> Path:
            return PROJECT_ROOT / "checkpoints/gems"

        @property
        def agent_checkpoint_dir(self) -> Path:
            return PROJECT_ROOT / "checkpoints/agents" / self.experiment_name

    ExperimentConfig.__module__ = "config"
    module.ExperimentConfig = ExperimentConfig
    module.PROJECT_ROOT = PROJECT_ROOT
    sys.modules["config"] = module


install_config_shim_if_needed()

from src.agents.iql.agent import IQLAgent
from src.data.trajectory_buffer import TrajectoryReplayBuffer
from src.rankers.gems.embeddings import ItemEmbeddings
from src.rankers.gems.ranker import GeMS
from src.utils.checkpoint import load_gems_ranker
from src.utils.common import set_seed


RUNS = {
    "mix_b8": "kl001_mix_b8_ideal_init",
    "mix_b0": "kl001_mix_b0_ideal_init",
    "topdown_b8": "kl001_topdown_b8_ideal_init",
    "topdown_b0": "kl001_topdown_b0_ideal_init",
}

DEFAULT_STEPS = [50, 250, 1000, 5000, 10000]
EXP_GROUP = "early10k_validation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default=",".join(RUNS.keys()))
    parser.add_argument("--steps", default=",".join(map(str, DEFAULT_STEPS)))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-transitions", type=int, default=3000)
    parser.add_argument("--elite-pool", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=58407201)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "experiments/qv_adv_signal_diagnostic/outputs"),
    )
    return parser.parse_args()


def resolve_run(token: str) -> str:
    token = token.strip()
    return RUNS.get(token, token)


def checkpoint_path(run_label: str, step: int) -> Path:
    return PROJECT_ROOT / "checkpoints/agents" / EXP_GROUP / run_label / f"iql_step{step}.pt"


def config_to_dict(cfg) -> Dict:
    if is_dataclass(cfg):
        return asdict(cfg)
    if hasattr(cfg, "__dict__"):
        return dict(vars(cfg))
    return {}


def resolve_early10k_gems_checkpoint(cfg) -> Path:
    name = (
        f"GeMS_{cfg.env_name}_{cfg.dataset_quality}_pretrained"
        f"_latent{cfg.latent_dim}_beta{cfg.lambda_KL}"
        f"_click{cfg.lambda_click}_seed{cfg.seed}_{cfg.gems_embedding_mode}.ckpt"
    )
    path = PROJECT_ROOT / "checkpoints/gems" / name
    if not path.exists():
        raise FileNotFoundError(f"early10k GeMS checkpoint not found: {path}")
    return path


def load_gems_ranker_with_fallback(cfg, device: torch.device):
    try:
        return load_gems_ranker(
            env_name=cfg.env_name,
            dataset_quality=cfg.dataset_quality,
            gems_embedding_mode=cfg.gems_embedding_mode,
            device=device,
            item_embedd_dim=cfg.item_embedd_dim,
            rec_size=cfg.rec_size,
            latent_dim=cfg.latent_dim,
            lambda_KL=cfg.lambda_KL,
            seed=cfg.seed,
        )
    except FileNotFoundError:
        ckpt_path = resolve_early10k_gems_checkpoint(cfg)
        temp_emb = ItemEmbeddings.from_pretrained(
            str(PROJECT_ROOT / "data/embeddings/item_embeddings_diffuse.pt"),
            device,
        )
        ranker = GeMS.load_from_checkpoint(
            str(ckpt_path),
            map_location=device,
            item_embeddings=temp_emb,
            item_embedd_dim=cfg.item_embedd_dim,
            device=device,
            rec_size=cfg.rec_size,
            latent_dim=cfg.latent_dim,
            lambda_click=cfg.lambda_click,
            lambda_KL=cfg.lambda_KL,
            lambda_prior=1.0,
            ranker_lr=3e-3,
            fixed_embedds="scratch",
            ranker_sample=False,
            hidden_layers_infer=[512, 256],
            hidden_layers_decoder=[256, 512],
        )
        ranker.freeze()
        ranker = ranker.to(device)
        item_embeddings = ItemEmbeddings(
            num_items=ranker.num_items,
            item_embedd_dim=cfg.item_embedd_dim,
            device=device,
            weights=ranker.item_embeddings.weight.data.clone(),
        )
        action_dim, _ = ranker.get_action_dim()
        print(f"[fallback] loaded early10k GeMS checkpoint: {ckpt_path.name}")
        return ranker, action_dim, item_embeddings


def load_agent_and_data(ckpt: Path, device: torch.device):
    raw_ckpt = torch.load(str(ckpt), map_location=device)
    cfg = raw_ckpt["config"]
    cfg.device = str(device)
    cfg.use_swanlab = False

    ranker, action_dim, item_embeddings = load_gems_ranker_with_fallback(cfg, device)
    ranker_params = {
        "action_center": raw_ckpt["action_center"].to(device),
        "action_scale": raw_ckpt["action_scale"].to(device),
        "dataset_center": raw_ckpt.get("dataset_center", raw_ckpt["action_center"]).to(device),
        "action_range": raw_ckpt.get("action_range", raw_ckpt["action_scale"] * 2).to(device),
        "item_embeddings": item_embeddings,
    }

    agent = IQLAgent(action_dim=action_dim, config=cfg, ranker_params=ranker_params, ranker=ranker)
    agent.load(str(ckpt))
    for module in [agent.belief, agent.actor, agent.critic_1, agent.critic_2, agent.value]:
        module.eval()

    data = np.load(str(cfg.dataset_path), allow_pickle=True)
    data_dict = {key: data[key] for key in data.files}
    if "rewards" in data_dict:
        data_dict["rewards"] = data_dict["rewards"] / cfg.reward_scale

    buffer = TrajectoryReplayBuffer(device="cpu")
    buffer.load_d4rl_dataset(data_dict)
    return cfg, ranker, agent, buffer, data_dict


def encode_dataset_actions(ranker, cfg, slates: torch.Tensor, clicks: torch.Tensor, agent) -> torch.Tensor:
    if getattr(cfg, "label_click_mode", "fake_zero") == "real":
        label_clicks = clicks.float()
    else:
        label_clicks = torch.zeros_like(slates, dtype=torch.float32)
    actions, _ = ranker.run_inference(slates, label_clicks)
    actions = (actions - agent.action_center) / agent.action_scale
    return torch.clamp(actions, min=-0.99, max=0.99)


def build_elite_actions(ranker, cfg, data: Dict[str, np.ndarray], agent, n: int, elite_pool: int, device, seed):
    rewards = data.get("rewards")
    if rewards is None:
        raise ValueError("Dataset has no rewards; cannot build elite_data_action.")
    pool = min(elite_pool, len(rewards))
    top_idx = np.argpartition(rewards, -pool)[-pool:]
    rng = np.random.default_rng(seed)
    chosen = rng.choice(top_idx, size=n, replace=True)

    slates = torch.tensor(data["slates"][chosen], dtype=torch.long, device=device)
    if getattr(cfg, "label_click_mode", "fake_zero") == "real":
        clicks = torch.tensor(data["clicks"][chosen], dtype=torch.float32, device=device)
    else:
        clicks = torch.zeros_like(slates, dtype=torch.float32)
    return encode_dataset_actions(ranker, cfg, slates, clicks, agent)


def flatten_batch(batch, device: torch.device):
    slates = torch.cat(batch.obs["slate"], dim=0).to(device)
    clicks = torch.cat(batch.obs["clicks"], dim=0).to(device)
    return slates, clicks


def subsample(items: Dict[str, torch.Tensor], max_n: int, seed: int) -> Dict[str, torch.Tensor]:
    first = next(iter(items.values()))
    n = first.shape[0]
    if n <= max_n:
        return items
    gen = torch.Generator(device=first.device)
    gen.manual_seed(seed)
    idx = torch.randperm(n, generator=gen, device=first.device)[:max_n]
    return {key: val[idx] for key, val in items.items()}


def quantile_stats(x: torch.Tensor) -> Dict[str, float]:
    x = x.detach().flatten().float()
    qs = torch.quantile(x, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=x.device))
    return {
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "q10": float(qs[0].item()),
        "q25": float(qs[1].item()),
        "q50": float(qs[2].item()),
        "q75": float(qs[3].item()),
        "q90": float(qs[4].item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
    }


def actor_log_std_mean(agent, s_actor: torch.Tensor) -> float:
    actor = agent.actor
    if hasattr(actor, "log_std") and callable(actor.log_std):
        hidden = actor.trunk(s_actor)
        return float(torch.clamp(actor.log_std(hidden), min=-5.0, max=2.0).mean().item())
    if hasattr(actor, "log_std"):
        return float(actor.log_std.mean().item())
    return 0.0


def score_actions(agent, s_critic: torch.Tensor, v: torch.Tensor, action: torch.Tensor):
    q1, q2 = agent.critic_1.both(s_critic, action)
    q = torch.min(q1, q2).flatten()
    v = v.flatten()
    adv = q - v
    return {"q": q, "adv": adv}


def summarize_source(run_label: str, step: int, source: str, values, ref, v, data_q, log_std_mean: float, n: int):
    q = values["q"]
    adv = values["adv"]
    row = {
        "run": run_label,
        "step": step,
        "source": source,
        "n": n,
        "log_std_mean": log_std_mean,
        "adv_positive_rate": float((adv > 0).float().mean().item()),
        "adv_near_zero_rate": float((adv.abs() < 0.1).float().mean().item()),
        "adv_effective_range": float((torch.quantile(adv, 0.9) - torch.quantile(adv, 0.1)).item()),
        "q_gap_vs_data_mean": float((q - ref["data_q"]).mean().item()),
        "q_gap_vs_policy_mu_mean": float((q - ref["policy_mu_q"]).mean().item()),
        "adv_gap_vs_data_mean": float((adv - ref["data_adv"]).mean().item()),
        "v_mean": float(v.mean().item()),
        "v_std": float(v.std(unbiased=False).item()),
        "current_q_data_minus_v_q50": float(torch.quantile((data_q - v).flatten(), 0.5).item()),
        "current_q_data_minus_v_q90": float(torch.quantile((data_q - v).flatten(), 0.9).item()),
    }
    for prefix, tensor in [("q", q), ("adv", adv)]:
        for key, val in quantile_stats(tensor).items():
            row[f"{prefix}_{key}"] = val
    return row


def probe_checkpoint(run_label: str, step: int, args, device: torch.device) -> List[Dict]:
    ckpt = checkpoint_path(run_label, step)
    if not ckpt.exists():
        print(f"[skip] missing checkpoint: {ckpt}")
        return []

    cfg, ranker, agent, buffer, data = load_agent_and_data(ckpt, device)
    set_seed(args.seed + step)
    batch = buffer.sample(args.batch_size).to(device)

    with torch.no_grad():
        states, _ = agent.belief.forward_batch(batch)
        s_actor = states["actor"]
        s_critic = states["critic_q"] if getattr(cfg, "gru_mode", "") == "q_independent" else states["critic_v"]
        s_value = states["critic_v"]
        slates, clicks = flatten_batch(batch, device)

        packed = subsample(
            {"s_actor": s_actor, "s_critic": s_critic, "s_value": s_value, "slates": slates, "clicks": clicks},
            args.max_transitions,
            args.seed + step,
        )
        s_actor = packed["s_actor"]
        s_critic = packed["s_critic"]
        s_value = packed["s_value"]
        slates = packed["slates"]
        clicks = packed["clicks"]
        n = slates.shape[0]

        v = agent.value(s_value).flatten()
        data_action = encode_dataset_actions(ranker, cfg, slates, clicks, agent)
        policy_mu, _ = agent.actor(s_actor, deterministic=True, need_log_prob=False)
        policy_sample, _ = agent.actor(s_actor, deterministic=False, need_log_prob=False)
        random_latent = torch.empty_like(data_action).uniform_(-0.99, 0.99)
        shuffled_data_action = data_action[torch.randperm(n, device=device)]
        elite_action = build_elite_actions(
            ranker, cfg, data, agent, n=n, elite_pool=args.elite_pool, device=device, seed=args.seed + step
        )

        sources = {
            "data_action": data_action,
            "policy_mu": policy_mu,
            "policy_sample": policy_sample,
            "random_latent": random_latent,
            "shuffled_data_action": shuffled_data_action,
            "elite_data_action": elite_action,
        }
        scored = {name: score_actions(agent, s_critic, v, action) for name, action in sources.items()}
        ref = {
            "data_q": scored["data_action"]["q"],
            "data_adv": scored["data_action"]["adv"],
            "policy_mu_q": scored["policy_mu"]["q"],
        }
        log_std = actor_log_std_mean(agent, s_actor)
        rows = [
            summarize_source(run_label, step, source, values, ref, v, ref["data_q"], log_std, n)
            for source, values in scored.items()
        ]

    out_json = Path(args.output_dir) / "per_checkpoint" / run_label / f"step{step}_qv_action_ranking.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w") as f:
        json.dump({"run": run_label, "step": step, "config": config_to_dict(cfg), "rows": rows}, f, indent=2)
    print(f"[ok] {run_label} step{step}: n={n} -> {out_json}")
    return rows


def write_csv(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or "cuda" not in args.device else "cpu")
    runs = [resolve_run(x) for x in args.runs.split(",") if x.strip()]
    steps = [int(x) for x in args.steps.split(",") if x.strip()]
    rows: List[Dict] = []
    for run_label in runs:
        for step in steps:
            rows.extend(probe_checkpoint(run_label, step, args, device))
    out_csv = Path(args.output_dir) / "qv_action_ranking.csv"
    write_csv(rows, out_csv)
    print(f"[done] wrote {len(rows)} rows -> {out_csv}")


if __name__ == "__main__":
    main()
