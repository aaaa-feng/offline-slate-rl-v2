#!/usr/bin/env python3
"""Offline agent training entry point for offline-slate-rl-v2.

This script is intentionally small and explicit: it wires the v2 config,
GeMS ranker, trajectory replay buffer, agent implementation, online eval, and
checkpoint/timeline outputs used by the experiment launchers.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import ExperimentConfig
from src.agents.bc import BCAgent
from src.agents.iql.agent import IQLAgent
from src.data.trajectory_buffer import TrajectoryReplayBuffer
from src.env.simulators import TopicRec
from src.rankers.gems.embeddings import ItemEmbeddings
from src.rankers.gems.ranker import GeMS
from src.utils.checkpoint import load_gems_ranker
from src.utils.common import set_seed


TIMELINE_FIELDS = [
    "step",
    "det_iqm_reward",
    "det_mean_reward",
    "det_median_reward",
    "det_std_reward",
    "det_min_reward",
    "det_max_reward",
    "det_combo_hit",
    "det_global_unique",
    "det_global_unique_items",
    "det_item_freq_pct",
    "samp_iqm_reward",
    "samp_mean_reward",
    "samp_median_reward",
    "samp_std_reward",
    "samp_min_reward",
    "samp_max_reward",
    "samp_combo_hit",
    "samp_global_unique",
    "samp_global_unique_items",
    "samp_item_freq_pct",
    "log_std_mean",
    "log_std_q90",
    "log_std_floor_hit_rate",
    "ood_det",
    "ood_samp",
    "train_adv_q90",
    "train_adv_near_zero_rate",
    "train_actor_loss",
    "train_critic_loss",
    "train_value_loss",
]


def str_to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def comma_ints(value: Optional[str]) -> List[int]:
    if not value:
        return []
    return sorted({int(x.strip()) for x in value.split(",") if x.strip()})


def should_run_by_schedule(step: int, base_freq: int, schedule: Optional[str]) -> bool:
    """Return whether an eval/log event should run at this step.

    Schedule format follows existing launchers: "25:1000,100" means every 25
    steps through 1000, then every 100 steps afterwards.
    """
    if step <= 0:
        return False
    if not schedule:
        return base_freq > 0 and step % base_freq == 0

    segments = [s.strip() for s in schedule.split(",") if s.strip()]
    for segment in segments:
        if ":" in segment:
            freq_s, until_s = segment.split(":", 1)
            freq, until = int(freq_s), int(until_s)
            if step <= until:
                return freq > 0 and step % freq == 0
        else:
            freq = int(segment)
            return freq > 0 and step % freq == 0
    return base_freq > 0 and step % base_freq == 0


def create_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Train offline RL/BC agent")
    parser.add_argument("--config", type=str, default=None)

    # Config-overridable fields. Keep defaults as None so YAML values survive.
    for name, typ in [
        ("experiment_name", str),
        ("seed", int),
        ("device", str),
        ("env_name", str),
        ("dataset_quality", str),
        ("ranker_type", str),
        ("gems_embedding_mode", str),
        ("latent_dim", int),
        ("lambda_KL", float),
        ("lambda_click", float),
        ("algo", str),
        ("beta", float),
        ("expectile", float),
        ("lambda_bc", float),
        ("gamma", float),
        ("iql_tau", float),
        ("reward_scale", float),
        ("label_click_mode", str),
        ("best_checkpoint_metric", str),
        ("max_timesteps", int),
        ("batch_size", int),
        ("eval_freq", int),
        ("eval_episodes", int),
        ("final_eval_episodes", int),
        ("save_freq", int),
        ("log_freq", int),
        ("eval_timeline_dir", str),
        ("geometry_probe_episodes", int),
        ("geometry_probe_dataset_samples", int),
        ("geometry_probe_dir", str),
        ("hidden_dim", int),
        ("n_hidden", int),
        ("actor_lr", float),
        ("critic_lr", float),
        ("value_lr", float),
        ("belief_hidden_dim", int),
        ("item_embedd_dim", int),
        ("num_items", int),
        ("rec_size", int),
        ("gru_mode", str),
        ("actor_type", str),
        ("fixed_std", float),
        ("swan_project", str),
        ("swan_workspace", str),
        ("swan_mode", str),
        ("swan_logdir", str),
    ]:
        parser.add_argument(f"--{name}", type=typ, default=None)

    for name in [
        "dual_eval",
        "enable_train_geometry_probe",
        "geometry_probe_save_samples",
        "use_swanlab",
        "eval_step_zero",
    ]:
        parser.add_argument(f"--{name}", type=str_to_bool, default=None)

    # Launcher-only compatibility options.
    parser.add_argument("--experiment_tag", type=str, default=None)
    parser.add_argument("--gems_checkpoint", type=str, default=None)
    parser.add_argument("--save_steps", type=str, default=None)
    parser.add_argument("--eval_freq_schedule", type=str, default=None)
    parser.add_argument("--log_freq_schedule", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--progress_bar", action="store_true")
    return parser


def load_config(args) -> ExperimentConfig:
    cfg = ExperimentConfig.from_yaml(args.config) if args.config else ExperimentConfig()
    for key, value in vars(args).items():
        if key == "config" or value is None:
            continue
        if hasattr(cfg, key):
            setattr(cfg, key, value)

    # Keep launcher-only values on the config for checkpoint metadata.
    for key in (
        "experiment_tag",
        "gems_checkpoint",
        "save_steps",
        "eval_freq_schedule",
        "log_freq_schedule",
        "eval_step_zero",
    ):
        setattr(cfg, key, getattr(args, key, None))
    return cfg


def setup_logging(cfg: ExperimentConfig) -> None:
    log_dir = PROJECT_ROOT / "logs/agents" / cfg.experiment_name.split("/")[0]
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


class SwanLogger:
    def __init__(self, cfg: ExperimentConfig):
        self.run = None
        if not getattr(cfg, "use_swanlab", True):
            return
        try:
            import swanlab

            exp_name = getattr(cfg, "experiment_tag", None) or cfg.experiment_name
            self.run = swanlab.init(
                project=cfg.swan_project,
                workspace=cfg.swan_workspace,
                experiment_name=exp_name,
                config=cfg.__dict__,
                mode=cfg.swan_mode,
                logdir=str(PROJECT_ROOT / cfg.swan_logdir),
            )
        except Exception as exc:  # pragma: no cover - logging fallback
            logging.warning("SwanLab disabled: %s", exc)
            self.run = None

    def log(self, metrics: Dict[str, Any], step: int) -> None:
        if self.run is None:
            return
        try:
            import swanlab

            swanlab.log(metrics, step=step)
        except Exception as exc:  # pragma: no cover - logging fallback
            logging.warning("SwanLab log failed: %s", exc)

    def finish(self) -> None:
        if self.run is None:
            return
        try:
            import swanlab

            swanlab.finish()
        except Exception:
            pass


def load_ranker_from_path(
    ckpt_path: Path, cfg: ExperimentConfig, device: torch.device
) -> Tuple[GeMS, int, ItemEmbeddings]:
    temp_embeddings = ItemEmbeddings.from_pretrained(str(cfg.item_embedds_path), device)
    ranker = GeMS.load_from_checkpoint(
        str(ckpt_path),
        map_location=device,
        item_embeddings=temp_embeddings,
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

    weights = ranker.item_embeddings.weight.data.clone()
    item_embeddings = ItemEmbeddings(
        num_items=ranker.num_items,
        item_embedd_dim=cfg.item_embedd_dim,
        device=device,
        weights=weights,
    )
    action_dim, _ = ranker.get_action_dim()
    return ranker, action_dim, item_embeddings


def discover_gems_checkpoint(cfg: ExperimentConfig) -> Optional[Path]:
    explicit = getattr(cfg, "gems_checkpoint", None)
    if explicit:
        path = Path(explicit)
        return path if path.is_absolute() else PROJECT_ROOT / path

    candidates = [
        cfg.gems_checkpoint_path,
        PROJECT_ROOT
        / "checkpoints/gems"
        / (
            f"GeMS_{cfg.env_name}_{cfg.dataset_quality}_pretrained"
            f"_latent{cfg.latent_dim}_beta{cfg.lambda_KL}"
            f"_click{cfg.lambda_click}_seed{cfg.seed}_{cfg.gems_embedding_mode}.ckpt"
        ),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def load_ranker(cfg: ExperimentConfig, device: torch.device):
    ckpt = discover_gems_checkpoint(cfg)
    if ckpt is not None:
        logging.info("Loading GeMS checkpoint: %s", ckpt)
        return load_ranker_from_path(ckpt, cfg, device)

    logging.info("Loading GeMS checkpoint via resolver")
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


def load_dataset(cfg: ExperimentConfig) -> Dict[str, np.ndarray]:
    if not cfg.dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {cfg.dataset_path}")
    data = np.load(str(cfg.dataset_path), allow_pickle=True)
    return {key: data[key] for key in data.files}


def build_frequency_stats(data: Dict[str, np.ndarray]) -> Tuple[Counter, int, Dict[Tuple[int, ...], int]]:
    item_freq = Counter(data["slates"].reshape(-1).astype(int).tolist())
    item_total = int(sum(item_freq.values()))
    if "combo_freq_keys" in data and "combo_freq_vals" in data:
        combo_freq = {
            tuple(int(x) for x in key): int(val)
            for key, val in zip(data["combo_freq_keys"], data["combo_freq_vals"])
        }
    else:
        combo_counter = Counter(tuple(int(x) for x in slate) for slate in data["slates"])
        combo_freq = dict(combo_counter.most_common(1000))
    return item_freq, item_total, combo_freq


def compute_action_norm(
    cfg: ExperimentConfig,
    ranker: GeMS,
    data: Dict[str, np.ndarray],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    sample_size = min(int(getattr(cfg, "geometry_probe_dataset_samples", 20_000)), len(data["slates"]))
    rng = np.random.default_rng(cfg.seed)
    idx = rng.choice(len(data["slates"]), sample_size, replace=False)

    slates = torch.tensor(data["slates"][idx], dtype=torch.long, device=device)
    if cfg.label_click_mode == "real":
        clicks = torch.tensor(data["clicks"][idx], dtype=torch.float32, device=device)
    else:
        clicks = torch.zeros_like(slates, dtype=torch.float32, device=device)

    parts = []
    batch_size = 5000
    with torch.no_grad():
        for start in range(0, sample_size, batch_size):
            end = min(start + batch_size, sample_size)
            mu, _ = ranker.run_inference(slates[start:end], clicks[start:end])
            parts.append(mu.detach())
    actions = torch.cat(parts, dim=0)

    action_min = actions.min(dim=0)[0]
    action_max = actions.max(dim=0)[0]
    action_center = (action_max + action_min) / 2
    action_scale = (action_max - action_min) / 2 + 1e-6
    return {
        "action_center": action_center,
        "action_scale": action_scale,
        "dataset_center": actions.mean(dim=0),
        "action_range": action_max - action_min,
    }


def create_env(cfg: ExperimentConfig, device: torch.device, seed_offset: int) -> TopicRec:
    env_params = cfg.get_env_params()
    return TopicRec(
        num_items=env_params["num_items"],
        rec_size=env_params["rec_size"],
        dataset_name="eval",
        sim_seed=cfg.seed + seed_offset,
        filename=None,
        device=device,
        env_embedds=str(PROJECT_ROOT / "data/embeddings" / env_params["env_embedds"]),
        click_model=env_params["click_model"],
        topic_size=env_params["topic_size"],
        num_topics=env_params["num_topics"],
        episode_length=env_params["episode_length"],
        env_alpha=1.0,
        env_propensities=[],
        click_only_once=False,
        rel_threshold=None,
        prop_threshold=None,
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


def iqm(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    ordered = np.sort(values)
    lo, hi = len(ordered) // 4, (3 * len(ordered)) // 4
    return float(ordered[lo:hi].mean()) if hi > lo else float(ordered.mean())


def evaluate_policy(
    agent,
    env: TopicRec,
    episodes: int,
    deterministic: bool,
    item_freq: Counter,
    item_total: int,
    combo_freq: Dict[Tuple[int, ...], int],
) -> Dict[str, float]:
    if hasattr(agent, "reset_eval_diag_buffers"):
        agent.reset_eval_diag_buffers()

    rewards = []
    lengths = []
    slate_counter = Counter()
    item_counter = Counter()
    combo_hits = []
    item_pcts = []

    for _ in range(episodes):
        obs, _info = env.reset()
        if hasattr(agent, "reset_hidden"):
            agent.reset_hidden()

        ep_reward = 0.0
        ep_len = 0
        done = False
        while not done:
            slate_np = agent.act(obs, deterministic=deterministic)
            if hasattr(agent, "collect_eval_step_metrics"):
                agent.collect_eval_step_metrics(obs)

            slate_tuple = tuple(int(x) for x in np.asarray(slate_np).tolist())
            slate_counter[slate_tuple] += 1
            combo_hits.append(1.0 if slate_tuple in combo_freq else 0.0)
            item_pcts.append(
                float(np.mean([item_freq.get(x, 0) / max(item_total, 1) * 100 for x in slate_tuple]))
            )
            item_counter.update(slate_tuple)

            slate_tensor = torch.tensor(slate_tuple, dtype=torch.long, device=env.device)
            obs, reward, done, _info = env.step(slate_tensor)
            ep_reward += float(reward.item() if isinstance(reward, torch.Tensor) else reward)
            ep_len += 1

        rewards.append(ep_reward)
        lengths.append(ep_len)

    r = np.asarray(rewards, dtype=np.float32)
    metrics = {
        "mean_reward": float(r.mean()) if len(r) else 0.0,
        "std_reward": float(r.std()) if len(r) else 0.0,
        "median_reward": float(np.median(r)) if len(r) else 0.0,
        "iqm_reward": iqm(r),
        "min_reward": float(r.min()) if len(r) else 0.0,
        "max_reward": float(r.max()) if len(r) else 0.0,
        "mean_episode_length": float(np.mean(lengths)) if lengths else 0.0,
        "combo_hit": float(np.mean(combo_hits)) if combo_hits else 0.0,
        "global_unique": float(len(slate_counter)),
        "global_unique_items": float(len(item_counter)),
        "item_freq_percentile_mean": float(np.mean(item_pcts)) if item_pcts else 0.0,
    }
    if hasattr(agent, "summarize_eval_diag_buffers"):
        metrics.update(agent.summarize_eval_diag_buffers())
    return metrics


def add_bc_actions_to_batch(batch, cfg: ExperimentConfig, ranker: GeMS, ranker_params: Dict[str, torch.Tensor]):
    flat_slates = torch.cat(batch.obs["slate"], dim=0)
    flat_clicks = torch.cat(batch.obs["clicks"], dim=0)
    if cfg.label_click_mode == "real":
        label_clicks = flat_clicks.float()
    else:
        label_clicks = torch.zeros_like(flat_slates, dtype=torch.float32)
    with torch.no_grad():
        actions, _ = ranker.run_inference(flat_slates, label_clicks)
        actions = (actions - ranker_params["action_center"]) / ranker_params["action_scale"]
        actions = torch.clamp(actions, -0.99, 0.99)
    batch.obs["action"] = [actions]
    return batch


def init_agent(cfg: ExperimentConfig, action_dim: int, ranker_params: Dict[str, Any], ranker: GeMS):
    if cfg.algo == "iql":
        return IQLAgent(action_dim=action_dim, config=cfg, ranker_params=ranker_params, ranker=ranker)
    if cfg.algo == "bc":
        return BCAgent(action_dim=action_dim, config=cfg, ranker_params=ranker_params, ranker=ranker)
    raise ValueError(f"Unsupported algo for this entry point: {cfg.algo}")


def write_timeline_row(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TIMELINE_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in TIMELINE_FIELDS})


def flatten_eval(prefix: str, metrics: Dict[str, float]) -> Dict[str, float]:
    return {
        f"{prefix}_iqm_reward": metrics["iqm_reward"],
        f"{prefix}_mean_reward": metrics["mean_reward"],
        f"{prefix}_median_reward": metrics["median_reward"],
        f"{prefix}_std_reward": metrics["std_reward"],
        f"{prefix}_min_reward": metrics["min_reward"],
        f"{prefix}_max_reward": metrics["max_reward"],
        f"{prefix}_combo_hit": metrics["combo_hit"],
        f"{prefix}_global_unique": metrics["global_unique"],
        f"{prefix}_global_unique_items": metrics["global_unique_items"],
        f"{prefix}_item_freq_pct": metrics["item_freq_percentile_mean"],
    }


def build_timeline_row(
    step: int,
    det_m: Dict[str, float],
    samp_m: Dict[str, float],
    train_m: Dict[str, float],
) -> Dict[str, Any]:
    row = {"step": step}
    row.update(flatten_eval("det", det_m))
    row.update(flatten_eval("samp", samp_m))
    row.update(
        {
            "log_std_mean": train_m.get("actor_log_std_mean", ""),
            "log_std_q90": train_m.get("actor_log_std_max", ""),
            "log_std_floor_hit_rate": train_m.get("actor_log_std_floor_hit_rate", ""),
            "ood_det": det_m.get("eval_adv_q90", ""),
            "ood_samp": samp_m.get("eval_adv_q90", ""),
            "train_adv_q90": train_m.get("adv_q90", ""),
            "train_adv_near_zero_rate": train_m.get("adv_near_zero_rate", ""),
            "train_actor_loss": train_m.get("actor_loss", ""),
            "train_critic_loss": train_m.get("critic_loss", ""),
            "train_value_loss": train_m.get("value_loss", ""),
        }
    )
    return row


def save_probe_json(
    cfg: ExperimentConfig,
    step: int,
    det_m: Dict[str, float],
    samp_m: Dict[str, float],
    train_m: Dict[str, float],
) -> None:
    if not getattr(cfg, "enable_train_geometry_probe", False):
        return
    out_dir = cfg.geometry_probe_output_dir / f"step_{step:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "config": cfg.__dict__,
        "eval_metrics": {"det": det_m, "samp": samp_m},
        "train_metrics": train_m,
    }
    with (out_dir / "eval_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload["eval_metrics"], f, indent=2, sort_keys=True)
    with (out_dir / "train_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(train_m, f, indent=2, sort_keys=True)
    with (out_dir / "policy_geometry_summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)


def save_checkpoint(agent, cfg: ExperimentConfig, name: str) -> Path:
    ckpt_dir = cfg.agent_checkpoint_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / name
    agent.save(str(path))
    return path


def run_eval_block(
    cfg: ExperimentConfig,
    agent,
    step: int,
    train_metrics: Dict[str, float],
    timeline_path: Path,
    item_freq: Counter,
    item_total: int,
    combo_freq: Dict[Tuple[int, ...], int],
    logger: SwanLogger,
    final: bool = False,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    episodes = cfg.final_eval_episodes if final else cfg.eval_episodes
    device = torch.device(cfg.device)

    det_env = create_env(cfg, device, seed_offset=10_000 + step)
    det_m = evaluate_policy(agent, det_env, episodes, True, item_freq, item_total, combo_freq)

    if cfg.dual_eval:
        samp_env = create_env(cfg, device, seed_offset=20_000 + step)
        samp_m = evaluate_policy(agent, samp_env, episodes, False, item_freq, item_total, combo_freq)
    else:
        samp_m = dict(det_m)

    row = build_timeline_row(step, det_m, samp_m, train_metrics)
    if cfg.eval_timeline_dir:
        write_timeline_row(timeline_path, row)
    save_probe_json(cfg, step, det_m, samp_m, train_metrics)

    swan_metrics = {f"eval/{k}": v for k, v in row.items() if k != "step" and v != ""}
    logger.log(swan_metrics, step)

    logging.info(
        "eval step=%s det_iqm=%.2f samp_iqm=%.2f det_unique=%s samp_unique=%s",
        step,
        det_m["iqm_reward"],
        samp_m["iqm_reward"],
        int(det_m["global_unique"]),
        int(samp_m["global_unique"]),
    )
    return det_m, samp_m


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    cfg = load_config(args)
    setup_logging(cfg)
    set_seed(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    cfg.device = str(device)

    logging.info("experiment=%s algo=%s env=%s beta=%s", cfg.experiment_name, cfg.algo, cfg.env_name, cfg.beta)
    data = load_dataset(cfg)
    item_freq, item_total, combo_freq = build_frequency_stats(data)

    ranker, action_dim, item_embeddings = load_ranker(cfg, device)
    norm_params = compute_action_norm(cfg, ranker, data, device)
    ranker_params = {**norm_params, "item_embeddings": item_embeddings}

    buffer = TrajectoryReplayBuffer(device=str(device))
    buffer.load_d4rl_dataset(data)

    agent = init_agent(cfg, action_dim, ranker_params, ranker)
    if args.resume:
        agent.load(args.resume)

    logger = SwanLogger(cfg)
    save_steps = set(comma_ints(getattr(cfg, "save_steps", None)))
    best_score = -float("inf")
    last_train_metrics: Dict[str, float] = {}

    timeline_root = Path(cfg.eval_timeline_dir)
    if cfg.eval_timeline_dir and not timeline_root.is_absolute():
        timeline_root = PROJECT_ROOT / timeline_root
    timeline_path = timeline_root / cfg.experiment_name.split("/")[-1] / "timeline.csv"

    if getattr(cfg, "eval_step_zero", False):
        if 0 in save_steps:
            save_checkpoint(agent, cfg, "iql_step0.pt" if cfg.algo == "iql" else "bc_step0.pt")
        det_m, _samp_m = run_eval_block(
            cfg, agent, 0, last_train_metrics, timeline_path, item_freq, item_total, combo_freq, logger
        )
        best_score = det_m["iqm_reward"]
        save_checkpoint(agent, cfg, f"{cfg.algo}_best.pt")
        save_checkpoint(agent, cfg, f"{cfg.algo}_best_step0.pt")

    for step in range(1, cfg.max_timesteps + 1):
        batch = buffer.sample(cfg.batch_size).to(device)
        if cfg.algo == "bc":
            batch = add_bc_actions_to_batch(batch, cfg, ranker, ranker_params)
        last_train_metrics = agent.train(batch)

        if should_run_by_schedule(step, cfg.log_freq, getattr(cfg, "log_freq_schedule", None)):
            logger.log({f"train/{k}": v for k, v in last_train_metrics.items()}, step)
            logging.info(
                "train step=%s actor_loss=%s critic_loss=%s value_loss=%s adv_q90=%s",
                step,
                last_train_metrics.get("actor_loss"),
                last_train_metrics.get("critic_loss"),
                last_train_metrics.get("value_loss"),
                last_train_metrics.get("adv_q90"),
            )

        should_eval = should_run_by_schedule(step, cfg.eval_freq, getattr(cfg, "eval_freq_schedule", None))
        if should_eval:
            det_m, _samp_m = run_eval_block(
                cfg, agent, step, last_train_metrics, timeline_path, item_freq, item_total, combo_freq, logger
            )
            score = det_m["iqm_reward"] if cfg.best_checkpoint_metric == "iqm" else det_m["mean_reward"]
            if score > best_score:
                best_score = score
                save_checkpoint(agent, cfg, f"{cfg.algo}_best.pt")
                save_checkpoint(agent, cfg, f"{cfg.algo}_best_step{step}.pt")

        if (cfg.save_freq > 0 and step % cfg.save_freq == 0) or step in save_steps:
            save_checkpoint(agent, cfg, f"{cfg.algo}_step{step}.pt")

    final_path = save_checkpoint(agent, cfg, f"{cfg.algo}_final.pt")
    logging.info("saved final checkpoint: %s", final_path)
    run_eval_block(
        cfg,
        agent,
        cfg.max_timesteps,
        last_train_metrics,
        timeline_path,
        item_freq,
        item_total,
        combo_freq,
        logger,
        final=True,
    )
    logger.finish()


if __name__ == "__main__":
    main()
