"""Unified experiment configuration for offline slate RL.

This module is intentionally lightweight: old checkpoints pickle
``config.ExperimentConfig`` directly, so keeping this class at the repo root is
required for checkpoint loading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


PROJECT_ROOT = Path(__file__).resolve().parent


ENV_PARAMS = {
    "mix_divpen": {
        "click_model": "mixPBM",
        "diversity_penalty": 5.0,
        "num_items": 1000,
        "rec_size": 10,
        "episode_length": 100,
        "num_topics": 10,
        "topic_size": 2,
        "env_offset": 0.28,
        "env_slope": 100,
        "env_omega": 0.9,
        "boredom_threshold": None,
        "recent_items_maxlen": 10,
        "boredom_moving_window": 5,
        "short_term_boost": 1.0,
        "diversity_threshold": 4,
        "item_embedd_dim": 20,
        "env_embedds": "item_embeddings_diffuse.pt",
    },
    "topdown_divpen": {
        "click_model": "tdPBM",
        "diversity_penalty": 5.0,
        "num_items": 1000,
        "rec_size": 10,
        "episode_length": 100,
        "num_topics": 10,
        "topic_size": 2,
        "env_offset": 0.28,
        "env_slope": 100,
        "env_omega": 0.9,
        "boredom_threshold": None,
        "recent_items_maxlen": 10,
        "boredom_moving_window": 5,
        "short_term_boost": 1.0,
        "diversity_threshold": 4,
        "item_embedd_dim": 20,
        "env_embedds": "item_embeddings_diffuse.pt",
    },
}


def _extract_boredom(dataset_quality: str) -> int:
    import re

    match = re.search(r"b(\d+)", str(dataset_quality))
    if match:
        return int(match.group(1))
    return 5


@dataclass
class ExperimentConfig:
    """Single config object used by training, eval, probes, and checkpoints."""

    # Experiment
    experiment_name: str = "default"
    seed: int = 58407201
    device: str = "cuda"

    # Environment
    env_name: str = "mix_divpen"
    dataset_quality: str = "b5"

    # GeMS ranker
    ranker_type: str = "gems"
    gems_embedding_mode: str = "scratch"
    latent_dim: int = 32
    lambda_KL: float = 1.0
    lambda_click: float = 1.0

    # Algorithm
    algo: str = "iql"
    beta: float = 3.0
    expectile: float = 0.8
    lambda_bc: float = 0.3
    gamma: float = 0.99
    iql_tau: float = 0.005
    reward_scale: float = 10.0
    label_click_mode: str = "fake_zero"
    best_checkpoint_metric: str = "iqm"

    # Training
    max_timesteps: int = 1_000_000
    batch_size: int = 256
    eval_freq: int = 500
    eval_episodes: int = 50
    final_eval_episodes: int = 100
    save_freq: int = 50_000
    log_freq: int = 500

    # Dual eval / timelines
    dual_eval: bool = False
    eval_timeline_dir: str = ""

    # Training-time geometry probe
    enable_train_geometry_probe: bool = False
    geometry_probe_episodes: int = 5
    geometry_probe_dataset_samples: int = 20_000
    geometry_probe_save_samples: bool = True
    geometry_probe_dir: str = "experiments/action_cloud/training_adv_geometry_monitor/outputs"

    # Networks
    hidden_dim: int = 256
    n_hidden: int = 2
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    value_lr: float = 3e-4

    # GRU belief
    belief_hidden_dim: int = 20
    item_embedd_dim: int = 20
    num_items: int = 1000
    rec_size: int = 10
    gru_mode: str = "qv_shared_detach"

    # Actor
    actor_type: str = "gaussian"
    fixed_std: float = 0.1

    # Flow matching / diffusion slate experiments
    flow_steps: int = 10
    guidance_scale: float = 0.0
    flow_dedup_knn: int = 1

    # Wolpertinger / greedy legacy options
    wolpertinger_k: int = 50
    wolpertinger_hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    greedy_s_no_click: float = -1.0

    # GeMS pretraining
    gems_item_embedds: str = "scratch"
    gems_embedding_path: str = ""
    gems_fixed_embedds: str = "scratch"
    gems_hidden_layers_infer: List[int] = field(default_factory=lambda: [512, 256])
    gems_hidden_layers_decoder: List[int] = field(default_factory=lambda: [256, 512])
    gems_lambda_prior: float = 0.0
    gems_lr: float = 1e-3
    gems_max_epochs: int = 15
    gems_val_split: float = 0.1

    # SwanLab
    use_swanlab: bool = True
    swan_project: str = "offline_slate_rl_gems_202606"
    swan_workspace: str = "Cliff"
    swan_mode: str = "cloud"
    swan_logdir: str = "logs/swanlog"

    @property
    def tau(self) -> float:
        """Alias used by older IQL code."""
        return self.expectile

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        cfg = cls()
        for key, value in raw.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        return cfg

    @classmethod
    def from_args(cls, args) -> "ExperimentConfig":
        cfg = cls()
        for key, value in vars(args).items():
            if key == "config" or value is None:
                continue
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        return cfg

    def get_env_params(self) -> dict:
        if self.env_name not in ENV_PARAMS:
            raise ValueError(f"Unknown env_name: {self.env_name}")
        params = dict(ENV_PARAMS[self.env_name])
        params["boredom_threshold"] = _extract_boredom(self.dataset_quality)
        params["item_embedd_dim"] = self.item_embedd_dim
        params["num_items"] = self.num_items
        params["rec_size"] = self.rec_size
        return params

    @property
    def dataset_path(self) -> Path:
        return (
            PROJECT_ROOT
            / "data/datasets/offline"
            / self.env_name
            / f"{self.env_name}_{self.dataset_quality}_data_d4rl.npz"
        )

    @property
    def oracle_path(self) -> Path:
        return (
            PROJECT_ROOT
            / "data/datasets/offline"
            / self.env_name
            / f"{self.env_name}_{self.dataset_quality}_oracle.npz"
        )

    @property
    def online_dataset_path(self) -> Path:
        return (
            PROJECT_ROOT
            / "data/datasets/offline"
            / self.env_name
            / f"{self.env_name}_{self.dataset_quality}.pt"
        )

    @property
    def item_embedds_path(self) -> Path:
        return PROJECT_ROOT / "data/embeddings/item_embeddings_diffuse.pt"

    @property
    def mf_embedding_path(self) -> Path:
        env_prefix = self.env_name.split("_")[0]
        return PROJECT_ROOT / "data/embeddings/mf" / f"mf_{env_prefix}_{self.dataset_quality}.pt"

    @property
    def gems_checkpoint_path(self) -> Path:
        mode = self.gems_embedding_mode
        name = (
            f"GeMS_{self.env_name}_{self.dataset_quality}_{mode}"
            f"_latent{self.latent_dim}_beta{self.lambda_KL}"
            f"_click{self.lambda_click}_seed{self.seed}.ckpt"
        )
        return PROJECT_ROOT / "checkpoints/gems" / name

    @property
    def exp_dir(self) -> Path:
        name = self.experiment_name.split("/")[0]
        return PROJECT_ROOT / "experiments" / name

    @property
    def gems_checkpoint_dir(self) -> Path:
        return PROJECT_ROOT / "checkpoints/gems"

    @property
    def agent_checkpoint_dir(self) -> Path:
        return PROJECT_ROOT / "checkpoints/agents" / self.experiment_name

    @property
    def geometry_probe_output_dir(self) -> Path:
        path = Path(self.geometry_probe_dir)
        if path.is_absolute():
            return path / self.experiment_name.split("/")[-1]
        return PROJECT_ROOT / path / self.experiment_name.split("/")[-1]
