"""Run registry for real_click_action_label_ablation post-processing."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
GEMS_DIR = PROJECT_ROOT / "checkpoints/gems"
EXP_GROUP = "real_click_action_label_ablation"

RUNS = [
    ("0.01", "mix_divpen", 8, "kl001_mix_b8_ideal_init_rc"),
    ("0.01", "mix_divpen", 0, "kl001_mix_b0_ideal_init_rc"),
    ("0.01", "topdown_divpen", 8, "kl001_topdown_b8_ideal_init_rc"),
    ("0.01", "topdown_divpen", 0, "kl001_topdown_b0_ideal_init_rc"),
]


def gems_checkpoint(env_name: str, kl: str) -> Path:
    name = (
        f"GeMS_{env_name}_b5_pretrained_latent32_beta{kl}_click1.0_"
        f"seed58407201_ideal_init.ckpt"
    )
    path = GEMS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"GeMS checkpoint not found: {path}")
    return path


def ckpt_dir(run_label: str) -> Path:
    return PROJECT_ROOT / "checkpoints/agents" / EXP_GROUP / run_label
