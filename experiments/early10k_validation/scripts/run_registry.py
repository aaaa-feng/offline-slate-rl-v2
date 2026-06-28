#!/usr/bin/env python3
"""Run registry for early10k_validation post-processing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP_ROOT = PROJECT_ROOT / "experiments" / "early10k_validation"


@dataclass(frozen=True)
class RunSpec:
    label: str
    kl: float
    env_group: str
    beta: int
    complete_10k: bool = True

    @property
    def env_name(self) -> str:
        if self.env_group == "mix":
            return "mix_divpen"
        if self.env_group == "topdown":
            return "topdown"
        raise ValueError(f"unknown env_group: {self.env_group}")


RUNS: tuple[RunSpec, ...] = (
    RunSpec("kl001_mix_b0_ideal_init", 0.01, "mix", 0),
    RunSpec("kl001_mix_b2_ideal_init", 0.01, "mix", 2),
    RunSpec("kl001_mix_b5_ideal_init", 0.01, "mix", 5),
    RunSpec("kl001_mix_b8_ideal_init", 0.01, "mix", 8),
    RunSpec("kl001_topdown_b0_ideal_init", 0.01, "topdown", 0),
    RunSpec("kl001_topdown_b2_ideal_init", 0.01, "topdown", 2),
    RunSpec("kl001_topdown_b5_ideal_init", 0.01, "topdown", 5),
    RunSpec("kl001_topdown_b8_ideal_init", 0.01, "topdown", 8),
    RunSpec("kl005_mix_b0_ideal_init", 0.05, "mix", 0),
    RunSpec("kl005_mix_b2_ideal_init", 0.05, "mix", 2),
    RunSpec("kl005_mix_b5_ideal_init", 0.05, "mix", 5),
    RunSpec("kl005_mix_b8_ideal_init", 0.05, "mix", 8),
    # This run stopped before 10k. Keep it in the registry, but plotting skips
    # missing checkpoint tags automatically.
    RunSpec("kl005_topdown_b0_ideal_init", 0.05, "topdown", 0, complete_10k=False),
    RunSpec("kl005_topdown_b2_ideal_init", 0.05, "topdown", 2),
    RunSpec("kl005_topdown_b5_ideal_init", 0.05, "topdown", 5),
    RunSpec("kl005_topdown_b8_ideal_init", 0.05, "topdown", 8),
)


RUN_BY_LABEL = {run.label: run for run in RUNS}


def get_run(label: str) -> RunSpec:
    try:
        return RUN_BY_LABEL[label]
    except KeyError as exc:
        valid = ", ".join(sorted(RUN_BY_LABEL))
        raise SystemExit(f"unknown run_label={label!r}; valid labels: {valid}") from exc


def iter_labels(which: str = "all") -> list[str]:
    if which == "all":
        return [run.label for run in RUNS]
    if which == "complete":
        return [run.label for run in RUNS if run.complete_10k]
    if which in RUN_BY_LABEL:
        return [which]
    raise SystemExit("first argument must be 'all', 'complete', or a run label")
