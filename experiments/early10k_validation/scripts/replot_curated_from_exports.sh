#!/bin/bash
# Rebuild early10k_validation action/belief panels from existing geometry_exports.
#
# Usage:
#   bash experiments/early10k_validation/scripts/replot_curated_from_exports.sh all
#   bash experiments/early10k_validation/scripts/replot_curated_from_exports.sh complete
#   bash experiments/early10k_validation/scripts/replot_curated_from_exports.sh kl001_mix_b8_ideal_init

set -euo pipefail

RUN_LABEL="${1:-all}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-/data/liyuefeng/miniconda3/envs/gems/bin/python}"
ACTION_LOG="${ACTION_LOG:-}"
BELIEF_LOG="${BELIEF_LOG:-}"

cd "$ROOT"

echo "=== early10k_validation replot ==="
echo "run_selector=$RUN_LABEL"
echo "input=experiments/early10k_validation/geometry_exports"
echo "output=experiments/early10k_validation/analysis/figures"

echo
echo "[1/2] action latent panels"
if [ -n "$ACTION_LOG" ]; then
  "$PYTHON" experiments/early10k_validation/scripts/plot_action_trajectory.py \
    --run_label "$RUN_LABEL" \
    --eval_mode both | tee "$ACTION_LOG"
else
  "$PYTHON" experiments/early10k_validation/scripts/plot_action_trajectory.py \
    --run_label "$RUN_LABEL" \
    --eval_mode both
fi

echo
echo "[2/2] GRU belief panels"
if [ -n "$BELIEF_LOG" ]; then
  "$PYTHON" experiments/early10k_validation/scripts/plot_belief_trajectory.py \
    --run_label "$RUN_LABEL" \
    --eval_mode both | tee "$BELIEF_LOG"
else
  "$PYTHON" experiments/early10k_validation/scripts/plot_belief_trajectory.py \
    --run_label "$RUN_LABEL" \
    --eval_mode both
fi

echo
echo "=== output counts ==="
"$PYTHON" - <<'PY'
from pathlib import Path
base = Path("experiments/early10k_validation")
print("action_figures=", len(list((base / "analysis/figures/action").glob("**/*.png"))))
print("belief_figures=", len(list((base / "analysis/figures/belief").glob("**/*.png"))))
print("action_npz=", len(list((base / "geometry_exports/action").glob("*/*/*.npz"))))
print("belief_npz=", len(list((base / "geometry_exports/belief").glob("*/*/*.npz"))))
PY

echo "Done."
