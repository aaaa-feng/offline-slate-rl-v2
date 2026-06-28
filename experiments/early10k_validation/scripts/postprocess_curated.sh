#!/bin/bash
# Compatibility entrypoint for the restored early10k_validation plotting workflow.
#
# The original postprocess script exported .npz files from checkpoints and then
# plotted them. The exported .npz files are still present, so this restored script
# only rebuilds figures from geometry_exports.

set -euo pipefail

RUN_LABEL="${1:-all}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-/data/liyuefeng/miniconda3/envs/gems/bin/python}"

cd "$ROOT"

ACTION_ONLY=0
BELIEF_ONLY=0
for arg in "${@:2}"; do
  case "$arg" in
    --action-only) ACTION_ONLY=1 ;;
    --belief-only) BELIEF_ONLY=1 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

if [ "$BELIEF_ONLY" -eq 0 ]; then
  "$PYTHON" experiments/early10k_validation/scripts/plot_action_trajectory.py \
    --run_label "$RUN_LABEL" \
    --eval_mode both
fi

if [ "$ACTION_ONLY" -eq 0 ]; then
  "$PYTHON" experiments/early10k_validation/scripts/plot_belief_trajectory.py \
    --run_label "$RUN_LABEL" \
    --eval_mode both
fi
