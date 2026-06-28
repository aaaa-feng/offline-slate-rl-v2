#!/bin/bash
# Replot early10k_validation figures for many runs.
#
# This compatibility script keeps the previous command shape. Since .npz exports
# already exist, each worker only rebuilds figures from disk.

set -euo pipefail

WHICH="${1:-all}"
MAX_JOBS="${MAX_JOBS:-4}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-/data/liyuefeng/miniconda3/envs/gems/bin/python}"

cd "$ROOT"

mapfile -t RUNS < <("$PYTHON" - <<PY
import sys
sys.path.insert(0, "experiments/early10k_validation/scripts")
from run_registry import iter_labels
for label in iter_labels("$WHICH"):
    print(label)
PY
)

active=0
for run_label in "${RUNS[@]}"; do
  echo "[launch] $run_label"
  bash experiments/early10k_validation/scripts/postprocess_curated.sh "$run_label" \
    > "experiments/early10k_validation/geometry_exports/${run_label}_replot.log" 2>&1 &
  active=$((active + 1))
  if [ "$active" -ge "$MAX_JOBS" ]; then
    wait -n
    active=$((active - 1))
  fi
done

wait
echo "Done. Replot logs are under experiments/early10k_validation/geometry_exports/*_replot.log"
