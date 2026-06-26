#!/bin/bash
# Dataset action-cloud quality PCA — extract + plot
#
# Usage:
#   bash run.sh [GPU]                         # all b5 GeMS ckpts
#   bash run.sh [GPU] plot-only               # replot from exports/
#   bash run.sh [GPU] mix_divpen_b5_beta0.1_ideal_init

set -euo pipefail
GPU=${1:-0}
MODE=${2:-all}
cd /data/liyuefeng/offline-slate-rl-v2
PYTHON=/data/liyuefeng/miniconda3/envs/gems/bin/python
DIR=experiments/action_cloud/static_analysis

if [ "${MODE}" = "plot-only" ]; then
    ${PYTHON} ${DIR}/plot_cloud_quality.py --all
    exit 0
fi

if [ "${MODE}" != "all" ]; then
    echo "=== ${MODE} ==="
    CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON} ${DIR}/extract_cloud_quality.py --slug "${MODE}"
    ${PYTHON} ${DIR}/plot_cloud_quality.py --slug "${MODE}"
    exit 0
fi

echo "=== Extract (all b5 GeMS) ==="
CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON} ${DIR}/extract_cloud_quality.py --all_b5 --n_samples 20000

echo "=== Plot ==="
${PYTHON} ${DIR}/plot_cloud_quality.py --all

echo "Done → ${DIR}/figures/"
