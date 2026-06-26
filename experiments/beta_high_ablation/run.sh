#!/bin/bash
# Beta High Ablation: beta=8/10, mix_divpen + topdown_divpen (复现 exp_beta_high_ablation_20260518)
# Usage:
#   bash run.sh mix_B8    # mix_divpen, beta=8
#   bash run.sh mix_B10   # mix_divpen, beta=10
#   bash run.sh td_B8     # topdown_divpen, beta=8
#   bash run.sh td_B10    # topdown_divpen, beta=10
#   bash run.sh all       # 全部 4 个

GROUP=${1:-all}
GPU=${2:-0}

VALID="all mix_B8 mix_B10 td_B8 td_B10"
if ! echo "$VALID" | grep -qw "$GROUP"; then
    echo "Usage: bash run.sh [all|mix_B8|mix_B10|td_B8|td_B10] [gpu_id]"
    exit 1
fi

cd /data/liyuefeng/offline-slate-rl-v2
PYTHON=/data/liyuefeng/miniconda3/envs/gems/bin/python

run_one() {
    local label=$1 env=$2 beta=$3 gpu=$4
    echo "=== $label: env=$env beta=$beta GPU=$gpu ==="
    mkdir -p logs/agents/beta_high_ablation
    CUDA_VISIBLE_DEVICES=${gpu} nohup ${PYTHON} -u scripts/train_agent.py \
        --config experiments/beta_high_ablation/config.yaml \
        --env_name ${env} \
        --beta ${beta} \
        --experiment_tag "${env}/IQL/gems_beta${beta}_tau_0.8_seed58407201/beta_ablation" \
        > logs/agents/beta_high_ablation/${label}.log 2>&1 &
    echo "  PID: $!"
}

case $GROUP in
    mix_B8)  run_one "mix_B8"  "mix_divpen"    8  $GPU ;;
    mix_B10) run_one "mix_B10" "mix_divpen"    10 $GPU ;;
    td_B8)   run_one "td_B8"   "topdown_divpen" 8  $GPU ;;
    td_B10)  run_one "td_B10"  "topdown_divpen" 10 $GPU ;;
    all)
        run_one "mix_B8"  "mix_divpen"    8  $GPU; sleep 2
        run_one "mix_B10" "mix_divpen"    10 $GPU; sleep 2
        run_one "td_B8"   "topdown_divpen" 8  $GPU; sleep 2
        run_one "td_B10"  "topdown_divpen" 10 $GPU ;;
esac

echo "Done launching."
