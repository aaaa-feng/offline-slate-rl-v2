#!/bin/bash
# GRU Architecture Ablation (复现 exp_gru_ablation_20260420)
# G0: qv_shared_detach | G1: qv_shared_all_update | G2: q_independent
# Usage: bash run.sh [G0|G1|G2|all] [gpu_id]

GROUP=${1:-all}
GPU=${2:-0}

cd /data/liyuefeng/offline-slate-rl-v2
PYTHON=/data/liyuefeng/miniconda3/envs/gems/bin/python

run_one() {
    local label=$1 mode=$2 gpu=$3
    echo "=== $label: gru_mode=$mode GPU=$gpu ==="
    mkdir -p logs/agents/gru_ablation
    CUDA_VISIBLE_DEVICES=${gpu} nohup ${PYTHON} -u scripts/train_agent.py \
        --config experiments/gru_ablation/config.yaml \
        --gru_mode ${mode} \
        --experiment_tag "mix_divpen/IQL/gems_gru_${mode}_beta3_tau_0.8_seed58407201/gru_ablation" \
        > logs/agents/gru_ablation/${label}.log 2>&1 &
    echo "  PID: $!"
}

case $GROUP in
    G0)  run_one "G0" "qv_shared_detach"      $GPU ;;
    G1)  run_one "G1" "qv_shared_all_update"  $GPU ;;
    G2)  run_one "G2" "q_independent"         $GPU ;;
    all)
        run_one "G0" "qv_shared_detach"      $GPU; sleep 2
        run_one "G1" "qv_shared_all_update"  $GPU; sleep 2
        run_one "G2" "q_independent"         $GPU ;;
esac

echo "Done launching."
