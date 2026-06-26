#!/bin/bash
# 实验4: 真实 click 替代 fake_zero
# 对比基线 beta_ablation_repreduce (fake_zero)
#
# Usage:
#   bash run.sh all        # 3 个 run (b0/b5/b8), 分散到 GPU 0/1/7
#   bash run.sh b0  0      # 仅 β=0, GPU 0
#   bash run.sh b5  1      # 仅 β=5, GPU 1
#   bash run.sh b8  7      # 仅 β=8, GPU 7

GROUP=${1:-all}
GPU=${2:-0}

cd /data/liyuefeng/offline-slate-rl-v2
PYTHON=/data/liyuefeng/miniconda3/envs/gems/bin/python

run_one() {
    local beta=$1 gpu=$2
    local label="mix_b${beta}_real"
    local exp_name="real_click/${label}"

    echo "=== ${label}: beta=${beta} GPU=${gpu} ==="
    mkdir -p logs/agents/real_click

    CUDA_VISIBLE_DEVICES=${gpu} nohup ${PYTHON} -u scripts/train_agent.py \
        --config experiments/action_cloud/real_click/config.yaml \
        --env_name mix_divpen \
        --beta ${beta} \
        --experiment_name "${exp_name}" \
        --experiment_tag "real_click/${label}" \
        > logs/agents/real_click/${label}.log 2>&1 &

    echo "  PID: $!"
}

case $GROUP in
    all)
        run_one 0 0;  sleep 3
        run_one 5 1;  sleep 3
        run_one 8 7
        ;;
    b0) run_one 0 $GPU ;;
    b5) run_one 5 $GPU ;;
    b8) run_one 8 $GPU ;;
    *)
        echo "Usage: bash run.sh [all|b0|b5|b8] [gpu_id]"
        exit 1
        ;;
esac

echo "Launched. Logs: logs/agents/real_click/"
echo "Checkpoints: checkpoints/agents/real_click/{mix_b0_real,mix_b5_real,mix_b8_real}/"
