#!/bin/bash
# reward_scale ablation: reward_scale=1/5/10/20/50 @ beta=8, mix_divpen
# Usage: bash run.sh [gpu_id]

GPU=${1:-0}
SCALES=(1 5 10 20 50)

cd /data/liyuefeng/offline-slate-rl-v2
PYTHON=/data/liyuefeng/miniconda3/envs/gems/bin/python

for rs in "${SCALES[@]}"; do
    label="rs${rs}"
    echo "=== ${label}: reward_scale=${rs} GPU=${GPU} ==="
    mkdir -p logs/agents/reward_scale_ablation

    CUDA_VISIBLE_DEVICES=${GPU} nohup ${PYTHON} -u scripts/train_agent.py \
        --config experiments/reward_scale_ablation/config.yaml \
        --reward_scale ${rs} \
        --experiment_name "reward_scale_ablation/${label}" \
        --experiment_tag "reward_scale_ablation/${label}" \
        > logs/agents/reward_scale_ablation/${label}.log 2>&1 &

    echo "  PID: $!"
    sleep 3
done

echo "Launched. Logs: logs/agents/reward_scale_ablation/"
echo "Checkpoints: checkpoints/agents/reward_scale_ablation/rs{1,5,10,20,50}/"
