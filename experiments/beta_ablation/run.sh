#!/bin/bash
# Beta Ablation: beta=0/1/2/5 (复现 exp_beta_ablation_20260420)
# Usage: bash run.sh [beta] [gpu_id]
#   bash run.sh 0 0    # beta=0, GPU 0
#   bash run.sh all    # 运行全部 4 个 beta 值

BETA=${1:-3.0}
GPU=${2:-0}

cd /data/liyuefeng/offline-slate-rl-v2

if [ "$BETA" == "all" ]; then
    for b in 0 1 2 5; do
        echo "=== Starting beta=${b} on GPU ${GPU} ==="
        CUDA_VISIBLE_DEVICES=${GPU} nohup /data/liyuefeng/miniconda3/envs/gems/bin/python -u scripts/train_agent.py \
            --config experiments/beta_ablation/config.yaml \
            --beta ${b} \
            --experiment_tag "mix_divpen/IQL/gems_beta${b}_tau_0.8_seed58407201/beta_ablation" \
            > logs/agents/beta_ablation/beta${b}.log 2>&1 &
        sleep 3
    done
    echo "All beta experiments launched."
else
    mkdir -p logs/agents/beta_ablation
    CUDA_VISIBLE_DEVICES=${GPU} /data/liyuefeng/miniconda3/envs/gems/bin/python -u scripts/train_agent.py \
        --config experiments/beta_ablation/config.yaml \
        --beta ${BETA} \
        --experiment_tag "mix_divpen/IQL/gems_beta${BETA}_tau_0.8_seed58407201/beta_ablation" \
        2>&1 | tee logs/agents/beta_ablation/beta${BETA}.log
fi
