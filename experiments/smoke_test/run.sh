#!/bin/bash
# Smoke Test: beta=10, mix_divpen, 5000 steps
# 对标旧实验 exp_beta_high_ablation_20260518 mix_B10
#
# SwanLab project: Offline_Slate_RL_202605（与旧实验同一项目，便于对比）
# Usage: bash run.sh [gpu_id]

GPU=${1:-0}

cd /data/liyuefeng/offline-slate-rl-v2
PYTHON=/data/liyuefeng/miniconda3/envs/gems/bin/python

mkdir -p logs/agents/smoke_test

CUDA_VISIBLE_DEVICES=${GPU} nohup ${PYTHON} -u scripts/train_agent.py \
    --config experiments/smoke_test/config.yaml \
    --experiment_tag "mix_divpen/IQL/gems_beta10_tau_0.8_seed58407201/smoke_test" \
    --experiment_name "smoke_test" \
    > logs/agents/smoke_test/smoke_test.log 2>&1 &

echo "Smoke test launched (PID $!)."
echo "  Log: logs/agents/smoke_test/smoke_test.log"
echo "  Tail: tail -f logs/agents/smoke_test/smoke_test.log"
