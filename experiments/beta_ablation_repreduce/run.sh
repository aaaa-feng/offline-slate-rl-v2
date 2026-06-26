#!/bin/bash
# beta_ablation_repreduce: 复现旧项目 beta ablation (0/1/2/5/8/10) × (mix/topdown)
# 矩阵: 2 env × 6 beta = 12 runs
#
# Usage:
#   bash run.sh all              # 全部 12 个 run
#   bash run.sh mix              # mix_divpen 全部 6 个 beta
#   bash run.sh td               # topdown_divpen 全部 6 个 beta
#   bash run.sh mix_b5 0         # 单个 run: mix_divpen, beta=5, GPU 0
#   bash run.sh td_b8 1          # 单个 run: topdown_divpen, beta=8, GPU 1

GROUP=${1:-all}
GPU=${2:-0}

cd /data/liyuefeng/offline-slate-rl-v2
PYTHON=/data/liyuefeng/miniconda3/envs/gems/bin/python
BETAS=(0 2 5 8 10)

run_one() {
    local env=$1 beta=$2 gpu=$3
    local label="${env}_b${beta}_tau0.8_lbc0.0_ideal_init_seed58407201"
    local exp_name="beta_ablation_repreduce/${label}"

    echo "=== ${label}: env=${env} beta=${beta} GPU=${gpu} ==="
    mkdir -p logs/agents/beta_ablation_repreduce

    CUDA_VISIBLE_DEVICES=${gpu} nohup ${PYTHON} -u scripts/train_agent.py \
        --config experiments/beta_ablation_repreduce/config.yaml \
        --env_name ${env} \
        --beta ${beta} \
        --experiment_name "${exp_name}" \
        --experiment_tag "beta_ablation_repreduce/${label}" \
        > logs/agents/beta_ablation_repreduce/${label}.log 2>&1 &

    echo "  PID: $!"
}

case $GROUP in
    all)
        # 10 runs 分散到 3 个 GPU (~2.5 GB/run)
        # GPU 0: mix_b0, mix_b5, td_b2
        run_one mix_divpen 0 0; sleep 3
        run_one mix_divpen 5 0; sleep 3
        run_one topdown_divpen 2 0; sleep 3
        # GPU 1: mix_b2, mix_b8, td_b5, td_b10
        run_one mix_divpen 2 1; sleep 3
        run_one mix_divpen 8 1; sleep 3
        run_one topdown_divpen 5 1; sleep 3
        run_one topdown_divpen 10 1; sleep 3
        # GPU 7: mix_b10, td_b0, td_b8
        run_one mix_divpen 10 7; sleep 3
        run_one topdown_divpen 0 7; sleep 3
        run_one topdown_divpen 8 7
        ;;
    mix)
        for beta in "${BETAS[@]}"; do
            run_one mix_divpen $beta $GPU
            sleep 3
        done
        ;;
    td)
        for beta in "${BETAS[@]}"; do
            run_one topdown_divpen $beta $GPU
            sleep 3
        done
        ;;
    mix_b*|td_b*)
        # 单个 run: e.g. mix_b5, td_b8
        env_name=$(echo $GROUP | sed 's/_b.*//' | sed 's/mix/mix_divpen/;s/td/topdown_divpen/')
        beta_val=$(echo $GROUP | sed 's/.*_b//')
        run_one $env_name $beta_val $GPU
        ;;
    *)
        echo "Usage: bash run.sh [all|mix|td|mix_b{N}|td_b{N}] [gpu_id]"
        echo "Examples:"
        echo "  bash run.sh all          # 全部 10 个, 分散到 GPU 0/1/7"
        echo "  bash run.sh mix          # mix_divpen 5 个"
        echo "  bash run.sh td           # topdown_divpen 5 个"
        echo "  bash run.sh mix_b5 0     # mix beta=5, GPU 0"
        echo "  bash run.sh td_b8 1     # topdown beta=8, GPU 1"
        exit 1
        ;;
esac

echo "Launched. Logs: logs/agents/beta_ablation_repreduce/"
echo "Checkpoints: checkpoints/agents/beta_ablation_repreduce/{mix,td}_b{N}/"
