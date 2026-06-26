#!/bin/bash
# Actor Temperature Ablation
# 扫 T ∈ {0.3, 0.4, 0.5, 0.6, 0.7, 1.0}
# 对 mix_b8 和 td_b8 的 final + best checkpoint 分别评估
#
# Usage: bash run.sh [gpu_id]
#   约 5-10 分钟跑完一个 (run, T) 组合，共 24 个评估

GPU=${1:-0}
cd /data/liyuefeng/offline-slate-rl-v2
PYTHON=/data/liyuefeng/miniconda3/envs/gems/bin/python

TEMPS=(0.3 0.4 0.5 0.6 0.7 1.0)
RUNS=(
    "mix_b8 mix_divpen"
    "topdown_b8 topdown_divpen"
)

mkdir -p logs/agents/temperature_ablation
RESULTS_FILE="experiments/action_cloud/temperature_ablation/results.txt"
echo "Temperature Ablation Results - $(date)" > $RESULTS_FILE
echo "============================================================" >> $RESULTS_FILE

for run_info in "${RUNS[@]}"; do
    run_name=$(echo $run_info | awk '{print $1}')
    env_name=$(echo $run_info | awk '{print $2}')

    # Final checkpoint
    CKPT="checkpoints/agents/beta_ablation_repreduce/${run_name}/iql_final.pt"
    if [ ! -f "$CKPT" ]; then
        echo "WARNING: $CKPT not found, skipping" | tee -a $RESULTS_FILE
        continue
    fi

    for T in "${TEMPS[@]}"; do
        echo "" | tee -a $RESULTS_FILE
        echo "=== ${run_name} final T=${T} ===" | tee -a $RESULTS_FILE
        CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON} -u scripts/eval.py \
            --algo iql \
            --env_name ${env_name} \
            --dataset_quality b5 \
            --gems_embedding_mode ideal_init \
            --checkpoint ${CKPT} \
            --temperature ${T} \
            --episodes 100 \
            > logs/agents/temperature_ablation/${run_name}_final_T${T}.log 2>&1

        # Extract results
        grep -E "Mean Reward|Median Reward|IQM Reward|Unique Items" logs/agents/temperature_ablation/${run_name}_final_T${T}.log | \
            tr '\n' ' ' >> $RESULTS_FILE
        echo "" >> $RESULTS_FILE
    done

    # Best checkpoint
    CKPT="checkpoints/agents/beta_ablation_repreduce/${run_name}/iql_best.pt"
    if [ ! -f "$CKPT" ]; then
        echo "WARNING: $CKPT not found, skipping" | tee -a $RESULTS_FILE
        continue
    fi

    for T in "${TEMPS[@]}"; do
        echo "" | tee -a $RESULTS_FILE
        echo "=== ${run_name} best T=${T} ===" | tee -a $RESULTS_FILE
        CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON} -u scripts/eval.py \
            --algo iql \
            --env_name ${env_name} \
            --dataset_quality b5 \
            --gems_embedding_mode ideal_init \
            --checkpoint ${CKPT} \
            --temperature ${T} \
            --episodes 100 \
            > logs/agents/temperature_ablation/${run_name}_best_T${T}.log 2>&1

        grep -E "Mean Reward|Median Reward|IQM Reward|Unique Items" logs/agents/temperature_ablation/${run_name}_best_T${T}.log | \
            tr '\n' ' ' >> $RESULTS_FILE
        echo "" >> $RESULTS_FILE
    done
done

echo "" | tee -a $RESULTS_FILE
echo "Done. Results: $RESULTS_FILE" | tee -a $RESULTS_FILE
echo "Individual logs: logs/agents/temperature_ablation/"
