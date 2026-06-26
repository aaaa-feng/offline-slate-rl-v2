#!/bin/bash
# mf_init GeMS KL 消融
# embedding: BPR MF (mf_{env}.pt), 可训练 (fixed_embedds=scratch)
# 变量: lambda_KL ∈ {0.01, 0.05, 0.1, 0.5, 1.0}
#
# Usage:
#   bash run.sh all        # 全部 10 个 (5 KL × 2 env)
#   bash run.sh mix        # mix_divpen 5 个
#   bash run.sh td         # topdown_divpen 5 个
#   bash run.sh mix_kl1    # 单个: mix, KL=1.0

GROUP=${1:-all}
GPU=${2:-0}

cd /data/liyuefeng/offline-slate-rl-v2
PYTHON=/data/liyuefeng/miniconda3/envs/gems/bin/python

KL_VALUES=(0.01 0.05 0.1 0.5 1.0)

# MF embedding 文件映射
get_mf_path() {
    local env=$1
    local short=$(echo $env | sed 's/_divpen//')  # mix_divpen → mix, topdown_divpen → topdown
    echo "data/embeddings/mf/mf_${short}_b5.pt"
}

run_one() {
    local env=$1 kl=$2 gpu=$3
    local dataset="${env}_b5"
    local kl_label=$(echo $kl | sed 's/\.//')  # 0.05 → 005, 1.0 → 10
    local label="${env}_kl${kl_label}"
    local mf_path=$(get_mf_path $env)

    echo "=== ${label}: env=${env} KL=${kl} GPU=${gpu} MF=${mf_path} ==="
    mkdir -p logs/gems

    CUDA_VISIBLE_DEVICES=${gpu} nohup ${PYTHON} -u scripts/train_gems.py \
        --dataset ${dataset} \
        --item_embedds pretrained \
        --embedding_path ${mf_path} \
        --fixed_embedds scratch \
        --lambda_KL ${kl} \
        --lambda_click 1.0 \
        --lambda_prior 1.0 \
        --ranker_lr 3e-3 \
        --max_epochs 50 \
        --seed 58407201 \
        --batch_size 256 \
        --experiment_tag mf_init \
        > logs/gems/mf_init_${label}.log 2>&1 &

    echo "  PID: $!"
}

case $GROUP in
    all)
        # 10 runs 分散到 GPU 0,5,6,7 (0和7各3个, 5和6各2个)
        GPUS=(0 0 1 1 4 4 5 5 6 7)
        i=0
        for env in mix_divpen topdown_divpen; do
            for kl in "${KL_VALUES[@]}"; do
                gpu=${GPUS[$i]}
                run_one $env $kl $gpu
                sleep 3
                i=$((i + 1))
            done
        done
        ;;
    mix)
        for kl in "${KL_VALUES[@]}"; do
            run_one mix_divpen $kl $GPU
            sleep 3
        done
        ;;
    td)
        for kl in "${KL_VALUES[@]}"; do
            run_one topdown_divpen $kl $GPU
            sleep 3
        done
        ;;
    mix_kl*|td_kl*)
        env=$(echo $GROUP | sed 's/_kl.*//' | sed 's/mix/mix_divpen/;s/td/topdown_divpen/')
        kl=$(echo $GROUP | sed 's/.*_kl//' | sed 's/^0/0./' | sed 's/^00/0.0/')
        run_one $env $kl $GPU
        ;;
    *)
        echo "Usage: bash run.sh [all|mix|td|mix_kl1|td_kl005|...] [gpu_id]"
        echo ""
        echo "  bash run.sh all        # 全部 10 个"
        echo "  bash run.sh mix        # mix_divpen 5 个 KL"
        echo "  bash run.sh mix_kl1    # mix KL=1.0"
        exit 1
        ;;
esac

echo ""
echo "Launched. Logs: logs/gems/mf_init_*.log"
echo "Checkpoints: checkpoints/gems/"
