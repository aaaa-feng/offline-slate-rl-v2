#!/bin/bash
# ideal_init GeMS KL 消融
# embedding: env ground-truth (item_embeddings_diffuse.pt), 可训练
# 变量: lambda_KL ∈ {0.01, 0.05, 0.1, 0.5}
# 已有: lambda_KL=1.0 (checkpoints/gems/GeMS_{env}_b5_ideal_init_latent32_beta1.0_click1.0_seed58407201.ckpt)
#
# Usage:
#   bash run.sh all        # 全部 8 个 (4 KL × 2 env)
#   bash run.sh mix        # mix_divpen 4 个
#   bash run.sh td         # topdown_divpen 4 个
#   bash run.sh mix_kl005  # 单个: mix, KL=0.05

GROUP=${1:-all}
GPU=${2:-0}

cd /data/liyuefeng/offline-slate-rl-v2
PYTHON=/data/liyuefeng/miniconda3/envs/gems/bin/python

KL_VALUES=(0.01 0.05 0.1 0.5)  # 不含 1.0（已有 ckpt）

run_one() {
    local env=$1 kl=$2 gpu=$3
    local dataset="${env}_b5"
    local kl_label=$(echo $kl | sed 's/\.//')  # 0.05 → 005, 0.1 → 01
    local label="${env}_kl${kl_label}"

    echo "=== ${label}: env=${env} KL=${kl} GPU=${gpu} ==="
    mkdir -p logs/gems

    CUDA_VISIBLE_DEVICES=${gpu} nohup ${PYTHON} -u scripts/train_gems.py \
        --dataset ${dataset} \
        --item_embedds pretrained \
        --embedding_path data/embeddings/item_embeddings_diffuse.pt \
        --fixed_embedds scratch \
        --lambda_KL ${kl} \
        --lambda_click 1.0 \
        --lambda_prior 1.0 \
        --ranker_lr 3e-3 \
        --max_epochs 50 \
        --seed 58407201 \
        --batch_size 256 \
        --experiment_tag ideal_init \
        > logs/gems/ideal_init_${label}.log 2>&1 &

    echo "  PID: $!"
}

case $GROUP in
    all)
        # 8 runs 分散到 GPU 0,5,6,7 (每 GPU 2 个)
        GPUS=(0 1 4 5 6 7 0 1)
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
        echo "Usage: bash run.sh [all|mix|td|mix_kl005|td_kl01|...] [gpu_id]"
        echo ""
        echo "  bash run.sh all        # 全部 8 个"
        echo "  bash run.sh mix        # mix_divpen 4 个 KL"
        echo "  bash run.sh mix_kl005  # mix KL=0.05"
        exit 1
        ;;
esac

echo ""
echo "Launched. Logs: logs/gems/ideal_init_*.log"
echo "Checkpoints: checkpoints/gems/"
