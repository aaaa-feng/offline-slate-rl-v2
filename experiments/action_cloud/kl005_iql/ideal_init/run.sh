#!/bin/bash
# Group A: ideal_init KL=0.05 IQL
# 对比 old ideal_init KL=1.0 (beta_ablation_repreduce)
#
# Usage:
#   bash run.sh all        # 10 runs
#   bash run.sh mix        # mix 5 beta
#   bash run.sh mix_b5 0   # single: mix beta=5 GPU=0

GROUP=${1:-all}
GPU=${2:-0}

cd /data/liyuefeng/offline-slate-rl-v2
PYTHON=/data/liyuefeng/miniconda3/envs/gems/bin/python
CKPT_DIR=checkpoints/gems
BETAS=(0 2 5 8 10)

run_one() {
    local env=$1 beta=$2 gpu=$3
    local short=$(echo $env | cut -d'_' -f1)
    local label="${short}_b${beta}"
    local ckpt="${CKPT_DIR}/GeMS_${env}_b5_pretrained_latent32_beta0.05_click1.0_seed58407201_ideal_init.ckpt"

    if [ ! -f "$ckpt" ]; then
        echo "ERROR: ckpt not found: $ckpt"
        exit 1
    fi

    echo "=== ${label}: env=${env} beta=${beta} GPU=${gpu} ==="
    mkdir -p logs/agents/kl005_ideal_init

    CUDA_VISIBLE_DEVICES=${gpu} nohup ${PYTHON} -u scripts/train_agent.py \
        --config experiments/action_cloud/kl005_iql/ideal_init/config.yaml \
        --env_name ${env} \
        --beta ${beta} \
        --gems_checkpoint ${ckpt} \
        --experiment_name "kl005_ideal_init/${label}" \
        --experiment_tag "kl005_ideal_init/${label}" \
        > logs/agents/kl005_ideal_init/${label}.log 2>&1 &

    echo "  PID: $!"
}

case $GROUP in
    all)
        # 10 runs distributed across GPU 1,3,5,6
        GPUS=(1 1 3 3 5 5 6 6 1 3)
        j=0
        for env in mix_divpen topdown_divpen; do
            for beta in "${BETAS[@]}"; do
                gpu=${GPUS[$j]}
                run_one $env $beta $gpu
                sleep 3
                j=$((j + 1))
            done
        done
        ;;
    mix)
        for beta in "${BETAS[@]}"; do run_one mix_divpen $beta $GPU; sleep 3; done
        ;;
    td)
        for beta in "${BETAS[@]}"; do run_one topdown_divpen $beta $GPU; sleep 3; done
        ;;
    mix_b*|td_b*)
        env=$(echo $GROUP | sed 's/_b.*//;s/mix/mix_divpen/;s/td/topdown_divpen/')
        b=$(echo $GROUP | sed 's/.*_b//')
        run_one $env $b $GPU
        ;;
    *)
        echo "Usage: bash run.sh [all|mix|td|mix_b5|td_b8] [gpu_id]"
        exit 1
        ;;
esac

echo ""
echo "Launched. Logs: logs/agents/kl005_ideal_init/"
echo "Checkpoints: checkpoints/agents/kl005_ideal_init/{mix,td}_b{N}/"