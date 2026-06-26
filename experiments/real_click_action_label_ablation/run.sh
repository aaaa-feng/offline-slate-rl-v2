#!/bin/bash
# Real Click Action Label Ablation launcher.
#
# Usage:
#   bash experiments/real_click_action_label_ablation/run.sh status
#   bash experiments/real_click_action_label_ablation/run.sh train

set -euo pipefail

GROUP=${1:-status}

cd /data/liyuefeng/offline-slate-rl-v2

PYTHON=/data/liyuefeng/miniconda3/envs/gems/bin/python
EXP_GROUP=real_click_action_label_ablation
CONFIG=experiments/${EXP_GROUP}/config.yaml
SWAN_PROJECT=Early10K_Validation_202606

LOG_DIR=logs/agents/${EXP_GROUP}
RUN_LOG_DIR=${LOG_DIR}/runs
LAUNCHER_LOG=${LOG_DIR}/launcher.log
LAUNCHER_STDOUT=${LOG_DIR}/launcher_stdout.log

TIMELINE_DIR=experiments/${EXP_GROUP}/eval_timeline
PROBE_DIR=experiments/${EXP_GROUP}/probe_outputs
CHECKPOINT_DIR=checkpoints/agents/${EXP_GROUP}

TRAIN_GPUS=(${TRAIN_GPUS_OVERRIDE:-0 1 2 3})
MAX_TRAIN_CONCURRENT=${MAX_TRAIN_CONCURRENT:-4}
SAVE_STEPS="0,25,50,75,100,125,150,175,200,225,250,300,400,500,750,1000,1500,2000,2500,3000,4000,5000,6000,7000,8000,9000,10000"
EVAL_SCHEDULE="25:1000,100"
LOG_SCHEDULE="25:1000,100"

PIDS=()
NAMES=()
GPUS_USED=()
GPU_IDX=0

mkdir -p "${RUN_LOG_DIR}" "${TIMELINE_DIR}" "${PROBE_DIR}" "${CHECKPOINT_DIR}"

ts() {
    date '+%Y-%m-%d %H:%M:%S'
}

log_launcher() {
    local line="[$(ts)] $*"
    echo "${line}"
    echo "${line}" >> "${LAUNCHER_LOG}"
}

pick_gpu() {
    NEXT_GPU=${TRAIN_GPUS[$GPU_IDX]}
    GPU_IDX=$(( (GPU_IDX + 1) % ${#TRAIN_GPUS[@]} ))
}

count_running() {
    local n=0
    for pid in "${PIDS[@]}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            n=$((n + 1))
        fi
    done
    echo "${n}"
}

wait_for_slot() {
    while [ "$(count_running)" -ge "${MAX_TRAIN_CONCURRENT}" ]; do
        log_launcher "WAIT running=$(count_running)/${MAX_TRAIN_CONCURRENT}"
        sleep 20
    done
}

gems_ckpt() {
    local env=$1 kl=$2
    echo "checkpoints/gems/GeMS_${env}_b5_pretrained_latent32_beta${kl}_click1.0_seed58407201_ideal_init.ckpt"
}

env_value() {
    case "$1" in
        mix) echo "mix_divpen" ;;
        topdown) echo "topdown_divpen" ;;
        *) echo "ERROR: unknown env tag $1" >&2; exit 1 ;;
    esac
}

record_pid() {
    local label=$1 gpu=$2 pid=$3
    PIDS+=("${pid}")
    NAMES+=("${label}")
    GPUS_USED+=("${gpu}")
    log_launcher "LAUNCH label=${label} gpu=${gpu} pid=${pid} log=${RUN_LOG_DIR}/${label}.log"
}

run_train() {
    local env_tag=$1 beta=$2 gpu=$3
    local kl env label gems
    kl="0.01"
    env=$(env_value "${env_tag}")
    label="kl001_${env_tag}_b${beta}_ideal_init_rc"
    gems=$(gems_ckpt "${env}" "${kl}")
    [ -f "${gems}" ] || { log_launcher "ERROR missing_gems=${gems}"; exit 1; }

    CUDA_VISIBLE_DEVICES=${gpu} nohup "${PYTHON}" -u scripts/train_agent.py \
        --config "${CONFIG}" \
        --env_name "${env}" \
        --lambda_KL "${kl}" \
        --beta "${beta}" \
        --label_click_mode real \
        --max_timesteps 10000 \
        --eval_freq 100 \
        --eval_freq_schedule "${EVAL_SCHEDULE}" \
        --log_freq 100 \
        --log_freq_schedule "${LOG_SCHEDULE}" \
        --save_freq 50000 \
        --save_steps "${SAVE_STEPS}" \
        --gems_checkpoint "${gems}" \
        --experiment_name "${EXP_GROUP}/${label}" \
        --experiment_tag "${label}" \
        --swan_project "${SWAN_PROJECT}" \
        --dual_eval 1 \
        --eval_timeline_dir "${TIMELINE_DIR}" \
        --eval_step_zero 1 \
        --enable_train_geometry_probe 1 \
        --geometry_probe_dir "${PROBE_DIR}" \
        > "${RUN_LOG_DIR}/${label}.log" 2>&1 &
    record_pid "${label}" "${gpu}" "$!"
}

launch_train() {
    log_launcher "SESSION start mode=train project=${SWAN_PROJECT} train_gpus=${TRAIN_GPUS[*]} max_concurrent=${MAX_TRAIN_CONCURRENT}"
    GPU_IDX=0
    for spec in "mix 8" "mix 0" "topdown 8" "topdown 0"; do
        wait_for_slot
        pick_gpu
        run_train ${spec} "${NEXT_GPU}"
        sleep 1
    done
}

wait_all() {
    local fail=0
    for i in "${!PIDS[@]}"; do
        if wait "${PIDS[$i]}"; then
            log_launcher "DONE label=${NAMES[$i]} gpu=${GPUS_USED[$i]} pid=${PIDS[$i]}"
        else
            fail=$((fail + 1))
            log_launcher "FAIL label=${NAMES[$i]} gpu=${GPUS_USED[$i]} pid=${PIDS[$i]}"
        fi
    done
    return "${fail}"
}

print_status() {
    echo "Launcher log: ${LAUNCHER_LOG}"
    echo "Run logs:     ${RUN_LOG_DIR}/{run_label}.log"
    echo ""
    echo "Running real-click jobs:"
    pgrep -af "real_click_action_label_ablation|kl001_.*_ideal_init_rc" || true
    echo ""
    echo "Timeline completion:"
    "${PYTHON}" - <<'PY'
from pathlib import Path
import pandas as pd

root = Path("experiments/real_click_action_label_ablation/eval_timeline")
for path in sorted(root.glob("*/timeline.csv")):
    df = pd.read_csv(path)
    print(f"{path.parent.name}: last_step={int(df.step.iloc[-1])} rows={len(df)}")
PY
}

case "${GROUP}" in
    train)
        launch_train > >(tee -a "${LAUNCHER_STDOUT}") 2>&1
        wait_all > >(tee -a "${LAUNCHER_STDOUT}") 2>&1
        ;;
    status)
        print_status
        ;;
    *)
        echo "Usage: $0 {train|status}" >&2
        exit 1
        ;;
esac
