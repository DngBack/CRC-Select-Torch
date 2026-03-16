#!/bin/bash
#
# Multi-seed CRC-Select Training for CIFAR-100
#
# Uses WideResNet-28-10 backbone with randcrop augmentation.
# Runs 5 seeds sequentially (or adjust GPU assignments for parallel).
#
# Usage:
#   ./train_cifar100_multi_seed.sh              # all 5 seeds
#   ./train_cifar100_multi_seed.sh --seeds "42 123"
#   CUDA_VISIBLE_DEVICES=1 ./train_cifar100_multi_seed.sh

set -e

GPU=${CUDA_VISIBLE_DEVICES:-0}
SEEDS=(42 123 456 789 999)

while [[ $# -gt 0 ]]; do
    case $1 in
        --seeds) IFS=' ' read -ra SEEDS <<< "$2"; shift 2 ;;
        *) shift ;;
    esac
done

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

mkdir -p checkpoints/CRC-Select-cifar100
mkdir -p logs/training/cifar100

echo "============================================================"
echo " CRC-Select: CIFAR-100 Multi-Seed Training"
echo "============================================================"
echo " GPU: ${GPU}"
echo " Seeds: ${SEEDS[*]}"
echo " Backbone: WideResNet-28-10"
echo " Epochs: 300"
echo "============================================================"

# Hyperparameters (WideResNet-28-10, CIFAR-100)
BACKBONE=wrn28_10
DIM_FEATURES=640
DATASET=cifar100
DATAROOT=./data
AUGMENTATION=randcrop
NUM_EPOCHS=300
LR=0.1
WD=0.0005
MOMENTUM=0.9
SCHEDULER_STEP=60
SCHEDULER_GAMMA=0.2
BATCH_SIZE=128
NUM_WORKERS=8

ALPHA_RISK=0.1
COVERAGE=0.8
LM=32.0
ALPHA_MIX=0.5
TAU=0.5
WARMUP_EPOCHS=30
RECALIBRATE_EVERY=5
MU_INIT=1.0
DUAL_LR=0.01
DELTA=0.1

errors=()

for seed in "${SEEDS[@]}"; do
    CKPT_OUT="checkpoints/CRC-Select-cifar100/seed_${seed}.pth"
    LOGFILE="logs/training/cifar100/crc_select_seed_${seed}.log"

    if [[ -f "$CKPT_OUT" ]]; then
        log "SKIP seed=${seed} — checkpoint already exists: ${CKPT_OUT}"
        continue
    fi

    log "START seed=${seed}"

    CUDA_VISIBLE_DEVICES=${GPU} python scripts/train_crc_select.py \
        --seed          ${seed} \
        --backbone      ${BACKBONE} \
        --dim_features  ${DIM_FEATURES} \
        --dataset       ${DATASET} \
        --dataroot      ${DATAROOT} \
        --augmentation  ${AUGMENTATION} \
        --num_epochs    ${NUM_EPOCHS} \
        --lr            ${LR} \
        --wd            ${WD} \
        --momentum      ${MOMENTUM} \
        --nesterov \
        --scheduler_step  ${SCHEDULER_STEP} \
        --scheduler_gamma ${SCHEDULER_GAMMA} \
        --batch_size    ${BATCH_SIZE} \
        --num_workers   ${NUM_WORKERS} \
        --alpha_risk    ${ALPHA_RISK} \
        --coverage      ${COVERAGE} \
        --lm            ${LM} \
        --alpha         ${ALPHA_MIX} \
        --tau           ${TAU} \
        --warmup_epochs ${WARMUP_EPOCHS} \
        --recalibrate_every ${RECALIBRATE_EVERY} \
        --mu_init       ${MU_INIT} \
        --dual_lr       ${DUAL_LR} \
        --delta         ${DELTA} \
        --use_dual_update \
        --unobserve \
        2>&1 | tee "${LOGFILE}"

    if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
        log "DONE seed=${seed}"
        # Copy checkpoint to cifar100-specific directory
        SHARED_CKPT="checkpoints/CRC-Select/seed_${seed}.pth"
        if [[ -f "$SHARED_CKPT" && ! -f "$CKPT_OUT" ]]; then
            cp "$SHARED_CKPT" "$CKPT_OUT"
            log "Copied checkpoint -> ${CKPT_OUT}"
        fi
    else
        log "FAILED seed=${seed}"
        errors+=("seed_${seed}")
    fi
done

echo ""
echo "============================================================"
if [[ ${#errors[@]} -eq 0 ]]; then
    echo " All seeds completed successfully."
else
    echo " FAILED seeds: ${errors[*]}"
fi
echo "============================================================"
echo ""
echo "Next step — run baseline evaluations:"
echo "  ./run_baselines_cifar100.sh"
