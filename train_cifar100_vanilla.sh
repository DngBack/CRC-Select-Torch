#!/bin/bash
#
# Vanilla SelectiveNet training for CIFAR-100 (multi-seed)
#
# Produces baseline checkpoints in checkpoints/vanilla-cifar100/
# These are needed before running run_baselines_cifar100.sh.
#
# Usage:
#   ./train_cifar100_vanilla.sh
#   ./train_cifar100_vanilla.sh --seeds "42 123"
#   CUDA_VISIBLE_DEVICES=1 ./train_cifar100_vanilla.sh

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

CKPT_DIR=checkpoints/vanilla-cifar100
mkdir -p ${CKPT_DIR}
mkdir -p logs/training/cifar100

echo "============================================================"
echo " Vanilla SelectiveNet: CIFAR-100 Multi-Seed Training"
echo "============================================================"
echo " GPU: ${GPU}"
echo " Seeds: ${SEEDS[*]}"
echo " Backbone: WideResNet-28-10"
echo " Epochs: 300"
echo " Checkpoint dir: ${CKPT_DIR}"
echo "============================================================"

errors=()

for seed in "${SEEDS[@]}"; do
    CKPT_OUT="${CKPT_DIR}/seed_${seed}.pth"
    LOGFILE="logs/training/cifar100/vanilla_seed_${seed}.log"

    if [[ -f "$CKPT_OUT" ]]; then
        log "SKIP seed=${seed} — checkpoint already exists"
        continue
    fi

    log "START vanilla seed=${seed}"

    CUDA_VISIBLE_DEVICES=${GPU} python scripts/train.py \
        --seed          ${seed} \
        --backbone      wrn28_10 \
        --dim_features  640 \
        --dataset       cifar100 \
        --dataroot      ./data \
        --augmentation  randcrop \
        --num_epochs    300 \
        --lr            0.1 \
        --wd            0.0005 \
        --momentum      0.9 \
        --nesterov \
        --batch_size    128 \
        --num_workers   8 \
        --coverage      0.8 \
        --alpha         0.5 \
        --checkpoint_dir ${CKPT_DIR} \
        --unobserve \
        2>&1 | tee "${LOGFILE}"

    if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
        log "DONE seed=${seed}"
        # train.py saves to checkpoint_dir/seed_{seed}.pth — verify
        if [[ -f "$CKPT_OUT" ]]; then
            log "Checkpoint verified: ${CKPT_OUT}"
        else
            log "WARNING: checkpoint not found at expected path ${CKPT_OUT}"
            errors+=("ckpt_missing_seed_${seed}")
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
    echo " FAILED: ${errors[*]}"
fi
echo "============================================================"
echo ""
echo "Next step:"
echo "  ./train_cifar100_multi_seed.sh   # train CRC-Select on CIFAR-100"
echo "  ./run_baselines_cifar100.sh      # evaluate all baselines"
