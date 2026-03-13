#!/bin/bash
#
# Sequential Training Script — 1 GPU
#
# Trains all models needed for the paper experiments in sequence:
#   Phase A: Vanilla SelectiveNet × 4 seeds  (enables MSP, Energy, TempScaled, PosthocCRC)
#   Phase B: DeepGambler × 4 seeds           (for DeepGambler baseline)
#   Phase C: CRC-Select seed_999 only        (to complete the CRC-Select 4-seed set)
#
# Skip already-done seeds by checking checkpoint existence.
#
# Usage:
#   ./run_training_sequential.sh           # Run all phases (A + B + C)
#   ./run_training_sequential.sh --phase A # Only vanilla
#   ./run_training_sequential.sh --phase B # Only DeepGambler
#   ./run_training_sequential.sh --phase C # Only CRC-Select seed_999
#
# Estimated total time on 1 GPU: ~54h (16h vanilla + 16h DeepGambler + ~8h CRC-Select)
# Run in background:  nohup ./run_training_sequential.sh > logs/training_all.log 2>&1 &

set -e

GPU=${CUDA_VISIBLE_DEVICES:-0}
PHASE=${2:-all}   # all | A | B | C (second positional or via --phase flag)

# Parse --phase flag
while [[ $# -gt 0 ]]; do
    case $1 in
        --phase) PHASE="$2"; shift 2 ;;
        *) shift ;;
    esac
done

SEEDS=(42 123 456 999)

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

mkdir -p checkpoints/{vanilla,CRC-Select,DeepGambler}
mkdir -p logs/{training,vanilla_training,evaluation}

# ============================================================
# PHASE A: Vanilla SelectiveNet
# ============================================================
train_vanilla() {
    local seed=$1
    local ckpt="checkpoints/vanilla/seed_${seed}.pth"
    if [[ -f "$ckpt" ]]; then
        log "SKIP vanilla seed=${seed} — checkpoint exists: $ckpt"
        return 0
    fi
    log "START vanilla seed=${seed}"
    CUDA_VISIBLE_DEVICES=${GPU} python3 scripts/train.py \
        --seed ${seed} \
        --num_epochs 300 \
        --coverage 0.8 \
        --batch_size 128 \
        --dataset cifar10 \
        --dataroot ./data \
        --nesterov \
        2>&1 | tee logs/vanilla_training/vanilla_seed_${seed}.log
    if [[ -f "$ckpt" ]]; then
        log "DONE vanilla seed=${seed} — checkpoint: $ckpt"
    else
        log "ERROR vanilla seed=${seed} — checkpoint NOT found after training!"
        exit 1
    fi
}

# ============================================================
# PHASE B: Deep Gambler
# ============================================================
train_gambler() {
    local seed=$1
    local ckpt="checkpoints/DeepGambler/seed_${seed}.pth"
    if [[ -f "$ckpt" ]]; then
        log "SKIP DeepGambler seed=${seed} — checkpoint exists: $ckpt"
        return 0
    fi
    log "START DeepGambler seed=${seed}"
    CUDA_VISIBLE_DEVICES=${GPU} python3 scripts/baseline_deep_gambler.py \
        --train \
        --seed ${seed} \
        --num_epochs 300 \
        --reward 2.0 \
        --dataset cifar10 \
        --dataroot ./data \
        --batch_size 128 \
        2>&1 | tee logs/training/deep_gambler_seed_${seed}.log
    if [[ -f "$ckpt" ]]; then
        log "DONE DeepGambler seed=${seed} — checkpoint: $ckpt"
    else
        log "ERROR DeepGambler seed=${seed} — checkpoint NOT found!"
        exit 1
    fi
}

# ============================================================
# PHASE C: CRC-Select seed_999 (completes the 4-seed set)
# ============================================================
train_crc_select() {
    local seed=$1
    local ckpt="checkpoints/CRC-Select/seed_${seed}.pth"
    if [[ -f "$ckpt" ]]; then
        log "SKIP CRC-Select seed=${seed} — checkpoint exists: $ckpt"
        return 0
    fi
    log "START CRC-Select seed=${seed}"
    CUDA_VISIBLE_DEVICES=${GPU} python3 scripts/train_crc_select.py \
        --seed ${seed} \
        --num_epochs 300 \
        --alpha_risk 0.1 \
        --coverage 0.8 \
        --mu_init 1.0 \
        --recalibrate_every 10 \
        --warmup_epochs 20 \
        --batch_size 128 \
        --dataset cifar10 \
        --dataroot ./data \
        --nesterov \
        2>&1 | tee logs/training/crc_select_seed_${seed}_retrain.log
    if [[ -f "$ckpt" ]]; then
        log "DONE CRC-Select seed=${seed} — checkpoint: $ckpt"
    else
        log "ERROR CRC-Select seed=${seed} — checkpoint NOT found!"
        exit 1
    fi
}

# ============================================================
# Run phases
# ============================================================
echo "============================================================"
echo " CRC-Select Paper: Sequential Training Pipeline"
echo "============================================================"
echo " GPU: ${GPU}"
echo " Phase: ${PHASE}"
echo " Seeds: ${SEEDS[*]}"
echo "============================================================"

if [[ "$PHASE" == "all" || "$PHASE" == "A" ]]; then
    echo ""
    echo "=== PHASE A: Vanilla SelectiveNet ==="
    for seed in "${SEEDS[@]}"; do
        train_vanilla $seed
    done
    echo "=== Phase A complete ==="
fi

if [[ "$PHASE" == "all" || "$PHASE" == "B" ]]; then
    echo ""
    echo "=== PHASE B: Deep Gambler ==="
    for seed in "${SEEDS[@]}"; do
        train_gambler $seed
    done
    echo "=== Phase B complete ==="
fi

if [[ "$PHASE" == "all" || "$PHASE" == "C" ]]; then
    echo ""
    echo "=== PHASE C: CRC-Select (seeds 123, 456, 999 — replaces old-format results) ==="
    for seed in 123 456 999; do
        train_crc_select $seed
    done
    echo "=== Phase C complete ==="
fi

echo ""
echo "============================================================"
echo " Training complete! Checkpoints:"
echo "============================================================"
ls checkpoints/vanilla/   2>/dev/null | sed 's/^/  vanilla: /'
ls checkpoints/DeepGambler/ 2>/dev/null | sed 's/^/  DeepGambler: /'
ls checkpoints/CRC-Select/  2>/dev/null | sed 's/^/  CRC-Select: /'
echo ""
echo " Next step: ./run_all_baselines.sh"
