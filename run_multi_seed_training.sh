#!/bin/bash
#
# Multi-seed Training Script for CRC-Select
# Runs 4 seeds in parallel on 4 GPUs (seed 42 already done)
#
# Usage: ./run_multi_seed_training.sh

set -e

echo "============================================================"
echo "🚀 Starting Multi-Seed Training for CRC-Select"
echo "============================================================"
echo "Seeds: 123, 456, 789, 999 (seed 42 skipped - already done)"
echo "GPUs: 4 available (0,1,2,3)"
echo "Est. time: 8-12 hours per seed (will run in parallel)"
echo "============================================================"
echo ""

# Create checkpoint directory
mkdir -p checkpoints/CRC-Select
mkdir -p logs/training

# Array of seeds to train (excluding 42 which is done)
SEEDS=(123 456 789 999)
GPU_IDS=(0 1 2 3)

# Training hyperparameters
NUM_EPOCHS=300
ALPHA_RISK=0.1
COVERAGE=0.8
MU_INIT=1.0
RECALIBRATE_EVERY=10
BATCH_SIZE=128
WARMUP_EPOCHS=20

# Function to train one seed
train_seed() {
    local seed=$1
    local gpu=$2
    local logfile="logs/training/crc_select_seed_${seed}.log"
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting training for seed ${seed} on GPU ${gpu}"
    
    CUDA_VISIBLE_DEVICES=${gpu} python scripts/train_crc_select.py \
        --seed ${seed} \
        --num_epochs ${NUM_EPOCHS} \
        --alpha_risk ${ALPHA_RISK} \
        --coverage ${COVERAGE} \
        --mu_init ${MU_INIT} \
        --recalibrate_every ${RECALIBRATE_EVERY} \
        --warmup_epochs ${WARMUP_EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --dataset cifar10 \
        --dataroot ./data \
        --nesterov \
        2>&1 | tee ${logfile}
    
    if [ $? -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ Seed ${seed} completed successfully"
        echo "✅ COMPLETED: seed ${seed}" >> logs/training/status.txt
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ Seed ${seed} failed"
        echo "❌ FAILED: seed ${seed}" >> logs/training/status.txt
    fi
}

# Clear previous status
> logs/training/status.txt

# Launch training for all seeds in parallel
echo "Launching training jobs in parallel..."
for i in "${!SEEDS[@]}"; do
    seed=${SEEDS[$i]}
    gpu=${GPU_IDS[$i]}
    train_seed $seed $gpu &
    sleep 5  # Stagger starts slightly
done

echo ""
echo "============================================================"
echo "⏳ All training jobs launched!"
echo "============================================================"
echo "Monitor progress:"
echo "  - Overall: watch -n 30 'cat logs/training/status.txt'"
echo "  - Seed 123: tail -f logs/training/crc_select_seed_123.log"
echo "  - GPU usage: watch -n 1 nvidia-smi"
echo ""
echo "Wait for completion with: wait"
echo "============================================================"

# Wait for all background jobs to complete
wait

echo ""
echo "============================================================"
echo "🎉 All training jobs completed!"
echo "============================================================"
cat logs/training/status.txt
echo ""
echo "Next steps:"
echo "  1. Check checkpoints: ls -lh checkpoints/CRC-Select/"
echo "  2. Run evaluations: ./run_all_evaluations.sh"
echo "============================================================"
