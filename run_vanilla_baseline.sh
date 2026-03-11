#!/bin/bash
#
# Vanilla SelectiveNet Baseline Training (all 5 seeds)
# Run AFTER CRC-Select training completes
#
# Usage: ./run_vanilla_baseline.sh

set -e

echo "============================================================"
echo "🚀 Starting Vanilla SelectiveNet Baseline Training"
echo "============================================================"
echo "Seeds: 42, 123, 456, 789, 999 (all 5)"
echo "GPUs: 4 available (will run in batches)"
echo "Est. time: 8-12 hours per seed"
echo "============================================================"
echo ""

# Create checkpoint directory
mkdir -p checkpoints/vanilla
mkdir -p logs/vanilla_training

# Array of all seeds
SEEDS=(42 123 456 789 999)

# Training hyperparameters (matching CRC-Select for fair comparison)
NUM_EPOCHS=300
COVERAGE=0.8
BATCH_SIZE=128

# Function to train one seed
train_vanilla_seed() {
    local seed=$1
    local gpu=$2
    local logfile="logs/vanilla_training/vanilla_seed_${seed}.log"
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting vanilla training for seed ${seed} on GPU ${gpu}"
    
    CUDA_VISIBLE_DEVICES=${gpu} python scripts/train.py \
        --seed ${seed} \
        --num_epochs ${NUM_EPOCHS} \
        --coverage ${COVERAGE} \
        --batch_size ${BATCH_SIZE} \
        --dataset cifar10 \
        --dataroot ./data \
        --nesterov \
        2>&1 | tee ${logfile}
    
    if [ $? -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ Vanilla seed ${seed} completed"
        echo "✅ COMPLETED: seed ${seed}" >> logs/vanilla_training/status.txt
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ Vanilla seed ${seed} failed"
        echo "❌ FAILED: seed ${seed}" >> logs/vanilla_training/status.txt
    fi
}

# Clear previous status
> logs/vanilla_training/status.txt

# Run first batch (4 seeds in parallel)
echo "Batch 1: Training seeds 42, 123, 456, 789 in parallel..."
train_vanilla_seed 42 0 &
train_vanilla_seed 123 1 &
train_vanilla_seed 456 2 &
train_vanilla_seed 789 3 &
wait

# Run second batch (1 seed)
echo "Batch 2: Training seed 999..."
train_vanilla_seed 999 0 &
wait

echo ""
echo "============================================================"
echo "🎉 All vanilla baseline training completed!"
echo "============================================================"
cat logs/vanilla_training/status.txt
echo ""
echo "Next steps:"
echo "  1. Check checkpoints: ls -lh checkpoints/vanilla/"
echo "  2. Run evaluations: ./run_all_evaluations.sh"
echo "============================================================"
