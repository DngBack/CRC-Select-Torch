#!/bin/bash
#
# Train Vanilla SelectiveNet for 3 seeds (123, 456, 999)
# Matching the seeds that have CRC-Select checkpoints
#
# Usage: ./train_vanilla_3seeds.sh

set -e

echo "============================================================"
echo "🚀 Training Vanilla SelectiveNet (3 seeds)"
echo "============================================================"
echo "Seeds: 123, 456, 999"
echo "GPUs: Will use GPU 0, 1, 2"
echo "Est. time: ~8-10 hours per seed"
echo "============================================================"
echo ""

# Create directories
mkdir -p checkpoints/vanilla
mkdir -p logs/vanilla_training

# Array of seeds
SEEDS=(123 456 999)

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

# Clear status file
> logs/vanilla_training/status.txt

# Train all 3 seeds in parallel (one per GPU)
echo "Training all 3 seeds in parallel..."
train_vanilla_seed 123 0 &
train_vanilla_seed 456 1 &
train_vanilla_seed 999 2 &

# Wait for all background jobs to finish
wait

echo ""
echo "============================================================"
echo "✅ All Vanilla Training Complete!"
echo "============================================================"
echo ""
echo "Training Status:"
cat logs/vanilla_training/status.txt
echo ""
echo "Available checkpoints:"
ls -lh checkpoints/vanilla/ 2>/dev/null || echo "No checkpoints found"
echo ""
echo "Next steps:"
echo "  1. Check training logs: ls logs/vanilla_training/"
echo "  2. Run evaluations: ./run_all_evaluations.sh"
echo "============================================================"
