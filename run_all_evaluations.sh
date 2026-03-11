#!/bin/bash
#
# Comprehensive Evaluation Script
# Evaluates CRC-Select, Vanilla, and Post-hoc CRC for all seeds
#
# Usage: ./run_all_evaluations.sh

set -e

echo "============================================================"
echo "📊 Starting Comprehensive Evaluation"
echo "============================================================"
echo "Methods: CRC-Select, Vanilla, Post-hoc CRC"
echo "Seeds: 42, 123, 456, 789, 999"
echo "Est. time: ~2-3 hours total"
echo "============================================================"
echo ""

SEEDS=(42 123 456 789 999)
mkdir -p logs/evaluation

# Function to evaluate CRC-Select
eval_crc_select() {
    local seed=$1
    echo "[$(date '+%H:%M:%S')] Evaluating CRC-Select seed ${seed}..."
    
    python scripts/evaluate_for_paper.py \
        --checkpoint checkpoints/CRC-Select/seed_${seed}.pth \
        --seed ${seed} \
        --method_name "CRC-Select" \
        --dataset cifar10 \
        --ood_dataset svhn \
        --output_dir results_paper \
        2>&1 | tee logs/evaluation/crc_select_seed_${seed}.log
}

# Function to evaluate Vanilla
eval_vanilla() {
    local seed=$1
    echo "[$(date '+%H:%M:%S')] Evaluating Vanilla seed ${seed}..."
    
    python scripts/evaluate_for_paper.py \
        --checkpoint checkpoints/vanilla/seed_${seed}.pth \
        --seed ${seed} \
        --method_name "vanilla" \
        --dataset cifar10 \
        --ood_dataset svhn \
        --output_dir results_paper \
        2>&1 | tee logs/evaluation/vanilla_seed_${seed}.log
}

# Function to evaluate Post-hoc CRC
eval_posthoc() {
    local seed=$1
    echo "[$(date '+%H:%M:%S')] Evaluating Post-hoc CRC seed ${seed}..."
    
    python scripts/baseline_posthoc_crc.py \
        --checkpoint checkpoints/vanilla/seed_${seed}.pth \
        --seed ${seed} \
        --alpha_risk 0.1 \
        --dataset cifar10 \
        --ood_dataset svhn \
        --output_dir results_paper/posthoc_crc \
        2>&1 | tee logs/evaluation/posthoc_seed_${seed}.log
}

echo "Step 1/3: Evaluating CRC-Select..."
for seed in "${SEEDS[@]}"; do
    if [ -f "checkpoints/CRC-Select/seed_${seed}.pth" ]; then
        eval_crc_select $seed
    else
        echo "⚠️  Skipping CRC-Select seed ${seed} (checkpoint not found)"
    fi
done

echo ""
echo "Step 2/3: Evaluating Vanilla..."
for seed in "${SEEDS[@]}"; do
    if [ -f "checkpoints/vanilla/seed_${seed}.pth" ]; then
        eval_vanilla $seed
    else
        echo "⚠️  Skipping Vanilla seed ${seed} (checkpoint not found)"
    fi
done

echo ""
echo "Step 3/3: Evaluating Post-hoc CRC..."
for seed in "${SEEDS[@]}"; do
    if [ -f "checkpoints/vanilla/seed_${seed}.pth" ]; then
        eval_posthoc $seed
    else
        echo "⚠️  Skipping Post-hoc CRC seed ${seed} (checkpoint not found)"
    fi
done

echo ""
echo "============================================================"
echo "✅ Evaluation Complete!"
echo "============================================================"
echo ""
echo "Results structure:"
find results_paper -name "*.csv" | head -20
echo ""
echo "Next steps:"
echo "  1. Aggregate results: python scripts/aggregate_results.py"
echo "  2. Compute violations: python scripts/compute_violation_rate.py"
echo "  3. Generate figures: python scripts/generate_paper_figures.py"
echo "============================================================"
