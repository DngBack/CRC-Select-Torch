#!/bin/bash
# run_eval_all_seeds.sh
# Run evaluation on all available seeds

cd /home/admin1/Desktop/CRC-Select-Torch

echo "========================================="
echo "Multi-Seed Evaluation"
echo "========================================="
echo ""

# Check available seeds
SEEDS=()
for checkpoint in checkpoints/seed_*.pth; do
    if [ -f "$checkpoint" ]; then
        seed=$(basename "$checkpoint" .pth | sed 's/seed_//')
        SEEDS+=($seed)
    fi
done

if [ ${#SEEDS[@]} -eq 0 ]; then
    echo "❌ No checkpoints found in checkpoints/"
    echo ""
    echo "Please organize checkpoints first."
    exit 1
fi

echo "Found ${#SEEDS[@]} seed(s): ${SEEDS[*]}"
echo ""
echo "Starting evaluations..."
echo ""

success_count=0
fail_count=0

for seed in "${SEEDS[@]}"; do
    echo "======================================="
    echo "Evaluating seed $seed..."
    echo "======================================="
    
    python3 scripts/evaluate_for_paper.py \
        --checkpoint checkpoints/seed_${seed}.pth \
        --seed $seed \
        --method_name "CRC-Select" \
        --dataset cifar10 \
        --output_dir ../results_paper
    
    if [ $? -eq 0 ]; then
        echo "✅ Seed $seed completed"
        success_count=$((success_count + 1))
    else
        echo "❌ Seed $seed failed"
        fail_count=$((fail_count + 1))
    fi
    echo ""
done

echo ""
echo "========================================="
echo "Summary: $success_count/${#SEEDS[@]} successful"
echo "========================================="
