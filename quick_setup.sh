#!/bin/bash
# quick_setup.sh - Quick setup for evaluation

cd /home/admin1/Desktop/CRC-Select-Torch

echo "========================================="
echo "Quick Setup: Organizing Checkpoints"
echo "========================================="

# Create checkpoints directory
mkdir -p checkpoints

# Copy latest checkpoint
echo "Copying latest checkpoint..."
cp scripts/wandb/latest-run/files/checkpoints/checkpoint_best_val.pth checkpoints/seed_42.pth

if [ $? -eq 0 ]; then
    echo "✓ Successfully copied checkpoint to checkpoints/seed_42.pth"
    echo ""
    echo "File size:"
    ls -lh checkpoints/seed_42.pth
    echo ""
    echo "========================================="
    echo "Ready to Run Evaluation!"
    echo "========================================="
    echo ""
    echo "Run this command:"
    echo ""
    echo "  python scripts/evaluate_for_paper.py \\"
    echo "      --checkpoint checkpoints/seed_42.pth \\"
    echo "      --seed 42 \\"
    echo "      --method_name CRC-Select"
    echo ""
else
    echo "✗ Failed to copy checkpoint"
    exit 1
fi
