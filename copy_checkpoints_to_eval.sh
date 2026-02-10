#!/bin/bash
#
# Copy trained checkpoints from wandb runs to evaluation directory
#

set -e

echo "============================================================"
echo "📦 Copying trained checkpoints for evaluation"
echo "============================================================"

# Create checkpoint directory
mkdir -p checkpoints/CRC-Select

# Function to extract seed from wandb config
get_seed_from_run() {
    local run_dir=$1
    local config_file="${run_dir}/files/config.yaml"
    
    if [ -f "$config_file" ]; then
        grep -A 1 "seed:" "$config_file" | tail -1 | awk '{print $2}'
    else
        echo "unknown"
    fi
}

# Process all recent runs
for run_dir in wandb/run-20260209_21*/; do
    if [ -d "${run_dir}files/checkpoints/" ]; then
        seed=$(get_seed_from_run "$run_dir")
        
        if [ "$seed" != "unknown" ]; then
            echo "Processing run: $(basename $run_dir) (seed: $seed)"
            
            # Copy checkpoint_best_val.pth as the main checkpoint
            if [ -f "${run_dir}files/checkpoints/checkpoint_best_val.pth" ]; then
                cp "${run_dir}files/checkpoints/checkpoint_best_val.pth" "checkpoints/CRC-Select/seed_${seed}.pth"
                echo "  ✓ Copied checkpoint for seed ${seed}"
            else
                echo "  ⚠️  No checkpoint_best_val.pth found"
            fi
        else
            echo "⚠️  Skipping $(basename $run_dir) - no seed found in config"
        fi
    fi
done

echo ""
echo "============================================================"
echo "✅ Checkpoint copy complete!"
echo "============================================================"
echo ""
echo "Available checkpoints:"
ls -lh checkpoints/CRC-Select/
echo ""
