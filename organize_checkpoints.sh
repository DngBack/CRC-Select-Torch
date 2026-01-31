#!/bin/bash
# organize_checkpoints.sh
# Copy checkpoint files from wandb runs to organized checkpoints folder

cd /home/admin1/Desktop/CRC-Select-Torch

# Create checkpoints directory
mkdir -p checkpoints

echo "========================================="
echo "Organizing Checkpoint Files"
echo "========================================="

# Find all checkpoint files in wandb runs
WANDB_DIR="scripts/wandb"

# List all wandb runs and their checkpoints
echo ""
echo "Available wandb runs:"
echo ""

counter=1
for run_dir in ${WANDB_DIR}/offline-run-*/; do
    if [ -d "$run_dir" ]; then
        run_name=$(basename "$run_dir")
        checkpoint_file="${run_dir}files/checkpoints/checkpoint_best_val.pth"
        
        if [ -f "$checkpoint_file" ]; then
            # Try to get seed from config
            config_file="${run_dir}files/config.yaml"
            if [ -f "$config_file" ]; then
                seed=$(grep -E "^  seed:" "$config_file" | awk '{print $2}' | head -1)
                dataset=$(grep -E "^  dataset:" "$config_file" | awk '{print $2}' | head -1)
                
                echo "[$counter] Run: $run_name"
                echo "    Seed: ${seed:-unknown}"
                echo "    Dataset: ${dataset:-unknown}"
                echo "    Checkpoint: $checkpoint_file"
                echo ""
                
                counter=$((counter + 1))
            fi
        fi
    fi
done

echo "========================================="
echo "Interactive Mode"
echo "========================================="
echo ""
echo "Choose an option:"
echo "  1. Copy latest run as seed_42.pth"
echo "  2. Copy all runs with detected seeds"
echo "  3. Manual: I'll tell you the run-to-seed mapping"
echo "  4. Exit"
echo ""
read -p "Enter choice (1-4): " choice

case $choice in
    1)
        # Copy latest run
        latest_checkpoint="scripts/wandb/latest-run/files/checkpoints/checkpoint_best_val.pth"
        if [ -f "$latest_checkpoint" ]; then
            cp "$latest_checkpoint" checkpoints/seed_42.pth
            echo "✓ Copied latest run to checkpoints/seed_42.pth"
        else
            echo "✗ Latest checkpoint not found"
        fi
        ;;
    2)
        # Auto-detect and copy
        for run_dir in ${WANDB_DIR}/offline-run-*/; do
            config_file="${run_dir}files/config.yaml"
            checkpoint_file="${run_dir}files/checkpoints/checkpoint_best_val.pth"
            
            if [ -f "$config_file" ] && [ -f "$checkpoint_file" ]; then
                seed=$(grep -E "^  seed:" "$config_file" | awk '{print $2}' | head -1)
                
                if [ -n "$seed" ]; then
                    cp "$checkpoint_file" "checkpoints/seed_${seed}.pth"
                    echo "✓ Copied seed $seed from $(basename $run_dir)"
                fi
            fi
        done
        ;;
    3)
        echo ""
        echo "Manual mapping mode:"
        echo "For each run, enter the seed number (or 'skip' to skip)"
        echo ""
        
        for run_dir in ${WANDB_DIR}/offline-run-*/; do
            checkpoint_file="${run_dir}files/checkpoints/checkpoint_best_val.pth"
            
            if [ -f "$checkpoint_file" ]; then
                run_name=$(basename "$run_dir")
                echo "Run: $run_name"
                read -p "  Enter seed number (or 'skip'): " seed_input
                
                if [ "$seed_input" != "skip" ] && [ -n "$seed_input" ]; then
                    cp "$checkpoint_file" "checkpoints/seed_${seed_input}.pth"
                    echo "  ✓ Copied to checkpoints/seed_${seed_input}.pth"
                fi
                echo ""
            fi
        done
        ;;
    4)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "========================================="
echo "Results"
echo "========================================="
ls -lh checkpoints/

echo ""
echo "Done! You can now run:"
echo "  python scripts/evaluate_for_paper.py --checkpoint checkpoints/seed_42.pth --seed 42"
