#!/bin/bash
# organize_multi_seed_checkpoints.sh
# Automatically organize checkpoints from wandb runs by detecting seeds

cd /home/admin1/Desktop/CRC-Select-Torch

echo "========================================="
echo "Auto-Organizing Multi-Seed Checkpoints"
echo "========================================="
echo ""

# Create checkpoints directory if not exists
mkdir -p checkpoints

WANDB_DIR="scripts/wandb"
counter=0

# Iterate through all wandb runs (newest first)
for run_dir in $(ls -t ${WANDB_DIR}/offline-run-*/); do
    if [ -d "$run_dir" ]; then
        run_name=$(basename "$run_dir")
        checkpoint_file="${run_dir}files/checkpoints/checkpoint_best_val.pth"
        
        if [ -f "$checkpoint_file" ]; then
            # Try to extract seed from wandb config or files
            config_file="${run_dir}files/wandb-summary.json"
            config_yaml="${run_dir}files/config.yaml"
            
            seed=""
            
            # Try different methods to get seed
            if [ -f "$config_yaml" ]; then
                # Method 1: From config.yaml
                seed=$(grep -E "^\s+seed:" "$config_yaml" | awk '{print $2}' | head -1)
            fi
            
            if [ -z "$seed" ] && [ -f "$config_file" ]; then
                # Method 2: From wandb-summary.json
                seed=$(grep -o '"seed":[[:space:]]*[0-9]*' "$config_file" | grep -o '[0-9]*' | head -1)
            fi
            
            if [ -z "$seed" ]; then
                # Method 3: From wandb-metadata.json
                metadata_file="${run_dir}files/wandb-metadata.json"
                if [ -f "$metadata_file" ]; then
                    seed=$(grep -o '"seed":[[:space:]]*[0-9]*' "$metadata_file" | grep -o '[0-9]*' | head -1)
                fi
            fi
            
            # Check if checkpoint already exists
            if [ -n "$seed" ]; then
                target_file="checkpoints/seed_${seed}.pth"
                
                if [ -f "$target_file" ]; then
                    echo "[$counter] Seed $seed: Already exists (skipping)"
                else
                    # Copy checkpoint
                    cp "$checkpoint_file" "$target_file"
                    echo "[$counter] Seed $seed: ✓ Copied from $run_name"
                    counter=$((counter + 1))
                fi
            else
                # No seed found - show for manual mapping
                echo "[$counter] Unknown seed in $run_name"
                echo "    Checkpoint: $checkpoint_file"
                echo "    (No seed found in config files)"
                echo ""
            fi
        fi
    fi
done

echo ""
echo "========================================="
echo "Results"
echo "========================================="
echo ""
echo "Checkpoints directory:"
ls -lh checkpoints/
echo ""

# Count seeds
num_seeds=$(ls checkpoints/seed_*.pth 2>/dev/null | wc -l)
echo "Total seeds found: $num_seeds"

if [ $num_seeds -ge 5 ]; then
    echo "✅ You have $num_seeds seeds - ready for statistical analysis!"
elif [ $num_seeds -ge 3 ]; then
    echo "⚠️  You have $num_seeds seeds - minimum for paper, but 5+ recommended"
elif [ $num_seeds -ge 1 ]; then
    echo "⚠️  You have $num_seeds seed(s) - need at least 3-5 for violation rate"
else
    echo "❌ No seeds found - please check your training outputs"
fi

echo ""
echo "Next step: Run evaluation on all seeds"
echo "  ./run_eval_all_seeds.sh"
