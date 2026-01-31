#!/bin/bash
# manual_checkpoint_setup.sh
# Manual mapping of wandb runs to seeds

cd /home/admin1/Desktop/CRC-Select-Torch

echo "========================================="
echo "Manual Checkpoint Organization"
echo "========================================="
echo ""
echo "Found 4 trained models in wandb:"
echo ""
echo "Available checkpoints:"
echo ""
echo "[1] offline-run-20260126_094732-1fotfotl (Jan 26 09:52)"
echo "[2] offline-run-20260126_134357-wub8gc2w (Jan 26 13:44)"
echo "[3] offline-run-20260126_135412-s0wv6r38 (Jan 26 13:54)"
echo "[4] offline-run-20260127_091317-5aulwvrk (Jan 27 10:35) <- Current seed_42"
echo ""
echo "========================================="
echo "Auto Setup: Assuming sequential training"
echo "========================================="
echo ""
echo "Mapping:"
echo "  Run [4] (Jan 27 10:35) -> seed_42 (already done)"
echo "  Run [3] (Jan 26 13:54) -> seed_123"
echo "  Run [2] (Jan 26 13:44) -> seed_456"
echo "  Run [1] (Jan 26 09:52) -> seed_789"
echo ""
read -p "Use this mapping? (y/n): " confirm

if [ "$confirm" != "y" ]; then
    echo "Cancelled. Please map manually:"
    echo ""
    echo "  cp scripts/wandb/offline-run-XXXXX/files/checkpoints/checkpoint_best_val.pth checkpoints/seed_XXX.pth"
    exit 0
fi

echo ""
echo "Copying checkpoints..."

# Run 4 -> seed_42 (already exists)
echo "✓ seed_42 (already exists)"

# Run 3 -> seed_123
cp scripts/wandb/offline-run-20260126_135412-s0wv6r38/files/checkpoints/checkpoint_best_val.pth \
   checkpoints/seed_123.pth
echo "✓ Copied seed_123"

# Run 2 -> seed_456
cp scripts/wandb/offline-run-20260126_134357-wub8gc2w/files/checkpoints/checkpoint_best_val.pth \
   checkpoints/seed_456.pth
echo "✓ Copied seed_456"

# Run 1 -> seed_789
cp scripts/wandb/offline-run-20260126_094732-1fotfotl/files/checkpoints/checkpoint_best_val.pth \
   checkpoints/seed_789.pth
echo "✓ Copied seed_789"

echo ""
echo "========================================="
echo "Results"
echo "========================================="
ls -lh checkpoints/

num_seeds=$(ls checkpoints/seed_*.pth 2>/dev/null | wc -l)
echo ""
echo "Total seeds: $num_seeds"

if [ $num_seeds -eq 4 ]; then
    echo "✅ Ready! You have 4 seeds (42, 123, 456, 789)"
    echo ""
    echo "Note: You trained 4 seeds, not 5 (missing seed_999)"
    echo ""
    echo "Next step: Run evaluation"
    echo "  ./run_eval_all_seeds.sh"
fi
