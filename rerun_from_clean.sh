#!/bin/bash
# rerun_from_clean.sh
# Removes stale checkpoints and results that were generated with bugs, then
# reruns training + evaluation from scratch so all comparisons are fair.
#
# Bugs fixed before running this:
#   1. train.py: was training on full 50K examples (data leakage into test set)
#                → now uses get_split_loaders (70% train, 10% cal, 20% test)
#   2. DeepGambler: reward=2.2 caused always-abstain → lowered to 2.0, 300 epochs
#   3. run_all_baselines.sh: skip-if-done kept stale seed_42 results from old
#                checkpoint → delete them so they run fresh

set -e
cd "$(dirname "$0")"

GPU=${GPU:-0}
echo "=== CLEAN RERUN (GPU $GPU) ==="
echo ""

# ── Step 1: Remove stale vanilla checkpoints (all 4 seeds trained without split)
echo "[1/5] Removing stale vanilla checkpoints..."
rm -f checkpoints/vanilla/seed_42.pth
rm -f checkpoints/vanilla/seed_123.pth
rm -f checkpoints/vanilla/seed_456.pth
rm -f checkpoints/vanilla/seed_999.pth
echo "  Done."

# ── Step 2: Remove stale DeepGambler checkpoints (reward=2.2 caused abstention)
echo "[2/5] Removing stale DeepGambler checkpoints..."
rm -f checkpoints/DeepGambler/seed_42.pth
rm -f checkpoints/DeepGambler/seed_123.pth
rm -f checkpoints/DeepGambler/seed_456.pth
rm -f checkpoints/DeepGambler/seed_999.pth
echo "  Done."

# ── Step 3: Remove all stale evaluation results so run_all_baselines.sh doesn't skip
echo "[3/5] Removing stale evaluation results..."
for method in vanilla posthoc_crc MSP Energy TempScaled_MSP DeepGambler; do
    for seed in 42 123 456 999; do
        rm -rf "results_paper/${method}/seed_${seed}"
    done
done
rm -rf results_paper/aggregated
rm -rf results_paper/violation_analysis
rm -rf figures/*.pdf figures/*.png 2>/dev/null || true
echo "  Done."

# ── Step 4: Retrain vanilla (4 seeds) + DeepGambler (4 seeds) sequentially
echo "[4/5] Running training (vanilla + DeepGambler only)..."
bash run_training_sequential.sh --phase A
bash run_training_sequential.sh --phase B
echo "  Training done."

# ── Step 5: Evaluate all methods for all seeds, then aggregate + figures
echo "[5/5] Running evaluations + analysis..."
bash run_all_baselines.sh
bash run_final_analysis.sh

echo ""
echo "=== ALL DONE ==="
echo "Results: results_paper/"
echo "Figures: figures/"
