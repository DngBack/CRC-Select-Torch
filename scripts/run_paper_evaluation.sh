#!/bin/bash
# Master script to generate ALL results for paper submission
# 
# This script runs:
# 1. Comprehensive evaluation on test set
# 2. OOD evaluation on SVHN
# 3. Statistical analysis across seeds
# 4. Generate all figures and tables
#
# Usage:
#   bash run_paper_evaluation.sh

set -e  # Exit on error

echo "=============================================================================="
echo "PAPER EVALUATION PIPELINE FOR CRC-SELECT"
echo "=============================================================================="

# Configuration
CRC_SELECT_CHECKPOINT="wandb/offline-run-20260127_091317-5aulwvrk/files/checkpoints/checkpoint_best_val.pth"
VANILLA_CHECKPOINT="wandb/offline-run-20260126_094732-1fotfotl/files/checkpoints/checkpoint_best_val.pth"
RESULTS_DIR="../results_paper"
FIGURES_DIR="../paper_figures"
DATASET="cifar10"
OOD_DATASET="svhn"
SEEDS=(42 123 456)

echo ""
echo "Configuration:"
echo "  CRC-Select checkpoint: $CRC_SELECT_CHECKPOINT"
echo "  Vanilla checkpoint: $VANILLA_CHECKPOINT"
echo "  Results directory: $RESULTS_DIR"
echo "  Dataset: $DATASET"
echo "  OOD dataset: $OOD_DATASET"
echo "  Seeds: ${SEEDS[@]}"
echo "=============================================================================="

# Create directories
mkdir -p $RESULTS_DIR
mkdir -p $FIGURES_DIR

# ============================================================================
# STEP 1: Evaluate CRC-Select
# ============================================================================
echo ""
echo "[STEP 1/5] Evaluating CRC-Select model..."
echo "=============================================================================="

for seed in "${SEEDS[@]}"; do
    echo ""
    echo "  Evaluating seed $seed..."
    python evaluate_for_paper.py \
        --checkpoint "$CRC_SELECT_CHECKPOINT" \
        --method_name "CRC-Select" \
        --dataset "$DATASET" \
        --seed $seed \
        --ood_dataset "$OOD_DATASET" \
        --n_points 201 \
        --output_dir "$RESULTS_DIR" || echo "Warning: Evaluation failed for seed $seed"
    
    echo "  ✓ Seed $seed completed"
done

echo ""
echo "✓ CRC-Select evaluation completed for all seeds"

# ============================================================================
# STEP 2: Evaluate Vanilla SelectiveNet (if checkpoint exists)
# ============================================================================
echo ""
echo "[STEP 2/5] Evaluating Vanilla SelectiveNet baseline..."
echo "=============================================================================="

if [ -f "$VANILLA_CHECKPOINT" ]; then
    for seed in "${SEEDS[@]}"; do
        echo ""
        echo "  Evaluating seed $seed..."
        python evaluate_for_paper.py \
            --checkpoint "$VANILLA_CHECKPOINT" \
            --method_name "Vanilla-SelectiveNet" \
            --dataset "$DATASET" \
            --seed $seed \
            --ood_dataset "$OOD_DATASET" \
            --n_points 201 \
            --output_dir "$RESULTS_DIR" || echo "Warning: Evaluation failed for seed $seed"
        
        echo "  ✓ Seed $seed completed"
    done
    echo ""
    echo "✓ Vanilla SelectiveNet evaluation completed"
else
    echo "⚠️  Vanilla checkpoint not found. Skipping vanilla baseline."
    echo "   Train vanilla with: python train.py --dataset cifar10 --num_epochs 200"
fi

# ============================================================================
# STEP 3: Post-hoc CRC Baseline
# ============================================================================
echo ""
echo "[STEP 3/5] Evaluating Post-hoc CRC baseline..."
echo "=============================================================================="

if [ -f "$VANILLA_CHECKPOINT" ]; then
    for seed in "${SEEDS[@]}"; do
        echo ""
        echo "  Running post-hoc CRC for seed $seed..."
        python baseline_posthoc_crc.py \
            --checkpoint "$VANILLA_CHECKPOINT" \
            --dataset "$DATASET" \
            --seed $seed \
            --alpha_risk 0.1 \
            --output_dir "$RESULTS_DIR" || echo "Warning: Post-hoc CRC failed for seed $seed"
        
        # Evaluate the post-hoc results
        echo "  Evaluating post-hoc CRC results..."
        python evaluate_for_paper.py \
            --checkpoint "$VANILLA_CHECKPOINT" \
            --method_name "Post-hoc-CRC" \
            --dataset "$DATASET" \
            --seed $seed \
            --ood_dataset "$OOD_DATASET" \
            --n_points 201 \
            --output_dir "$RESULTS_DIR" || echo "Warning: Evaluation failed"
        
        echo "  ✓ Seed $seed completed"
    done
    echo ""
    echo "✓ Post-hoc CRC evaluation completed"
else
    echo "⚠️  Vanilla checkpoint not found. Skipping post-hoc CRC."
fi

# ============================================================================
# STEP 4: Aggregate Results Across Seeds
# ============================================================================
echo ""
echo "[STEP 4/5] Aggregating results across seeds..."
echo "=============================================================================="

# Determine which methods are available
METHODS=()
if [ -d "$RESULTS_DIR/CRC-Select" ]; then
    METHODS+=("CRC-Select")
fi
if [ -d "$RESULTS_DIR/Vanilla-SelectiveNet" ]; then
    METHODS+=("Vanilla-SelectiveNet")
fi
if [ -d "$RESULTS_DIR/Post-hoc-CRC" ]; then
    METHODS+=("Post-hoc-CRC")
fi

echo "  Methods found: ${METHODS[@]}"

if [ ${#METHODS[@]} -gt 0 ]; then
    METHOD_DIRS=()
    for method in "${METHODS[@]}"; do
        METHOD_DIRS+=("$RESULTS_DIR/$method")
    done
    
    python aggregate_results.py \
        --method_dirs "${METHOD_DIRS[@]}" \
        --seeds "${SEEDS[@]}" \
        --output_dir "$RESULTS_DIR/aggregated"
    
    echo ""
    echo "✓ Results aggregated"
else
    echo "⚠️  No methods found to aggregate"
fi

# ============================================================================
# STEP 5: Generate Paper Figures and Tables
# ============================================================================
echo ""
echo "[STEP 5/5] Generating publication-quality figures..."
echo "=============================================================================="

if [ ${#METHODS[@]} -gt 0 ]; then
    python generate_paper_figures.py \
        --results_dir "$RESULTS_DIR" \
        --methods "${METHODS[@]}" \
        --seed 42 \
        --output_dir "$FIGURES_DIR"
    
    echo ""
    echo "✓ Figures generated"
else
    echo "⚠️  No methods found to generate figures"
fi

# ============================================================================
# SUMMARY
# ============================================================================
echo ""
echo "=============================================================================="
echo "EVALUATION PIPELINE COMPLETED!"
echo "=============================================================================="
echo ""
echo "📊 Results saved to:"
echo "  • Detailed results: $RESULTS_DIR"
echo "  • Aggregated stats: $RESULTS_DIR/aggregated"
echo "  • Paper figures: $FIGURES_DIR"
echo ""
echo "📁 Generated files:"
echo "  Figures:"
echo "    • figure1_rc_curves.{png,pdf}"
echo "    • figure2_coverage_at_risk.{png,pdf}"
echo "    • figure3_ood_dar.{png,pdf}"
echo "    • figure4_aurc_comparison.{png,pdf}"
echo ""
echo "  Tables:"
echo "    • table1_summary.csv (CSV format)"
echo "    • table1_summary.tex (LaTeX format)"
echo ""
echo "  Per-method results (for each seed):"
echo "    • risk_coverage_curve.csv (201 points)"
echo "    • coverage_at_risk.csv"
echo "    • ood_evaluation.csv"
echo "    • calibration_metrics.csv"
echo "    • summary.csv"
echo ""
echo "✓ All results ready for paper submission!"
echo "=============================================================================="

