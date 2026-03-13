#!/bin/bash
#
# Final Analysis Script — aggregation, violation rates, and paper figures
#
# Prerequisites:
#   - results_paper/CRC-Select/seed_{N}/all_metrics.csv  (≥3 seeds)
#   - results_paper/posthoc_crc/seed_{N}/all_metrics.csv (≥3 seeds)
#   - results_paper/MSP/seed_{N}/all_metrics.csv         (≥3 seeds)
#   - results_paper/Energy/seed_{N}/all_metrics.csv      (≥3 seeds)
#   Optional: TempScaled_MSP, DeepGambler
#
# Outputs:
#   - results_paper/aggregated/   — per-method mean±std tables
#   - results_paper/violation_analysis/ — violation rates + LaTeX table
#   - figures/   — publication-quality plots (PNG + PDF)
#
# Usage:
#   ./run_final_analysis.sh
#   ./run_final_analysis.sh --seeds "42 123 456"   # override default seeds

set -e

SEEDS=(42 123 456 999)
SKIP_OOD="--skip_ood"

while [[ $# -gt 0 ]]; do
    case $1 in
        --seeds) IFS=' ' read -ra SEEDS <<< "$2"; shift 2 ;;
        --with-ood) SKIP_OOD=""; shift ;;
        *) shift ;;
    esac
done

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

mkdir -p results_paper/aggregated results_paper/violation_analysis figures

echo "============================================================"
echo " CRC-Select Paper: Final Analysis Pipeline"
echo "============================================================"
echo " Seeds: ${SEEDS[*]}"
echo "============================================================"

# Detect which methods have at least 1 seed of results
detect_methods() {
    local methods=()
    for method in CRC-Select vanilla posthoc_crc MSP Energy TempScaled_MSP DeepGambler; do
        for seed in "${SEEDS[@]}"; do
            if [[ -f "results_paper/${method}/seed_${seed}/all_metrics.csv" ]]; then
                methods+=("$method")
                break
            fi
        done
    done
    echo "${methods[@]}"
}

METHODS=($(detect_methods))
log "Detected methods with results: ${METHODS[*]}"

METHOD_DIRS=()
for m in "${METHODS[@]}"; do
    METHOD_DIRS+=("results_paper/${m}")
done

# ============================================================
# Step 1: Aggregate results across seeds
# ============================================================
echo ""
echo "=== Step 1: Aggregating results ==="
log "Running aggregate_results.py ..."

python3 scripts/aggregate_results.py \
    --method_dirs "${METHOD_DIRS[@]}" \
    --seeds "${SEEDS[@]}" \
    -o results_paper/aggregated \
    2>&1 | tee results_paper/aggregated/aggregate_log.txt

log "Aggregation complete → results_paper/aggregated/"

# ============================================================
# Step 2: Compute violation rates
# ============================================================
echo ""
echo "=== Step 2: Violation rate analysis ==="
log "Running compute_violation_rate.py ..."

python3 scripts/compute_violation_rate.py \
    --method_dirs "${METHOD_DIRS[@]}" \
    --seeds "${SEEDS[@]}" \
    --alphas 0.01 0.02 0.05 0.1 0.15 0.2 \
    --generate_latex \
    -o results_paper/violation_analysis \
    2>&1 | tee results_paper/violation_analysis/violation_log.txt

log "Violation analysis complete → results_paper/violation_analysis/"

# ============================================================
# Step 3: Generate paper figures
# ============================================================
echo ""
echo "=== Step 3: Generating figures ==="
log "Running generate_paper_figures.py ..."

python3 scripts/generate_paper_figures.py \
    --results_dir results_paper/ \
    --methods "${METHODS[@]}" \
    --seeds "${SEEDS[@]}" \
    ${SKIP_OOD} \
    -o figures/ \
    2>&1 | tee figures/figure_generation_log.txt

log "Figures complete → figures/"

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================================"
echo " Analysis complete!"
echo "============================================================"

echo " Aggregated tables:"
ls results_paper/aggregated/*.csv 2>/dev/null | sed 's/^/   /' || echo "   (none)"

echo " Violation analysis:"
ls results_paper/violation_analysis/*.csv results_paper/violation_analysis/*.tex 2>/dev/null | sed 's/^/   /' || echo "   (none)"

echo " Figures:"
ls figures/*.png figures/*.pdf 2>/dev/null | sed 's/^/   /' || echo "   (none)"

echo ""
echo " Key files for paper:"
echo "   Main results table:    results_paper/aggregated/summary_table.csv"
echo "   Violation LaTeX table: results_paper/violation_analysis/violation_rate_table.tex"
echo "   RC curve figure:       figures/fig_risk_coverage_curve.pdf"
