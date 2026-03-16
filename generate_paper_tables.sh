#!/bin/bash
#
# ============================================================
#  Generate Paper Tables & Figures — CIFAR-10 + CIFAR-100
# ============================================================
#
#  Chạy sau khi tất cả training và evaluation đã xong.
#  Output:
#    figures/table1_cifar10.csv        — CIFAR-10 main table
#    figures/table1_cifar100.csv       — CIFAR-100 main table
#    figures/table1_combined.csv       — Cả hai gộp lại
#    figures/cifar10/                  — Figures CIFAR-10
#    figures/cifar100/                 — Figures CIFAR-100
#
#  Usage:
#    ./generate_paper_tables.sh
#    ./generate_paper_tables.sh --dataset cifar100   # chỉ cifar100

set -e

PYTHON=/home/duong.xuan.bach/anaconda3/envs/crc-select/bin/python

DATASET="both"    # both | cifar10 | cifar100
while [[ $# -gt 0 ]]; do
    case $1 in --dataset) DATASET="$2"; shift 2 ;; *) shift ;; esac
done

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

METHODS="CRC-Select vanilla posthoc_crc MSP Energy TempScaled_MSP DeepGambler"
CIFAR10_SEEDS="42 123 456 999"
CIFAR100_SEEDS="42 123 456 789 999"

mkdir -p figures/cifar10 figures/cifar100 results_paper/aggregated results_paper/cifar100/aggregated

# ============================================================
#  Kiểm tra kết quả có đủ không
# ============================================================
check_results() {
    local dataset=$1
    local seeds=($2)
    local results_dir=$3
    local missing=0

    echo ""
    echo "=== Kiểm tra kết quả: ${dataset} ==="
    for method in ${METHODS}; do
        local count=0
        for seed in "${seeds[@]}"; do
            [[ -f "${results_dir}/${method}/seed_${seed}/all_metrics.csv" ]] && count=$((count+1))
        done
        local total=${#seeds[@]}
        if [[ $count -lt $total ]]; then
            echo "  ⚠  ${method}: ${count}/${total} seeds"
            missing=$((missing+1))
        else
            echo "  ✓  ${method}: ${count}/${total} seeds"
        fi
    done
    return $missing
}

# ============================================================
#  Aggregate + generate table (một dataset)
# ============================================================
run_for_dataset() {
    local dataset=$1          # cifar10 | cifar100
    local seeds_str=$2        # "42 123 456 999"
    local results_dir=$3      # results_paper  OR  results_paper/cifar100
    local figures_dir=$4      # figures/cifar10 OR figures/cifar100
    local agg_dir="${results_dir}/aggregated"

    mkdir -p "${figures_dir}" "${agg_dir}"

    # Build method_dirs list
    local method_dirs=""
    for method in ${METHODS}; do
        local d="${results_dir}/${method}"
        [[ -d "$d" ]] && method_dirs="${method_dirs} ${d}"
    done

    if [[ -z "${method_dirs}" ]]; then
        log "ERROR: No result directories found in ${results_dir}"
        return 1
    fi

    echo ""
    echo "────────────────────────────────────────"
    echo " Dataset: ${dataset}"
    echo " Results: ${results_dir}"
    echo " Figures: ${figures_dir}"
    echo "────────────────────────────────────────"

    # 1. Aggregate across seeds
    log "Aggregating results across seeds..."
    ${PYTHON} scripts/aggregate_results.py \
        --method_dirs ${method_dirs} \
        --seeds ${seeds_str} \
        -o "${agg_dir}" \
        2>&1 | tail -20

    # 2. Generate figures + tables
    log "Generating figures & tables..."
    ${PYTHON} scripts/generate_paper_figures.py \
        --results_dir   "${results_dir}" \
        --methods       CRC-Select vanilla posthoc_crc MSP TempScaled_MSP Energy DeepGambler \
        --seeds         ${seeds_str} \
        --output_dir    "${figures_dir}" \
        --skip_ood \
        2>&1 | tail -30

    # 3. Build main table CSV (mean ± std across seeds at alpha=0.1)
    log "Building main table for ${dataset}..."
    ${PYTHON} - <<PYEOF
import os, sys, numpy as np, pandas as pd

results_dir   = "${results_dir}"
dataset       = "${dataset}"
seeds         = [int(s) for s in "${seeds_str}".split()]
methods_order = ["CRC-Select", "vanilla", "posthoc_crc", "MSP",
                 "TempScaled_MSP", "Energy", "DeepGambler"]
ALPHA         = 0.1

rows = []
for method in methods_order:
    method_rows = []
    for seed in seeds:
        p = os.path.join(results_dir, method, f"seed_{seed}", "all_metrics.csv")
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        # normalise column names
        df.columns = [c.replace("test_", "") for c in df.columns]
        r = df[df["alpha"].round(3) == round(ALPHA, 3)]
        if len(r) == 0:
            continue
        method_rows.append(r.iloc[0])

    if not method_rows:
        continue

    vals = pd.DataFrame(method_rows)
    def fmt(col): 
        if col not in vals.columns: return "N/A"
        return f"{vals[col].mean():.3f} ±{vals[col].std():.3f}"

    violation_rate = (vals["violation_gap"] > 0).mean() if "violation_gap" in vals.columns else float("nan")
    rows.append({
        "dataset":          dataset,
        "method":           method,
        "n_seeds":          len(method_rows),
        "coverage":         fmt("coverage"),
        "accepted_loss_mass": fmt("accepted_loss_mass"),
        "selective_risk":   fmt("selective_risk"),
        "selective_acc":    fmt("selective_acc"),
        "auroc":            fmt("auroc"),
        "aupr":             fmt("aupr"),
        "violation_rate":   f"{violation_rate:.2f}",
        "violation_gap":    fmt("violation_gap"),
        "conservativeness": fmt("conservativeness"),
    })

table = pd.DataFrame(rows)
out = os.path.join("figures", f"table1_{dataset}.csv")
table.to_csv(out, index=False)
print(table.to_string(index=False))
print(f"\nSaved: {out}")
PYEOF

    log "Done — ${dataset}"
}

# ============================================================
#  Main
# ============================================================
echo "============================================================"
echo " Generate Paper Tables & Figures"
echo "============================================================"
echo " Dataset mode: ${DATASET}"
echo "============================================================"

if [[ "$DATASET" == "both" || "$DATASET" == "cifar10" ]]; then
    check_results "CIFAR-10" "${CIFAR10_SEEDS}" "results_paper" || true
    run_for_dataset \
        "cifar10" "${CIFAR10_SEEDS}" \
        "results_paper" "figures/cifar10"
fi

if [[ "$DATASET" == "both" || "$DATASET" == "cifar100" ]]; then
    check_results "CIFAR-100" "${CIFAR100_SEEDS}" "results_paper/cifar100" || true
    run_for_dataset \
        "cifar100" "${CIFAR100_SEEDS}" \
        "results_paper/cifar100" "figures/cifar100"
fi

# ============================================================
#  Merge cả hai dataset thành bảng combined
# ============================================================
if [[ "$DATASET" == "both" ]]; then
    log "Merging CIFAR-10 + CIFAR-100 into combined table..."
    /home/duong.xuan.bach/anaconda3/envs/crc-select/bin/python - <<PYEOF
import pandas as pd, os

files = ["figures/table1_cifar10.csv", "figures/table1_cifar100.csv"]
dfs = [pd.read_csv(f) for f in files if os.path.exists(f)]
if dfs:
    combined = pd.concat(dfs, ignore_index=True)
    out = "figures/table1_combined.csv"
    combined.to_csv(out, index=False)
    print(combined.to_string(index=False))
    print(f"\nSaved: {out}")
PYEOF
fi

echo ""
echo "============================================================"
echo " Kết quả đầu ra:"
echo "  figures/table1_cifar10.csv    — CIFAR-10 main table"
echo "  figures/table1_cifar100.csv   — CIFAR-100 main table"
echo "  figures/table1_combined.csv   — Cả hai gộp lại"
echo "  figures/cifar10/              — Plots CIFAR-10"
echo "  figures/cifar100/             — Plots CIFAR-100"
echo "============================================================"
