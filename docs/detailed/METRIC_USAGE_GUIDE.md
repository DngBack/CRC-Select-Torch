# Metric Usage Guide: Computing All Load-Bearing Metrics

This guide shows you how to compute all the essential metrics for comparing CRC-Select with baselines.

## Quick Start

### 1. Run Evaluation on Multiple Seeds

First, run the evaluation script on multiple seeds to enable statistical analysis:

```bash
cd scripts

# Run evaluation on 5 seeds
for seed in 42 123 456 789 999; do
    python evaluate_for_paper.py \
        --checkpoint /path/to/checkpoint_seed_${seed}.pth \
        --method_name "CRC-Select" \
        --dataset cifar10 \
        --seed $seed \
        --n_points 201 \
        --output_dir ../results_paper
done
```

### 2. Compute Risk Violation Rate

After running evaluations on multiple seeds:

```bash
# Compute violation rate for CRC-Select
python compute_violation_rate.py \
    --method_dirs ../results_paper/CRC-Select \
    --seeds 42 123 456 789 999 \
    --alphas 0.05 0.1 0.15 0.2 \
    --output_dir ../results/violation_rate \
    --generate_latex
```

### 3. Compare OOD Safety Across Methods

After running evaluations for all methods (CRC-Select, post-hoc CRC, vanilla):

```bash
# Compare OOD safety metrics
python compare_ood_safety.py \
    --results_dir ../results_paper \
    --methods CRC-Select posthoc_crc vanilla \
    --seeds 42 123 456 789 999 \
    --plot \
    --latex \
    --output_dir ../results/ood_comparison
```

---

## Detailed Workflow

### Step 1: Train Models on Multiple Seeds

Train your models with different seeds for statistical robustness:

```bash
# Train CRC-Select on 5 seeds
for seed in 42 123 456 789 999; do
    python train_crc_select.py \
        --dataset cifar10 \
        --seed $seed \
        --num_epochs 200 \
        --alpha_risk 0.1 \
        --coverage 0.8 \
        --warmup_epochs 20 \
        --recalibrate_every 5 \
        --use_dual_update \
        --unobserve
done

# Train vanilla SelectiveNet baseline
for seed in 42 123 456 789 999; do
    python train.py \
        --dataset cifar10 \
        --seed $seed \
        --coverage 0.8 \
        --num_epochs 200 \
        --unobserve
done

# Apply post-hoc CRC to vanilla models
for seed in 42 123 456 789 999; do
    python baseline_posthoc_crc.py \
        --checkpoint /path/to/vanilla_checkpoint_seed_${seed}.pth \
        --dataset cifar10 \
        --seed $seed \
        --alpha_risk 0.1 \
        --output_dir ../results_paper
done
```

### Step 2: Comprehensive Evaluation

Run the comprehensive evaluation script on each seed:

```bash
# Evaluate CRC-Select
for seed in 42 123 456 789 999; do
    python evaluate_for_paper.py \
        --checkpoint checkpoints/crc_select_seed_${seed}.pth \
        --method_name "CRC-Select" \
        --dataset cifar10 \
        --seed $seed \
        --n_points 201 \
        --output_dir ../results_paper
done

# Evaluate vanilla baseline
for seed in 42 123 456 789 999; do
    python evaluate_for_paper.py \
        --checkpoint checkpoints/vanilla_seed_${seed}.pth \
        --method_name "vanilla" \
        --dataset cifar10 \
        --seed $seed \
        --n_points 201 \
        --output_dir ../results_paper
done
```

### Step 3: Compute All Metrics

#### Metric 1: Coverage@Risk(α) ✅ (Already in evaluate_for_paper.py)

This is automatically computed by `evaluate_for_paper.py` and saved to `coverage_at_risk.csv`.

**Check your results:**
```bash
cat results_paper/CRC-Select/seed_42/coverage_at_risk.csv
```

#### Metric 2: Risk-Coverage Curve ✅ (Already in evaluate_for_paper.py)

Also automatically computed and saved to `risk_coverage_curve.csv`.

**Check your results:**
```bash
cat results_paper/CRC-Select/seed_42/risk_coverage_curve.csv
```

#### Metric 3: Risk Violation Rate ⚠️ (Use new script)

Compute violation rate across seeds:

```bash
# Single method
python compute_violation_rate.py \
    --method_dirs ../results_paper/CRC-Select \
    --seeds 42 123 456 789 999 \
    --alphas 0.05 0.1 0.15 0.2 \
    --margin 0.0 \
    --output_dir ../results/violation_rate \
    --generate_latex

# Compare multiple methods
python compute_violation_rate.py \
    --method_dirs \
        ../results_paper/CRC-Select \
        ../results_paper/posthoc_crc \
        ../results_paper/vanilla \
    --seeds 42 123 456 789 999 \
    --alphas 0.05 0.1 0.15 0.2 \
    --output_dir ../results/violation_rate \
    --generate_latex
```

**Output files:**
- `violation_rate_comparison.csv` - Comparison table
- `{method}_alpha_{alpha}_violation_summary.csv` - Per-method summary
- `{method}_alpha_{alpha}_violation_details.csv` - Per-seed details
- `violation_rate_table.tex` - LaTeX table

#### Metric 4: OOD-Acceptance@ID-Coverage ⚠️ (Already computed in updated evaluate_for_paper.py)

The updated `evaluate_for_paper.py` now computes this automatically and saves to `ood_at_fixed_id_coverage.csv`.

**Check your results:**
```bash
cat results_paper/CRC-Select/seed_42/ood_at_fixed_id_coverage.csv
```

**Compare across methods:**
```bash
python compare_ood_safety.py \
    --results_dir ../results_paper \
    --methods CRC-Select posthoc_crc vanilla \
    --seeds 42 123 456 789 999 \
    --plot \
    --latex \
    --output_dir ../results/ood_comparison
```

**Output files:**
- `ood_safety_comparison.csv` - Comparison table with mean±std
- `ood_comparison_plot.png` - Line plot comparison
- `ood_comparison_heatmap.png` - Heatmap visualization
- `ood_comparison_table.tex` - LaTeX table

---

## Expected Outputs

### Directory Structure After Running All Scripts

```
results_paper/
├── CRC-Select/
│   ├── seed_42/
│   │   ├── risk_coverage_curve.csv         ✅ RC curve
│   │   ├── coverage_at_risk.csv            ✅ Coverage@Risk
│   │   ├── ood_evaluation.csv              ✅ DAR sweep
│   │   ├── ood_at_fixed_id_coverage.csv    ✅ NEW: OOD@fixed-cov
│   │   ├── calibration_metrics.csv
│   │   └── summary.csv
│   ├── seed_123/ ...
│   └── seed_456/ ...
├── posthoc_crc/
│   └── ... (same structure)
└── vanilla/
    └── ... (same structure)

results/
├── violation_rate/
│   ├── violation_rate_comparison.csv       ✅ NEW: Violation rates
│   ├── CRC-Select_alpha_0.100_violation_summary.csv
│   ├── CRC-Select_alpha_0.100_violation_details.csv
│   └── violation_rate_table.tex            ✅ LaTeX table
└── ood_comparison/
    ├── ood_safety_comparison.csv           ✅ NEW: OOD comparison
    ├── ood_comparison_plot.png             ✅ Plots
    ├── ood_comparison_heatmap.png
    └── ood_comparison_table.tex            ✅ LaTeX table
```

---

## Verification Checklist

Use this checklist to ensure all metrics are computed correctly:

### ✅ Basic Metrics (Single Seed)

- [ ] Risk-Coverage curve computed (201 points)
- [ ] AURC computed
- [ ] Coverage@Risk computed for α ∈ {0.01, 0.02, 0.05, 0.1, 0.15, 0.2}
- [ ] OOD evaluation (DAR sweep) computed
- [ ] OOD@fixed-ID-coverage computed

**Verify:**
```bash
ls results_paper/CRC-Select/seed_42/
# Should see: risk_coverage_curve.csv, coverage_at_risk.csv, 
#            ood_evaluation.csv, ood_at_fixed_id_coverage.csv
```

### ✅ Multi-Seed Metrics

- [ ] Evaluations run on ≥5 seeds
- [ ] Risk violation rate computed across seeds
- [ ] OOD comparison computed across seeds

**Verify:**
```bash
ls results_paper/CRC-Select/
# Should see: seed_42, seed_123, seed_456, seed_789, seed_999

ls results/violation_rate/
# Should see: violation_rate_comparison.csv

ls results/ood_comparison/
# Should see: ood_safety_comparison.csv
```

### ✅ Baseline Comparisons

- [ ] Vanilla SelectiveNet evaluated
- [ ] Post-hoc CRC evaluated
- [ ] Comparison tables generated

**Verify:**
```bash
cat results/violation_rate/violation_rate_comparison.csv
cat results/ood_comparison/ood_safety_comparison.csv
```

---

## Interpreting Results

### Coverage@Risk(α)

**What it measures:** Maximum coverage achievable while keeping risk ≤ α

**Example:**
```
alpha  coverage  risk
0.10   0.874     0.031
```
→ At α=0.1, CRC-Select achieves 87.4% coverage with risk=0.031 (well below 0.1)

**For paper:** Report coverage@risk(0.1) as the headline metric

### Risk Violation Rate

**What it measures:** Fraction of test sets where risk exceeds α

**Example:**
```
method      alpha  violation_rate  mean_risk  std_risk
CRC-Select  0.10   0.20           0.095      0.012
```
→ 20% of seeds violated α=0.1, with mean risk=0.095±0.012

**For paper:** Lower is better. CRC should have ~10-20% violation rate (theory: ≤δ)

### OOD-Acceptance@ID-Coverage

**What it measures:** OOD acceptance rate at fixed ID coverage

**Example:**
```
id_coverage  ood_accept_rate  safety_ratio
0.70         0.085           8.2
0.80         0.112           7.1
```
→ At 80% ID coverage, only 11.2% of OOD samples are accepted (7.1× safer than random)

**For paper:** Lower OOD acceptance = better. Compare across methods at same ID coverage.

---

## Common Issues & Solutions

### Issue 1: Missing `ood_at_fixed_id_coverage.csv`

**Cause:** Old version of `evaluate_for_paper.py`

**Solution:** 
```bash
# Re-run evaluation with updated script
python evaluate_for_paper.py \
    --checkpoint checkpoint.pth \
    --method_name "CRC-Select" \
    --seed 42
```

### Issue 2: Violation rate script says "No results available"

**Cause:** Missing evaluation results for some seeds

**Solution:**
```bash
# Check which seeds are missing
ls results_paper/CRC-Select/

# Run evaluation for missing seeds
python evaluate_for_paper.py --seed 123 ...
```

### Issue 3: OOD comparison fails with "File not found"

**Cause:** OOD evaluation was skipped with `--skip_ood`

**Solution:**
```bash
# Re-run without --skip_ood
python evaluate_for_paper.py \
    --checkpoint checkpoint.pth \
    --method_name "CRC-Select" \
    --seed 42
    # Don't use --skip_ood
```

---

## Paper Writing Tips

### Recommended Tables

**Table 1: Main Results**
```
Method         | Coverage@0.1 | Violation Rate | AURC
---------------|--------------|----------------|-------
Vanilla        | 65.3 ± 2.1%  | 38.2%         | 0.025
Post-hoc CRC   | 72.8 ± 1.8%  | 14.5%         | 0.018
CRC-Select     | 78.5 ± 1.5%  | 8.2%          | 0.013
```

**Table 2: OOD Safety**
```
Method         | OOD@70% | OOD@80% | OOD@90%
---------------|---------|---------|--------
Vanilla        | 18.7%   | 23.4%   | 28.9%
Post-hoc CRC   | 12.3%   | 16.5%   | 21.2%
CRC-Select     | 8.5%    | 11.2%   | 15.8%
```

### Recommended Figures

1. **Figure 1:** Risk-Coverage curves (all methods)
2. **Figure 2:** Coverage@Risk bar chart (multiple α)
3. **Figure 3:** OOD acceptance vs ID coverage (line plot)
4. **Figure 4:** Violation rate heatmap (methods × α)

### Key Claims to Support

1. **"CRC-Select achieves 5-15% higher coverage than post-hoc CRC at same risk level"**
   - Evidence: `coverage_at_risk.csv` comparison
   
2. **"Risk violations are controlled at ~10%"**
   - Evidence: `violation_rate_comparison.csv`

3. **"OOD acceptance is reduced by 30-50% compared to baselines"**
   - Evidence: `ood_safety_comparison.csv`

4. **"Full risk-coverage curves show consistent improvement"**
   - Evidence: `risk_coverage_curve.csv` + AURC

---

## Example: Complete Workflow

Here's a complete example from training to paper tables:

```bash
#!/bin/bash
# complete_evaluation_pipeline.sh

cd scripts

# 1. Train models (3 seeds for demo)
for seed in 42 123 456; do
    echo "Training seed $seed..."
    
    # Train CRC-Select
    python train_crc_select.py \
        --dataset cifar10 \
        --seed $seed \
        --num_epochs 200 \
        --alpha_risk 0.1 \
        --unobserve
    
    # Train vanilla
    python train.py \
        --dataset cifar10 \
        --seed $seed \
        --coverage 0.8 \
        --num_epochs 200 \
        --unobserve
done

# 2. Evaluate all models
for seed in 42 123 456; do
    echo "Evaluating seed $seed..."
    
    # Evaluate CRC-Select
    python evaluate_for_paper.py \
        --checkpoint checkpoints/crc_select_seed_${seed}.pth \
        --method_name "CRC-Select" \
        --seed $seed \
        --output_dir ../results_paper
    
    # Evaluate vanilla
    python evaluate_for_paper.py \
        --checkpoint checkpoints/vanilla_seed_${seed}.pth \
        --method_name "vanilla" \
        --seed $seed \
        --output_dir ../results_paper
    
    # Post-hoc CRC
    python baseline_posthoc_crc.py \
        --checkpoint checkpoints/vanilla_seed_${seed}.pth \
        --seed $seed \
        --output_dir ../results_paper
done

# 3. Compute violation rates
python compute_violation_rate.py \
    --method_dirs \
        ../results_paper/CRC-Select \
        ../results_paper/posthoc_crc \
        ../results_paper/vanilla \
    --seeds 42 123 456 \
    --alphas 0.1 \
    --output_dir ../results/violation_rate \
    --generate_latex

# 4. Compare OOD safety
python compare_ood_safety.py \
    --results_dir ../results_paper \
    --methods CRC-Select posthoc_crc vanilla \
    --seeds 42 123 456 \
    --plot \
    --latex \
    --output_dir ../results/ood_comparison

echo "Done! Check results in:"
echo "  - results_paper/       (per-seed results)"
echo "  - results/violation_rate/  (violation analysis)"
echo "  - results/ood_comparison/  (OOD safety comparison)"
```

Run it:
```bash
chmod +x complete_evaluation_pipeline.sh
./complete_evaluation_pipeline.sh
```

---

## Quick Reference

| Metric | Script | Output File | Status |
|--------|--------|-------------|--------|
| Coverage@Risk | `evaluate_for_paper.py` | `coverage_at_risk.csv` | ✅ Auto |
| RC Curve | `evaluate_for_paper.py` | `risk_coverage_curve.csv` | ✅ Auto |
| AURC | `evaluate_for_paper.py` | `summary.csv` | ✅ Auto |
| Risk Violation Rate | `compute_violation_rate.py` | `violation_rate_comparison.csv` | ⚠️ Manual |
| OOD@Fixed-Coverage | `evaluate_for_paper.py` | `ood_at_fixed_id_coverage.csv` | ✅ Auto (new) |
| OOD Comparison | `compare_ood_safety.py` | `ood_safety_comparison.csv` | ⚠️ Manual |

**Legend:**
- ✅ Auto: Automatically computed by evaluation script
- ⚠️ Manual: Requires separate script after evaluation

---

## Questions?

If you have issues or questions:

1. Check the implementation status: `METRIC_IMPLEMENTATION_STATUS.md`
2. Verify your files match the expected structure above
3. Check script help: `python script_name.py --help`

## Next Steps

After computing all metrics:

1. Generate figures: `python generate_paper_figures.py`
2. View results: `python view_results.py`
3. Write paper sections using the LaTeX tables generated
