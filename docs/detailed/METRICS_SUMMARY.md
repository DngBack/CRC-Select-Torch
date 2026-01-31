# CRC-Select Metrics Summary

Quick reference for all load-bearing metrics in your paper.

## ✅ Status Overview

| # | Metric | Status | Location | Used in Paper? |
|---|--------|--------|----------|----------------|
| 1 | Coverage@Risk(α) | ✅ Correct | `evaluator_crc.py:118-161` | ✅ Yes |
| 2 | RC Curve | ✅ Correct | `evaluate_for_paper.py:68-113` | ✅ Yes |
| 3 | Risk Violation Rate | ⚠️ Needs multi-seed | `evaluator_crc.py:282-334` | ❌ Not yet |
| 4 | OOD-Accept@ID-Cov | ✅ Just added | `evaluator_crc.py:200-259` | ✅ Yes (new) |

## 🎯 Quick Actions Needed

### Priority 1: Run on Multiple Seeds

```bash
# Run evaluation on 5 seeds to enable violation rate calculation
for seed in 42 123 456 789 999; do
    python scripts/evaluate_for_paper.py \
        --checkpoint checkpoints/seed_${seed}.pth \
        --seed $seed \
        --method_name "CRC-Select"
done
```

### Priority 2: Compute Violation Rate

```bash
# After running on multiple seeds
python scripts/compute_violation_rate.py \
    --method_dirs results_paper/CRC-Select \
    --seeds 42 123 456 789 999 \
    --alphas 0.1 \
    --generate_latex
```

### Priority 3: Compare OOD Safety

```bash
# Compare all methods
python scripts/compare_ood_safety.py \
    --methods CRC-Select posthoc_crc vanilla \
    --seeds 42 123 456 789 999 \
    --plot --latex
```

## 📊 What Each Metric Tells You

### 1. Coverage@Risk(α) - Headline Metric

**Definition:** Maximum coverage achievable while keeping risk ≤ α

**Current Result (seed 42, α=0.1):** Coverage = 100%, Risk = 0.088

**For Paper:**
```
CRC-Select achieves X% coverage at risk α=0.1
vs Y% for post-hoc CRC (Z% improvement)
```

**File:** `results_paper/CRC-Select/seed_42/coverage_at_risk.csv`

---

### 2. Risk Violation Rate - Statistical Guarantee

**Definition:** Fraction of seeds where risk(test) > α

**Expected:** ~10-20% (theory: ≤δ where δ=0.1 typically)

**Status:** ⚠️ NOT COMPUTED YET (need multi-seed)

**For Paper:**
```
Risk violations occur in X% of runs (Y/5 seeds)
Mean risk: 0.095 ± 0.012
```

**After fix:** `results/violation_rate/violation_rate_comparison.csv`

---

### 3. RC Curve - Overall Quality

**Definition:** Full risk-coverage tradeoff curve

**Current Result:** AURC = 0.0126 (excellent, lower is better)

**For Paper:**
- Plot full curves for all methods
- Report AURC in table
- Show CRC-Select dominates baselines

**File:** `results_paper/CRC-Select/seed_42/risk_coverage_curve.csv`

---

### 4. OOD-Accept@ID-Coverage - Safety Metric

**Definition:** OOD acceptance rate at fixed ID coverage

**Current Result (seed 42):**
- 70% ID: OOD = ?% (need to re-run)
- 80% ID: OOD = ?%
- 90% ID: OOD = ?%

**For Paper:**
```
At 80% ID coverage:
- CRC-Select accepts X% OOD
- Post-hoc accepts Y% OOD (Z% worse)
```

**After fix:** `results_paper/CRC-Select/seed_42/ood_at_fixed_id_coverage.csv`

---

## 📋 Paper Tables to Generate

### Table 1: Main Results

| Method | Coverage@0.1 ↑ | Viol. Rate ↓ | AURC ↓ |
|--------|----------------|--------------|--------|
| Vanilla | ??.? ± ?.?% | ??.?% | 0.??? |
| Post-hoc CRC | ??.? ± ?.?% | ??.?% | 0.??? |
| **CRC-Select** | **??.? ± ?.?%** | **?.?%** | **0.???** |

**Files needed:**
- `coverage_at_risk.csv` (all methods, all seeds)
- `violation_rate_comparison.csv`
- `summary.csv`

---

### Table 2: OOD Safety

| Method | 70% ID | 80% ID | 90% ID |
|--------|--------|--------|--------|
| Vanilla | ??.?% | ??.?% | ??.?% |
| Post-hoc CRC | ??.?% | ??.?% | ??.?% |
| **CRC-Select** | **?.?%** | **?.?%** | **?.?%** |

**File needed:**
- `ood_safety_comparison.csv`

---

## 🔧 What Changed / Was Fixed

### ✅ Added: OOD-Acceptance@ID-Coverage

**File:** `selectivenet/evaluator_crc.py`

**New function:**
```python
def compute_ood_acceptance_at_fixed_id_coverage(
    self, id_loader, ood_loader, 
    target_id_coverages=[0.7, 0.8, 0.9]
):
    # For each target ID coverage:
    # 1. Find threshold τ that gives that coverage
    # 2. Measure OOD acceptance at that τ
    # 3. Compute safety ratio
```

**Usage:** Automatically called by `evaluate_for_paper.py` (updated)

---

### ✅ Added: Violation Rate Computation

**File:** `scripts/compute_violation_rate.py` (NEW)

**What it does:**
- Loads results from multiple seeds
- Checks: risk(test) > α for each seed
- Computes violation_rate = violations / total_seeds
- Generates comparison tables and LaTeX

**Usage:**
```bash
python compute_violation_rate.py \
    --method_dirs results_paper/CRC-Select \
    --seeds 42 123 456 789 999 \
    --alphas 0.1
```

---

### ✅ Added: OOD Safety Comparison

**File:** `scripts/compare_ood_safety.py` (NEW)

**What it does:**
- Aggregates OOD@fixed-coverage across seeds
- Compares multiple methods
- Generates plots and LaTeX tables

**Usage:**
```bash
python compare_ood_safety.py \
    --methods CRC-Select posthoc_crc vanilla \
    --seeds 42 123 456 789 999 \
    --plot --latex
```

---

## 🚀 Complete Workflow (TL;DR)

```bash
# 1. Run evaluation on multiple seeds
for seed in 42 123 456 789 999; do
    python scripts/evaluate_for_paper.py \
        --checkpoint checkpoints/seed_${seed}.pth \
        --seed $seed --method_name "CRC-Select"
done

# 2. Compute violation rate
python scripts/compute_violation_rate.py \
    --method_dirs results_paper/CRC-Select \
    --seeds 42 123 456 789 999 \
    --alphas 0.1 --generate_latex

# 3. Compare OOD safety
python scripts/compare_ood_safety.py \
    --methods CRC-Select posthoc_crc vanilla \
    --seeds 42 123 456 789 999 \
    --plot --latex

# 4. Check results
cat results/violation_rate/violation_rate_comparison.csv
cat results/ood_comparison/ood_safety_comparison.csv
```

---

## 📁 Expected Files After Running

```
results_paper/
├── CRC-Select/
│   ├── seed_42/
│   │   ├── risk_coverage_curve.csv          ← RC curve
│   │   ├── coverage_at_risk.csv             ← Coverage@Risk
│   │   ├── ood_evaluation.csv               ← DAR sweep
│   │   ├── ood_at_fixed_id_coverage.csv     ← NEW: OOD@fixed
│   │   └── summary.csv                      ← AURC + summary
│   ├── seed_123/ ...
│   └── seed_456/ ...

results/
├── violation_rate/
│   ├── violation_rate_comparison.csv        ← NEW: Violation rates
│   └── violation_rate_table.tex             ← LaTeX table
└── ood_comparison/
    ├── ood_safety_comparison.csv            ← NEW: OOD comparison
    ├── ood_comparison_plot.png              ← Plots
    └── ood_comparison_table.tex             ← LaTeX table
```

---

## ✅ Verification Checklist

Before submitting your paper, verify:

- [ ] Evaluated on ≥5 seeds
- [ ] `coverage_at_risk.csv` exists for all seeds
- [ ] `ood_at_fixed_id_coverage.csv` exists for all seeds
- [ ] `violation_rate_comparison.csv` generated
- [ ] `ood_safety_comparison.csv` generated
- [ ] All LaTeX tables generated
- [ ] Compared with at least 1 baseline (post-hoc CRC)
- [ ] Plots generated and look correct

---

## ❓ FAQ

### Q: Why do I need multiple seeds?

**A:** Risk violation rate is a statistical property - you need multiple test sets to measure how often risk > α occurs.

### Q: What's the difference between DAR and OOD-Accept@ID-Coverage?

**A:** 
- DAR: OOD acceptance rate at a specific threshold τ
- OOD-Accept@ID-Coverage: OOD acceptance at threshold that gives specific ID coverage

The second is better for fair comparison because all methods are evaluated at the same ID coverage.

### Q: My violation rate is 40%, is that bad?

**A:** Depends on your δ parameter. Theory says violation rate ≤ δ (typically δ=0.1 or 0.2). So 40% might indicate:
1. Not enough calibration data
2. α is too tight
3. Statistical fluctuation (need more seeds)

### Q: How many seeds do I need?

**A:** Minimum 3, recommended 5-10. More seeds = more reliable statistics.

---

## 📚 Documentation Files

- `METRIC_IMPLEMENTATION_STATUS.md` - Detailed implementation analysis
- `METRIC_USAGE_GUIDE.md` - Step-by-step usage guide
- `METRICS_SUMMARY.md` - This file (quick reference)

---

## 🎯 Bottom Line

**What's working:** Coverage@Risk, RC Curve, AURC, OOD sweep

**What's new:** OOD-Accept@ID-Coverage (just added)

**What needs action:** 
1. Run on multiple seeds (5+)
2. Compute violation rate
3. Compare OOD safety

**Time needed:** ~5-7 hours (mostly training/eval time)

**Impact:** Complete set of metrics for strong paper submission ✨
