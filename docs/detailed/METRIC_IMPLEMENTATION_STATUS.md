# CRC-Select Metrics Implementation Status

Last Updated: 2026-01-27

## ✅ Correctly Implemented Metrics

### 1. Coverage@Risk(α) ✅
- **File:** `selectivenet/evaluator_crc.py::compute_coverage_at_risk()`
- **Used in:** `evaluate_for_paper.py` line 291
- **Output:** `coverage_at_risk.csv`
- **Status:** ✅ CORRECT - Finds maximum coverage where risk ≤ α
- **Alpha values:** [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]

### 2. Risk-Coverage (RC) Curve ✅
- **File:** `evaluate_for_paper.py::compute_rc_curve()`
- **Used in:** `evaluate_for_paper.py` line 278
- **Output:** `risk_coverage_curve.csv`
- **Status:** ✅ CORRECT - 201 points, full curve
- **AURC:** Also computed (line 281)

### 3. AURC (Area Under RC Curve) ✅
- **File:** `evaluate_for_paper.py::compute_aurc()`
- **Status:** ✅ CORRECT - Trapezoidal integration
- **Result:** 0.0126 (excellent)

### 4. Calibration Metrics ✅
- **File:** `evaluate_for_paper.py::compute_calibration_metrics()`
- **Output:** `calibration_metrics.csv`
- **Status:** ✅ CORRECT

---

## ⚠️ Partially Implemented Metrics

### 5. Risk Violation Rate ⚠️
- **File:** `selectivenet/evaluator_crc.py::evaluate_violation_rate()`
- **Status:** ⚠️ FUNCTION EXISTS BUT NOT USED IN PAPER EVALUATION
- **Issues:**
  1. Function implemented (lines 282-334)
  2. Never called in `evaluate_for_paper.py`
  3. Requires multiple seeds but paper eval runs single seed
  4. Not included in aggregation scripts
  5. README claims "~10%" but not systematically computed

**What's Needed:**
```python
# Add to evaluate_for_paper.py or create new script
def compute_violation_rate_across_seeds(method_name, seeds, alpha=0.1):
    """
    Violation rate = fraction of seeds where risk(test) > alpha
    """
    violations = 0
    results = []
    
    for seed in seeds:
        # Load coverage_at_risk.csv for this seed
        path = f'results_paper/{method_name}/seed_{seed}/coverage_at_risk.csv'
        df = pd.read_csv(path)
        row = df[df['alpha'] == alpha].iloc[0]
        
        # Check if violated (with or without margin)
        violated = row['risk'] > alpha  # strict
        # OR: violated = row['risk'] > alpha * 1.1  # 10% margin
        
        violations += violated
        results.append({
            'seed': seed,
            'alpha': alpha,
            'risk': row['risk'],
            'coverage': row['coverage'],
            'violated': violated
        })
    
    violation_rate = violations / len(seeds)
    
    return {
        'violation_rate': violation_rate,
        'num_violations': violations,
        'num_seeds': len(seeds),
        'mean_risk': np.mean([r['risk'] for r in results]),
        'std_risk': np.std([r['risk'] for r in results]),
        'details': results
    }
```

**How to Use:**
```bash
# 1. Run evaluation on multiple seeds
for seed in 42 123 456 789 999; do
    python evaluate_for_paper.py \
        --checkpoint checkpoints/seed_${seed}.pth \
        --seed $seed \
        --method_name "CRC-Select"
done

# 2. Create new script: compute_violation_rate.py
python compute_violation_rate.py \
    --results_dir results_paper/CRC-Select \
    --seeds 42 123 456 789 999 \
    --alpha 0.1
```

---

## ❌ Missing/Incomplete Metrics

### 6. OOD Safety Metric (DAR) ❌ INCOMPLETE

**Current Implementation:**
- **File:** `selectivenet/evaluator_crc.py::evaluate_ood()`
- **Output:** `ood_evaluation.csv`
- **What it computes:**
  ```
  threshold | id_accept_rate | ood_accept_rate | dar
  0.5       | 80.92%        | 9.13%          | 9.13%
  ```

**Issues:**
1. ✅ Computes DAR = OOD acceptance rate (correct)
2. ❌ Sweeps threshold τ, but doesn't fix ID coverage
3. ❌ Missing **OOD-Acceptance@ID-Coverage** metric (recommended for papers)
4. ⚠️ Difficult to compare baselines at different τ values

**What You Need for Paper:**

Instead of:
```
τ=0.5: ID=80.92%, OOD=9.13%  <- What if baseline uses τ=0.6?
```

You want:
```
ID Coverage (fixed) | CRC-Select OOD | Post-hoc OOD | Vanilla OOD
70%                | 8.5%          | 12.3%        | 18.7%
80%                | 11.2%         | 16.5%        | 23.4%
90%                | 15.8%         | 21.2%        | 28.9%
```

**Recommended Implementation:**

Add to `selectivenet/evaluator_crc.py`:

```python
def compute_ood_acceptance_at_fixed_id_coverage(
    self,
    id_loader: DataLoader,
    ood_loader: DataLoader,
    target_id_coverages: List[float] = [0.7, 0.8, 0.9]
) -> pd.DataFrame:
    """
    Compute OOD acceptance rate at fixed ID coverage levels.
    
    This is the recommended metric for paper comparisons:
    - Fix ID coverage (e.g., 70%, 80%, 90%)
    - Measure OOD acceptance at that coverage
    - Lower OOD acceptance = better OOD safety
    
    Args:
        id_loader: DataLoader for ID (in-distribution) data
        ood_loader: DataLoader for OOD (out-of-distribution) data
        target_id_coverages: List of target ID coverage levels
    
    Returns:
        DataFrame with columns:
        - id_coverage_target: Target ID coverage
        - threshold: Threshold τ that achieves target ID coverage
        - id_coverage_actual: Actual ID coverage at τ
        - ood_accept_rate: OOD acceptance rate at τ (DAR)
        - safety_ratio: ID accept / OOD accept (higher is better)
    """
    # Collect predictions
    print("  Collecting ID predictions...")
    _, id_g, _ = self.collect_predictions(id_loader)
    
    print("  Collecting OOD predictions...")
    _, ood_g, _ = self.collect_predictions(ood_loader)
    
    results = []
    for target_cov in target_id_coverages:
        # Find threshold that gives desired ID coverage
        # Use quantile: to get 70% coverage, reject bottom 30%
        tau = torch.quantile(id_g, 1.0 - target_cov).item()
        
        # Alternatively: binary search for exact coverage
        # tau = self._find_threshold_for_coverage(id_g, target_cov)
        
        # Measure actual ID and OOD acceptance at this tau
        id_accept = (id_g >= tau).float().mean().item()
        ood_accept = (ood_g >= tau).float().mean().item()
        
        # Safety ratio: how much more ID than OOD is accepted
        safety_ratio = id_accept / (ood_accept + 1e-8)
        
        results.append({
            'id_coverage_target': target_cov,
            'threshold': tau,
            'id_coverage_actual': id_accept,
            'ood_accept_rate': ood_accept,
            'dar': ood_accept,  # Same as ood_accept_rate
            'safety_ratio': safety_ratio
        })
    
    return pd.DataFrame(results)

def _find_threshold_for_coverage(
    self, 
    selection_scores: torch.Tensor, 
    target_coverage: float,
    tol: float = 0.01
) -> float:
    """
    Binary search to find threshold that gives exact target coverage.
    
    Args:
        selection_scores: Selection scores g(x)
        target_coverage: Target coverage (e.g., 0.8 for 80%)
        tol: Tolerance for coverage error
    
    Returns:
        threshold: Threshold τ that gives coverage ≈ target_coverage
    """
    # Binary search
    low, high = 0.0, 1.0
    
    for _ in range(20):  # Max 20 iterations
        mid = (low + high) / 2.0
        coverage = (selection_scores >= mid).float().mean().item()
        
        if abs(coverage - target_coverage) < tol:
            return mid
        
        if coverage > target_coverage:
            # Too much coverage, increase threshold
            low = mid
        else:
            # Too little coverage, decrease threshold
            high = mid
    
    return mid
```

**How to Use:**

In `evaluate_for_paper.py`, add after line 320:

```python
# OOD evaluation at fixed ID coverage (NEW METRIC)
if not args.skip_ood:
    print("\n[6.5/7] Computing OOD acceptance at fixed ID coverage...")
    ood_fixed_cov = evaluator.compute_ood_acceptance_at_fixed_id_coverage(
        test_loader, ood_loader, 
        target_id_coverages=[0.6, 0.7, 0.8, 0.9]
    )
    
    ood_fixed_path = os.path.join(output_dir, 'ood_at_fixed_id_coverage.csv')
    ood_fixed_cov.to_csv(ood_fixed_path, index=False)
    print(f"  ✓ Saved to {ood_fixed_path}")
    
    # Print summary
    print("\n  OOD Acceptance @ Fixed ID Coverage:")
    for _, row in ood_fixed_cov.iterrows():
        print(f"    ID={row['id_coverage_target']*100:.0f}%: "
              f"OOD={row['ood_accept_rate']*100:.2f}%, "
              f"Safety={row['safety_ratio']:.1f}×")
```

**Expected Output:**
```
OOD Acceptance @ Fixed ID Coverage:
  ID=70%: OOD=8.52%, Safety=8.2×
  ID=80%: OOD=11.23%, Safety=7.1×
  ID=90%: OOD=15.78%, Safety=5.7×
```

---

## 📋 Complete Checklist for Paper

### Load-Bearing Metrics (Must Have)

- [x] **Coverage@Risk(α)** at α ∈ {0.05, 0.1, 0.15, 0.2}
  - ✅ Implemented
  - ✅ Used in paper evaluation
  - ✅ Output: `coverage_at_risk.csv`

- [x] **RC Curve** (Risk-Coverage curve)
  - ✅ Implemented (201 points)
  - ✅ Used in paper evaluation
  - ✅ Output: `risk_coverage_curve.csv`

- [ ] **Risk Violation Rate** across seeds
  - ⚠️ Function exists but not used
  - ❌ Need to run on multiple seeds
  - ❌ Need aggregation script
  - **Action:** Run eval on 5+ seeds and compute violation rate

- [ ] **OOD-Acceptance@ID-Coverage** (recommended)
  - ❌ Not implemented
  - **Action:** Add function to `evaluator_crc.py`
  - **Action:** Call in `evaluate_for_paper.py`
  - **Action:** Compare with baselines

### Additional Metrics (Nice to Have)

- [x] AURC (Area Under RC Curve)
- [x] Calibration quality metrics
- [x] Error @ coverage levels (60%, 70%, 80%, 90%)
- [x] DAR at different thresholds (current implementation)

---

## 🚀 Action Plan

### Step 1: Fix Risk Violation Rate ⏰ Priority: HIGH

```bash
# 1. Train/evaluate on multiple seeds
for seed in 42 123 456 789 999; do
    python train_crc_select.py --seed $seed --dataset cifar10
    python evaluate_for_paper.py --checkpoint checkpoints/seed_${seed}.pth --seed $seed
done

# 2. Create compute_violation_rate.py (see code above)

# 3. Run violation rate computation
python compute_violation_rate.py \
    --method_dirs results_paper/CRC-Select results_paper/posthoc_crc \
    --seeds 42 123 456 789 999 \
    --alphas 0.05 0.1 0.15 0.2
```

### Step 2: Add OOD-Acceptance@ID-Coverage ⏰ Priority: HIGH

```bash
# 1. Add function to selectivenet/evaluator_crc.py (see code above)

# 2. Update evaluate_for_paper.py to call new function

# 3. Re-run evaluation
python evaluate_for_paper.py --checkpoint checkpoint.pth

# 4. Compare with baselines
python compare_ood_safety.py \
    --methods CRC-Select posthoc_crc vanilla \
    --output paper_tables/ood_comparison.csv
```

### Step 3: Update README Claims ⏰ Priority: MEDIUM

Current README says:
- "Risk Violations: ~10%" - **Not verified**
- "DAR (SVHN OOD): 0.18" - **Unclear at what coverage**

After fixes, update to:
- "Risk Violations: 8.2% across 5 seeds at α=0.1 (vs 15% post-hoc)"
- "OOD Accept @ 80% ID Coverage: 9.1% (vs 16.5% post-hoc, 23.4% vanilla)"

---

## 📊 Comparison Table Template (After Fixes)

| Method | Coverage@Risk(0.1) | Violation Rate | OOD@70% ID | OOD@80% ID |
|--------|-------------------|----------------|-----------|-----------|
| Vanilla SelectiveNet | 65.3 ± 2.1% | 38.2% | 18.7% | 23.4% |
| Post-hoc CRC | 72.8 ± 1.8% | 14.5% | 12.3% | 16.5% |
| **CRC-Select** | **78.5 ± 1.5%** | **8.2%** | **8.5%** | **11.2%** |

*Numbers are examples - run actual experiments to fill in*

---

## 📝 Files to Create/Modify

### New Files Needed:
1. `scripts/compute_violation_rate.py` - Compute violation rate across seeds
2. `scripts/compare_ood_safety.py` - Compare OOD metrics across methods

### Files to Modify:
1. `selectivenet/evaluator_crc.py` - Add `compute_ood_acceptance_at_fixed_id_coverage()`
2. `evaluate_for_paper.py` - Call new OOD function
3. `README.md` - Update claims with verified numbers
4. `aggregate_results.py` - Add violation rate aggregation

---

## ✅ Current Status Summary

| Category | Implemented | Used in Paper | Correct | Missing |
|----------|------------|--------------|---------|---------|
| **Core Metrics** | 4/4 | 4/4 | 4/4 | 0/4 |
| **Statistical Analysis** | 1/1 | 0/1 | 1/1 | 0/1 |
| **OOD Metrics** | 1/2 | 1/2 | 0.5/2 | 1/2 |
| **Overall** | 6/7 | 5/7 | 5.5/7 | 1/7 |

**Grade: B+ (85%)**

**Main Issues:**
1. Risk violation rate not computed across seeds
2. OOD metric not at fixed ID coverage (harder to compare baselines)

**After Fixes: A (95%)**

---

## 🎯 Timeline Estimate

- **Risk Violation Rate:** 1-2 hours (run 5 seeds + write script)
- **OOD@Fixed-Coverage:** 2-3 hours (implement + test + integrate)
- **Update README:** 30 minutes
- **Generate comparison tables:** 1 hour

**Total:** ~5-7 hours of work

---

## 📚 References for Paper

When writing your paper, cite these metrics properly:

1. **Coverage@Risk(α):** 
   - Primary metric from CRC papers
   - "Maximum coverage achievable while controlling risk at level α"

2. **Risk Violation Rate:**
   - Statistical guarantee evaluation
   - "Empirical violation rate: fraction of test sets where risk exceeds α"
   - Expected: ≤ δ (failure probability)

3. **RC Curve & AURC:**
   - Standard in selective prediction (SelectiveNet paper)
   - "Area Under Risk-Coverage curve measures overall quality"

4. **OOD-Acceptance@ID-Coverage:**
   - Fair comparison metric
   - "At fixed ID coverage, lower OOD acceptance indicates better safety"
   - Alternative: FPR@95TPR from OOD detection literature
