# 📋 CRC-Select Paper Completion Checklist

**Status**: 🚧 In Progress  
**Target**: Complete paper-ready results  
**Last Updated**: Feb 9, 2026

---

## 📊 Current Status Overview

- [x] CRC-Select implementation complete
- [x] Single seed evaluation (seed 42) done
- [ ] Multi-seed experiments (0/5 seeds)
- [ ] Baseline comparisons (0/2 baselines)
- [ ] All metrics computed
- [ ] Figures generated
- [ ] Paper draft written

**Estimated Completion**: 2-3 weeks

---

## 🔴 PHASE 1: Core Experiments (CRITICAL - ~3 days)

### 1.1 Multi-Seed Training - CRC-Select ⏱️ ~10-15 hours GPU time

- [ ] **Seed 42** (Already done ✅)
  ```bash
  # Skip - already have results in results_paper/CRC-Select/seed_42/
  ```

- [ ] **Seed 123** 
  ```bash
  python scripts/train_crc_select.py \
      --seed 123 \
      --epochs 300 \
      --alpha_risk 0.1 \
      --coverage_target 0.8 \
      --mu_risk_init 1.0 \
      --calibrate_every 10 \
      --output_dir checkpoints/CRC-Select
  ```
  **Output**: `checkpoints/CRC-Select/seed_123.pth`

- [ ] **Seed 456**
  ```bash
  python scripts/train_crc_select.py \
      --seed 456 \
      --epochs 300 \
      --alpha_risk 0.1 \
      --coverage_target 0.8 \
      --mu_risk_init 1.0 \
      --calibrate_every 10 \
      --output_dir checkpoints/CRC-Select
  ```
  **Output**: `checkpoints/CRC-Select/seed_456.pth`

- [ ] **Seed 789**
  ```bash
  python scripts/train_crc_select.py \
      --seed 789 \
      --epochs 300 \
      --alpha_risk 0.1 \
      --coverage_target 0.8 \
      --mu_risk_init 1.0 \
      --calibrate_every 10 \
      --output_dir checkpoints/CRC-Select
  ```
  **Output**: `checkpoints/CRC-Select/seed_789.pth`

- [ ] **Seed 999**
  ```bash
  python scripts/train_crc_select.py \
      --seed 999 \
      --epochs 300 \
      --alpha_risk 0.1 \
      --coverage_target 0.8 \
      --mu_risk_init 1.0 \
      --calibrate_every 10 \
      --output_dir checkpoints/CRC-Select
  ```
  **Output**: `checkpoints/CRC-Select/seed_999.pth`

**Quick parallel execution**:
```bash
# Run all seeds in parallel (if you have multiple GPUs)
for seed in 123 456 789 999; do
    CUDA_VISIBLE_DEVICES=$((seed % 4)) python scripts/train_crc_select.py \
        --seed $seed --epochs 300 --alpha_risk 0.1 \
        --output_dir checkpoints/CRC-Select &
done
wait
```

---

### 1.2 Baseline 1: Vanilla SelectiveNet ⏱️ ~10-15 hours GPU time

Train vanilla SelectiveNet WITHOUT CRC (for comparison)

- [ ] **Seed 42**
  ```bash
  python scripts/train.py \
      --seed 42 \
      --epochs 300 \
      --coverage_target 0.8 \
      --dataset cifar10 \
      --dataroot ../data \
      --output_dir checkpoints/vanilla
  ```

- [ ] **Seed 123**
  ```bash
  python scripts/train.py --seed 123 --epochs 300 --coverage_target 0.8 \
      --output_dir checkpoints/vanilla
  ```

- [ ] **Seed 456**
  ```bash
  python scripts/train.py --seed 456 --epochs 300 --coverage_target 0.8 \
      --output_dir checkpoints/vanilla
  ```

- [ ] **Seed 789**
  ```bash
  python scripts/train.py --seed 789 --epochs 300 --coverage_target 0.8 \
      --output_dir checkpoints/vanilla
  ```

- [ ] **Seed 999**
  ```bash
  python scripts/train.py --seed 999 --epochs 300 --coverage_target 0.8 \
      --output_dir checkpoints/vanilla
  ```

**Parallel execution**:
```bash
for seed in 42 123 456 789 999; do
    CUDA_VISIBLE_DEVICES=$((seed % 4)) python scripts/train.py \
        --seed $seed --epochs 300 --coverage_target 0.8 \
        --output_dir checkpoints/vanilla &
done
wait
```

---

### 1.3 Evaluate All Methods × All Seeds ⏱️ ~2-4 hours

#### CRC-Select Evaluation

- [ ] **Evaluate seed 42** (might be done already, re-run to be sure)
  ```bash
  python scripts/evaluate_for_paper.py \
      --checkpoint checkpoints/CRC-Select/seed_42.pth \
      --seed 42 \
      --method_name "CRC-Select" \
      --output_dir results_paper
  ```

- [ ] **Evaluate seed 123**
  ```bash
  python scripts/evaluate_for_paper.py \
      --checkpoint checkpoints/CRC-Select/seed_123.pth \
      --seed 123 \
      --method_name "CRC-Select" \
      --output_dir results_paper
  ```

- [ ] **Evaluate seed 456**
  ```bash
  python scripts/evaluate_for_paper.py \
      --checkpoint checkpoints/CRC-Select/seed_456.pth \
      --seed 456 \
      --method_name "CRC-Select" \
      --output_dir results_paper
  ```

- [ ] **Evaluate seed 789**
  ```bash
  python scripts/evaluate_for_paper.py \
      --checkpoint checkpoints/CRC-Select/seed_789.pth \
      --seed 789 \
      --method_name "CRC-Select" \
      --output_dir results_paper
  ```

- [ ] **Evaluate seed 999**
  ```bash
  python scripts/evaluate_for_paper.py \
      --checkpoint checkpoints/CRC-Select/seed_999.pth \
      --seed 999 \
      --method_name "CRC-Select" \
      --output_dir results_paper
  ```

#### Vanilla Evaluation (used as both baseline and for Post-hoc CRC)

- [ ] **Evaluate vanilla seed 42**
  ```bash
  python scripts/evaluate_for_paper.py \
      --checkpoint checkpoints/vanilla/seed_42.pth \
      --seed 42 \
      --method_name "vanilla" \
      --output_dir results_paper
  ```

- [ ] **Evaluate vanilla seed 123**
  ```bash
  python scripts/evaluate_for_paper.py \
      --checkpoint checkpoints/vanilla/seed_123.pth \
      --seed 123 \
      --method_name "vanilla" \
      --output_dir results_paper
  ```

- [ ] **Evaluate vanilla seed 456**
  ```bash
  python scripts/evaluate_for_paper.py \
      --checkpoint checkpoints/vanilla/seed_456.pth \
      --seed 456 \
      --method_name "vanilla" \
      --output_dir results_paper
  ```

- [ ] **Evaluate vanilla seed 789**
  ```bash
  python scripts/evaluate_for_paper.py \
      --checkpoint checkpoints/vanilla/seed_789.pth \
      --seed 789 \
      --method_name "vanilla" \
      --output_dir results_paper
  ```

- [ ] **Evaluate vanilla seed 999**
  ```bash
  python scripts/evaluate_for_paper.py \
      --checkpoint checkpoints/vanilla/seed_999.pth \
      --seed 999 \
      --method_name "vanilla" \
      --output_dir results_paper
  ```

#### Post-hoc CRC Baseline

- [ ] **Post-hoc CRC seed 42**
  ```bash
  python scripts/baseline_posthoc_crc.py \
      --checkpoint checkpoints/vanilla/seed_42.pth \
      --seed 42 \
      --alpha_risk 0.1 \
      --output_dir results_paper/posthoc_crc
  ```

- [ ] **Post-hoc CRC seed 123**
  ```bash
  python scripts/baseline_posthoc_crc.py \
      --checkpoint checkpoints/vanilla/seed_123.pth \
      --seed 123 \
      --alpha_risk 0.1 \
      --output_dir results_paper/posthoc_crc
  ```

- [ ] **Post-hoc CRC seed 456**
  ```bash
  python scripts/baseline_posthoc_crc.py \
      --checkpoint checkpoints/vanilla/seed_456.pth \
      --seed 456 \
      --alpha_risk 0.1 \
      --output_dir results_paper/posthoc_crc
  ```

- [ ] **Post-hoc CRC seed 789**
  ```bash
  python scripts/baseline_posthoc_crc.py \
      --checkpoint checkpoints/vanilla/seed_789.pth \
      --seed 789 \
      --alpha_risk 0.1 \
      --output_dir results_paper/posthoc_crc
  ```

- [ ] **Post-hoc CRC seed 999**
  ```bash
  python scripts/baseline_posthoc_crc.py \
      --checkpoint checkpoints/vanilla/seed_999.pth \
      --seed 999 \
      --alpha_risk 0.1 \
      --output_dir results_paper/posthoc_crc
  ```

**Automated script** (create if needed):
```bash
# Save as: run_all_evaluations.sh
#!/bin/bash
seeds=(42 123 456 789 999)

for seed in "${seeds[@]}"; do
    echo "Evaluating seed $seed..."
    
    # CRC-Select
    python scripts/evaluate_for_paper.py \
        --checkpoint checkpoints/CRC-Select/seed_${seed}.pth \
        --seed $seed --method_name "CRC-Select" \
        --output_dir results_paper
    
    # Vanilla
    python scripts/evaluate_for_paper.py \
        --checkpoint checkpoints/vanilla/seed_${seed}.pth \
        --seed $seed --method_name "vanilla" \
        --output_dir results_paper
    
    # Post-hoc CRC
    python scripts/baseline_posthoc_crc.py \
        --checkpoint checkpoints/vanilla/seed_${seed}.pth \
        --seed $seed --alpha_risk 0.1 \
        --output_dir results_paper/posthoc_crc
done
```

---

## 🟡 PHASE 2: Metric Computation (REQUIRED - ~2 hours)

### 2.1 Aggregate Results (Mean ± Std)

- [ ] **Aggregate Coverage@Risk**
  ```bash
  python scripts/aggregate_results.py \
      --method_dirs results_paper/CRC-Select \
                   results_paper/posthoc_crc \
                   results_paper/vanilla \
      --seeds 42 123 456 789 999 \
      --metric coverage_at_risk \
      --output_dir results/aggregated
  ```
  **Expected output**: `results/aggregated/coverage_at_risk_comparison.csv`

- [ ] **Aggregate RC Curves**
  ```bash
  python scripts/aggregate_results.py \
      --method_dirs results_paper/CRC-Select \
                   results_paper/posthoc_crc \
                   results_paper/vanilla \
      --seeds 42 123 456 789 999 \
      --metric rc_curve \
      --output_dir results/aggregated
  ```
  **Expected output**: `results/aggregated/rc_curve_mean_std.csv`

- [ ] **Aggregate OOD metrics**
  ```bash
  python scripts/aggregate_results.py \
      --method_dirs results_paper/CRC-Select \
                   results_paper/posthoc_crc \
                   results_paper/vanilla \
      --seeds 42 123 456 789 999 \
      --metric ood_evaluation \
      --output_dir results/aggregated
  ```

---

### 2.2 Compute Risk Violation Rate

- [ ] **Violation rate @ α=0.1**
  ```bash
  python scripts/compute_violation_rate.py \
      --method_dirs results_paper/CRC-Select \
                   results_paper/posthoc_crc \
                   results_paper/vanilla \
      --seeds 42 123 456 789 999 \
      --alphas 0.1 \
      --output_dir results/violation_analysis
  ```
  **Expected output**: 
  - `results/violation_analysis/violation_rate_alpha_0.1.csv`
  - Shows: CRC-Select ~10-15%, Post-hoc ~20-30%, Vanilla ~40-50%

- [ ] **Violation rate @ multiple alphas**
  ```bash
  python scripts/compute_violation_rate.py \
      --method_dirs results_paper/CRC-Select \
                   results_paper/posthoc_crc \
                   results_paper/vanilla \
      --seeds 42 123 456 789 999 \
      --alphas 0.05 0.1 0.15 0.2 \
      --output_dir results/violation_analysis
  ```

- [ ] **Verify results make sense**
  - Violation rate should be ≤ 20% (theoretical bound with δ=0.1)
  - CRC-Select should have lower violation rate than baselines
  - Document any anomalies

---

### 2.3 OOD Safety Analysis

- [ ] **OOD-Acceptance @ Fixed ID Coverage**
  ```bash
  python scripts/compare_ood_safety.py \
      --results_dir results_paper \
      --methods CRC-Select posthoc_crc vanilla \
      --seeds 42 123 456 789 999 \
      --id_coverages 0.7 0.8 0.9 \
      --output_dir results/ood_comparison
  ```
  **Expected output**:
  - `results/ood_comparison/ood_at_fixed_coverage.csv`
  - Table showing OOD acceptance rate for each method @ 70%, 80%, 90% ID coverage

- [ ] **Safety Ratio Analysis**
  ```bash
  python scripts/compare_ood_safety.py \
      --results_dir results_paper \
      --methods CRC-Select posthoc_crc vanilla \
      --seeds 42 123 456 789 999 \
      --compute_safety_ratio \
      --output_dir results/ood_comparison
  ```

---

## 🟢 PHASE 3: Visualization (FOR PAPER - ~3 hours)

### 3.1 Main Result Figures

- [ ] **Figure 1: Risk-Coverage Curves**
  ```bash
  python scripts/generate_paper_figures.py \
      --figure rc_curves \
      --results_dir results_paper \
      --methods CRC-Select posthoc_crc vanilla \
      --seeds 42 123 456 789 999 \
      --output figures/figure1_rc_curves.pdf
  ```
  **Expected**: 
  - RC curves for all 3 methods (mean ± std shaded area)
  - CRC-Select dominates (higher coverage at same risk)
  - Legend, axis labels, grid

- [ ] **Figure 2: Coverage@Risk Comparison**
  ```bash
  python scripts/generate_paper_figures.py \
      --figure coverage_bars \
      --results_dir results/aggregated \
      --alphas 0.05 0.1 0.15 \
      --output figures/figure2_coverage_comparison.pdf
  ```
  **Expected**:
  - Bar chart: 3 groups (α=0.05, 0.1, 0.15)
  - Each group: 3 bars (CRC-Select, Post-hoc, Vanilla)
  - Error bars (std across seeds)
  - Percentage labels on bars

- [ ] **Figure 3: OOD Dangerous Acceptance Rate**
  ```bash
  python scripts/generate_paper_figures.py \
      --figure ood_dar \
      --results_dir results/ood_comparison \
      --output figures/figure3_ood_safety.pdf
  ```
  **Expected**:
  - Line plot: X-axis = ID coverage (70%, 80%, 90%)
  - Y-axis = OOD acceptance rate (lower is better)
  - 3 lines for 3 methods with error bars
  - Shows CRC-Select has lowest OOD leakage

- [ ] **Figure 4: Violation Rate Distribution**
  ```bash
  python scripts/generate_paper_figures.py \
      --figure violation_dist \
      --results_dir results/violation_analysis \
      --alpha 0.1 \
      --output figures/figure4_violation_rate.pdf
  ```
  **Expected**:
  - Box plot or violin plot showing risk distribution across seeds
  - Horizontal line at α=0.1
  - Shows CRC-Select violations are controlled

---

### 3.2 Supplementary Figures

- [ ] **Supp Fig 1: Training Curves**
  ```bash
  python scripts/plot_results.py \
      --plot training_curves \
      --wandb_runs results_paper/*/wandb/ \
      --output figures/supp_training_curves.pdf
  ```

- [ ] **Supp Fig 2: Calibration Plots**
  ```bash
  python scripts/generate_paper_figures.py \
      --figure calibration \
      --results_dir results_paper \
      --methods CRC-Select posthoc_crc vanilla \
      --output figures/supp_calibration.pdf
  ```

- [ ] **Supp Fig 3: Threshold τ vs Metrics**
  ```bash
  python scripts/generate_paper_figures.py \
      --figure threshold_sensitivity \
      --results_dir results_paper \
      --output figures/supp_threshold_analysis.pdf
  ```

---

### 3.3 LaTeX Tables Generation

- [ ] **Table 1: Main Results (Coverage@Risk)**
  ```bash
  python scripts/aggregate_results.py \
      --generate_latex \
      --metric coverage_at_risk \
      --alphas 0.05 0.1 0.15 \
      --output results/latex_tables/table1_main_results.tex
  ```
  **Format**:
  ```latex
  Method          & α=0.05      & α=0.1       & α=0.15     \\
  CRC-Select      & 95.2 ± 1.3  & 98.5 ± 0.8  & 99.2 ± 0.5 \\
  Post-hoc CRC    & 87.3 ± 2.1  & 92.4 ± 1.5  & 95.8 ± 1.2 \\
  Vanilla         & 75.6 ± 3.4  & 82.1 ± 2.8  & 88.3 ± 2.1 \\
  ```

- [ ] **Table 2: Risk Violation Rate**
  ```bash
  python scripts/compute_violation_rate.py \
      --generate_latex \
      --output results/latex_tables/table2_violation_rate.tex
  ```

- [ ] **Table 3: OOD Safety**
  ```bash
  python scripts/compare_ood_safety.py \
      --generate_latex \
      --output results/latex_tables/table3_ood_safety.tex
  ```

- [ ] **Table 4: AURC Comparison**
  ```bash
  python scripts/aggregate_results.py \
      --generate_latex \
      --metric aurc \
      --output results/latex_tables/table4_aurc.tex
  ```

---

## 🔵 PHASE 4: Ablation Studies (RECOMMENDED - ~2 days)

### 4.1 Effect of Target Risk α

- [ ] **Train with α=0.05**
  ```bash
  python scripts/train_crc_select.py \
      --seed 42 --alpha_risk 0.05 --epochs 300 \
      --output_dir checkpoints/ablation_alpha/alpha_0.05
  ```

- [ ] **Train with α=0.15**
  ```bash
  python scripts/train_crc_select.py \
      --seed 42 --alpha_risk 0.15 --epochs 300 \
      --output_dir checkpoints/ablation_alpha/alpha_0.15
  ```

- [ ] **Train with α=0.20**
  ```bash
  python scripts/train_crc_select.py \
      --seed 42 --alpha_risk 0.20 --epochs 300 \
      --output_dir checkpoints/ablation_alpha/alpha_0.20
  ```

- [ ] **Evaluate all α variants**
  ```bash
  for alpha in 0.05 0.10 0.15 0.20; do
      python scripts/evaluate_for_paper.py \
          --checkpoint checkpoints/ablation_alpha/alpha_${alpha}/seed_42.pth \
          --seed 42 \
          --output_dir results/ablation_alpha/alpha_${alpha}
  done
  ```

- [ ] **Plot α sensitivity**
  ```bash
  python scripts/generate_paper_figures.py \
      --figure alpha_ablation \
      --results_dir results/ablation_alpha \
      --output figures/ablation_alpha.pdf
  ```

---

### 4.2 Effect of Risk Penalty Weight μ

- [ ] **Train with μ=0.5**
  ```bash
  python scripts/train_crc_select.py \
      --seed 42 --mu_risk_init 0.5 --epochs 300 \
      --output_dir checkpoints/ablation_mu/mu_0.5
  ```

- [ ] **Train with μ=2.0**
  ```bash
  python scripts/train_crc_select.py \
      --seed 42 --mu_risk_init 2.0 --epochs 300 \
      --output_dir checkpoints/ablation_mu/mu_2.0
  ```

- [ ] **Train with μ=5.0**
  ```bash
  python scripts/train_crc_select.py \
      --seed 42 --mu_risk_init 5.0 --epochs 300 \
      --output_dir checkpoints/ablation_mu/mu_5.0
  ```

- [ ] **Evaluate all μ variants**

- [ ] **Plot μ sensitivity**

---

### 4.3 Effect of Calibration Frequency

- [ ] **Train with calibrate_every=5**
  ```bash
  python scripts/train_crc_select.py \
      --seed 42 --calibrate_every 5 --epochs 300 \
      --output_dir checkpoints/ablation_calib/freq_5
  ```

- [ ] **Train with calibrate_every=20**
  ```bash
  python scripts/train_crc_select.py \
      --seed 42 --calibrate_every 20 --epochs 300 \
      --output_dir checkpoints/ablation_calib/freq_20
  ```

- [ ] **Train with calibrate_every=50**
  ```bash
  python scripts/train_crc_select.py \
      --seed 42 --calibrate_every 50 --epochs 300 \
      --output_dir checkpoints/ablation_calib/freq_50
  ```

- [ ] **Evaluate all calibration frequency variants**

- [ ] **Create ablation table**
  ```bash
  python scripts/generate_ablation_table.py \
      --results_dirs results/ablation_* \
      --output results/latex_tables/table_ablations.tex
  ```

---

### 4.4 Component Ablation

Test the contribution of each loss component:

- [ ] **No risk penalty (μ=0)** → Should match post-hoc CRC
  ```bash
  python scripts/train_crc_select.py \
      --seed 42 --mu_risk_init 0.0 --no_risk_penalty \
      --output_dir checkpoints/ablation_component/no_risk
  ```

- [ ] **No coverage constraint (β=0)**
  ```bash
  python scripts/train_crc_select.py \
      --seed 42 --beta_coverage 0.0 \
      --output_dir checkpoints/ablation_component/no_coverage
  ```

- [ ] **No CRC calibration** (fixed threshold)
  ```bash
  python scripts/train_crc_select.py \
      --seed 42 --no_calibration --tau_fixed 0.5 \
      --output_dir checkpoints/ablation_component/no_calib
  ```

- [ ] **Full model** (baseline - already done)

- [ ] **Create component ablation table**

---

## 🟣 PHASE 5: Additional Experiments (NICE TO HAVE - ~3-5 days)

### 5.1 Different Datasets

- [ ] **CIFAR-100 as ID**
  ```bash
  python scripts/train_crc_select.py \
      --dataset cifar100 \
      --seed 42 \
      --epochs 300 \
      --output_dir checkpoints/cifar100
  ```

- [ ] **SVHN as ID, CIFAR-10 as OOD (reverse)**
  ```bash
  python scripts/train_crc_select.py \
      --dataset svhn \
      --ood_dataset cifar10 \
      --seed 42 \
      --epochs 300 \
      --output_dir checkpoints/svhn
  ```

- [ ] **Evaluate cross-dataset generalization**

---

### 5.2 Multiple OOD Datasets

Evaluate on various OOD datasets to test robustness:

- [ ] **SVHN** (already done)

- [ ] **CIFAR-100**
  ```bash
  python scripts/evaluate_for_paper.py \
      --checkpoint checkpoints/CRC-Select/seed_42.pth \
      --ood_dataset cifar100 \
      --output_suffix _ood_cifar100
  ```

- [ ] **Textures (DTD)**
  ```bash
  # Needs dataset download first
  python scripts/evaluate_for_paper.py \
      --checkpoint checkpoints/CRC-Select/seed_42.pth \
      --ood_dataset textures \
      --output_suffix _ood_textures
  ```

- [ ] **TinyImageNet**
  ```bash
  python scripts/evaluate_for_paper.py \
      --checkpoint checkpoints/CRC-Select/seed_42.pth \
      --ood_dataset tiny_imagenet \
      --output_suffix _ood_tinyimagenet
  ```

- [ ] **Create multi-OOD comparison table**

---

### 5.3 Different Architectures

- [ ] **ResNet-18**
  ```bash
  python scripts/train_crc_select.py \
      --arch resnet18 \
      --seed 42 \
      --output_dir checkpoints/resnet18
  ```

- [ ] **ResNet-34**
  ```bash
  python scripts/train_crc_select.py \
      --arch resnet34 \
      --seed 42 \
      --output_dir checkpoints/resnet34
  ```

- [ ] **Compare architectures**

---

### 5.4 Computational Cost Analysis

- [ ] **Measure training time**
  - CRC-Select vs Vanilla vs Post-hoc
  - Log to `results/timing_analysis.csv`

- [ ] **Measure inference time**
  - Average prediction time per sample
  - Overhead from selection mechanism

- [ ] **Create efficiency table**

---

## 📝 PHASE 6: Paper Writing (1 week)

### 6.1 Draft Structure

- [ ] **Title & Abstract**
  - Clear problem statement
  - Key contribution (joint training for CRC)
  - Main results (X% coverage improvement)

- [ ] **Introduction (2 pages)**
  - [ ] Motivation: Why selective prediction + risk control?
  - [ ] Problem: Post-hoc CRC limitations
  - [ ] Solution: CRC-Select
  - [ ] Contributions (3-4 bullets)
  - [ ] Paper organization

- [ ] **Related Work (1.5 pages)**
  - [ ] Selective prediction & reject option
  - [ ] Conformal prediction
  - [ ] Risk control methods
  - [ ] OOD detection
  - [ ] Position relative to prior work

- [ ] **Background (1 page)**
  - [ ] SelectiveNet formulation
  - [ ] Conformal Risk Control theory
  - [ ] Problem setup

- [ ] **Method: CRC-Select (2-3 pages)**
  - [ ] Overall architecture
  - [ ] Loss function design
    - [ ] L_pred: Selective classification
    - [ ] L_cov: Coverage constraint
    - [ ] L_risk: CRC-aware penalty (NEW!)
  - [ ] Alternating optimization algorithm
  - [ ] Implementation details
  - [ ] Theoretical justification (optional)

- [ ] **Experiments (3-4 pages)**
  - [ ] Experimental setup
    - [ ] Datasets (CIFAR-10, SVHN)
    - [ ] Baselines (Vanilla, Post-hoc CRC)
    - [ ] Training details
    - [ ] Evaluation metrics
  - [ ] Main results
    - [ ] Table 1: Coverage@Risk comparison
    - [ ] Figure 1: RC curves
    - [ ] Table 2: Risk violation rate
  - [ ] OOD safety analysis
    - [ ] Table 3: OOD acceptance rates
    - [ ] Figure 2: OOD safety comparison
  - [ ] Ablation studies
    - [ ] Effect of α, μ, calibration frequency
    - [ ] Component analysis
  - [ ] Analysis & discussion
    - [ ] Why does joint training help?
    - [ ] Visualization of learned selector

- [ ] **Discussion (0.5 page)**
  - [ ] Key insights
  - [ ] Limitations
  - [ ] Future directions

- [ ] **Conclusion (0.5 page)**
  - [ ] Summary of contributions
  - [ ] Impact statement

- [ ] **References**
  - [ ] Collect all citations
  - [ ] Format consistently

---

### 6.2 Paper Checklist

- [ ] All figures have captions
- [ ] All tables have captions
- [ ] All claims are supported by evidence
- [ ] Statistical significance tests performed
- [ ] Code will be released (mention in paper)
- [ ] Reproducibility: all hyperparameters documented
- [ ] Limitations discussed
- [ ] Broader impact considered
- [ ] Formatting matches venue (e.g., NeurIPS, ICML)

---

### 6.3 Supplementary Material

- [ ] **Detailed proofs** (if theoretical contributions)
- [ ] **Additional ablations**
- [ ] **More experimental details**
- [ ] **Extended related work**
- [ ] **Hyperparameter search details**
- [ ] **Additional visualizations**
- [ ] **Failure cases analysis**

---

## 🚀 PHASE 7: Finalization (3-5 days)

### 7.1 Internal Review

- [ ] Self-review checklist:
  - [ ] Read paper start to finish
  - [ ] Check all numbers match results
  - [ ] Verify all references cited
  - [ ] Check grammar & spelling
  - [ ] Ensure figures are high quality (300+ dpi)

- [ ] Peer review:
  - [ ] Get feedback from advisor/colleagues
  - [ ] Address comments
  - [ ] Revise unclear sections

---

### 7.2 Code & Reproducibility

- [ ] **Clean up code**
  - [ ] Remove debug prints
  - [ ] Add docstrings
  - [ ] Consistent naming
  - [ ] Remove unused files

- [ ] **Create reproduction package**
  ```bash
  # Create archive with:
  # - Trained checkpoints
  # - Evaluation results
  # - Generated figures
  # - Scripts to reproduce
  ```

- [ ] **README for reproduction**
  - [ ] Step-by-step instructions
  - [ ] Expected outputs
  - [ ] Hardware requirements
  - [ ] Time estimates

- [ ] **Requirements.txt pinned versions**
  ```bash
  pip freeze > requirements_pinned.txt
  ```

---

### 7.3 Submission Preparation

- [ ] **Format for venue**
  - [ ] Page limit compliance
  - [ ] Margin requirements
  - [ ] Figure quality
  - [ ] Citation format

- [ ] **Supplementary material**
  - [ ] PDF generation
  - [ ] File size < limit
  - [ ] All referenced in main text

- [ ] **Code submission** (if required)
  - [ ] Anonymize code
  - [ ] Include trained models or reproduction scripts
  - [ ] Test on fresh environment

- [ ] **Final checks**
  - [ ] PDF renders correctly
  - [ ] No compilation errors/warnings
  - [ ] All co-authors reviewed
  - [ ] Conflict of interest declared

- [ ] **Submit!** 🎉

---

## 📊 Quick Progress Tracker

### Summary Statistics

**Core Experiments**: 0/30 tasks complete (0%)
- Training: 0/10 models
- Evaluation: 0/15 runs
- Metrics: 0/5 computed

**Visualization**: 0/10 figures complete (0%)

**Paper Writing**: 0/7 sections complete (0%)

**Overall Progress**: ~5% (only single-seed CRC-Select done)

---

## ⏱️ Time Estimates

| Phase | Tasks | GPU Time | CPU Time | Total |
|-------|-------|----------|----------|-------|
| 1. Core Experiments | Training + Eval | 20-30h | 2-4h | ~2-3 days |
| 2. Metrics | Aggregation | - | 2h | 2h |
| 3. Visualization | Figures + Tables | - | 3h | 3h |
| 4. Ablations | Training + Eval | 10-15h | 1h | ~1-2 days |
| 5. Additional | Optional | 10-20h | 2h | ~2-3 days |
| 6. Writing | Paper draft | - | 40h | 5-7 days |
| 7. Finalization | Review + revise | - | 20h | 3-5 days |
| **TOTAL** | | **40-65h GPU** | **70h CPU** | **~2-3 weeks** |

---

## 🎯 Recommended Priority Order

### Week 1: Critical Experiments
1. ✅ Train remaining 4 seeds for CRC-Select (123, 456, 789, 999)
2. ✅ Train 5 seeds for Vanilla baseline
3. ✅ Run all evaluations
4. ✅ Compute core metrics (violation rate, OOD safety)
5. ✅ Generate main figures (Figures 1-4)

### Week 2: Paper + Ablations
6. ✅ Write paper draft (Introduction → Method → Experiments)
7. ⭐ Run key ablations (α, μ)
8. ✅ Generate LaTeX tables
9. ✅ First internal review

### Week 3: Finalization
10. ✅ Address review comments
11. ⭐ Additional experiments if needed
12. ✅ Polish writing
13. ✅ Prepare submission package
14. 🎉 Submit!

---

## 📌 Notes & Tips

### Training Tips
- Use `tmux` or `screen` for long training runs
- Monitor with `watch -n 1 nvidia-smi`
- Check wandb logs regularly
- Save checkpoints every 10 epochs (in case of crashes)

### Debugging
- If violation rate > 30%, check calibration implementation
- If OOD acceptance too high, check normalization
- If coverage very low, check μ is not too large

### Writing Tips
- Start with experiments section (easier, have results)
- Then method (describe what you did)
- Then intro/related work (motivate and position)
- Conclusion last

### Submission Venues
Target venues (deadlines in 2026):
- **NeurIPS**: May deadline
- **ICML**: January deadline (missed)
- **ICLR**: September deadline
- **AISTATS**: October deadline

---

## ✅ Quick Status Commands

```bash
# Check what's done
find results_paper -name "*.csv" | wc -l  # Number of result files

# Check seeds completed
ls checkpoints/CRC-Select/  # Should see seed_*.pth

# Check evaluations done
ls results_paper/*/seed_*/  # Should see 5 seeds × 3 methods

# Verify metrics
ls results/aggregated/  # Should have aggregated CSVs

# Check figures
ls figures/*.pdf  # Should have 4+ figures
```

---

**Last Updated**: Feb 9, 2026  
**Status**: Ready to start Phase 1  
**Next Action**: Run multi-seed training for CRC-Select

---

*This TODO list will be updated as tasks are completed. Mark items with [x] when done.*
