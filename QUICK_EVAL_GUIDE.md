# Quick Evaluation Guide

Hướng dẫn chạy đánh giá CRC-Select nhanh nhất — từ training đến figures cho paper.

---

## 1. Training (5 seeds)

```bash
cd /path/to/CRC-Select-Torch

for seed in 42 123 456 789 999; do
    python scripts/train_crc_select.py \
        --dataset cifar10 \
        --seed $seed \
        --num_epochs 200 \
        --alpha_risk 0.1 \
        --backbone vgg16 \
        --warmup_epochs 20 \
        --recalibrate_every 5 \
        --use_dual_update \
        --unobserve
done
```

Kết quả checkpoint lưu tại `results/CRC-Select/seed_<s>/`.

---

## 2. Evaluation (tất cả methods)

### CRC-Select

```bash
for seed in 42 123 456 789 999; do
    python scripts/evaluate_for_paper.py \
        --checkpoint_dir results/CRC-Select/seed_$seed \
        --dataset cifar10 \
        --backbone vgg16
done
```

### Baselines

```bash
for seed in 42 123 456 789 999; do
    python scripts/baseline_msp.py          --dataset cifar10 --seed $seed
    python scripts/baseline_temp_scaled.py   --dataset cifar10 --seed $seed
    python scripts/baseline_posthoc_crc.py   --checkpoint_dir results/CRC-Select/seed_$seed --dataset cifar10
    python scripts/baseline_deep_gambler.py  --dataset cifar10 --seed $seed
    python scripts/baseline_energy.py        --dataset cifar10 --seed $seed
done
```

---

## 3. Analysis

### Violation rate (accepted-loss mass)

```bash
python scripts/compute_violation_rate.py \
    --method_dirs results/CRC-Select results/posthoc_crc results/msp \
        results/temp_scaled results/deep_gambler results/energy \
    --seeds 42 123 456 789 999 \
    --alphas 0.05 0.1 0.15 0.2 \
    --generate_latex \
    -o results/violation_rate
```

### Aggregate mean ± std

```bash
python scripts/aggregate_results.py \
    --method_dirs results/CRC-Select results/posthoc_crc results/msp \
        results/temp_scaled results/deep_gambler results/energy \
    --seeds 42 123 456 789 999 \
    -o results/aggregated
```

---

## 4. Figures & Tables

```bash
python scripts/generate_paper_figures.py \
    --results_dir results \
    --methods CRC-Select posthoc_crc msp temp_scaled deep_gambler energy \
    --seeds 42 123 456 789 999 \
    --output_dir figures
```

Output:
- `figure2_rc_frontier.pdf` — Accepted-loss mass + selective risk vs coverage
- `figure3_coverage_at_alpha.pdf` — Grouped bars per α
- `figure4_violation_gap.pdf` — Box plot across seeds
- `figure5_ood_dar.pdf` — OOD acceptance at fixed ID coverage
- `figure7_threshold_efficiency.pdf` — Conservativeness & coverage efficiency
- `table1_main_results.csv` / `.tex` — All metrics (mean ± std)
- `table4_violation_rates.csv` / `.tex`

---

## 5. Ablations (optional)

```bash
# All 7 ablations at once
python scripts/run_ablations.py --ablation all --dataset cifar10 \
    --seeds 42 123 456 789 999 --output_dir results/ablations

# Or one at a time
python scripts/run_ablations.py --ablation A1  # λ-risk sensitivity
python scripts/run_ablations.py --ablation A6  # backbone comparison
```

---

## Output Structure

```
results/
├── CRC-Select/
│   └── seed_42/
│       ├── all_metrics.csv              ← Primary: all metrics per alpha
│       ├── risk_coverage_curve.csv      ← RC curve data
│       ├── coverage_at_risk.csv         ← Coverage after CRC calibration
│       ├── calibration_metrics.csv      ← CRC calibration details
│       └── summary.csv
├── posthoc_crc/seed_42/...
├── msp/seed_42/...
├── temp_scaled/seed_42/...
├── deep_gambler/seed_42/...
├── energy/seed_42/...
├── aggregated/
│   ├── CRC-Select/
│   │   ├── coverage_at_risk_aggregated.csv
│   │   ├── risk_coverage_curve_aggregated.csv
│   │   └── ood_results_aggregated.csv
│   └── summary_table.csv
├── violation_rate/
│   ├── violation_rate_comparison.csv
│   └── violation_rate_table.tex
└── ablations/
    ├── ablation_A1_lambda_risk_sensitivity/
    └── ...

figures/
├── figure2_rc_frontier.pdf
├── figure3_coverage_at_alpha.pdf
├── ...
├── table1_main_results.csv
└── table1_main_results.tex
```

---

## Key Metrics

| Metric | File column | Description |
|--------|------------|-------------|
| Accepted-loss mass $\hat A$ | `accepted_loss_mass` | CRC-certified quantity ≤ α |
| Violation gap | `violation_gap` | max(0, Â − α) |
| Coverage | `coverage` | Fraction accepted |
| Selective risk | `selective_risk` | A/C (descriptive only) |
| AUROC | `auroc` | Error detection quality |
| AUPR | `aupr` | Error detection (imbalanced) |
| RC-AUC | `rc_auc` | Area under risk-coverage curve |
| Conservativeness | `conservativeness` | α − Â (slack) |
| DAR | `dar` | OOD dangerous acceptance rate |

---

## Verify Results

```bash
# Check all seeds evaluated
ls results/CRC-Select/

# Check all_metrics.csv for a seed
cat results/CRC-Select/seed_42/all_metrics.csv

# Quick summary table
cat results/aggregated/summary_table.csv

# Violation rate
cat results/violation_rate/violation_rate_comparison.csv
```

---

## Troubleshooting

### "No checkpoint found"
```bash
# Check checkpoint directories
ls results/CRC-Select/seed_*/

# If using wandb, copy checkpoint manually
cp scripts/wandb/offline-run-*/files/checkpoints/checkpoint_best_val.pth \
   results/CRC-Select/seed_42/checkpoint.pth
```

### "Missing alpha in all_metrics.csv"
Evaluation uses `--alphas 0.05 0.1 0.15 0.2` by default. Check:
```bash
head results/CRC-Select/seed_42/all_metrics.csv
```

### Memory issues with Tiny-ImageNet
Reduce batch size: `--batch_size 64`
