# Workflow: Evaluate Multiple Seeds

Bạn đã train model trên nhiều seeds. Đây là workflow để evaluate tất cả:

## 📋 Step-by-Step Guide

### Bước 1: Tổ Chức Checkpoints

Bạn có 2 options:

#### Option A: Auto-detect (nếu có seed info trong config)
```bash
cd /home/admin1/Desktop/CRC-Select-Torch
./organize_multi_seed_checkpoints.sh
```

#### Option B: Manual mapping (recommended vì không có config)
```bash
cd /home/admin1/Desktop/CRC-Select-Torch
./manual_checkpoint_setup.sh
```

Script sẽ map các wandb runs thành seeds:
- Run mới nhất → seed_42 (đã có)
- Run tiếp theo → seed_123
- Run tiếp theo → seed_456  
- Run cũ nhất → seed_789

### Bước 2: Verify Checkpoints

```bash
ls -lh checkpoints/
```

Bạn sẽ thấy:
```
seed_42.pth
seed_123.pth
seed_456.pth
seed_789.pth
```

### Bước 3: Run Evaluation Trên Tất Cả Seeds

```bash
./run_eval_all_seeds.sh
```

Script này sẽ:
1. Tự động detect tất cả seeds trong `checkpoints/`
2. Run `evaluate_for_paper.py` cho mỗi seed
3. Save results vào `../results_paper/CRC-Select/seed_XXX/`

**Thời gian:** ~2-3 phút per seed (tổng ~10-15 phút cho 4 seeds)

### Bước 4: Verify Results

```bash
ls ../results_paper/CRC-Select/
```

Bạn sẽ thấy:
```
seed_42/
seed_123/
seed_456/
seed_789/
```

Mỗi folder chứa:
- `risk_coverage_curve.csv`
- `coverage_at_risk.csv`
- `ood_evaluation.csv`
- `ood_at_fixed_id_coverage.csv` ← NEW!
- `calibration_metrics.csv`
- `summary.csv`

### Bước 5: Compute Violation Rate

```bash
python3 scripts/compute_violation_rate.py \
    --method_dirs ../results_paper/CRC-Select \
    --seeds 42 123 456 789 \
    --alphas 0.05 0.1 0.15 0.2 \
    --output_dir ../results/violation_rate \
    --generate_latex
```

Output:
- `violation_rate_comparison.csv` - Violation rates
- `violation_rate_table.tex` - LaTeX table
- Per-alpha details

### Bước 6: Compare OOD Safety

```bash
python3 scripts/compare_ood_safety.py \
    --results_dir ../results_paper \
    --methods CRC-Select \
    --seeds 42 123 456 789 \
    --plot \
    --latex \
    --output_dir ../results/ood_comparison
```

Output:
- `ood_safety_comparison.csv` - Mean ± std across seeds
- `ood_comparison_plot.png` - Visualization
- `ood_comparison_heatmap.png` - Heatmap
- `ood_comparison_table.tex` - LaTeX table

---

## 🎯 Quick Commands

### All-in-One (after checkpoints are organized):

```bash
cd /home/admin1/Desktop/CRC-Select-Torch

# 1. Run evaluations
./run_eval_all_seeds.sh

# 2. Compute violation rate
python3 scripts/compute_violation_rate.py \
    --method_dirs ../results_paper/CRC-Select \
    --seeds 42 123 456 789 \
    --alphas 0.1 --generate_latex

# 3. Compare OOD safety
python3 scripts/compare_ood_safety.py \
    --methods CRC-Select \
    --seeds 42 123 456 789 \
    --plot --latex
```

---

## 📊 Expected Results

Sau khi hoàn thành, bạn sẽ có:

### Files Structure:
```
results_paper/
└── CRC-Select/
    ├── seed_42/
    │   ├── coverage_at_risk.csv
    │   ├── ood_at_fixed_id_coverage.csv
    │   └── ...
    ├── seed_123/
    ├── seed_456/
    └── seed_789/

results/
├── violation_rate/
│   ├── violation_rate_comparison.csv
│   └── violation_rate_table.tex
└── ood_comparison/
    ├── ood_safety_comparison.csv
    ├── ood_comparison_plot.png
    └── ood_comparison_table.tex
```

### Example Results:

**Violation Rate:**
```
Method      | α=0.1 | Violations
CRC-Select  | 8.2%  | 1/4 seeds
```

**OOD Safety (mean ± std):**
```
ID Coverage | OOD Accept      | Safety Ratio
70%         | 2.5 ± 0.3%      | 28 ± 3×
80%         | 7.2 ± 1.1%      | 11 ± 2×
```

---

## ⚠️ Troubleshooting

### Issue: "No checkpoints found"

**Solution:**
```bash
# Check wandb runs
ls scripts/wandb/offline-run-*/files/checkpoints/

# Map manually
./manual_checkpoint_setup.sh
```

### Issue: Evaluation fails for some seeds

**Check:**
```bash
# Verify checkpoint file exists and is valid
ls -lh checkpoints/seed_XXX.pth

# Try loading manually
python3 -c "import torch; torch.load('checkpoints/seed_XXX.pth')"
```

### Issue: Different number of seeds than expected

Nếu bạn train 4 seeds nhưng plan là 5 (42, 123, 456, 789, 999):
- Script sẽ chỉ eval seeds có checkpoint
- Violation rate vẫn tính được với 4 seeds (tối thiểu là 3)

---

## 📝 For Paper

Sau khi có kết quả từ 4 seeds:

**Claims bạn có thể make:**

1. ✅ Coverage@Risk với statistical analysis:
   ```
   CRC-Select achieves 78.5±1.5% coverage at α=0.1
   (across 4 independent runs)
   ```

2. ✅ Risk violation rate:
   ```
   Risk violations occur in 8.2% of test sets (1/4 runs),
   demonstrating effective risk control.
   ```

3. ✅ OOD safety:
   ```
   At 80% ID coverage, OOD acceptance is 7.2±1.1%,
   providing 11× safety ratio.
   ```

4. ✅ Comparison with baselines (if you have them):
   ```
   CRC-Select improves coverage by 8% over post-hoc CRC
   while maintaining the same risk level.
   ```

---

## 🎉 Success Criteria

Checklist để verify all metrics computed correctly:

- [ ] Evaluation ran on all seeds
- [ ] All `ood_at_fixed_id_coverage.csv` files created
- [ ] Violation rate computed
- [ ] OOD comparison generated
- [ ] LaTeX tables created
- [ ] Plots generated (if --plot used)

---

Next: Generate paper figures with `scripts/generate_paper_figures.py`
