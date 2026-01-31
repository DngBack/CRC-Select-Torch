# Quick Evaluation Guide

Bạn đã train models trên nhiều seeds. Đây là cách eval nhanh nhất:

## 🚀 Quick Start (3 Commands)

```bash
cd /home/admin1/Desktop/CRC-Select-Torch

# 1. Setup checkpoints (chọn y khi được hỏi)
./manual_checkpoint_setup.sh

# 2. Run all evaluations
./run_eval_all_seeds.sh

# 3. Compute metrics
python3 scripts/compute_violation_rate.py \
    --method_dirs ../results_paper/CRC-Select \
    --seeds 42 123 456 789 \
    --alphas 0.1 --generate_latex

python3 scripts/compare_ood_safety.py \
    --methods CRC-Select \
    --seeds 42 123 456 789 \
    --plot --latex
```

**Thời gian:** ~15-20 phút total

---

## 📊 What You'll Get

### Files Created:

```
results_paper/CRC-Select/
├── seed_42/
│   ├── coverage_at_risk.csv
│   ├── ood_at_fixed_id_coverage.csv  ← NEW!
│   └── risk_coverage_curve.csv
├── seed_123/
├── seed_456/
└── seed_789/

results/
├── violation_rate/
│   ├── violation_rate_comparison.csv  ← For paper Table 1
│   └── violation_rate_table.tex       ← LaTeX
└── ood_comparison/
    ├── ood_safety_comparison.csv      ← For paper Table 2
    ├── ood_comparison_plot.png        ← Figure
    └── ood_comparison_table.tex       ← LaTeX
```

### Metrics You Can Report:

1. **Coverage@Risk (with std)**
   - Mean ± std across 4 seeds
   - "78.5 ± 1.5% coverage at α=0.1"

2. **Risk Violation Rate**
   - "8.2% of runs violate risk constraint"
   - "(1/4 seeds had risk > 0.1)"

3. **OOD Safety (with std)**
   - "At 80% ID coverage: 7.2 ± 1.1% OOD acceptance"
   - "Safety ratio: 11 ± 2×"

4. **AURC**
   - "0.0125 ± 0.001"

---

## 🔍 Verify Results

```bash
# Check all evaluations completed
ls ../results_paper/CRC-Select/

# View violation rate
cat ../results/violation_rate/violation_rate_comparison.csv

# View OOD comparison
cat ../results/ood_comparison/ood_safety_comparison.csv

# Check summary for each seed
cat ../results_paper/CRC-Select/seed_*/summary.csv
```

---

## ⚠️ Important Notes

1. **Bạn có 4 seeds (42, 123, 456, 789), không phải 5**
   - Đủ cho statistical analysis
   - Violation rate vẫn tính được
   - Paper recommended: ≥3 seeds

2. **Nếu bạn muốn thêm seed 999:**
   ```bash
   # Train thêm
   python3 scripts/train_crc_select.py --seed 999 --dataset cifar10
   
   # Copy checkpoint
   cp scripts/wandb/latest-run/files/checkpoints/checkpoint_best_val.pth \
      checkpoints/seed_999.pth
   
   # Eval
   python3 scripts/evaluate_for_paper.py \
       --checkpoint checkpoints/seed_999.pth --seed 999
   
   # Re-run analysis với seeds 42 123 456 789 999
   ```

3. **Seeds mapping (auto-detected từ timestamps):**
   - Latest run (Jan 27) → seed_42
   - Jan 26 13:54 → seed_123
   - Jan 26 13:44 → seed_456
   - Jan 26 09:52 → seed_789

---

## 📝 For Paper Writing

### Table 1: Main Results
```
Method      | Coverage@0.1  | Violation Rate | AURC
CRC-Select  | 78.5 ± 1.5%   | 8.2%          | 0.0125 ± 0.001
```

### Table 2: OOD Safety
```
ID Coverage | OOD Accept (%)  | Safety Ratio
70%         | 2.5 ± 0.3       | 28 ± 3×
80%         | 7.2 ± 1.1       | 11 ± 2×
90%         | 45.2 ± 5.2      | 2.0 ± 0.2×
```

*Note: Numbers above are examples - use actual values from your results*

---

## 🆘 Troubleshooting

### "No checkpoints found"
```bash
# Check wandb runs
ls -lh scripts/wandb/offline-run-*/files/checkpoints/

# Run manual setup
./manual_checkpoint_setup.sh
```

### "Evaluation failed for seed XXX"
```bash
# Check checkpoint
ls -lh checkpoints/seed_XXX.pth

# Try loading
python3 -c "import torch; print(torch.load('checkpoints/seed_XXX.pth').keys())"

# Re-copy if corrupted
cp scripts/wandb/offline-run-XXXXX/files/checkpoints/checkpoint_best_val.pth \
   checkpoints/seed_XXX.pth
```

### Missing OOD files
```bash
# OOD files should be created automatically
# If missing, re-run evaluation:
python3 scripts/evaluate_for_paper.py \
    --checkpoint checkpoints/seed_XXX.pth \
    --seed XXX
```

---

## 🎯 Success Checklist

- [ ] Checkpoints organized (4 files in `checkpoints/`)
- [ ] All evaluations completed (4 folders in `results_paper/CRC-Select/`)
- [ ] Each folder has `ood_at_fixed_id_coverage.csv`
- [ ] Violation rate computed
- [ ] OOD comparison generated
- [ ] LaTeX tables created

---

## 📚 More Info

- Detailed workflow: `EVAL_WORKFLOW.md`
- Metric explanations: `METRICS_SUMMARY.md`
- Implementation status: `METRIC_IMPLEMENTATION_STATUS.md`
- Full guide: `METRIC_USAGE_GUIDE.md`
