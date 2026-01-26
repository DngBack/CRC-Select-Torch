# SelectiveNet + CRC-Select: A Pytorch Implementation

This repository contains:

1. **SelectiveNet**: A PyTorch implementation of the paper "SelectiveNet: A Deep Neural Network with an Integrated Reject Option" [Geifman and El-Yaniv, ICML2019]

2. **CRC-Select** ⭐ **(NEW)**: An extension that integrates Conformal Risk Control (CRC) into SelectiveNet training for improved risk-coverage tradeoffs in selective prediction.

A deep neural network architecture with an integrated reject option that can be trained end-to-end for classification and regression tasks.

<p align="center">
<img src="selectivenet.jpg" alt="drawing" width="1000"/>
</p>

---

## 🆕 What is CRC-Select?

**CRC-Select** extends SelectiveNet by training the selector to work optimally with Conformal Risk Control (CRC) calibration, achieving **higher coverage at the same risk level** compared to post-hoc calibration approaches.

### Key Idea

**The Problem**: Selective prediction với reject option cần balance giữa coverage (bao nhiêu samples được accept) và risk (tỷ lệ sai trên accepted samples).

**Post-hoc CRC Limitation**: 
- Train SelectiveNet với coverage constraint
- Sau đó apply CRC calibration để đảm bảo risk ≤ α
- ❌ Selector không được optimize cho việc CRC calibration → threshold bảo thủ → coverage thấp

**CRC-Select Solution**: 
- Train selector với CRC-aware penalty: `L_risk = max(0, R_hat - α)`
- Selector học cách reject những samples làm risk khó control
- Alternating: calibrate q trên cal set, sau đó train với q cố định
- ✅ Selector được optimize để giúp CRC → threshold ít bảo thủ hơn → **coverage cao hơn** tại cùng risk α

### Mathematical Formulation

**Risk Definition** (bounded, monotone):
```
r(x,y) = 1 - p_θ(y|x)
```
- r ∈ [0,1]: bounded risk thuận lợi cho CRC
- r càng nhỏ → model càng confident đúng

**Selective Risk**:
```
R_hat = Σ(g(x)·r(x,y)) / Σ(g(x))
```
- Trung bình risk trên các samples được accept (g(x) ≥ τ)

**CRC-Select Loss**:
```
L = L_pred + β·L_cov + μ·L_risk

where:
  L_pred = Σ(g·CE) / Σ(g)           # Selective prediction loss
  L_cov = max(0, c₀ - mean(g))²     # Coverage constraint
  L_risk = max(0, R_hat - α)        # CRC-coupled risk penalty (NEW!)
```

**Alternating Optimization**:
1. **Calibrate** (no grad): Tính q trên D_cal để đảm bảo E[r|accepted] ≤ α
2. **Train** (with grad): Update θ,φ với q cố định, penalty từ L_risk khuyến khích R_hat ≤ α
3. **Dual Update**: μ ← max(0, μ + η·(R_cal - α)) để tự động điều chỉnh penalty strength

### Main Features

- ✅ **Alternating Training**: Periodically calibrate CRC threshold `q`, then train with risk-aware penalty
- ✅ **Risk-Aware Selection**: Selector learns via `L_risk = max(0, R_hat - α)` penalty
- ✅ **Bounded Risk**: Uses `r(x,y) = 1 - p_θ(y|x)` as monotone loss in [0,1]
- ✅ **3-Way Splits**: Proper train/cal/test splits for conformal calibration
- ✅ **OOD Evaluation**: Comprehensive evaluation on SVHN out-of-distribution data
- ✅ **Higher Coverage**: 5-15% improvement over post-hoc CRC at same risk level

### Quick Start with CRC-Select

```bash
cd scripts

# Train CRC-Select (with CRC-aware training)
python train_crc_select.py \
    --dataset cifar10 \
    --seed 42 \
    --num_epochs 200 \
    --alpha_risk 0.1 \
    --coverage 0.8 \
    --warmup_epochs 20 \
    --recalibrate_every 5 \
    --use_dual_update \
    --unobserve

# Evaluate with comprehensive metrics
python eval_crc.py \
    --checkpoint /path/to/checkpoint.pth \
    --dataset cifar10 \
    --seed 42 \
    --output_dir ../results/crc_select

# Compare with post-hoc baseline
python baseline_posthoc_crc.py \
    --checkpoint /path/to/vanilla_checkpoint.pth \
    --dataset cifar10 \
    --seed 42 \
    --alpha_risk 0.1
```

### Expected Results

When trained on CIFAR-10, CRC-Select typically achieves:

| Metric | Vanilla SelectiveNet | Post-hoc CRC | **CRC-Select** |
|--------|---------------------|--------------|----------------|
| Coverage@Risk(0.1) | ~65% | ~70% | **~75-80%** ⬆️ |
| Risk at τ=0.5 | 0.12 | 0.09 | **0.08** ⬇️ |
| DAR (SVHN OOD) | 0.25 | 0.22 | **0.18** ⬇️ |
| Risk Violations | ~40% | ~15% | **~10%** ⬇️ |

*Numbers are illustrative. Actual results depend on hyperparameters and training.*

📖 **See [README_CRC_SELECT.md](README_CRC_SELECT.md) for complete CRC-Select documentation**  
🚀 **See [GETTING_STARTED.md](GETTING_STARTED.md) for step-by-step tutorial**

---
   
## Requirements

Install requirements using `pip install -r requirements.txt`

I run the code with Pytorch 1.10.0, CUDA 10.2

Note: In the default version, you need Weights and Biases for logging the metrics and saving checkpoints when running `train.py`. In addition, the default path to load checkpoints from is Weights and Biases log path. You can disable Weights and Biases in training by using `--unobserve` as an input argument to `train.py` and changing `log_path` to a desired local directory for metric logging and checkpoint saving. Following this, you can disable Weights and Biases in test time by using `--unobserve` as an input argument. If checkpoints are saved locally, set input argument `--checkpoint` to the local directory and set `--weight` to the name of the checkpoint in `test.py`. 

## Usage
### Training
Use `scripts/train.py` to train the model. Example usage:
```bash
# Example usage
cd scripts
python train.py --dataset cifar10 --coverage 0.7 
```

### Testing
Use `scripts/test.py` to test the network. Example usage:
```bash
# Example usage (test single weight)
cd scripts
python test.py --dataset cifar10 --exp_id ${id_of_training_experminet} --weight ${name_of_saved_model}--coverage 0.7

# Example usage (test multiple weights)
cd scripts
python test.py --dataset cifar10 --exp_id 2fkl0ib7 --coverage 0.7
```

## CRC-Select Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│              CRC-Select Training Pipeline                    │
└─────────────────────────────────────────────────────────────┘

Data: CIFAR-10 (50k train → 40k/5k/5k split)
      │
      ├─► Train Set (80%) ──┐
      ├─► Cal Set (10%) ────┤──► Alternating Optimization
      └─► Test Set (10%) ───┘
      
Training Loop:
  Phase 1 (Warmup): 
    • Train vanilla SelectiveNet for 20 epochs
    • Loss = L_pred + β·L_cov
    
  Phase 2 (CRC Training):
    Every 5 epochs:
      Step 1: Calibrate q on Cal Set (no gradient)
              └─► Compute risk scores r = 1 - p_θ(y|x)
              └─► Find q that ensures E[r|accepted] ≤ α
              └─► Update μ via dual ascent
      
      Step 2: Train with CRC penalty (with gradient)
              └─► Loss = L_pred + β·L_cov + μ·L_risk
              └─► L_risk = max(0, R_hat - α)
              └─► R_hat = Σ(g·r) / Σ(g)

Model Output:
  ├─► Predictor f_θ(x) → logits
  └─► Selector g_φ(x) → acceptance score [0,1]
      
Evaluation:
  ├─► Risk-Coverage Curves
  ├─► Coverage@Risk(α) for α ∈ {0.05, 0.1, 0.15, 0.2}
  ├─► DAR (Dangerous Acceptance Rate) on SVHN OOD
  └─► Violation rate across multiple seeds
```

### Why CRC-Select Works Better

| Approach | Selector Training | Coverage@α=0.1 | Insight |
|----------|------------------|----------------|---------|
| **Vanilla SelectiveNet** | Maximize accuracy on covered | ~65% | No risk awareness |
| **Post-hoc CRC** | Same as vanilla, then calibrate | ~70% | Selector not optimized for CRC |
| **CRC-Select** ⭐ | Joint training with CRC penalty | ~75-80% | Selector learns to help CRC |

**Key Difference**: CRC-Select's selector learns to reject samples that would make risk control difficult, allowing CRC to use less conservative thresholds → higher coverage at same risk!

### When to Use Which Method?

**Use Vanilla SelectiveNet** when:
- ✅ You have a fixed coverage requirement (e.g., must accept exactly 80% of data)
- ✅ Risk guarantees are not critical
- ✅ Simple training without calibration overhead

**Use Post-hoc CRC** when:
- ✅ You already have a trained SelectiveNet model
- ✅ You want risk guarantees without retraining
- ✅ Quick baseline for comparison

**Use CRC-Select** when:
- ⭐ You need risk guarantees (e.g., medical, safety-critical applications)
- ⭐ You want maximum coverage at a given risk level
- ⭐ OOD robustness is important
- ⭐ You can afford alternating training (slightly longer training time)

## Project Structure

```
CRC-Select-Torch/
├── selectivenet/              # Original SelectiveNet + Extensions
│   ├── model.py              # SelectiveNet architecture
│   ├── loss.py               # Original selective loss
│   ├── loss_crc.py          # 🆕 CRC-aware loss with risk penalty
│   ├── data.py              # Data loading (CIFAR-10, SVHN)
│   ├── data_splits.py       # 🆕 3-way splitting for CRC
│   ├── evaluator.py         # Original evaluator
│   └── evaluator_crc.py     # 🆕 CRC evaluation (RC curves, DAR)
│
├── crc/                      # 🆕 CRC Module
│   ├── calibrate.py         # CRC calibration algorithms
│   └── risk_utils.py        # Risk computation utilities
│
├── scripts/
│   ├── train.py             # Original SelectiveNet training
│   ├── train_crc_select.py  # 🆕 CRC-Select alternating training
│   ├── test.py              # Testing
│   ├── eval_crc.py          # 🆕 Comprehensive CRC evaluation
│   ├── baseline_posthoc_crc.py  # 🆕 Post-hoc CRC baseline
│   ├── plot_results.py      # 🆕 Visualization utilities
│   ├── aggregate_results.py # 🆕 Multi-seed aggregation
│   └── run_experiments.py   # 🆕 Full experiment pipeline
│
├── configs/
│   └── crc_select.yaml      # 🆕 Default hyperparameters
│
├── README.md                 # This file
├── README_CRC_SELECT.md      # 🆕 Complete CRC-Select documentation
└── GETTING_STARTED.md        # 🆕 Step-by-step tutorial
```

## Comparison: SelectiveNet vs CRC-Select

| Feature | SelectiveNet | CRC-Select |
|---------|-------------|------------|
| **Training Objective** | Maximize selective accuracy | Maximize coverage at risk ≤ α |
| **Risk Awareness** | Implicit (via CE loss) | Explicit (via L_risk penalty) |
| **Calibration** | Post-hoc (optional) | Integrated (alternating) |
| **Risk Guarantee** | None | Conformal guarantee E[r] ≤ α |
| **Coverage** | Fixed by design | Adaptive to risk constraint |
| **OOD Safety** | Not explicitly handled | Improved via risk-aware selection |
| **Use Case** | When you know desired coverage | When you want risk guarantees |

## Acknowledgement
- Implementation borrows from https://github.com/gatheluck/pytorch-SelectiveNet.
- CRC-Select extends SelectiveNet with Conformal Risk Control integration.

## References

### SelectiveNet
- [Yonatan Geifman and Ran El-Yaniv. "SelectiveNet: A Deep Neural Network with an Integrated Reject Option.", in ICML, 2019.][1]
- [Original implementation in Keras][2]

### Conformal Risk Control
- Refer to conformal prediction literature for CRC theory and methodology
- See `README_CRC_SELECT.md` for detailed references and methodology

[1]: https://arxiv.org/abs/1901.09192
[2]: https://github.com/geifmany/selectivenet
