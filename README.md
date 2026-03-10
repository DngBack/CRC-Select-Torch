# CRC-Select: Learning Rejection Policies Optimized for Conformal Risk Control

A PyTorch implementation of **CRC-Select**, which integrates Conformal Risk Control (CRC) into SelectiveNet training to achieve higher coverage at the same audited risk level. Built on top of [SelectiveNet](https://arxiv.org/abs/1901.09192) (Geifman & El-Yaniv, ICML 2019).

---

## Key Idea

Standard CRC treats the score function as fixed and only calibrates a threshold post-hoc. If the selector assigns high scores to error-prone examples, the CRC threshold must be conservative, reducing coverage. **CRC-Select** trains the selector so that the family of CRC-calibrated rules achieves higher coverage at the same audited accepted-loss mass target.

**What CRC certifies** — the *accepted-loss mass*:

$$A(\tau) = \mathbb{E}\bigl[r(X,Y)\,\mathbf{1}\{g(X)\ge\tau\}\bigr] \le \alpha$$

This is **not** the conditional selective risk $R_{\text{sel}} = A/C$. The training loss, calibration, and evaluation are all aligned with $A(\tau)$.

---

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Train CRC-Select on CIFAR-10
python scripts/train_crc_select.py \
    --dataset cifar10 --seed 42 --num_epochs 200 \
    --alpha_risk 0.1 --warmup_epochs 20 --backbone vgg16 --unobserve

# Evaluate
python scripts/evaluate_for_paper.py \
    --checkpoint_dir checkpoints/seed_42 \
    --dataset cifar10 --backbone vgg16

# Run all baselines
python scripts/baseline_msp.py        --dataset cifar10 --seed 42
python scripts/baseline_temp_scaled.py --dataset cifar10 --seed 42
python scripts/baseline_posthoc_crc.py --checkpoint_dir checkpoints/seed_42 --dataset cifar10
python scripts/baseline_deep_gambler.py --dataset cifar10 --seed 42
python scripts/baseline_energy.py       --dataset cifar10 --seed 42
```

---

## Features

| Feature | Description |
|---------|-------------|
| **Accepted-loss mass guarantee** | Finite-sample $A(\hat\tau)\le\alpha$ via CRC calibration |
| **CRC-aware training** | Loss penalty $\max(0, \hat A_{\text{mb}} - \alpha)$ aligned with the certified quantity |
| **3-way data splits** | 70 / 10 / 20 (train / cal / test) for proper conformal calibration |
| **5 random seeds** | `[42, 123, 456, 789, 999]` for reliable statistics |
| **3 datasets** | CIFAR-10, CIFAR-100, Tiny-ImageNet |
| **3 backbones** | VGG-16, ResNet-18, WRN-28-10 |
| **6 baselines** | MSP, Temp-scaled, SelectiveNet, Deep Gambler, Energy, Post-hoc CRC |
| **7 ablations** | λ-risk, cal fraction, warmup, recal frequency, loss variant, backbone, α grid |
| **Comprehensive metrics** | Accepted-loss mass, coverage, violation gap, AUROC, AUPR, RC-AUC, DAR |
| **Paper-ready output** | Figures (PNG/PDF), LaTeX tables, aggregated CSVs |

---

## Mathematical Formulation

### Risk and Selector

- Predictor: $f_\theta(x) \to \mathbb{R}^K$ with $p_\theta(y|x) = \mathrm{softmax}(f_\theta(x))_y$
- Selector: $g_\phi(x) \to [0,1]$ (acceptance score)
- Risk: $r(x,y) = 1 - p_\theta(y|x) \in [0,1]$

### CRC Calibration

On calibration set $\{(X_i,Y_i)\}_{i=1}^n$:

$$\hat\tau = \inf\Bigl\{\tau : \frac{n}{n+1}\hat A_n(\tau) + \frac{1}{n+1} \le \alpha\Bigr\}$$

where $\hat A_n(\tau) = \frac{1}{n}\sum_{i=1}^n r_i \cdot \mathbf{1}\{g_i \ge \tau\}$.

**Theorem** (finite-sample): Under exchangeability, the deployed policy satisfies $\mathbb{E}[r(X_{n+1},Y_{n+1})\cdot\mathbf{1}\{g(X_{n+1})\ge\hat\tau\}] \le \alpha$.

### Training Objective

$$\mathcal{L} = \underbrace{\frac{\sum g_i \cdot \mathrm{CE}_i}{\sum g_i + \varepsilon}}_{\mathcal{L}_{\text{pred}}} + \beta \underbrace{(\hat C - c_0)^2}_{\mathcal{L}_{\text{cov}}} + \mu \underbrace{\max(0, \hat A_{\text{mb}} - \alpha)}_{\mathcal{L}_{\text{risk}}}$$

where $\hat A_{\text{mb}} = \frac{1}{m}\sum_{i=1}^m g_i \cdot r_i$ is the minibatch accepted-loss mass — matching the quantity CRC certifies.

### Training Algorithm

1. **Warm-up**: Train vanilla SelectiveNet for `warmup_epochs` (default 20)
2. **Calibrate** (no grad): Compute $\hat\tau$ on $\mathcal{D}_{\text{cal}}$ using CRC formula
3. **Train** (with grad): Update $(\theta, \phi)$ with $\mathcal{L}$ above for `recalibrate_every` epochs
4. Repeat steps 2–3 until convergence
5. **Final calibration**: Freeze model, calibrate once more on $\mathcal{D}_{\text{cal}}$ → deployed threshold

---

## Full Experiment Pipeline

### 1. Multi-Seed Training

```bash
for seed in 42 123 456 789 999; do
    python scripts/train_crc_select.py \
        --dataset cifar10 --seed $seed --num_epochs 200 \
        --alpha_risk 0.1 --backbone vgg16 --unobserve
done
```

### 2. Comprehensive Evaluation

```bash
# Evaluate CRC-Select across seeds
for seed in 42 123 456 789 999; do
    python scripts/evaluate_for_paper.py \
        --checkpoint_dir results/CRC-Select/seed_$seed \
        --dataset cifar10 --backbone vgg16
done

# Run baselines
for seed in 42 123 456 789 999; do
    python scripts/baseline_msp.py         --dataset cifar10 --seed $seed
    python scripts/baseline_temp_scaled.py  --dataset cifar10 --seed $seed
    python scripts/baseline_deep_gambler.py --dataset cifar10 --seed $seed
    python scripts/baseline_energy.py       --dataset cifar10 --seed $seed
    python scripts/baseline_posthoc_crc.py  --checkpoint_dir results/CRC-Select/seed_$seed \
        --dataset cifar10
done
```

### 3. Aggregate & Analyse

```bash
# Violation rate across seeds
python scripts/compute_violation_rate.py \
    --method_dirs results/CRC-Select results/posthoc_crc results/msp \
    --seeds 42 123 456 789 999 \
    --alphas 0.05 0.1 0.15 0.2 \
    --generate_latex

# Aggregate mean ± std
python scripts/aggregate_results.py \
    --method_dirs results/CRC-Select results/posthoc_crc results/msp \
        results/temp_scaled results/deep_gambler results/energy \
    --seeds 42 123 456 789 999
```

### 4. Figures & Tables

```bash
python scripts/generate_paper_figures.py \
    --results_dir results \
    --methods CRC-Select posthoc_crc msp temp_scaled deep_gambler energy \
    --seeds 42 123 456 789 999 \
    --output_dir figures
```

Generates: RC frontier (Fig 2), coverage-at-α bars (Fig 3), violation gap boxplot (Fig 4), OOD DAR (Fig 5), threshold efficiency (Fig 7), main results table (Table 1), violation rates table (Table 4).

### 5. Ablation Studies

```bash
# Run all 7 ablations
python scripts/run_ablations.py --ablation all --dataset cifar10 --seeds 42 123 456 789 999

# Or run one at a time
python scripts/run_ablations.py --ablation A1 --dataset cifar10  # λ-risk sensitivity
python scripts/run_ablations.py --ablation A6 --dataset cifar10  # backbone comparison
```

| Ablation | Parameter | Values |
|----------|-----------|--------|
| A1 | `mu_init` (λ-risk) | 0.1, 0.5, 1.0, 2.0, 5.0 |
| A2 | `cal_frac` | 0.05, 0.10, 0.15, 0.20, 0.30 |
| A3 | `warmup_epochs` | 0, 10, 20, 40, 80 |
| A4 | `recalibrate_every` | 1, 5, 10, 20, 50 |
| A5 | `loss_variant` | hinge, squared |
| A6 | `backbone` | vgg16, resnet18, wrn28_10 |
| A7 | `alpha_risk` | 0.01, 0.05, 0.10, 0.15, 0.20, 0.30 |

---

## Metrics

### Primary (CRC-certified)

| Metric | Formula | Description |
|--------|---------|-------------|
| **Accepted-loss mass** $A(\tau)$ | $\frac{1}{n}\sum r_i \cdot \mathbf{1}\{g_i \ge \tau\}$ | The quantity CRC certifies ≤ α |
| **Violation gap** | $\max(0, \hat A - \alpha)$ | How much the guarantee is exceeded |
| **Violation rate** | fraction of seeds with $\hat A > \alpha$ | Should be ≤ 1/(n+1) ≈ 5–20% |

### Descriptive

| Metric | Description |
|--------|-------------|
| **Coverage** $C(\tau)$ | Fraction of examples accepted |
| **Selective risk** $R_{\text{sel}}$ | $A/C$ — conditional risk on accepted |
| **Selective accuracy** | $1 - R_{\text{sel}}$ |
| **AUROC / AUPR** | Error detection quality of $g(x)$ |
| **RC-AUC** | Area under risk-coverage curve |
| **Threshold conservativeness** | $\alpha - \hat A$ (how much slack) |

### OOD Safety

| Metric | Description |
|--------|-------------|
| **DAR** (Dangerous Acceptance Rate) | OOD acceptance at CRC threshold |
| **OOD-Accept @ fixed ID-Coverage** | OOD acceptance at matching ID operating point |

---

## Evaluation Output

For each method × seed × α, `evaluate_for_paper.py` saves:

```
results/<method>/seed_<s>/
├── all_metrics.csv              # All metrics per alpha
├── risk_coverage_curve.csv      # RC curve (accepted_loss_mass, selective_risk, coverage vs tau)
├── coverage_at_risk.csv         # Coverage at each alpha after CRC calibration
├── calibration_metrics.csv      # CRC calibration details
└── summary.csv                  # Quick overview
```

Aggregated outputs from `aggregate_results.py`:

```
results/aggregated/
├── <method>/
│   ├── coverage_at_risk_aggregated.csv
│   ├── risk_coverage_curve_aggregated.csv
│   └── ood_results_aggregated.csv
└── summary_table.csv            # Cross-method comparison
```

---

## Project Structure

```
CRC-Select-Torch/
├── crc/                           # CRC core module
│   ├── calibrate.py              #   CRC calibration (crc_calibrate_threshold)
│   ├── risk_utils.py             #   Risk computation (accepted_loss_mass, AUROC/AUPR)
│   └── __init__.py
│
├── selectivenet/                  # SelectiveNet + extensions
│   ├── model.py                  #   SelectiveNet architecture (f, g, h)
│   ├── loss.py                   #   Vanilla selective loss
│   ├── loss_crc.py               #   CRC-aware loss (A_hat_mb penalty)
│   ├── data.py                   #   CIFAR-10/100, TinyImageNet, SVHN, corruptions
│   ├── data_splits.py            #   70/10/20 splits
│   ├── vgg_variant.py            #   VGG-16 backbone
│   ├── resnet_variant.py         #   ResNet-18, WRN-28-10 backbones
│   ├── evaluator.py              #   Original evaluator
│   └── evaluator_crc.py          #   Full CRC evaluation suite
│
├── scripts/
│   ├── train_crc_select.py       #   CRC-Select training (alternating optimisation)
│   ├── train.py                  #   Vanilla SelectiveNet training
│   ├── evaluate_for_paper.py     #   Comprehensive paper evaluation
│   ├── baseline_msp.py           #   MSP baseline
│   ├── baseline_temp_scaled.py   #   Temperature-scaled baseline
│   ├── baseline_posthoc_crc.py   #   Post-hoc CRC baseline
│   ├── baseline_deep_gambler.py  #   Deep Gambler baseline
│   ├── baseline_energy.py        #   Energy-based baseline
│   ├── compute_violation_rate.py #   Violation analysis across seeds
│   ├── aggregate_results.py      #   Multi-seed aggregation
│   ├── generate_paper_figures.py #   Figures & LaTeX tables
│   └── run_ablations.py          #   Ablation studies A1–A7
│
├── configs/
│   └── crc_select.yaml           #   All hyperparameters, seeds, datasets, ablations
│
├── docs/
│   ├── crc_select_corrected_mini_paper.md
│   └── detailed/                 #   Metric guides, workflows
│
├── requirements.txt
├── QUICK_EVAL_GUIDE.md           #   Fast evaluation recipe
└── README.md                     #   This file
```

---

## Configuration

All defaults live in `configs/crc_select.yaml`. Key sections:

| Section | Highlights |
|---------|-----------|
| `model` | `backbone: vgg16`, `dim_features: 512` |
| `data` | `split_ratios: [0.70, 0.10, 0.20]` |
| `crc_select` | `alpha_risk: 0.1`, `warmup_epochs: 20`, `loss_variant: hinge` |
| `evaluation` | `seeds: [42,123,456,789,999]`, `alphas: [0.01..0.30]` |
| `baselines` | msp, temp_scaled, selectivenet, deep_gambler, energy, posthoc_crc |
| `ablations` | A1–A7 with parameter grids |
| `datasets` | cifar10, cifar100, tinyimagenet with OOD pairs |

---

## Requirements

- Python 3.11+
- PyTorch 2.4.0, torchvision 0.19.0 (CUDA 12.1)
- `pip install -r requirements.txt`

Key dependencies: numpy, pandas, scipy, matplotlib, seaborn, scikit-learn, PyYAML, wandb, tqdm.

---

## Comparison: SelectiveNet vs Post-hoc CRC vs CRC-Select

| Feature | SelectiveNet | Post-hoc CRC | **CRC-Select** |
|---------|-------------|--------------|----------------|
| Training objective | Selective accuracy | Same | CRC-aligned (A_hat_mb) |
| Risk guarantee | None | $A(\hat\tau)\le\alpha$ | $A(\hat\tau)\le\alpha$ |
| Selector optimised for CRC | No | No | **Yes** |
| Expected coverage at same α | Lowest | Medium | **Highest** |
| OOD rejection | Implicit | Implicit | Risk-aware |

---

## References

- Y. Geifman and R. El-Yaniv, "SelectiveNet: A Deep Neural Network with an Integrated Reject Option", ICML 2019. [[paper]](https://arxiv.org/abs/1901.09192)
- A. N. Angelopoulos, S. Bates, et al., "Conformal Risk Control", ICLR 2024.
- L. Ziyin et al., "Deep Gamblers: Learning to Abstain with Portfolio Theory", NeurIPS 2019.
- W. Liu et al., "Energy-based Out-of-Distribution Detection", NeurIPS 2020.

## Acknowledgement

Implementation builds on [pytorch-SelectiveNet](https://github.com/gatheluck/pytorch-SelectiveNet).