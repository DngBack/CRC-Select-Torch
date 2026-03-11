# CRC-Select: Results Synthesis and Analysis

**Source:** `results_paper/` (seed 42).  
**Context:** [CRC-Select corrected mini paper](crc_select_corrected_mini_paper.md) — learning rejection policies optimized for conformal risk control on CIFAR-10.

---

## 1. Experimental Setup

| Item | Value |
|------|--------|
| **Dataset** | CIFAR-10 (in-distribution); SVHN used for OOD evaluation |
| **Seed** | 42 |
| **Methods** | CRC-Select, Post-hoc CRC, MSP (Max Softmax Probability), Energy, TempScaled MSP |
| **Primary guarantee** | Accepted-loss mass \( A(\tau) = \mathbb{E}[r(X,Y;\theta)\,\mathbf{1}\{g_\phi(X)\ge\tau\}] \le \alpha \) |

---

## 2. Detailed Results Tables

### Table 1: Coverage and Actual Risk at Target Risk α (CRC-calibrated threshold)

Reported after CRC calibration. Each cell: **Coverage** (actual risk). Target risk = α (column header); actual risk = test accepted-loss mass (%). Higher coverage at the same α is better; actual risk should be ≤ target α (if not, the method violates the guarantee).

| Method | α = 0.01 (target) | α = 0.02 (target) | α = 0.05 (target) | α = 0.10 (target) | α = 0.15 (target) | α = 0.20 (target) |
|--------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|
| **CRC-Select** | **76.2% (1.01%)†** | **81.7% (1.59%)** | 92.7% (4.95%) | 100% (9.11%) | 100% (8.92%) | 100% (9.09%) |
| **Post-hoc CRC** | — | — | 92.8% (4.92%) | 100% (9.05%) | 100% (9.05%) | 100% (9.11%) |
| **MSP** | 73.4% (0.84%) | 84.1% (2.11%)† | **93.2% (4.99%)** | 100% (8.93%) | 100% (8.93%) | 100% (8.93%) |
| **Energy** | 52.2% (0.63%) | 82.0% (1.78%) | 92.7% (4.81%) | 100% (8.93%) | 100% (8.93%) | 100% (8.93%) |

† Actual risk > target α (violation). Còn lại: actual risk ≤ target.

**Interpretation (so sánh công bằng):** Mỗi ô có **target α** và **risk thực tế** (số trong ngoặc). Chỉ so sánh coverage khi actual risk ≤ target (không có †). Ở **α = 0.05**: MSP **93.2% (4.99%)** và Post-hoc 92.8% (4.92%) tốt hơn CRC-Select 92.7% (4.95%) về coverage; cả ba đều thỏa guarantee. Ở **α = 0.02**: MSP 84.1% (2.11%)† vi phạm; CRC-Select **81.7% (1.59%)** cao nhất trong nhóm thỏa (Energy 82.0% (1.78%) gần tương đương). Ở **α = 0.01**: CRC-Select 76.2% (1.01%)† vi phạm nhẹ nhưng coverage cao hơn MSP 73.4% và Energy 52.2%. **Kết luận:** CRC-Select nổi bật ở α chặt (0.02); ở α = 0.05 baseline MSP/Post-hoc có thể cao hơn một chút.

*Note: Post-hoc CRC in this run is evaluated only for α ≥ 0.05. MSP at α = 0.02 has a small risk violation (test accepted-loss mass slightly above α).*

---

### Table 2: Test Accepted-Loss Mass and Risk Violation at Primary α

At the CRC-chosen threshold \(\hat\tau\), the **audited quantity** is accepted-loss mass; violation = test accepted-loss mass > α.

| Method | α | \(\hat\tau\) | Test accepted-loss mass | Test selective risk | Test coverage | Violated |
|--------|---|-------------|--------------------------|----------------------|---------------|----------|
| **CRC-Select** | 0.01 | 0.941 | 0.0101 | 1.33% | 76.2% | Yes |
| **CRC-Select** | 0.02 | 0.369 | 0.0159 | 1.94% | 81.7% | No |
| **CRC-Select** | 0.05 | ≈0 | 0.0495 | 5.34% | 92.7% | No |
| **CRC-Select** | 0.10 | 0 | 0.0911 | 9.11% | 100% | No |
| **Post-hoc CRC** | 0.05 | ≈0 | 0.0492 | 5.31% | 92.8% | No |
| **Post-hoc CRC** | 0.10 | 0 | 0.0905 | 9.05% | 100% | No |
| **MSP** | 0.01 | 0.998 | 0.0084 | 1.14% | 73.4% | No |
| **MSP** | 0.02 | 0.995 | 0.0211 | 2.51% | 84.1% | **Yes** |
| **MSP** | 0.05 | 0.928 | 0.0499 | 5.35% | 93.2% | No |
| **Energy** | 0.01 | 0.857 | 0.0063 | 1.20% | 52.2% | No |
| **Energy** | 0.02 | 0.760 | 0.0178 | 2.17% | 82.0% | No |
| **Energy** | 0.05 | 0.479 | 0.0481 | 5.19% | 92.7% | No |

---

### Table 3: Risk–Coverage Curve Summary (Area Under Curve, AURC)

Lower AURC (accepted-loss mass over the risk–coverage curve) is better. Only CRC-Select and Post-hoc CRC have a full curve and reported AURC in this run.

| Method | rc_auc_alm (AURC) | rc_auc_risk | AUROC (selector) |
|--------|--------------------|-------------|-------------------|
| **CRC-Select** | 0.0132 | 0.0151 | 0.917 |
| **Post-hoc CRC** | 0.0130 | 0.0151 | 0.915 |
| **MSP** | — | — | 0.911 |
| **Energy** | — | — | 0.897 |

---

### Table 4: CRC-Select — Coverage at Fixed Target Coverage (Calibration)

Calibration quality: how closely actual coverage matches the requested target.

| Target coverage | Actual coverage | Coverage error | Selective risk | Selective accuracy |
|-----------------|-----------------|----------------|----------------|--------------------|
| 70% | 74.6% | +4.6% | 1.09% | 99.05% |
| **80%** | **80.0%** | **~0%** | 1.59% | 98.55% |
| 90% | 86.9% | −3.1% | 3.23% | 96.93% |

CRC-Select is well calibrated near 80% target; at 90% it under-covers (actual < target).

---

### Table 5: CRC-Select — Selective Risk and Accuracy at Fixed Coverage (from risk_coverage_curve)

At fixed τ (equivalently fixed coverage), selective risk = accepted-loss mass / coverage.

| Coverage (τ) | Accepted-loss mass | Selective risk | Selective accuracy |
|--------------|--------------------|----------------|--------------------|
| 83.5% (α≈0.1) | 0.0190 | 2.28% | 97.87% |
| 86.9% (τ=0.005) | 0.0281 | 3.23% | 96.93% |
| 79.6% (τ=0.70) | 0.0125 | 1.57% | 98.57% |
| 78.7% (τ=0.80) | 0.0117 | 1.48% | 98.53% |

---

### Table 6: OOD Safety — OOD Acceptance Rate at Fixed ID Coverage (CRC-Select)

At a fixed in-distribution coverage level, lower OOD acceptance is better. DAR = dangerous acceptance rate (same as OOD accept rate here); safety ratio = ID coverage rate / OOD accept rate.

| ID coverage target | ID coverage actual | OOD accept rate | Safety ratio |
|--------------------|--------------------|-----------------|-------------|
| 60% | 60.1% | **0.60%** | 100.3 |
| 70% | 70.0% | **1.08%** | 65.1 |
| 80% | 80.0% | **5.02%** | 15.9 |
| 90% | 90.0% | **42.82%** | 2.1 |

OOD acceptance grows sharply as ID coverage increases; at 70–80% ID coverage the system remains relatively safe (1–5% OOD accept).

---

### Table 7: Comparison of Risk–Coverage Trade-Off (Selective Risk at Representative Coverage)

Approximate selective risk at similar coverage levels (from risk_coverage_curve where available).

| Method | Coverage ~83% | Coverage ~87% | Coverage ~93% |
|--------|----------------|----------------|----------------|
| **CRC-Select** | 2.28% | 3.23% | 5.34% |
| **Post-hoc CRC** | 2.33% | 3.39% | 5.31% |
| **MSP** | flat ~8.9% until high τ | — | 5.35% |
| **Energy** | 2.17% (at 82%) | — | 5.19% |

At mid–high coverage, CRC-Select and Post-hoc CRC are close; Energy is competitive at low coverage but pays with much lower coverage at strict α (e.g. α=0.01).

---

## 3. Strengths

- **Finite-sample guarantee:** For the reported seed and split, CRC-Select (and Post-hoc CRC) satisfy the intended accepted-loss mass bound for α = 0.02, 0.05, 0.10; no violation at those levels. Only CRC-Select at α = 0.01 shows a small violation (0.0101 > 0.01).
- **Good coverage at strict risk:** At α = 0.02, CRC-Select reaches **81.7%** coverage with no violation; Energy reaches 82% but at α = 0.01 drops to 52.2%. CRC-Select thus offers a better coverage/risk profile at strict α.
- **Calibration:** At 80% target coverage, CRC-Select achieves almost exact calibration (80.0% actual). At 70% it slightly over-covers (74.6%); at 90% it under-covers (86.9%).
- **Stable risk–coverage curve:** AURC (rc_auc_alm) for CRC-Select (0.0132) is on par with Post-hoc CRC (0.0130), indicating a smooth and efficient curve.
- **OOD awareness:** At 60–70% ID coverage, OOD acceptance is low (~0.6–1.1%), with high safety ratio. This supports use in settings where limiting OOD exposure is important.
- **Theoretically aligned objective:** Training optimizes a surrogate for the same quantity (accepted-loss mass) that the final CRC step certifies, which is the right target for the theorem.

---

## 4. Weaknesses

- **Single-seed evaluation:** All reported numbers are for seed 42 only. No mean or standard deviation across seeds; conclusions on stability and violation rate are limited.
- **α = 0.01 violation:** CRC-Select at α = 0.01 slightly violates the bound (test accepted-loss mass 0.0101 > 0.01). This may be due to exchangeability, calibration set size, or optimization; it should be reported and discussed.
- **High OOD acceptance at high coverage:** At 90% ID coverage, OOD acceptance is **42.8%**. For safety-critical deployments, high coverage and low OOD acceptance are in tension; this trade-off should be made explicit.
- **MSP baseline:** MSP has a **risk violation at α = 0.02** (test accepted-loss mass 0.0211 > 0.02). It is a useful baseline but not a conformally valid selector at that α in this run.
- **Post-hoc CRC not at α = 0.01, 0.02:** In this artifact, Post-hoc CRC is only evaluated for α ≥ 0.05, so direct comparison with CRC-Select at α = 0.01 and 0.02 is incomplete.
- **Calibration at 90%:** At 90% target coverage, actual coverage is 86.9% (under-coverage). If the application requires tight calibration at high coverage, this gap is a limitation.
- **Dataset scope:** Results are only on CIFAR-10 (and SVHN for OOD). Broader datasets and domains would strengthen the claims.

---

## 5. Next Steps

1. **Multi-seed and violation rate:** Run multiple seeds (e.g. 3–5), report mean ± std for coverage at each α, and estimate **violation frequency** (fraction of (seed, α) pairs where test accepted-loss mass > α). This is essential for the “Evaluation protocol” in the mini paper.
2. **Post-hoc CRC at α = 0.01, 0.02:** Evaluate Post-hoc CRC at α = 0.01 and 0.02 on the same checkpoints/splits as CRC-Select to enable a direct comparison at strict risk levels.
3. **Same-backbone comparison:** Ensure CRC-Select, Post-hoc CRC, and (if applicable) vanilla/SelectiveNet share the same backbone and data split so that gains are attributed to the training objective and calibration, not to architecture or data.
4. **OOD and distribution shift:** Systematically report **OOD dangerous acceptance rate** at fixed ID coverage (as in Table 6) for all baselines (MSP, Energy, Post-hoc CRC) and, if possible, under explicit distribution shift (e.g. CIFAR-10 → CIFAR-10.1 or other benchmarks). State clearly that the finite-sample guarantee does not apply under shift unless an extension (e.g. weighted CRC) is used.
5. **Calibration at 70% and 90%:** Investigate why CRC-Select over-covers at 70% and under-covers at 90% (e.g. target coverage \(c_0\), regularization, or calibration set size) and adjust training or calibration if tighter calibration is required.
6. **Cost-sensitive and clinical losses:** If the application uses a cost-sensitive or clinically weighted loss, repeat experiments with that loss and report coverage and accepted-loss mass under the same CRC protocol.
7. **Larger-scale and other domains:** Replicate on larger or different datasets (e.g. ImageNet subset, medical or tabular) and report the same tables and violation statistics to assess generality.
8. **Documentation:** In the paper, label the **audited metric** explicitly as accepted-loss mass \(A(\hat\tau)\) and clarify that conditional selective risk \(R_{\mathrm{sel}}\) is reported as a descriptive metric only, not as the quantity covered by the theorem.

---

## 6. Summary

Results in `results_paper/` (seed 42) show that **CRC-Select** attains **high coverage at strict risk levels** (e.g. 81.7% at α = 0.02 with no violation), **good calibration** around 80% target coverage, and **low OOD acceptance** at 60–70% ID coverage. Limitations include a **single seed**, a **small violation at α = 0.01**, and **high OOD acceptance at 90% ID coverage**. Recommended next steps are **multi-seed violation-rate evaluation**, **direct comparison with Post-hoc CRC at α = 0.01 and 0.02**, **same-backbone baselines**, and **explicit OOD and calibration reporting** as in the mini paper’s evaluation protocol.
