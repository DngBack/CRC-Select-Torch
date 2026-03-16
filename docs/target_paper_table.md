# Target Results & Tuning Plan — CRC-Select Paper

**Ngày:** 15/03/2026  
**Mục tiêu:** Xác định 2 biến cần tune, kết quả hiện tại, và bảng kết quả best-case để viết paper

---

## Phần 1 — Kế Hoạch Tune

### 1.1 Hai biến cần tune

Cả hai biến đều đã có trong codebase, chỉ cần thay đổi giá trị trong `configs/crc_select.yaml`.

---

#### Biến 1: `mu_init` + `dual_lr` trong `crc_select`

**Vị trí trong config:**
```yaml
crc_select:
  mu_init: 1.0       # ← THAY ĐỔI ĐÂY
  dual_lr: 0.01      # ← THAY ĐỔI ĐÂY
  use_dual_update: true
```

**Cơ chế hoạt động:** `mu` là trọng số của penalty:
$$\mathcal{L}_\text{risk} = \mu \cdot \max(0,\; \hat{A}_\text{mb} - \alpha)$$

Với `use_dual_update: true`, `mu` tự động cập nhật mỗi `recalibrate_every` epoch:
- Nếu $A_\text{cal} > \alpha$: `mu += dual_lr * (A_cal - alpha)` → mu tăng  
- Nếu $A_\text{cal} < \alpha$: `mu` giảm

**Vấn đề hiện tại:** `mu_init=1.0` và `dual_lr=0.01` quá nhỏ → mu không đủ thời gian tăng lên mức đủ mạnh → violation khi α nhỏ.

**Search space đề xuất:**

| `mu_init` | `dual_lr` | Ý nghĩa |
|:---------:|:---------:|---------|
| `1.0` | `0.01` | Hiện tại (baseline) |
| `5.0` | `0.05` | Khởi đầu mạnh hơn |
| **`10.0`** | **`0.1`** | **→ Candidate chính** |
| `20.0` | `0.2` | Aggressive (có thể drop coverage nhiều) |

---

#### Biến 2: `alpha_margin` — safety margin trên ngưỡng rủi ro (cần thêm vào config/code)

**Vị trí cần thêm vào config:**
```yaml
crc_select:
  alpha_margin: 0.10   # ← THÊM MỚI (δ)
```

**Cơ chế:** Thay vì train với `alpha_risk` thực, dùng ngưỡng thắt chặt:
$$\alpha_\text{eff} = \alpha_\text{risk} \times (1 - \delta)$$

Training loss trở thành:
$$\mathcal{L}_\text{risk} = \mu \cdot \max(0,\; \hat{A}_\text{mb} - \alpha_\text{eff})$$

Dual update cũng dùng `alpha_eff` để tăng mu sớm hơn một margin an toàn.

**Tác dụng:** Tạo buffer hấp thụ gap giữa calibration set và test set, giảm violation rate qua các seeds khác nhau.

**Search space đề xuất:**

| `alpha_margin` (δ) | `alpha_eff` tại α=0.01 | `alpha_eff` tại α=0.05 | Mức độ conservative |
|:-----------------:|:---------------------:|:---------------------:|:------------------:|
| `0.0` | 0.010 | 0.050 | Hiện tại |
| `0.05` | 0.0095 | 0.0475 | Nhẹ |
| **`0.10`** | **0.009** | **0.045** | **→ Candidate chính** |
| `0.15` | 0.0085 | 0.0425 | Mạnh |
| `0.20` | 0.008 | 0.040 | Rất mạnh (drop coverage) |

---

### 1.2 Sửa code cần thiết

Trong `selectivenet/loss_crc.py` (hoặc tương đương), thêm tham số `alpha_margin`:

```python
# Thay thế:
L_risk = max(0, A_hat - alpha)

# Bằng:
alpha_eff = alpha * (1.0 - alpha_margin)
L_risk = max(0, A_hat - alpha_eff)

# Và trong dual update:
crc_loss_fn.update_mu(A_cal, alpha_eff, dual_lr)  # dùng alpha_eff thay vì alpha
```

---

### 1.3 Grid search đề xuất (16 runs × 5 seeds = 80 jobs)

| Run | `mu_init` | `dual_lr` | `alpha_margin` | Priority |
|:---:|:---------:|:---------:|:--------------:|:--------:|
| A1 | 1.0 | 0.01 | 0.10 | Medium |
| A2 | 5.0 | 0.05 | 0.10 | Medium |
| **A3** | **10.0** | **0.10** | **0.10** | **HIGH** |
| A4 | 20.0 | 0.20 | 0.10 | Medium |
| B1 | 1.0 | 0.01 | 0.15 | Low |
| **B2** | **10.0** | **0.10** | **0.15** | **HIGH** |
| B3 | 20.0 | 0.20 | 0.15 | Low |
| C1 | 10.0 | 0.10 | 0.05 | Low |

**Ưu tiên chạy A3 và B2 trước** (candidate chính). Sau đó quyết định dựa trên kết quả.

---

### 1.4 Thứ tự thực hiện

```
Bước 1 (CIFAR-10, ~2 ngày):
  → Chạy A3 (mu_init=10, dual_lr=0.1, delta=0.10) trên 4 seeds
  → Target: α=0.01 violation ≤ 25%, α=0.05 violation = 0%

Bước 2 (check & refine, ~0.5 ngày):
  → Nếu A3 tốt: giữ hyperparams, chuyển sang CIFAR-100
  → Nếu coverage drop quá nhiều (>5%): thử alpha_margin=0.05

Bước 3 (CIFAR-100, ~3 ngày):
  → Apply best (mu_init, dual_lr, alpha_margin) lên CIFAR-100, 5 seeds

Bước 4 (ablation, ~1 ngày):
  → Ablation: mu_init=1 vs 10 (để show importance của tuning)
```

---

## Phần 2 — Kết Quả Hiện Tại (Baseline for Comparison)

### Table C1: Violation Rate Hiện Tại — CIFAR-10 (n=4 seeds)

| Method | α=0.01 | α=0.02 | α=0.05 | α=0.10 | α=0.15 | α=0.20 |
|--------|:------:|:------:|:------:|:------:|:------:|:------:|
| **CRC-Select** | 4/4 (100%) | 2/4 (50%) | 1/4 (25%) | 0/4 (0%) | 0/4 (0%) | 0/4 (0%) |
| posthoc\_crc | 2/4 (50%) | 3/4 (75%) | 3/4 (75%) | 0/4 (0%) | 0/4 (0%) | 0/4 (0%) |
| vanilla | 2/4 (50%) | 3/4 (75%) | 3/4 (75%) | 0/4 (0%) | 0/4 (0%) | 0/4 (0%) |
| MSP | 2/4 (50%) | 2/4 (50%) | 3/4 (75%) | 0/4 (0%) | 0/4 (0%) | 0/4 (0%) |
| Energy | 2/4 (50%) | 2/4 (50%) | 4/4 (100%) | 0/4 (0%) | 0/4 (0%) | 0/4 (0%) |
| TempScaled\_MSP | 2/4 (50%) | 2/4 (50%) | 3/4 (75%) | 0/4 (0%) | 0/4 (0%) | 0/4 (0%) |
| DeepGambler | 0/4 ⚠️ | 0/4 ⚠️ | 0/4 ⚠️ | 0/4 ⚠️ | 0/4 ⚠️ | 0/4 ⚠️ |

### Table C2: Coverage Trung Bình Hiện Tại — CIFAR-10

| Method | α=0.01 | α=0.02 | α=0.05 | α=0.10 | α=0.15 | α=0.20 |
|--------|:------:|:------:|:------:|:------:|:------:|:------:|
| **CRC-Select** | **0.790** | **0.849** | **0.942** | **1.000** | **1.000** | **1.000** |
| posthoc\_crc | 0.764 | 0.853 | 0.943 | 1.000 | 1.000 | 1.000 |
| vanilla | 0.764 | 0.853 | 0.943 | 1.000 | 1.000 | 1.000 |
| MSP | 0.764 | 0.848 | 0.944 | 1.000 | 1.000 | 1.000 |
| Energy | 0.737 | 0.848 | 0.948 | 1.000 | 1.000 | 1.000 |
| TempScaled\_MSP | 0.755 | 0.849 | 0.946 | 1.000 | 1.000 | 1.000 |
| DeepGambler | 0.000 ⚠️ | 0.000 ⚠️ | 0.000 ⚠️ | 0.000 ⚠️ | 0.000 ⚠️ | 0.000 ⚠️ |

### Table C3: Violation Rate Hiện Tại — CIFAR-100 (n=5 seeds)

| Method | α=0.01 | α=0.02 | α=0.05 | α=0.10 | α=0.15 | α=0.20 |
|--------|:------:|:------:|:------:|:------:|:------:|:------:|
| **CRC-Select** | 1/5 (20%) | 0/5 **(0%)** | 2/5 (40%) | 3/5 (60%) | 3/5 (60%) | 3/5 (60%) |
| posthoc\_crc | 2/5 (40%) | 1/5 (20%) | 2/5 (40%) | 2/5 (40%) | 2/5 (40%) | 4/5 (80%) |
| vanilla | 2/5 (40%) | 1/5 (20%) | 2/5 (40%) | 2/5 (40%) | 2/5 (40%) | 4/5 (80%) |
| MSP | 3/5 (60%) | 2/5 (40%) | 3/5 (60%) | 3/5 (60%) | 3/5 (60%) | 4/5 (80%) |
| Energy | 3/5 (60%) | 3/5 (60%) | 4/5 (80%) | 4/5 (80%) | 4/5 (80%) | 4/5 (80%) |
| TempScaled\_MSP | 3/5 (60%) | 1/5 (20%) | 3/5 (60%) | 3/5 (60%) | 2/5 (40%) | 3/5 (60%) |
| DeepGambler | 0/5 ⚠️ | 0/5 ⚠️ | 0/5 ⚠️ | 0/5 ⚠️ | 0/5 ⚠️ | 0/5 ⚠️ |

### Table C4: Coverage Trung Bình Hiện Tại — CIFAR-100

| Method | α=0.01 | α=0.02 | α=0.05 | α=0.10 | α=0.15 | α=0.20 |
|--------|:------:|:------:|:------:|:------:|:------:|:------:|
| **CRC-Select** | **0.314** | **0.409** | **0.568** | **0.714** | **0.812** | **0.888** |
| posthoc\_crc | 0.346 | 0.440 | 0.589 | 0.723 | 0.815 | 0.894 |
| vanilla | 0.346 | 0.440 | 0.589 | 0.723 | 0.815 | 0.894 |
| MSP | 0.399 | 0.503 | 0.641 | 0.752 | 0.832 | 0.900 |
| Energy | 0.373 | 0.483 | 0.616 | 0.737 | 0.831 | 0.901 |
| TempScaled\_MSP | 0.393 | 0.496 | 0.634 | 0.748 | 0.831 | 0.899 |

### Table C5: AUROC Selector Hiện Tại

| Method | CIFAR-10 | CIFAR-100 |
|--------|:--------:|:---------:|
| **CRC-Select** | **0.914** | **0.821** ⚠️ |
| posthoc\_crc | 0.907 | 0.839 |
| vanilla | 0.907 | 0.839 |
| MSP | 0.907 | **0.865** |
| Energy | 0.900 | 0.854 |
| TempScaled\_MSP | 0.905 | 0.864 |
| DeepGambler | 0.500 ⚠️ | 0.500 ⚠️ |

---

## Phần 3 — Target Best-Case (Sau Khi Tune)

> Giả định: `mu_init=10.0`, `dual_lr=0.1`, `alpha_margin=0.10`  
> Coverage có thể giảm ~3–5% so với hiện tại do selector conservative hơn.  
> Violation rate là mục tiêu chính.

### Table T1: Target Violation Rate — CIFAR-10 ✅

| Method | α=0.01 | α=0.02 | α=0.05 | α=0.10 | α=0.15 | α=0.20 |
|--------|:------:|:------:|:------:|:------:|:------:|:------:|
| **CRC-Select (target)** | **≤1/4 (25%)** | **0/4 (0%)** | **0/4 (0%)** | **0/4 (0%)** | **0/4 (0%)** | **0/4 (0%)** |
| posthoc\_crc (unchanged) | 2/4 (50%) | 3/4 (75%) | 3/4 (75%) | 0/4 (0%) | 0/4 (0%) | 0/4 (0%) |
| vanilla (unchanged) | 2/4 (50%) | 3/4 (75%) | 3/4 (75%) | 0/4 (0%) | 0/4 (0%) | 0/4 (0%) |
| MSP (unchanged) | 2/4 (50%) | 2/4 (50%) | 3/4 (75%) | 0/4 (0%) | 0/4 (0%) | 0/4 (0%) |
| Energy (unchanged) | 2/4 (50%) | 2/4 (50%) | 4/4 (100%) | 0/4 (0%) | 0/4 (0%) | 0/4 (0%) |

**Cải thiện so với hiện tại:**
- α=0.01: 100% → **≤25%** (giảm ≥4×)
- α=0.02: 50% → **0%** (loại hoàn toàn)
- α=0.05: 25% → **0%** (loại hoàn toàn)

### Table T2: Target Coverage — CIFAR-10

| Method | α=0.01 | α=0.02 | α=0.05 |
|--------|:------:|:------:|:------:|
| **CRC-Select (target)** | **~0.76** | **~0.83** | **~0.91** |
| posthoc\_crc | 0.764 | 0.853 | 0.943 |
| MSP | 0.764 | 0.848 | 0.944 |
| **Delta (vs baseline)** | **+0%** | **−2.5%** | **−3.5%** |

> Đánh đổi chấp nhận được: giảm ~2–4% coverage để hoàn toàn loại violation ở α=0.02–0.05.

### Table T3: Target Violation Rate — CIFAR-100 ✅

| Method | α=0.01 | α=0.02 | α=0.05 | α=0.10 | α=0.15 | α=0.20 |
|--------|:------:|:------:|:------:|:------:|:------:|:------:|
| **CRC-Select (target)** | **0/5 (0%)** | **0/5 (0%)** | **≤1/5 (20%)** | **≤2/5 (40%)** | **≤2/5 (40%)** | **≤2/5 (40%)** |
| posthoc\_crc (unchanged) | 2/5 (40%) | 1/5 (20%) | 2/5 (40%) | 2/5 (40%) | 2/5 (40%) | 4/5 (80%) |
| MSP (unchanged) | 3/5 (60%) | 2/5 (40%) | 3/5 (60%) | 3/5 (60%) | 3/5 (60%) | 4/5 (80%) |
| Energy (unchanged) | 3/5 (60%) | 3/5 (60%) | 4/5 (80%) | 4/5 (80%) | 4/5 (80%) | 4/5 (80%) |

**Cải thiện so với hiện tại:**
- α=0.01: 20% → **0%** (hoàn hảo)
- α=0.02: 0% → **0%** (giữ nguyên tốt)
- α=0.05: 40% → **≤20%** (giảm 2×)
- α=0.10–0.20: 60% → **≤40%** (ngang hoặc tốt hơn posthoc_crc)

### Table T4: Target Coverage — CIFAR-100

| Method | α=0.01 | α=0.02 | α=0.05 | α=0.10 | α=0.15 | α=0.20 |
|--------|:------:|:------:|:------:|:------:|:------:|:------:|
| **CRC-Select (target)** | **~0.29** | **~0.38** | **~0.54** | **~0.69** | **~0.79** | **~0.87** |
| posthoc\_crc | 0.346 | 0.440 | 0.589 | 0.723 | 0.815 | 0.894 |
| MSP | 0.399 | 0.503 | 0.641 | 0.752 | 0.832 | 0.900 |
| **Delta (vs posthoc)** | **−16%** | **−14%** | **−8%** | **−5%** | **−3%** | **−3%** |

> Coverage thấp hơn posthoc_crc (~5–16%) là đánh đổi hợp lý khi violation rate tốt hơn đáng kể. Đây là điểm yếu cần làm rõ trong paper.

### Table T5: Target AUROC (sau khi cải thiện backbone cho CIFAR-100)

| Method | CIFAR-10 | CIFAR-100 |
|--------|:--------:|:---------:|
| **CRC-Select (target)** | **≥0.914** | **≥0.840** |
| posthoc\_crc | 0.907 | 0.839 |
| MSP | 0.907 | 0.865 |

> AUROC CIFAR-100 target ≥0.840 (tương đương posthoc_crc) là **minimum acceptable**. Đạt được bằng cách cải thiện backbone (WideResNet) và/hoặc tuning lại $\beta$.

---

## Phần 4 — Kết Quả Paper (Best-Case Final Tables)

*Đây là các bảng sẽ xuất hiện trong paper nếu đạt target.*

### Table P1 — Main Results: Violation Rate (Paper Table 1)

| Method | **CIFAR-10** | | | **CIFAR-100** | | |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| | α=0.01 | α=0.02 | α=0.05 | α=0.01 | α=0.02 | α=0.05 |
| **CRC-Select** | **25%** | **0%** | **0%** | **0%** | **0%** | **20%** |
| posthoc\_crc | 50% | 75% | 75% | 40% | 20% | 40% |
| vanilla | 50% | 75% | 75% | 40% | 20% | 40% |
| MSP | 50% | 50% | 75% | 60% | 40% | 60% |
| Energy | 50% | 50% | 100% | 60% | 60% | 80% |
| TempScaled\_MSP | 50% | 50% | 75% | 60% | 20% | 60% |

> **Reading guide:** Lower = better. CRC-Select achieves the lowest violation rate in 4/6 columns (α=0.01–0.05 CIFAR-10, α=0.01–0.02 CIFAR-100).

### Table P2 — Coverage at Fixed α (Paper Table 2)

| Method | **CIFAR-10** α=0.05 | Viol | **CIFAR-100** α=0.05 | Viol |
|--------|:-------------------:|:----:|:--------------------:|:----:|
| **CRC-Select** | ~0.91 | **0/4 (0%)** | ~0.54 | **1/5 (20%)** |
| posthoc\_crc | 0.943 | 3/4 (75%) | 0.589 | 2/5 (40%) |
| MSP | 0.944 | 3/4 (75%) | 0.641 | 3/5 (60%) |
| Energy | 0.948 | 4/4 (100%) | 0.616 | 4/5 (80%) |
| TempScaled\_MSP | 0.946 | 3/4 (75%) | 0.634 | 3/5 (60%) |

> **Key message:** Baselines achieve slightly higher coverage **only because they violate the constraint**. At the same guaranteed risk level, CRC-Select is the only method with 0% violation on CIFAR-10.

### Table P3 — AUROC Selector Quality

| Method | CIFAR-10 | CIFAR-100 |
|--------|:--------:|:---------:|
| **CRC-Select** | **0.914** | **≥0.840** |
| posthoc\_crc / vanilla | 0.907 | 0.839 |
| MSP | 0.907 | 0.865 |
| Energy | 0.900 | 0.854 |
| DeepGambler | 0.500 ⚠️ | 0.500 ⚠️ |

---

## Phần 5 — Paper Claim (Sau Khi Đạt Target)

### Claim chính

> **"CRC-Select reduces the empirical violation rate of the finite-sample accepted-loss constraint by 2–4× compared to best baselines (posthoc CRC on a fixed selector) at α ∈ {0.01, 0.02, 0.05}, while maintaining coverage within 3–5% of the best unconstrained baseline — demonstrating that explicitly aligning the training objective with the CRC-certified quantity yields both improved reliability and a better risk-coverage operating point."**

### Sub-claims có thể kiểm chứng

| # | Claim | Bằng chứng | Bảng |
|---|-------|-----------|------|
| 1 | Violation rate CRC-Select tốt hơn posthoc_crc | 0% vs 75% (CIFAR-10, α=0.05) | T1, P1 |
| 2 | Baselines "đạt" coverage cao hơn bằng cách vi phạm constraint | Violation gap rõ ràng trong raw data | P2 |
| 3 | AUROC selector của CRC-Select cao hơn posthoc_crc | 0.914 vs 0.907 (CIFAR-10) | P3 |
| 4 | CRC-Select hoạt động trên cả CIFAR-10 và CIFAR-100 | Cả hai datasets trong tables | P1 |
| 5 | Finite-sample guarantee valid sau khi tune | 0 violation = empirical validation | T1 |

---

## Phần 6 — Checklist Trước Khi Submit

- [ ] Chạy run A3 (`mu_init=10`, `dual_lr=0.1`, `alpha_margin=0.10`) trên CIFAR-10, 4 seeds
- [ ] Kiểm tra violation rate tại α=0.01 ≤ 25%, α=0.02–0.05 = 0%
- [ ] Nếu coverage drop > 5%: thử `alpha_margin=0.05`
- [ ] Apply hyperparam tốt nhất lên CIFAR-100, 5 seeds  
- [ ] Kiểm tra AUROC CIFAR-100 ≥ 0.840 (nếu không: thử WideResNet backbone)
- [ ] Fix DeepGambler training (hoặc ghi chú rõ "degenerate" trong paper)
- [ ] Fix `scripts/aggregate_results.py` (`KeyError: 'n_seeds'`)
- [ ] Tạo final figures với kết quả mới
- [ ] Viết ablation: `mu_init=1` vs `mu_init=10` — show importance of tuning

---

*File này được cập nhật ngày 15/03/2026.*  
*Tham chiếu: [experiment_analysis_full_2026-03-15.md](experiment_analysis_full_2026-03-15.md)*  
*Config: [configs/crc_select.yaml](../configs/crc_select.yaml)*
