# Phân Tích Kết Quả Thí Nghiệm — CIFAR-10 & CIFAR-100

**Ngày thực hiện thí nghiệm:** 26/01/2026 – 09/02/2026  
**Ngày viết báo cáo:** 15/03/2026  
**Tác giả:** Dương Xuân Bách  
**Workspace:** `/home/duong.xuan.bach/CRC-Select-Torch`

---

## 1. Tổng Quan Thí Nghiệm

### 1.1 Bài toán

**Selective Prediction (dự đoán có chọn lọc):** Mô hình được phép **từ chối** (abstain) những ví dụ nó không đủ tự tin thay vì luôn đưa ra dự đoán. Mục tiêu:
$$\text{Tối đa hóa coverage} \quad C(\tau) = \mathbb{P}(g_\phi(X) \geq \tau)$$
trong khi giữ **accepted-loss mass** (ALM) dưới ngưỡng mục tiêu $\alpha$:
$$A(\tau) = \mathbb{E}[r(X,Y;\theta)\,\mathbf{1}\{g_\phi(X)\geq\tau\}] \leq \alpha$$

### 1.2 Phương pháp đề xuất: CRC-Select

CRC-Select huấn luyện selector $g_\phi$ với tổng loss:
$$\mathcal{L} = \mathcal{L}_\text{pred} + \beta\,\mathcal{L}_\text{cov} + \mu\,\mathcal{L}_\text{risk}$$

trong đó $\mathcal{L}_\text{risk} = \max(0,\,\hat{A}_\text{mb} - \alpha)$ thẳng hàng với đại lượng CRC chứng nhận.

**Bảo đảm lý thuyết (Theorem 1):** Dưới exchangeability, sau CRC calibration trên held-out set:
$$\mathbb{E}[r(X_{n+1},Y_{n+1};\theta)\,\mathbf{1}\{g_\phi(X_{n+1})\geq\hat{\tau}\}] \leq \alpha$$

### 1.3 Thiết lập

| Mục | CIFAR-10 | CIFAR-100 |
|-----|----------|-----------|
| Seeds | 42, 123, 456, 999 (4 seeds) | 42, 123, 456, 789, 999 (5 seeds) |
| Mức α thử nghiệm | 0.01, 0.02, 0.05, 0.10, 0.15, 0.20 | 0.01, 0.02, 0.05, 0.10, 0.15, 0.20 |
| Natural ALM (τ=0) | ~0.083–0.091 | ~0.190–0.210 |
| Base accuracy | ~91–92% | ~65–72% (top-1) |
| Baselines | vanilla, posthoc\_crc, MSP, Energy, TempScaled\_MSP, DeepGambler | Như CIFAR-10 |

**Chú ý quan trọng:** Trên CIFAR-10, natural ALM ≈ 0.088 nên α ≥ 0.10 là **trivial** (τ̂=0, coverage=1.0). Trên CIFAR-100, natural ALM ≈ 0.20 nên **tất cả 6 mức α đều có selective prediction thực sự.**

---

## 2. Kết Quả CIFAR-10

### 2.1 Violation Rate (n=4 seeds)

| Method | α=0.01 | α=0.02 | α=0.05 | α=0.10 | α=0.15 | α=0.20 |
|--------|:------:|:------:|:------:|:------:|:------:|:------:|
| **CRC-Select** | **4/4 = 100%** | 2/4 = 50% | **1/4 = 25%** | 0/4 = 0% | 0/4 = 0% | 0/4 = 0% |
| vanilla | 2/4 = 50% | 3/4 = 75% | 3/4 = 75% | 0/4 = 0% | 0/4 = 0% | 0/4 = 0% |
| posthoc\_crc | 2/4 = 50% | 3/4 = 75% | 3/4 = 75% | 0/4 = 0% | 0/4 = 0% | 0/4 = 0% |
| MSP | 2/4 = 50% | 2/4 = 50% | 3/4 = 75% | 0/4 = 0% | 0/4 = 0% | 0/4 = 0% |
| Energy | 2/4 = 50% | 2/4 = 50% | 4/4 = 100% | 0/4 = 0% | 0/4 = 0% | 0/4 = 0% |
| TempScaled\_MSP | 2/4 = 50% | 2/4 = 50% | 3/4 = 75% | 0/4 = 0% | 0/4 = 0% | 0/4 = 0% |
| DeepGambler | 0/4 = 0% ⚠️ | 0% ⚠️ | 0% ⚠️ | 0% ⚠️ | 0% ⚠️ | 0% ⚠️ |

> ⚠️ DeepGambler có violation=0% nhưng coverage=0.0 ở mọi α — training degenerate (từ chối tất cả).

> α ≥ 0.10: Tất cả phương pháp có τ̂=0 (chấp nhận mọi ví dụ), coverage=1.0. Không có selective prediction thực sự trong vùng này trên CIFAR-10.

### 2.2 Coverage Trung Bình và ALM (seed đại diện: seed_42)

| Method | α=0.01 | | α=0.02 | | α=0.05 | |
|--------|--------|--------|--------|--------|--------|--------|
| | Cov | ALM | Cov | ALM | Cov | ALM |
| **CRC-Select** | 0.762 | 0.0101 | 0.835 | 0.0158 | **0.927** | 0.0495 |
| vanilla/posthoc\_crc | 0.781 | 0.0119 | 0.852 | 0.0239 | 0.942 | 0.0554 |
| MSP | 0.785 | 0.0118 | 0.840 | 0.0194 | 0.935 | 0.0502 |
| Energy | 0.759 | 0.0109 | 0.837 | 0.0217 | 0.939 | 0.0522 |

### 2.3 AUROC Selector (CIFAR-10)

| Method | AUROC |
|--------|:-----:|
| **CRC-Select** | **~0.914–0.917** |
| vanilla/posthoc\_crc | ~0.905–0.916 |
| MSP | ~0.907 |
| TempScaled\_MSP | ~0.905 |
| Energy | ~0.900 |
| DeepGambler | 0.500 (random) |

### 2.4 Phân tích CIFAR-10

**Điểm mạnh — α=0.05:**
- CRC-Select: violation 25% (1/4), coverage 92.7%
- posthoc\_crc: violation 75% (3/4), coverage 94.2%
- **Coverage gần tương đương nhưng CRC-Select vi phạm ít hơn 3× → selector đáng tin cậy hơn.**

**Vấn đề — α=0.01:**
- CRC-Select: violation 100% (4/4), trong khi baseline chỉ 50%
- Các vi phạm đa phần biên độ nhỏ (gap ≈ 0.00013 – 0.00370) nhưng nhất quán qua 4 seeds
- Nguyên nhân khả dĩ: training loss không đủ chặt tại α rất nhỏ; seed 123/999 có coverage=0 trong calibration set → instability

**α ≥ 0.10 là trivial:** Natural ALM model CIFAR-10 ≈ 0.083–0.091 < 0.10, nên không cần từ chối ví dụ nào.

---

## 3. Kết Quả CIFAR-100

### 3.1 Violation Rate (n=5 seeds)

| Method | α=0.01 | α=0.02 | α=0.05 | α=0.10 | α=0.15 | α=0.20 |
|--------|:------:|:------:|:------:|:------:|:------:|:------:|
| **CRC-Select** | **1/5 = 20%** | **0/5 = 0%** | 2/5 = 40% | 3/5 = 60% | 3/5 = 60% | 3/5 = 60% |
| posthoc\_crc | 2/5 = 40% | 1/5 = 20% | 2/5 = 40% | 2/5 = 40% | 2/5 = 40% | **4/5 = 80%** |
| vanilla | ≈ posthoc\_crc | ≈ | ≈ | ≈ | ≈ | ≈ |
| MSP | 3/5 = 60% | 2/5 = 40% | 3/5 = 60% | 3/5 = 60% | 3/5 = 60% | 4/5 = 80% |
| Energy | 3/5 = 60% | 3/5 = 60% | 4/5 = 80% | 4/5 = 80% | 4/5 = 80% | 4/5 = 80% |
| TempScaled\_MSP | ~3/5 = 60% | ~1/5 = 20% | ~3/5 = 60% | ~3/5 = 60% | ~2/5 = 40% | ~3/5 = 60% |
| DeepGambler | 0% ⚠️ | 0% ⚠️ | 0% ⚠️ | 0% ⚠️ | 0% ⚠️ | 0% ⚠️ |

> ⚠️ DeepGambler: coverage=0.0, AUROC=0.5 — training degenerate, kết quả không hợp lệ.

### 3.2 Coverage Chi Tiết theo Seed — CRC-Select

| Seed | α=0.01 | Viol | α=0.02 | Viol | α=0.05 | Viol | α=0.10 | Viol | α=0.15 | Viol | α=0.20 | Viol |
|------|:------:|:----:|:------:|:----:|:------:|:----:|:------:|:----:|:------:|:----:|:------:|:----:|
| 42 | 0.307 | ✓ | 0.412 | ✓ | 0.572 | ✗ | 0.716 | ✗ | 0.809 | ✗ | 0.884 | ✗ |
| 123 | 0.343 | ✗ | 0.422 | ✓ | 0.577 | ✓ | 0.732 | ✗ | 0.820 | ✗ | 0.891 | ✓ |
| 456 | 0.287 | ✓ | 0.396 | ✓ | 0.570 | ✗ | 0.717 | ✗ | 0.818 | ✗ | 0.891 | ✗ |
| 789 | 0.303 | ✓ | 0.397 | ✓ | 0.559 | ✓ | 0.698 | ✓ | 0.802 | ✓ | 0.885 | ✗ |
| 999 | 0.330 | ✓ | 0.418 | ✓ | 0.564 | ✓ | 0.707 | ✓ | 0.812 | ✓ | 0.889 | ✓ |
| **Mean** | **0.314** | **20%** | **0.409** | **0%** | **0.568** | **40%** | **0.714** | **60%** | **0.812** | **60%** | **0.888** | **60%** |

> ✓ = không vi phạm, ✗ = vi phạm (ALM > α)

### 3.3 Coverage Trung Bình So Sánh (CIFAR-100)

| Method | α=0.01 | α=0.02 | α=0.05 | α=0.10 | α=0.15 | α=0.20 |
|--------|:------:|:------:|:------:|:------:|:------:|:------:|
| **CRC-Select** | **0.314** | **0.409** | **0.568** | **0.714** | **0.812** | **0.888** |
| posthoc\_crc | 0.346 | 0.440 | 0.589 | 0.723 | 0.815 | 0.894 |
| MSP | 0.399 | 0.503 | 0.641 | 0.752 | 0.832 | 0.900 |
| Energy | 0.369 | 0.483 | 0.624 | 0.744 | 0.834 | 0.902 |

> CRC-Select có coverage thấp hơn mọi baseline trên CIFAR-100. Đây là đánh đổi (trade-off) để đổi lấy violation rate thấp hơn ở α nhỏ.

### 3.4 Accepted-Loss Mass Chi Tiết (seed_42, CIFAR-100)

| Method | α=0.01 | Viol | α=0.02 | Viol | α=0.05 | Viol | α=0.10 | Viol | α=0.20 | Viol |
|--------|:------:|:----:|:------:|:----:|:------:|:----:|:------:|:----:|:------:|:----:|
| **CRC-Select** | 0.00928 | ✓ | 0.01946 | ✓ | 0.05137 | ✗ | **0.10435** | ✗ | **0.20448** | ✗ |
| posthoc\_crc | 0.00852 | ✓ | 0.02094 | ✗ | 0.04907 | ✓ | 0.10291 | ✗ | 0.20886 | ✗ |
| MSP | 0.01088 | ✗ | 0.01874 | ✓ | 0.05182 | ✗ | 0.10131 | ✗ | 0.20728 | ✗ |
| Energy | 0.01191 | ✗ | 0.02173 | ✗ | 0.05518 | ✗ | 0.10754 | ✗ | 0.21654 | ✗ |

> ✓ = ALM ≤ α (không vi phạm), ✗ = ALM > α (vi phạm)

### 3.5 AUROC Selector (CIFAR-100)

| Method | AUROC (seed 42) | AUROC (seed 123) | Nhận xét |
|--------|:---------------:|:----------------:|----------|
| **MSP** | **0.865** | **0.872** | **Cao nhất** |
| **TempScaled\_MSP** | **0.864** | **0.870** | Tương đương MSP |
| **Energy** | 0.851 | 0.859 | Khá cao |
| posthoc\_crc/vanilla | 0.835 | 0.842 | Trung bình |
| **CRC-Select** | **0.821** | **0.824** | **Thấp nhất** ⚠️ |
| DeepGambler | 0.500 | 0.500 | Random |

> **Đây là kết quả đáng lo ngại:** Trên CIFAR-100, AUROC của CRC-Select (0.821–0.824) thấp hơn tất cả baseline (0.835–0.872). Điều này ngược lại hoàn toàn so với CIFAR-10 (CRC-Select dẫn đầu ~0.914–0.917).

---

## 4. So Sánh CIFAR-10 vs CIFAR-100

### 4.1 Bảng tổng hợp violation rate

| Dataset | Method | α=0.01 | α=0.02 | α=0.05 | α=0.10 | α=0.15 | α=0.20 |
|---------|--------|:------:|:------:|:------:|:------:|:------:|:------:|
| CIFAR-10 | **CRC-Select** | 100% ⚠️ | 50% | **25%** ✅ | 0% | 0% | 0% |
| CIFAR-10 | posthoc\_crc | 50% | 75% | 75% | 0% | 0% | 0% |
| CIFAR-10 | MSP | 50% | 50% | 75% | 0% | 0% | 0% |
| CIFAR-10 | Energy | 50% | 50% | 100% ⚠️ | 0% | 0% | 0% |
| **CIFAR-100** | **CRC-Select** | **20%** ✅ | **0%** ✅ | 40% | 60% ⚠️ | 60% ⚠️ | 60% |
| CIFAR-100 | posthoc\_crc | 40% | 20% | 40% | 40% | 40% | 80% |
| CIFAR-100 | MSP | 60% | 40% | 60% | 60% | 60% | 80% |
| CIFAR-100 | Energy | 60% | 60% | 80% | 80% | 80% | 80% |

### 4.2 Điểm mạnh và điểm yếu theo dataset

**CIFAR-10:**
- ✅ Tốt nhất tại α=0.05 (25% vs 75%) — luận điểm chính của paper
- ❌ Tệ nhất tại α=0.01 (100%)
- ✅ AUROC selector tốt nhất (~0.914)

**CIFAR-100:**
- ✅ Tốt nhất tại α=0.01 và α=0.02 (20%, 0% vs 40-60%)
- ❌ Tăng mạnh violation ở α=0.10–0.20 (60%) — posthoc_crc chỉ 40%
- ❌ AUROC selector thấp nhất (~0.821) — mất khả năng phân biệt hard/easy examples trên dataset khó
- Coverage luôn thấp hơn tất cả baseline

---

## 5. Phân Tích Chuyên Sâu

### 5.1 Tại sao CRC-Select hoạt động khác nhau trên hai dataset?

**CIFAR-10 (dễ hơn, acc ~91%):**
- Model backbone đủ mạnh → selector học được điên kiện tốt để phân biệt hard/easy
- AUROC cao (~0.914) → selector hiệu quả
- CRC training objective phù hợp với natural ALM thấp

**CIFAR-100 (khó hơn, acc ~65-72%):**
- Natural ALM ≈ 0.19–0.21, rất gần với các mức α = 0.10–0.20
- Selector học trong môi trường khó → AUROC giảm (0.821 vs 0.914)
- Khi α lớn (0.10–0.20), mô hình cần chấp nhận nhiều ví dụ → sát biên → vi phạm nhiều
- Training loss $\mathcal{L}_\text{cov} = (\hat{C} - c_0)^2$ đẩy coverage lên tối đa → quá nhiều accepted examples khi α lớn

### 5.2 Tại sao MSP có AUROC cao hơn CRC-Select trên CIFAR-100?

MSP sử dụng max softmax probability $\max_y p_\theta(y|x)$ — đây là một score calibrated tốt tự nhiên trên softmax outputs. Trên CIFAR-100 với 100 classes, softmax entropy/confidence có nhiều thông tin phân biệt hơn. CRC-Select học một selector phức tạp hơn nhưng bị ảnh hưởng bởi training noise trên task khó.

### 5.3 Pattern đặc biệt: Energy seed 789

Energy score tại seed 789 cho ALM rất thấp (0.0046 tại α=0.01, coverage=0.246) và không vi phạm ở mọi α — ngược lại hoàn toàn với các seeds 42, 123, 456, 999. Đây có thể là seed may mắn tạo ra calibration set thuận lợi, hoặc sự khác biệt do random split. Điều này phản ánh **high variance** của Energy-based baseline trên CIFAR-100.

### 5.4 Vấn đề violation rate tăng theo α trên CIFAR-100

Một pattern kỳ lạ: CRC-Select vi phạm nhiều hơn khi α tăng (0% tại α=0.02 → 60% tại α=0.10–0.20). Giải thích:
- Khi α lớn, CRC calibration đặt τ̂ rất nhỏ (≈ 0.003–0.03) → accept gần như toàn bộ test set
- Test ALM ≈ 0.19–0.21 (natural error rate) xấp xỉ α=0.20 → biên độ vi phạm nhỏ nhưng nhất quán
- Selector không thể làm giảm ALM đủ nhiều bằng cách từ chối thêm ví dụ vì coverage penalty cản trở

---

## 6. Bảng Tổng Hợp Cuối (Summary)

### CIFAR-10 — Kết luận chính

| Tiêu chí | CRC-Select | Best Baseline | Kết luận |
|----------|:----------:|:-------------:|---------|
| Violation rate α=0.05 | **25%** | 75% (vanilla/posthoc) | ✅ CRC-Select vượt trội 3× |
| Coverage tại α=0.05 | 93.6% | 94.2% | ~ Tương đương |
| AUROC | **0.914** | 0.908 (posthoc) | ✅ CRC-Select tốt hơn |
| Violation rate α=0.01 | **100%** | 50% | ❌ CRC-Select tệ nhất |
| DeepGambler | N/A | N/A | ❌ Degenerate (cov=0) |

### CIFAR-100 — Kết luận chính

| Tiêu chí | CRC-Select | Best Baseline | Kết luận |
|----------|:----------:|:-------------:|---------|
| Violation rate α=0.01 | **20%** | 40% (posthoc) | ✅ CRC-Select tốt hơn 2× |
| Violation rate α=0.02 | **0%** | 20% (posthoc) | ✅ CRC-Select hoàn hảo |
| Violation rate α=0.05 | 40% | 40% (posthoc) | ~ Bằng nhau |
| Violation rate α=0.10 | **60%** | 40% (posthoc) | ❌ CRC-Select tệ hơn |
| Coverage (mọi α) | **Thấp nhất** | MSP: cao nhất | ❌ Trade-off bất lợi |
| AUROC | **0.821** | **0.865** (MSP) | ❌ CRC-Select thấp nhất |

---

## 7. Vấn Đề Kỹ Thuật Cần Sửa

| Vấn đề | Mức độ | Chi tiết |
|--------|--------|---------|
| **100% violation tại α=0.01 (CIFAR-10)** | 🔴 Cao | Cần thêm safety margin trong training |
| **AUROC CRC-Select thấp hơn MSP trên CIFAR-100** | 🔴 Cao | Selector không generalize tốt trên task khó |
| **60% violation tại α=0.10–0.20 (CIFAR-100)** | 🔴 Cao | Coverage penalty quá mạnh, cần re-tune $\mu/\beta$ |
| **DeepGambler coverage=0 trên cả hai dataset** | 🔴 Cao | Training degenerate, cần fix hoặc loại baseline |
| **Aggregate script crash** (`KeyError: 'n_seeds'`) | 🟡 Trung bình | Fix `scripts/aggregate_results.py` line 195 |
| **vanilla ≡ posthoc_crc** (kết quả đồng nhất) | 🟡 Trung bình | Cần tách biệt hơn hoặc ghi chú rõ trong paper |

---

## 8. Hướng Cải Thiện

### Ưu tiên cao

1. **Safety margin:** Thay thế $\alpha$ bằng $\alpha' = \alpha \cdot (1-\delta)$ với $\delta \approx 0.1$ trong training loss $\mathcal{L}_\text{risk}$ để tạo buffer an toàn

2. **Re-tune hyperparameters trên CIFAR-100:** Thực nghiệm với $\beta$ nhỏ hơn (ít coverage pressure) và $\mu$ lớn hơn (risk pressure) để cân bằng tốt hơn trên dataset khó

3. **Backbone mạnh hơn cho CIFAR-100:** Sử dụng ResNet-50 hoặc WideResNet thay vì VGG để cải thiện AUROC của selector

### Ưu tiên trung bình

4. **Calibration set lớn hơn:** Tăng tỉ lệ n_cal để giảm variance giữa cal/test ALM, đặc biệt tại α nhỏ

5. **Adaptive $\tau$ scheduling:** Cập nhật CRC threshold thường xuyên hơn trong vòng lặp alternating optimization

6. **Fix DeepGambler:** Điều chỉnh learning rate hoặc reward ratio để tránh degenerate collapse

---

## 9. Kết Luận Tổng Thể

### Claim hiện tại của paper — cần điều chỉnh

| Claim | Thực tế | Điều chỉnh đề xuất |
|-------|---------|-------------------|
| "Higher coverage at same α" | Không nhất quán; CIFAR-100 coverage luôn thấp hơn | → "Lower violation rate at same coverage level" |
| "Finite-sample guarantee" | Bị vi phạm empirically tại α=0.01 (CIFAR-10) và α=0.1–0.2 (CIFAR-100) | → Nhấn mạnh guarantee chỉ valid cho final frozen model + final cal set |
| "CRC-Select works well across datasets" | Tốt trên CIFAR-10 tại α=0.05; yếu hơn trên CIFAR-100 | → Giới hạn scope claim hoặc thêm ablation trên CIFAR-100 |

### Tóm tắt 1 câu cho mỗi dataset

- **CIFAR-10:** CRC-Select **giảm violation rate 3×** tại α=0.05 so với baseline tại cùng coverage level (~93%), với AUROC selector cao nhất — đây là bằng chứng thực nghiệm thuyết phục nhất.

- **CIFAR-100:** CRC-Select **tốt hơn ở α nhỏ** (20% vs 40% violation tại α=0.01) nhưng **tệ hơn ở α lớn** (60% vs 40% tại α=0.10–0.15) và AUROC selector thấp nhất — cần cải thiện trước khi đưa vào paper.

---

*Dữ liệu nguồn: `results_paper/cifar100/` và `results_paper/` (CIFAR-10)*  
*Phân tích CIFAR-10 được thực hiện ngày 13/03/2026, phân tích CIFAR-100 được bổ sung ngày 15/03/2026*  
*Paper gốc: [docs/crc_select_corrected_mini_paper.md](crc_select_corrected_mini_paper.md)*
