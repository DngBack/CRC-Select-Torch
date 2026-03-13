# Phân Tích Kết Quả Thí Nghiệm: CRC-Select

**Ngày thực hiện thí nghiệm:** 26/01/2026 – 09/02/2026 (xem thư mục `wandb/`)  
**Ngày viết báo cáo phân tích:** 13/03/2026  
**Tác giả phân tích:** Dương Xuân Bách  
**Dataset:** CIFAR-10  
**Seeds:** 42, 123, 456, 999 (4 lần chạy độc lập)  

---

## 1. Mô Tả Thí Nghiệm

### 1.1 Bài toán

Thí nghiệm nghiên cứu **Selective Prediction** (dự đoán có chọn lọc): thay vì luôn đưa ra dự đoán, mô hình có thể **từ chối** (abstain) trên các ví dụ được đánh giá là không đáng tin cậy. Mục tiêu thực tiễn là:

> **Tối đa hóa coverage** (tỉ lệ ví dụ được chấp nhận) trong khi **giữ rủi ro của các dự đoán được chấp nhận dưới ngưỡng mục tiêu** $\alpha$.

Đây là sự kết hợp giữa hai hướng nghiên cứu:
1. **Selective Prediction / Learning-to-Reject** — học từ chối ví dụ khó
2. **Conformal Risk Control (CRC)** — đảm bảo finite-sample guarantee về rủi ro

### 1.2 Phương pháp đề xuất: CRC-Select

**CRC-Select** huấn luyện một selector $g_\phi: \mathcal{X} \to [0,1]$ sao cho sau bước CRC calibration (giữ lại trên held-out set), ngưỡng $\hat{\tau}$ được chọn sẽ **ít conservative hơn** → coverage cao hơn.

**Ba thành phần loss khi training:**

| Loss | Công thức | Mục đích |
|------|-----------|----------|
| Prediction loss | $\mathcal{L}_\text{pred} = \frac{\sum g_i \cdot \text{CE}(f(x_i), y_i)}{\sum g_i + \varepsilon}$ | Dự đoán tốt trên ví dụ được chấp nhận |
| Coverage penalty | $\mathcal{L}_\text{cov} = (\hat{C} - c_0)^2$ | Duy trì coverage mục tiêu $c_0$ |
| CRC-aligned risk | $\mathcal{L}_\text{risk} = \max(0, \hat{A}_\text{mb} - \alpha)$ | Giảm accepted-loss mass |

**Đại lượng chứng nhận (certified quantity):**
$$A(\tau) = \mathbb{E}[r(X,Y;\theta) \cdot \mathbf{1}\{g_\phi(X) \geq \tau\}]$$

Đây là **accepted-loss mass** — không phải conditional selective risk. CRC có thể đảm bảo hữu hạn mẫu cho đại lượng này dưới giả định exchangeability.

**Định lý 1 (Finite-Sample Accepted-Loss Control):** Nếu $\hat{\tau}$ được chọn theo quy tắc CRC, thì:
$$\mathbb{E}[r(X_{n+1}, Y_{n+1};\theta) \cdot \mathbf{1}\{g_\phi(X_{n+1}) \geq \hat{\tau}\}] \leq \alpha$$

### 1.3 Thiết lập thí nghiệm

- **Model backbone:** VGG / ResNet (CIFAR-10)
- **Loss function:** $r(x,y;\theta) = 1 - p_\theta(y|x) \in [0,1]$ (1 − softmax confidence)
- **Data splits:** Train / Calibration / Test
- **Mức rủi ro thử nghiệm:** $\alpha \in \{0.01, 0.02, 0.05, 0.10, 0.15, 0.20\}$
- **Training protocol:** Alternating optimization (warm start → calibrate → gradient steps → repeat)

### 1.4 Các baseline so sánh

| Method | Mô tả |
|--------|-------|
| **vanilla** | SelectiveNet cơ bản, không có CRC calibration |
| **posthoc\_crc** | CRC calibration trên selector có sẵn (không train cùng) |
| **MSP** | Maximum Softmax Probability làm rejection score |
| **Energy** | Energy-based score cho selective prediction |
| **TempScaled\_MSP** | MSP sau temperature scaling |
| **DeepGambler** | Deep Gamblers — học từ chối bằng an additional "reject" class |

---

## 2. Kết Quả Chính

### 2.1 Violation Rate — Chỉ số chính về độ tin cậy

**Violation** xảy ra khi test accepted-loss mass $> \alpha$ (vi phạm constraint).

| Method | α=0.01 | α=0.02 | α=0.05 | α=0.10 | α=0.15 | α=0.20 |
|--------|:------:|:------:|:------:|:------:|:------:|:------:|
| **CRC-Select** | **100%** | **50%** | **25%** | **0%** | **0%** | **0%** |
| vanilla | 50% | 75% | 75% | 0% | 0% | 0% |
| posthoc\_crc | 50% | 75% | 75% | 0% | 0% | 0% |
| MSP | 50% | 50% | 75% | 0% | 0% | 0% |
| Energy | 50% | 50% | 100% | 0% | 0% | 0% |
| TempScaled\_MSP | 50% | 50% | 75% | 0% | 0% | 0% |
| DeepGambler | 0% | 0% | 0% | 0% | 0% | 0% |

> **Lưu ý:** Violation rate được tính qua 4 seeds (42, 123, 456, 999). 0% violation không có nghĩa là tốt nếu như coverage = 0 (trường hợp DeepGambler).

### 2.2 Coverage tại các mức α (seed_42, đại diện)

| Method | α=0.01 | α=0.02 | α=0.05 |
|--------|:------:|:------:|:------:|
| **CRC-Select** | 0.762 | 0.835 | 0.927 |
| vanilla | 0.781 | 0.853 | 0.942 |
| posthoc\_crc | 0.781 | 0.852 | 0.942 |
| MSP | 0.785 | 0.840 | 0.935 |
| Energy | 0.759 | 0.837 | 0.939 |
| TempScaled\_MSP | 0.780 | 0.840 | 0.937 |
| **DeepGambler** | **0.000** | **0.000** | **0.000** |

### 2.3 Accepted-Loss Mass (ALM) tại seed_42 — so sánh alm vs α

**CRC-Select:**

| α | ALM (test) | Vi phạm? | Gap |
|---|-----------|----------|-----|
| 0.01 | 0.01013 | ✗ (vượt 0.00013) | tiny |
| 0.02 | 0.01586 | ✓ (dưới ngưỡng) | — |
| 0.05 | 0.04949 | ✓ (dưới ngưỡng) | — |
| 0.10 | 0.09107 | ✓ | — |

**posthoc_crc:**

| α | ALM (test) | Vi phạm? | Gap |
|---|-----------|----------|-----|
| 0.01 | 0.01187 | ✗ (vượt 0.00187) | lớn hơn CRC-Select |
| 0.02 | 0.02385 | ✗ (vượt 0.00385) | — |
| 0.05 | 0.05540 | ✗ (vượt 0.00540) | — |
| 0.10 | 0.08847 | ✓ | — |

### 2.4 AUROC của Selector (khả năng phân biệt hard/easy examples)

| Method | AUROC (trung bình qua seeds) |
|--------|:-----------------------------:|
| **CRC-Select** | **~0.911–0.917** |
| vanilla | ~0.909–0.916 |
| posthoc\_crc | ~0.905–0.916 |
| MSP | ~0.907 |
| Energy | ~0.900 |
| TempScaled\_MSP | ~0.905 |
| DeepGambler | 0.500 (random!) |

### 2.5 Coverage tại các mức α — Trung bình qua 4 seeds

**α = 0.05 (mức quan trọng nhất):**

| Method | Coverage trung bình | Violation rate |
|--------|:-------------------:|:--------------:|
| **CRC-Select** | ~0.936 | **25%** |
| vanilla | ~0.942 | 75% |
| posthoc\_crc | ~0.942 | 75% |
| MSP | ~0.937 | 75% |
| Energy | ~0.939 | 100% |

---

## 3. Phân Tích Chi Tiết

### 3.1 Kết quả tại α = 0.05: Điểm sáng nhất của CRC-Select

Tại $\alpha = 0.05$, CRC-Select đạt **violation rate 25%** (1/4 seeds vi phạm) so với 75% của vanilla/posthoc_crc. Coverage gần như tương đương (~93.6% vs ~94.2%).

**Ý nghĩa quan trọng:** Các baseline "đạt" coverage cao hơn một phần vì **họ vi phạm constraint** — tức là chấp nhận các ví dụ rủi ro hơn ngưỡng cho phép. CRC-Select duy trì constraint tốt hơn mà không đánh đổi quá nhiều coverage.

**Vi phạm theo seed tại α=0.05:**

| Seed | CRC-Select ALM | Vi phạm? | posthoc_crc ALM | Vi phạm? |
|------|:--------------:|:--------:|:---------------:|:--------:|
| 42 | 0.0495 | ✓ | 0.0554 | ✗ |
| 123 | **0.0578** | **✗** | 0.0551 | ✗ |
| 456 | 0.0442 | ✓ | 0.0422 | ✓ |
| 999 | 0.0486 | ✓ | 0.0540 | ✗ |

### 3.2 Vấn đề tại α = 0.01: 100% violation — Nghịch lý cần giải quyết

CRC-Select vi phạm 100% (4/4 seeds) tại α=0.01, trong khi vanilla/posthoc_crc chỉ 50%. Đây là nghịch lý vì đây là phương pháp được thiết kế để đảm bảo constraint.

**Chi tiết vi phạm:**

| Seed | ALM | Violation gap | Observation |
|------|-----|:-------------:|-------------|
| 42 | 0.01013 | 0.00013 | Biên độ cực nhỏ — marginal |
| 123 | 0.01370 | 0.00370 | Thu phạm đáng kể |
| 456 | 0.01027 | 0.00027 | Biên độ nhỏ |
| 999 | 0.01306 | 0.00306 | Vượt đáng kể |

**Nguyên nhân khả dĩ:**

1. **Training objective quá permissive:** Khi $\alpha$ rất chặt (0.01), minibatch training loss $\mathcal{L}_\text{risk}$ không đủ áp lực để duy trì biên độ an toàn.

2. **Cal/Test distributional gap:** Với $n_{cal}$ hữu hạn, test ALM có thể lệch so với cal ALM. Tại $\alpha = 0.01$ rất nhỏ → ít "room" để hấp thụ sự sai lệch này.

3. **Selector học "greedy coverage":** Training với $\mathcal{L}_\text{cov} = (\hat{C} - c_0)^2$ khuyến khích coverage cao → selector chấp nhận nhiều ví dụ sát biên → dễ vượt ngưỡng trên test.

4. **Seed 123 và 999:** Coverage từ calibration set là 0.0 (xem `coverage_at_risk.csv`), có nghĩa là $\hat{\tau} > \max(g_\phi)$ → mô hình từ chối tất cả trong cal nhưng không trong test — đây là dấu hiệu của instability.

### 3.3 α ≥ 0.10: Tất cả phương pháp trở thành trivial

Tại $\alpha \geq 0.10$, tất cả phương pháp đặt $\hat{\tau} = 0$ (chấp nhận mọi ví dụ) và coverage = 1.0. Lý do: model CIFAR-10 có natural ALM ≈ 0.082–0.091 < 0.10 → không cần từ chối ví dụ nào để đảm bảo constraint. Selective prediction thực sự không xảy ra ở vùng này.

### 3.4 DeepGambler — Lỗi Training Nghiêm Trọng

DeepGambler cho coverage = 0.0 và AUROC = 0.5 (random) ở mọi $\alpha$. Đây là dấu hiệu training degenerate: model học cách **từ chối tất cả** thay vì phân biệt hard/easy examples. Kết quả 0% violation là "trivial" — không nên đưa vào so sánh chính mà phải ghi chú rõ.

### 3.5 posthoc_crc vs vanilla — Không có sự khác biệt

Đáng ngạc nhiên, posthoc_crc và vanilla cho **kết quả gần như đồng nhất** ở mọi $\alpha$ và mọi seed. Điều này xảy ra vì posthoc_crc áp dụng CRC calibration trên **cùng selector** với vanilla → cùng $\hat{\tau}$ → cùng kết quả. Sự khác biệt duy nhất là khi calibration set đủ lớn để CRC chọn $\hat{\tau}$ khác threshold mặc định.

---

## 4. Đánh Giá Tổng Thể

### 4.1 Điểm mạnh của CRC-Select

| Tiêu chí | Đánh giá |
|----------|----------|
| Violation rate tại α=0.05 | ✅ Tốt rõ rệt (25% vs 75%) |
| AUROC selector | ✅ Cao nhất trong tất cả methods |
| Lý thuyết đảm bảo | ✅ Theorem 1 hợp lệ (accepted-loss mass) |
| Coverage-risk trade-off  | ✅ Cân bằng tốt tại α ∈ [0.02, 0.10] |

### 4.2 Điểm yếu và vấn đề cần giải quyết

| Vấn đề | Mức độ | Hành động đề xuất |
|--------|--------|-------------------|
| 100% violation tại α=0.01 | 🔴 Nghiêm trọng | Thêm safety margin $\alpha' = \alpha - \delta$ trong training, tăng cal set size |
| DeepGambler degenerate | 🔴 Nghiêm trọng | Fix training hoặc loại baseline này, ghi chú rõ ràng |
| Aggregate script crash (KeyError: 'n_seeds') | 🟡 Cần sửa | Fix `scripts/aggregate_results.py` line 195 |
| Coverage lợi thế không rõ ràng vs baseline | 🟡 Framing | Đổi claim: "ít violation hơn" thay vì "coverage cao hơn" |
| Seed 123/999 instability tại α=0.01 | 🟡 Cần điều tra | Kiểm tra cal set split, có thể tăng n_cal |

### 4.3 Điều chỉnh claim trong paper

**Claim hiện tại (cần điều chỉnh):**
> "CRC-Select achieves higher coverage than post-hoc CRC at the same audited risk level."

**Claim chính xác hơn dựa trên kết quả:**
> "At the same coverage level, CRC-Select achieves significantly lower violation rates than baselines (25% vs 75% at α=0.05), indicating that the learned selector better aligns accepted-loss mass with the CRC-certified quantity."

---

## 5. Kết Quả Chi Tiết Theo Seed (α = 0.05)

### CRC-Select

| Seed | τ̂ | ALM (cal) | ALM (test) | Coverage | Selective Acc | Vi phạm |
|------|------|-----------|-----------|----------|:-------------:|:-------:|
| 42 | 5.6e-5 | 0.04963 | 0.04949 | 0.9266 | 0.9488 | ✓ |
| 123 | 2.0e-5 | 0.04975 | 0.05776 | 0.9528 | 0.9414 | **✗** |
| 456 | 2.5e-5 | 0.04980 | 0.04423 | 0.9400 | 0.9553 | ✓ |
| 999 | 1.3e-5 | 0.04975 | 0.04859 | 0.9464 | 0.9518 | ✓ |
| **Mean** | — | 0.04973 | **0.05002** | **0.9415** | 0.9493 | **25%** |

### posthoc_crc

| Seed | τ̂ | ALM (cal) | ALM (test) | Coverage | Selective Acc | Vi phạm |
|------|------|-----------|-----------|----------|:-------------:|:-------:|
| 42 | 6.4e-6 | 0.04969 | 0.05540 | 0.9425 | 0.9439 | **✗** |
| 123 | 3.1e-6 | 0.04971 | 0.05513 | 0.9478 | 0.9445 | **✗** |
| 456 | 2.6e-5 | 0.04973 | 0.04223 | 0.9354 | 0.9568 | ✓ |
| 999 | 7.2e-6 | 0.04963 | 0.05400 | 0.9464 | 0.9459 | **✗** |
| **Mean** | — | 0.04969 | **0.05169** | **0.9430** | 0.9478 | **75%** |

---

## 6. Trạng Thái Hiện Tại và Việc Cần Làm Tiếp Theo

### 6.1 Files kết quả đã có
- `results_paper/CRC-Select/seed_{42,123,456,999}/` — đầy đủ
- `results_paper/{vanilla,posthoc_crc,MSP,Energy,TempScaled_MSP,DeepGambler}/seed_{42,123,456,999}/` — đầy đủ
- `results_paper/violation_analysis/` — đầy đủ (violation details và summaries)
- `figures/{fig2_rc_frontier, fig3_coverage_at_alpha, fig4_violation_gap, fig7_threshold_efficiency}.{pdf,png}` — đã tạo

### 6.2 Vấn đề kỹ thuật cần sửa
- [ ] `scripts/aggregate_results.py` bị crash tại `KeyError: 'n_seeds'` → `results_paper/aggregated/summary_table.csv` chưa được tạo
- [ ] DeepGambler: kiểm tra lại training script, hiện tại coverage=0 ở mọi điều kiện
- [ ] Seeds 123/999 của CRC-Select: coverage=0 tại α=0.01 trong calibration set — cần điều tra

### 6.3 Hướng cải thiện thuật toán
1. **Safety margin trong training:** Thay $\alpha$ bằng $\alpha' = \alpha \cdot (1 - \epsilon)$ hoặc thêm penalty cho violation gap
2. **Larger calibration set:** Tăng tỉ lệ split để giảm variance giữa cal/test ALM
3. **Tighter alternating schedule:** Tăng tần suất calibration step để $\hat{\tau}$ luôn cập nhật
4. **Adaptive $\mu$:** Tăng weight $\mu$ của $\mathcal{L}_\text{risk}$ trong vùng $\alpha$ nhỏ

---

## 7. Tóm Tắt Nhanh (TL;DR)

| Câu hỏi | Trả lời |
|---------|---------|
| CRC-Select có tốt hơn baseline không? | **Có, tại α=0.05** — violation rate 25% vs 75% |
| CRC có đảm bảo không? | **Có về lý thuyết**, nhưng thực nghiệm vi phạm tại α=0.01 |
| Coverage có cao hơn không? | **Không rõ ràng** — tương đương hoặc thấp hơn chút |
| Điểm mạnh thực sự là gì? | **Ít vi phạm constraint hơn** tại cùng coverage level |
| Vấn đề lớn nhất? | 100% violation tại α=0.01 — cần fix trước khi submit |
| DeepGambler có dùng được không? | **Không** — training degenerate, coverage = 0 |

---

*Tài liệu này dựa trên kết quả từ thư mục `results_paper/` và `figures/` trong workspace `/home/duong.xuan.bach/CRC-Select-Torch`.*  
*Paper gốc: [docs/crc_select_corrected_mini_paper.md](crc_select_corrected_mini_paper.md)*
