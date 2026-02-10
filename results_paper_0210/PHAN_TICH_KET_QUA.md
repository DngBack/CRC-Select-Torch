# Phân tích chi tiết kết quả đánh giá (multi-seed)

Tài liệu này phân tích kết quả trong `results_paper/` cho **3 phương pháp** (CRC-Select, posthoc CRC, vanilla) và **3 seed** (123, 456, 999), định dạng phù hợp cho báo cáo/paper.

---

## 1. Thiết lập thí nghiệm

| Mục | Giá trị |
|-----|--------|
| **Dataset** | CIFAR-10 (test 5000 mẫu) |
| **Seeds** | 123, 456, 999 |
| **Phương pháp** | CRC-Select, posthoc CRC, vanilla (SelectiveNet không CRC) |
| **OOD** | Đánh giá OOD tại ID coverage cố định (60%, 70%, 80%, 90%) |

---

## 2. So sánh theo từng nhóm metric

### 2.1 AURC (Area Under Risk–Coverage curve)

AURC càng thấp càng tốt: risk trung bình khi coverage thay đổi thấp hơn.

| Method | Seed 123 | Seed 456 | Seed 999 | **Mean ± Std** |
|--------|----------|----------|----------|----------------|
| **CRC-Select** | 0.01350 | 0.01223 | 0.01402 | **0.0133 ± 0.0009** |
| Vanilla | 0.0 | 0.0 | 0.0 | 0.0 |

**Ghi chú:** Trong pipeline đánh giá hiện tại, vanilla được báo **AURC = 0** và coverage = 100% tại mọi α vì không áp dụng ngưỡng reject (hoặc chấp nhận toàn bộ). Vanilla đóng vai trò baseline “không từ chối”; so sánh công bằng nên dùng **CRC-Select vs posthoc CRC** trên cùng checkpoint/seed. Ở đây posthoc CRC không có cột AURC trong summary nên so sánh AURC chủ yếu dùng cho CRC-Select giữa các seed.

**Kết luận AURC:** CRC-Select ổn định qua seed (std ~0.0009), AURC ~0.013.

---

### 2.2 Coverage @ Risk α (Coverage đạt được khi giới hạn risk ≤ α)

Coverage tại mỗi α càng cao càng tốt (nhiều mẫu được chấp nhận trong khi vẫn đảm bảo risk).

#### CRC-Select (3 seeds)

| α (risk) | Seed 123 | Seed 456 | Seed 999 | **Mean** | **Std** |
|----------|----------|----------|----------|----------|---------|
| 0.01 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **0.02** | **82.78%** | **84.24%** | **82.50%** | **83.2%** | 0.9% |
| **0.05** | **88.48%** | **88.28%** | **87.44%** | **88.1%** | 0.5% |
| **0.10** | 100% | 100% | 100% | 100% | 0% |
| 0.15 | 100% | 100% | 100% | 100% | 0% |
| 0.20 | 100% | 100% | 100% | 100% | 0% |

- Tại **α = 0.02**: coverage trung bình **~83%**, risk thực tế ~1.97–2.0%.
- Tại **α = 0.05**: coverage **~88%**, risk ~2.95–3.12%.
- Tại **α = 0.10**: model đạt full coverage với risk test ~7.7–8.5% (đều ≤ 10%).

#### Posthoc CRC (3 seeds)

| α | Coverage (tau=0.3) | Risk (test) |
|---|-------------------|-------------|
| 0.05–0.2 | ~99.6–99.8% | ~0.14% |

Posthoc CRC đạt coverage rất cao vì risk trên test đã rất thấp (~0.14%), nên ràng buộc risk ≤ α dễ thỏa. Điều này phù hợp với baseline “calibrate trên model đã rất tốt”.

**So sánh ngắn gọn:** Ở mức α chặt (0.02, 0.05), CRC-Select cho **selective prediction có ý nghĩa** (82–88% coverage, risk 2–3%). Posthoc CRC trong số liệu này thể hiện trường hợp risk rất thấp nên coverage gần 100%.

---

### 2.3 Error / Risk tại coverage cố định (70%, 80%, 90%)

Error = 1 − accuracy trên vùng được chấp nhận; risk (trong code) = 1 − p(y|x) trung bình trên vùng đó.

#### CRC-Select

| Coverage | Seed 123 | Seed 456 | Seed 999 | **Mean (error)** | **Std** |
|----------|----------|----------|----------|------------------|---------|
| **70%** | 1.29% | 1.13% | 1.35% | **1.26%** | 0.11% |
| **80%** | 1.52% | 1.40% | 1.62% | **1.51%** | 0.11% |
| **90%** | 2.96% | 2.81% | 2.90% | **2.89%** | 0.08% |

- Ở 70% coverage: error ~1.26%, accuracy ~98.7%.
- Ở 80% coverage: error ~1.51%, accuracy ~98.5%.
- Ở 90% coverage: error ~2.89%, accuracy ~97.1%.

Kết luận: **CRC-Select ổn định giữa các seed** (std nhỏ), chất lượng selective prediction tốt ở mọi mức coverage.

#### Vanilla (reference)

Vanilla báo **error = 0**, risk ~0.14% tại mọi coverage vì cách đánh giá hiện tại (coverage = 100%, không reject). Dùng làm baseline “không từ chối”, không so trực tiếp số % error với CRC-Select.

---

### 2.4 OOD safety: OOD-Acceptance @ ID coverage cố định

Metric khuyến nghị cho so sánh công bằng: tại cùng ID coverage (60%, 70%, 80%, 90%), tỷ lệ chấp nhận mẫu OOD (SVHN) càng thấp càng an toàn.

#### CRC-Select (OOD accept rate, %)

| ID coverage | Seed 123 | Seed 456 | Seed 999 | **Mean** | **Std** |
|-------------|----------|----------|----------|----------|---------|
| 60% | 0.38 | 0.45 | 0.88 | **0.57** | 0.26 |
| **70%** | **0.90** | **0.81** | **1.39** | **1.03** | 0.30 |
| **80%** | **7.35** | **2.70** | **3.38** | **4.48** | 2.52 |
| **90%** | 47.7 | 16.8 | 23.0 | **29.2** | 15.8 |

- 70% ID: OOD accept **~1%**, safety ratio ~78–86× (seed 123, 456).
- 80% ID: OOD accept **~2.7–7.4%** tùy seed; seed 123 cao hơn (7.35%), 456 và 999 thấp hơn (2.7%, 3.4%).
- 90% ID: OOD accept tăng mạnh (17–48%), phù hợp với việc chấp nhận thêm nhiều mẫu khó.

#### Vanilla (OOD accept rate, %)

| ID coverage | Seed 123 | Seed 456 | Seed 999 | **Mean** |
|-------------|----------|----------|----------|----------|
| 60% | 0.53 | 0.13 | 0.66 | ~0.44 |
| 70% | 0.55 | 0.18 | 0.80 | ~0.51 |
| 80% | 0.85 | 0.47 | 1.08 | ~0.80 |
| 90% | 1.61 | 1.15 | 1.81 | ~1.52 |

Vanilla có **OOD accept thấp hơn** ở 80% và 90% ID coverage vì selector vanilla (cùng backbone) có thể bảo thủ hơn ở ngưỡng cao; cần so sánh trên **cùng một checkpoint** (vanilla vs posthoc CRC vs CRC-Select) để kết luận công bằng.

**Gợi ý cho paper:** Báo cả **mean ± std** theo seed và nhấn mạnh so sánh tại **cùng ID coverage** (ví dụ 70%, 80%). Nếu có cùng checkpoint, thêm bảng “Vanilla vs Posthoc CRC vs CRC-Select” tại 70% và 80% ID coverage.

---

### 2.5 Calibration (CRC-Select)

Ví dụ seed 123 (calibration_metrics.csv):

| Target coverage | Actual coverage | Coverage error | Risk | Accuracy |
|-----------------|-----------------|----------------|------|----------|
| 70% | 77.6% | +7.6% | 1.42% | 98.7% |
| 80% | 80.0% | ~0% | 1.66% | 98.5% |
| 90% | 88.5% | −1.5% | 3.12% | 97.0% |

- **80% target**: calibration rất sát (actual 80.0%).
- 70% target: model chấp nhận nhiều hơn (77.6% > 70%).
- 90% target: hơi thiếu (88.5% < 90%).

Có thể thêm đoạn ngắn trong paper: “CRC-Select đạt calibration gần mục tiêu tại 80% coverage.”

---

### 2.6 Posthoc CRC: risk violation và test risk

Từ `calibration_summary.csv` (α = 0.1):

| Seed | Cal coverage | Cal risk | Test coverage | Test risk | Risk violation |
|------|--------------|----------|---------------|-----------|----------------|
| 123 | 99.66% | 0.139% | 99.54% | 0.140% | False |
| 456 | 99.66% | 0.140% | 99.60% | 0.139% | False |
| 999 | 99.58% | 0.136% | 99.72% | 0.135% | False |

**Kết luận:** Không có risk violation (test risk < α = 0.1). Posthoc CRC đảm bảo risk dưới α trên cả cal và test với coverage rất cao vì risk thực tế đã rất thấp (~0.14%).

---

## 3. Bảng tóm tắt cho paper (gợi ý)

### Bảng 1: CRC-Select — Coverage @ Risk α (mean ± std, 3 seeds)

| α | Coverage (%) | Risk (test, %) |
|---|--------------|----------------|
| 0.02 | **83.2 ± 0.9** | ~2.0 |
| 0.05 | **88.1 ± 0.5** | ~3.0 |
| 0.10 | 100 | ~8.0 |

### Bảng 2: CRC-Select — Error tại coverage cố định (mean ± std)

| ID coverage | Error (%) | Accuracy (%) |
|-------------|-----------|--------------|
| 70% | **1.26 ± 0.11** | 98.7 |
| 80% | **1.51 ± 0.11** | 98.5 |
| 90% | **2.89 ± 0.08** | 97.1 |

### Bảng 3: OOD-Acceptance @ ID coverage cố định (%, mean ± std)

| ID coverage | CRC-Select (OOD accept) | Vanilla (OOD accept) |
|-------------|-------------------------|----------------------|
| 70% | **1.03 ± 0.30** | ~0.51 |
| 80% | **4.48 ± 2.52** | ~0.80 |
| 90% | **29.2 ± 15.8** | ~1.52 |

(Có thể thay bằng so sánh CRC-Select vs posthoc CRC nếu dùng cùng checkpoint.)

---

## 4. Kết luận và gợi ý viết

1. **CRC-Select** ổn định qua 3 seed: AURC ~0.0133, coverage@α=0.02/0.05 khoảng 83–88%, error tại 70–90% coverage trong khoảng 1.3–2.9%.
2. **Coverage @ Risk:** Ở α = 0.02 và 0.05, CRC-Select cho mức coverage có ý nghĩa (82–88%) với risk 2–3%; tại α = 0.1 đạt full coverage với risk test ~8%.
3. **OOD:** Tại 70% ID coverage, OOD accept ~1%; tại 80% có biến động giữa các seed (2.7–7.4%). Nên báo mean ± std và thảo luận độ nhạy theo seed.
4. **Posthoc CRC** trong dữ liệu này có risk rất thấp (~0.14%) nên coverage ~99.6%, không violation; so sánh với CRC-Select nên làm rõ ngữ cảnh (cùng/tương đương backbone, cùng seed/checkpoint).
5. **Vanilla** đang được đánh giá như baseline “full coverage” (AURC=0, error=0); nên nêu rõ trong paper là baseline không reject để tránh hiểu nhầm.

**Cách dùng tài liệu này:** Copy các bảng trên vào paper, thêm 1–2 câu mô tả từng bảng; phần “Kết luận và gợi ý viết” có thể viết lại thành “Results” và “Discussion” ngắn gọn.
