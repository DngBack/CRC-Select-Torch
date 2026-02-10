# Strengths and Limitations — CRC-Select

Phân tích dựa trên codebase và kết quả đánh giá (3 seeds: 123, 456, 999; CIFAR-10, OOD: SVHN).

---

## Strengths

| # | Strength | Evidence / Ghi chú |
|---|----------|--------------------|
| 1 | **Risk được kiểm soát rõ ràng** | CRC-Select tối ưu selector cùng với ràng buộc risk ≤ α (L_risk penalty + calibration). Tại α = 0.02, 0.05 đạt coverage 83–88% với risk thực tế ~2–3%, phù hợp bài toán selective prediction có đảm bảo risk. |
| 2 | **Ổn định qua nhiều seed** | AURC, coverage@risk, error@coverage có std nhỏ (AURC 0.0133 ± 0.0009; coverage@α=0.05 là 88.1 ± 0.5%; error@80% cov 1.51 ± 0.11%) → kết quả không phụ thuộc mạnh vào random seed. |
| 3 | **Chất lượng selective prediction cao** | Tại 70–90% coverage, error chỉ 1.26–2.89%, accuracy 97–98.7%; AURC ~0.013 cho thấy trade-off risk–coverage tốt. |
| 4 | **Selector được huấn luyện CRC-aware** | Khác posthoc CRC (chỉ calibrate sau), selector học reject những mẫu làm risk khó kiểm soát → về lý thuyết cho phép coverage cao hơn tại cùng α khi so với chỉ calibrate sau. |
| 5 | **OOD an toàn ở mức coverage thực tế** | Tại 70% ID coverage, OOD accept ~1%; safety ratio cao. Phù hợp ứng dụng cần vừa coverage tốt vừa hạn chế chấp nhận OOD. |
| 6 | **Calibration gần mục tiêu** | Ví dụ tại target coverage 80%, actual coverage ~80%; có thể dùng để đảm bảo hành vi vận hành đúng mức coverage mong muốn. |
| 7 | **Pipeline đánh giá đầy đủ** | Có risk-coverage curve, coverage@risk, OOD @ fixed ID coverage, calibration metrics, multi-seed → dễ báo cáo và so sánh trong paper. |

---

## Limitations

| # | Limitation | Chi tiết / Gợi ý |
|---|------------|-------------------|
| 1 | **Chỉ đánh giá trên CIFAR-10 (+ OOD SVHN)** | Chưa có kết quả trên ImageNet, medical, hay domain khác. Khái quát hóa sang dataset lớn/khó hơn chưa được chứng minh. |
| 2 | **Biến động OOD theo seed ở coverage cao** | Tại 80% ID coverage, OOD accept dao động 2.7–7.4%; tại 90% dao động 17–48%. Cần báo mean ± std và thận trọng khi so sánh OOD giữa các method. |
| 3 | **So sánh với posthoc CRC chưa cùng điều kiện** | Posthoc CRC trong số liệu có risk ~0.14% (model rất tốt) nên coverage ~99.6%. Để so sánh công bằng cần cùng backbone, cùng seed/checkpoint, và có thể thêm setting risk cao hơn. |
| 4 | **Chi phí huấn luyện** | Alternating optimization (calibrate mỗi vài epoch + train với L_risk) tốn thời gian hơn train vanilla SelectiveNet một lần; cần warmup + recalibrate_every. |
| 5 | **Số seed còn ít** | Chỉ 3 seed (123, 456, 999). Để tuyên bố mạnh về ổn định và violation rate, nên chạy thêm seed (ví dụ ≥5). |
| 6 | **Vanilla baseline không selective** | Trong pipeline hiện tại vanilla báo AURC=0, full coverage; đóng vai trò "không reject". So sánh selective vs selective cần rõ ràng (CRC-Select vs posthoc CRC trên cùng model). |
| 7 | **Lý thuyết CRC và coverage** | Phần calibration trong code dùng quantile/approximation; guarantee conformal (e.g. risk ≤ α với xác suất 1−δ) phụ thuộc giả định và cách chọn δ. Nên nêu rõ trong paper. |
| 8 | **Hyperparameter** | α_risk, coverage target, warmup_epochs, recalibrate_every, μ, dual_lr ảnh hưởng kết quả. Chưa có ablation đầy đủ; nên báo setting và thảo luận độ nhạy. |

---

## Tóm tắt một dòng

- **Strengths:** CRC-Select kiểm soát risk tốt, ổn định qua seed, selective quality cao, selector CRC-aware, OOD an toàn ở coverage thực tế, và pipeline đánh giá rõ ràng.
- **Limitations:** Chỉ thí nghiệm CIFAR-10/SVHN, OOD dao động ở coverage cao, so sánh với posthoc cần cùng điều kiện, chi phí train cao hơn, ít seed, và cần làm rõ guarantee/ablation.
