# ERNP — Chốt thuật toán, RQs và kịch bản thực nghiệm
*(bản đã duyệt 26/09/2026: 5 RQ và tiêu chí thành công được chấp nhận; SOTA #3 = CADE; nhắm tới Computer Networks – Elsevier)*

---

## PHẦN A — THUẬT TOÁN CHỐT: ERNP (Evidence-Retracted Novelty Prior)

### A1. Bài toán
Open-world NIDS vận hành liên tục trên luồng flow:
- (i) phát hiện tấn công **zero-day** (họ chưa từng thấy);
- (ii) chịu được **benign drift**;
- (iii) chỉ có **ngân sách nhãn nhỏ** từ analyst (B nhãn/ngày);
- (iv) không bị **hấp thụ**: không được học tấn công thành "bình thường" trong quá trình thích nghi.

### A2. Ý tưởng cốt lõi (một câu)
Bộ nhận diện là một **bộ ước lượng tỉ số mật độ** giữa bằng chứng benign và một **giả định novelty có cấu trúc**. **Kênh thích nghi duy nhất** là **rút lại khối lượng của giả định** tại nơi có bằng chứng benign tin cậy. Mô hình không bao giờ tự gán nhãn (self-training) và không tự nạp dữ liệu vào bộ nhớ.

### A3. Ký hiệu và công thức
| Ký hiệu | Ý nghĩa |
|---|---|
| φ(x) | vector đặc trưng flow = đặc trưng per-flow + đặc trưng ngữ cảnh thời gian (đếm/bytes theo src, dst, cặp trong 10 s và 120 s; fan-out) |
| P_t | tập bằng chứng benign tại ngày t: anchor benign + nhóm cảnh báo được analyst xác nhận benign |
| A_t | tập bằng chứng tấn công: tấn công đã biết ở giai đoạn anchor + nhóm cảnh báo được xác nhận tấn công |
| q_0 | **giả định novelty cấu trúc**: phân phối của flow benign bị *tráo từng đặc trưng với xác suất ρ* sang một flow benign khác (giữ phân phối biên, phá quan hệ giữa các đặc trưng) |
| κ(x; E_t) | độ phủ của bằng chứng: κ = exp(−d(x, E_t)² / 2ε²), với d = khoảng cách tới flow benign đã xác nhận gần nhất |
| ε | độ hạt tự nhiên của benign = phân vị 90 của khoảng cách 1-NN trong tập tham chiếu (tính từ dữ liệu, không tinh chỉnh) |

**Luật rút lại giả định (cơ chế lõi):**
```
q_{t+1}(x) = q_t(x) · (1 − κ(x; E_t))          — chỉ thay đổi khi có bằng chứng TIN CẬY
```

**Bộ nhận diện:**
```
g_t = argmin_g  E_{P_t}[ℓ(g,0)]·½ + E_{A_t}[ℓ(g,1)]·½ + E_{q_t}[ℓ(g,1)]·½
s_t(x) = logit g_t(x);   cảnh báo nếu s_t(x) > τ,  τ = ngưỡng 1% FPR trên holdout benign của anchor
```

Với ℓ là log-loss, g_t hội tụ về p_B / (p_B + ½ p_A + ½ q_t), nên điểm s_t chính là log-tỉ số mật độ.

### A4. Thủ tục vận hành (mỗi ngày t)
1. **Dự đoán** mọi flow trong ngày bằng g_t (nhân quả).
2. **Gom cảnh báo** của ngày thành tối đa B nhóm (k-means trong φ); analyst gán nhãn **một đại diện mỗi nhóm**. Đây là quy trình triage nhóm cảnh báo phổ biến ở SOC.
3. Nhóm được xác nhận benign → thêm vào P và E; nhóm tấn công → thêm vào A.
4. **Rút lại** q theo công thức ở A3, huấn luyện lại g (GBM 300 cây, lr 0,1), giữ nguyên τ.

### A5. Siêu tham số (cố định cho MỌI dataset, lấy từ mock test, không tinh chỉnh riêng)
ρ = 0,3; |mẫu giả định| = |P|; B = 200 nhãn/ngày; ε = phân vị 90 của khoảng cách 1-NN; τ ở 1% FPR.

### A6. Tính chất lý thuyết (sẽ viết chứng minh đầy đủ)
- **P1 — Không hấp thụ:** nếu không có nhãn tin cậy mới, q và g không đổi. Vùng chấp nhận **không bao giờ tự nở ra**, bất kể kẻ tấn công gửi bao nhiêu flow và trong bao lâu. Mở rộng trực tiếp từ Theorem 1 của QCA.
- **P2 — Hội tụ cục bộ:** tại vùng có bằng chứng benign dày đặc, q_t → 0, nên g_t → benign và FPR cục bộ → 0.
- **P3 — Bảo toàn zero-day:** tại vùng cách mọi bằng chứng benign hơn cε, q_t ≥ (1 − e^{−c²/2}) q_0. Độ nhạy novelty mất đi không quá một hệ số có thể tính trước.

### A7. Khác biệt với công trình gần nhất
| Công trình | Cơ chế | ERNP khác ở đâu |
|---|---|---|
| Sipple, ICML 2020 (negative sampling) | Phân loại dữ liệu thật với mẫu âm đồng nhất, **tĩnh** | Giả định có cấu trúc (phá quan hệ) và **được rút lại theo thời gian bằng bằng chứng** |
| SSF (INFOCOM 2025), NetSight, SOUL | Thích nghi bằng **pseudo-label / self-training** | Không self-training → có P1; mock test cho thấy SSF và AE tự thích nghi bị hấp thụ |
| OWAD (NDSS 2023) | Fine-tune AE với mẫu giải thích shift được gán nhãn | Không fine-tune detector một lớp; thích nghi bằng rút lại giả định |
| AAD (Das et al. ICDM 2016) | Học trọng số thành phần detector từ phản hồi, dữ liệu **tĩnh** | Cập nhật *phân phối novelty*, trong bối cảnh drift liên tục |
| DGAM, NNG-Mix | Chọn hoặc lọc pseudo-anomaly **một lần** | Rút lại liên tục, kèm bảo đảm P1–P3 |
| QCA (phần đầu dự án) | Admission có chứng nhận, không có khả năng nhận diện mạnh | ERNP kế thừa nguyên lý "chỉ thích nghi qua bằng chứng tin cậy" để có P1 |

---

## PHẦN B — CÂU HỎI NGHIÊN CỨU (RQs)

| RQ | Câu hỏi | Loại |
|---|---|---|
| **RQ1** | Trong điều kiện vận hành thực tế (stream nhân quả, benign drift, cùng ngân sách nhãn), ERNP có **phát hiện zero-day tốt hơn 5 SOTA** không? | **So sánh 5 SOTA** |
| **RQ2** | Khi kẻ tấn công lợi dụng chính quá trình thích nghi (luộc ếch, laundering qua server), ERNP có **bền vững hơn 5 SOTA** không? | **So sánh 5 SOTA** |
| **RQ3** | ERNP **hiệu quả về nhãn** đến mức nào so với các SOTA dùng nhãn khi ngân sách thay đổi? | So sánh (nhóm SOTA dùng nhãn) |
| **RQ4** | Thành phần nào của ERNP tạo ra hiệu năng, và các tính chất P1–P3 có **đúng trên dữ liệu thật** không? | Ablation + kiểm chứng lý thuyết |
| **RQ5** | ERNP **nhạy** với siêu tham số tới đâu, và có **triển khai được** không (thông lượng, bộ nhớ, thời gian huấn luyện lại)? | Độ nhạy + chi phí |

### 5 SOTA dùng cho RQ1, RQ2 (code gốc, venue hàng đầu)
| # | SOTA | Venue | Họ phương pháp | Nhãn | Trạng thái |
|---|---|---|---|---|---|
| 1 | **SSF** | IEEE INFOCOM 2025 | Học liên tục + chọn/quên mẫu | anchor + B/ngày | ✅ đã chạy |
| 2 | **OWAD** | NDSS 2023 | Phát hiện, giải thích và thích nghi shift | B/ngày | ✅ đã chạy |
| 3 | **CADE** | USENIX Security 2021 | Contrastive autoencoder + phát hiện drift/unknown theo khoảng cách MAD tới tâm lớp | anchor | ✅ port PyTorch; tái lập Table 3 của paper trên dữ liệu IDS2018 của tác giả (F1 0,958 so với 0,96) |
| 4 | **Trident** | ACM WWW 2024 | Phát hiện unknown chi tiết (AE mỗi lớp + EVT) | anchor | ✅ đã chạy |
| 5 | **MemStream** | ACM WWW 2022 | Phát hiện bất thường streaming có bộ nhớ | 0 | ✅ đã chạy (+ bản sửa chuẩn hóa) |

- **Bổ sung** (tham khảo, không bắt buộc): KitNET (NDSS 2018).
- **Không dùng được:** FOSS (IEEE/ACM ToN 2024), vì repo công khai thiếu mã lõi. Sẽ ghi rõ trong paper.

---

## PHẦN C — KỊCH BẢN THỰC NGHIỆM TỪNG RQ

### RQ1 — Phát hiện zero-day so với 5 SOTA
- **Dataset:** CIDDS-001 (chính), NF-ToN-IoT-v3, CICIDS2017-improved, CSE-CIC-IDS2018-improved.
- **Giao thức zero-day theo thời gian:** anchor = giai đoạn đầu; họ tấn công trong anchor là "đã biết", **mọi họ xuất hiện sau là zero-day**.
  - CIDDS: 2 ngày; ToN: 1 ngày.
  - CIC17: thứ Hai–thứ Tư (đã biết: Patator, DoS); zero-day: web, infiltration, botnet, portscan, DDoS.
  - CIC18: 14–16/02 (đã biết: SSH brute, DoS); zero-day: DDoS, web, infiltration, botnet.
- **Hai giao thức:**
  - (a) **Nhân quả** (chính) cho mọi phương pháp: dự đoán trước, học sau.
  - (b) **Giao thức gốc** (phụ): chỉ SSF có giao thức gốc khác (học nhãn trong ngày rồi mới dự đoán). Báo cáo SSF-gốc và ERNP chạy cùng giao thức đó để so ngang.
- **MemStream:** báo cáo bản gốc (chính). Bản giữ cố định bước chuẩn hóa (MemStream-fixnorm) chỉ ghi chú như phân tích độ nhạy, vì bản gốc sụp đổ do chuẩn hóa theo nội dung bộ nhớ trên dữ liệu có nhiều flow trùng.
- **Công bằng:** cùng đặc trưng, cùng anchor, cùng ngân sách B = 200 nhãn/ngày cho mọi phương pháp dùng nhãn. SOTA dùng siêu tham số gốc. Ngưỡng: 1% FPR trên holdout cho phương pháp chấm điểm; phương pháp có luật quyết định riêng (SSF p > 0,5, Trident EVT) giữ luật gốc.
- **Chỉ số:**
  - chính: **recall zero-day @ FPR 1%**, **F1**, **FPR**;
  - phụ: AUROC, AUPRC, recall từng họ, FPR theo ngày, số nhãn thực dùng.
  - **Recall zero-day @ FPR 1%** được đo tại **cùng một điểm vận hành cho mọi phương pháp**: ngưỡng = phân vị 99 của điểm số của chính phương pháp đó trên flow benign của stream. Nhờ vậy không phụ thuộc luật quyết định riêng (SSF p > 0,5, EVT của Trident). F1/FPR/recall zero-day tại ngưỡng riêng của từng phương pháp vẫn được báo cáo song song.
  - (Riêng ToN: không claim chỉ bằng F1, vì cảnh báo tất cả đã cho F1 ≈ 0,70.)
  - **Kiểm tra lộ nhãn:** baseline `new_ip` (cảnh báo mọi flow có IP nguồn chưa từng thấy trong anchor) được báo cáo cạnh RQ1/RQ2.
- **Thống kê:** 5 seed (mỗi seed là một mẫu flow khác lấy từ dữ liệu gốc). Kiểm định chính: **Wilcoxon một phía (ERNP > SOTA) trên toàn bộ n = 20 cặp (dataset × seed)**, cộng Friedman + Nemenyi (biểu đồ critical difference). Theo từng dataset chỉ báo mean ± std và số seed thắng. *(Lý do: với 5 seed, Wilcoxon hai phía trên từng dataset không bao giờ đạt p < 0,05, vì p nhỏ nhất = 0,0625.)*
- **Tiêu chí thành công:** ERNP đứng hạng 1 trung bình; recall zero-day hơn mọi SOTA có ý nghĩa thống kê trên ≥ 3/4 dataset; F1 không thua SOTA tốt nhất quá 1 điểm trên bất kỳ dataset nào.
  - *Cách chấm (chốt sau bản xem trước seed 0, trước khi có kết quả seed 1–4; `exp/check_criteria.py`):*
    - Xếp hạng chỉ gồm ERNP và 5 SOTA theo giao thức nhân quả, tính trên F1 và recall zero-day @ FPR 1%.
    - "Có ý nghĩa" trên một dataset nghĩa là Wilcoxon một phía chính xác với n = 5 cho p < 0,05, tương đương ERNP thắng cả 5 seed (p = 1/32).
    - Với F1, báo cáo cả bản so với mọi SOTA lẫn bản loại các điểm vận hành gần như cảnh báo tất cả (FPR > 50%).

### RQ2 — Bền vững trước tấn công vào quá trình thích nghi, so với 5 SOTA
- **Kịch bản tiêm có kiểm soát trên CIDDS:** host bị chiếm nhích dần từ hành vi bình thường của nó sang payload thật trong 7 ngày, rồi tấn công 2 ngày.
  - Biến thể: {initiator, responder-laundering} × payload {DoS (đã biết), bruteForce (zero-day)} = 4 kịch bản.
  - Thêm biến thể về **tốc độ ramp** (3, 7, 14 ngày) và **tốc độ gửi** (10, 30, 100 flow/giờ).
- **Hấp thụ tự nhiên:** DoS trong CIDDS (đã quan sát: AE tự thích nghi 0,975 → 0,34).
- **Chỉ số:** recall trên đợt tấn công cuối (payload); **ngày bị hấp thụ** = ngày đầu tiên mà từ đó recall trên luồng ramp luôn < 10% (−1 nếu không bao giờ bị hấp thụ); FPR; F1 tổng.
- **Quy mô:** 4 kịch bản chính × 5 seed cho ERNP và cả 5 SOTA. Phần quét (ramp {3, 7, 14} ngày × tốc độ {10, 30, 100} flow/giờ) chỉ chạy trên kịch bản initiator-bruteForce, cho ERNP và các SOTA tự thích nghi (SSF, OWAD, MemStream), 3 seed.
- **Kiểm chứng P1:** chạy ERNP **không có nhãn online**. Vùng chấp nhận (đo trên lưới probe) không được nở ra.
- **Tiêu chí:** ERNP giữ recall payload ≥ 90% mức khi không bị tấn công trong mọi kịch bản; ít nhất 3/5 SOTA bị hấp thụ trong ít nhất một kịch bản.

### RQ3 — Hiệu quả nhãn
- **Phương pháp:** ERNP, SSF, OWAD (dùng nhãn online); CADE, Trident và MemStream làm mốc không có nhãn online.
- **Ngân sách B/ngày:** {0, 25, 50, 100, 200, 400} (B = 200 lấy từ RQ1; B = 0 chỉ áp dụng cho ERNP, lấy từ `ernp_B0` của RQ4, vì SSF/OWAD không có nhãn online thì không còn là phương pháp thích nghi).
- **Dataset:** ToN (drift mạnh) và CIC18 (nhiều họ zero-day), 3 seed (seed 0–2, khớp ước tính chi phí ở D3).
- **Chỉ số:** đường cong recall zero-day @ 1% FPR và F1 theo B; diện tích dưới đường cong; **số nhãn cần để đạt hiệu năng của SSF tại B = 200**.
- **Tiêu chí:** ERNP đạt hiệu năng của SSF (B = 200) với B ≤ 100.

### RQ3b — Nhịp cập nhật (phân tích bổ sung, thêm ngày 27/09/2026)
- **Lý do:** thêm sau bản xem trước RQ1 (seed 0–1).
  - Trên ToN, mỗi ngày có một tập benign gần như mới hoàn toàn. Ví dụ ngày 29/04: ERNP báo nhầm 95% trong 21 nghìn flow benign.
  - Với giao thức nhân quả cập nhật theo ngày, mọi phương pháp phải chấm cả ngày trước khi có nhãn.
- **Thiết kế:**
  - Cập nhật mỗi 24h (RQ1), 4h hoặc 1h, với **cùng tổng ngân sách 200 nhãn/ngày** chia đều chính xác cho các chu kỳ trong ngày.
  - Phương pháp: các phương pháp có dùng nhãn online (ERNP, SSF-nhân quả, OWAD).
  - Dataset: cả 4, seed 0–2.
  - **Giới hạn do chi phí:** SSF huấn luyện lại 20 epoch trên toàn bộ buffer mỗi lần cập nhật, khoảng 5 phút/lần trên ToN. Với chu kỳ 1h, một lần chạy mất khoảng 8 giờ.
    - SSF-1h chỉ chạy trên ToN seed 0.
    - SSF-4h chạy đủ 4 dataset × 3 seed.
    - ERNP và OWAD chạy cả 1h và 4h ở mọi nơi.
    - So sánh cùng chu kỳ giữa cả 3 phương pháp là ở mức **4h**.
  - **Sửa tương thích cho SSF:** code gốc crash (ghép tensor 0 chiều) khi đúng một mẫu mới còn lại để gán nhãn giả. Trường hợp này chỉ xảy ra với lô dưới 1 ngày. Wrapper nâng tensor 0 chiều lên 1 chiều và không làm đổi bất kỳ lời gọi hợp lệ nào.
  - CADE, Trident và MemStream không dùng nhãn online nên không bị ảnh hưởng.
- **Chỉ số:** F1, recall zero-day @ FPR 1% (điểm vận hành chung), FPR, số nhãn thực dùng.
- **Không có tiêu chí thành công đặt trước.** Kết quả được báo cáo như phân tích bổ sung, không thay thế RQ1. RQ1 vẫn dùng chu kỳ ngày như thiết kế ban đầu.
- **Code:** `ernp.py --period h`; hậu tố `_P{h}` cho SOTA trong `baselines/sota.py`.
  - Với h = 24, kết quả trùng khớp tuyệt đối với bản trước khi thêm tham số (đã kiểm chứng trên ERNP và OWAD).
  - `run_jobs.py cad`.

### RQ4 — Ablation và kiểm chứng lý thuyết
| Trục | Biến thể |
|---|---|
| Giả định novelty | không có / đồng nhất (Sipple) / tích biên toàn phần / **tráo từng phần ρ** (ERNP) |
| Rút lại | không / cứng (bán kính ε) / **mềm κ** (ERNP) |
| Bằng chứng | chỉ đại diện / cả nhóm cảnh báo (ERNP) |
| Đặc trưng | per-flow / + ngữ cảnh (ERNP) |

- **Kiểm chứng P2:** FPR cục bộ quanh các vùng có bằng chứng giảm dần theo số nhãn.
- **Kiểm chứng P3:** recall zero-day theo khoảng cách tới bằng chứng benign gần nhất, so với cận dưới lý thuyết.
- **Dataset:** cả 4, 5 seed.

### RQ5 — Độ nhạy và chi phí triển khai
- **Quét tham số** (quét một chiều quanh cấu hình mặc định; 3 seed; trên CIDDS, ToN, CIC18): ρ ∈ {0,1; 0,2; 0,3; 0,5}; khối lượng giả định ∈ {0,25; 0,5; 1; 2}×|P|; ε × {0,5; 1; 2}; bộ phân loại {GBM, RF}.
- **Chi phí:** thông lượng suy luận (flow/giây), bộ nhớ, thời gian huấn luyện lại mỗi ngày. So với 5 SOTA trên cùng máy (CPU 12 lõi, RTX 3050).
- **Tiêu chí:** hiệu năng thay đổi dưới 2 điểm F1 trong vùng tham số hợp lý; suy luận ≥ 100k flow/giây trên CPU.

---

## PHẦN D — THIẾT LẬP CHUNG
- **Dữ liệu:** stream theo thời gian, **khoảng 300k flow mỗi seed** (CIDDS, ToN, CIC17) và **400k** (CIC18), tức 1–14% dữ liệu gốc; mỗi seed là một mẫu khác. (Paper CADE cũng lấy mẫu 10% IDS2018 và cho thấy tỉ lệ lấy mẫu không làm đổi kết quả.) Đặc trưng ngữ cảnh luôn tính trên **toàn bộ** dữ liệu trước khi lấy mẫu. *Quy mô 1M flow ban đầu bị loại vì ước tính chi phí vượt 150 giờ máy trên laptop (SSF chiếm phần lớn).*
- **Nhãn "Attempted"** (CIC improved) gộp vào BENIGN theo khuyến nghị của Distrinet.
- **Kiểm soát lộ nhãn qua IP:** luôn báo cáo baseline `new_ip`; không dùng IP làm đặc trưng.
- **5 seed** (lấy mẫu, khởi tạo, k-means); báo cáo mean ± std.
- **Tái lập:** code, cấu hình, log của mọi lần chạy; SOTA chạy qua wrapper gọi code gốc (không sửa thuật toán; mọi sai lệch như epoch_1 = 20 của SSF hay fallback EVT của Trident đều ghi rõ).

## PHẦN D2 — ÁNH XẠ SANG CODE (`exp/`)
| Thành phần | File |
|---|---|
| Giao thức chung (load, anchor, hiệu chỉnh ngưỡng, lưu kết quả) | `exp/common.py` |
| **ERNP** (thuật toán đề xuất, mọi công tắc ablation) | `exp/ernp.py` |
| 5 SOTA (MemStream, OWAD, SSF, Trident, CADE) | `exp/baselines/sota.py`, `exp/baselines/cade_port.py` |
| Dựng stream, đặc trưng ngữ cảnh, tiêm tấn công | `exp/data/*.py` |
| Chỉ số, gộp seed, kiểm định | `exp/evaluate.py` |
| Chấm tiêu chí thành công; kiểm chứng P2/P3; baseline lộ nhãn `new_ip` | `exp/check_criteria.py`, `exp/analysis_p23.py`, `exp/baselines/sanity.py` |
| Chạy toàn bộ báo cáo | `exp/report_all.sh` → `exp/reports/` |
`exp/ernp.py` đã được kiểm chứng: tái tạo **chính xác từng phần tử** kết quả ERNP của mock test (sai khác lớn nhất = 0).

## PHẦN D3 — ƯỚC TÍNH CHI PHÍ (máy hiện tại: 12 CPU, RTX 3050)
| RQ | Số lần chạy | Thời gian ước tính |
|---|---|---|
| RQ1 | 4 dataset × 5 seed × (ERNP ×2 giao thức + 5 SOTA + SSF-gốc) | ≈ 25–30 giờ (SSF khoảng 35 phút/lần) |
| RQ2 | 4 kịch bản × 5 seed × 7 phương pháp + phần quét | ≈ 20 giờ |
| RQ3 | 2 dataset × 5 ngân sách × 3 seed × (ERNP, SSF, OWAD) | ≈ 12 giờ |
| RQ4 + RQ5 | chỉ ERNP (khoảng 3–10 phút/lần) | ≈ 8 giờ |
| **Tổng** | | **≈ 65–70 giờ máy**, chạy nền theo từng RQ |

## PHẦN E — RỦI RO CÒN MỞ
1. **Novelty:** phép rút lại tự nó đơn giản. Đóng góp phải được bảo vệ bằng P1–P3 và bằng kết quả RQ2.
2. **FPR trên dữ liệu benign ổn định** cao hơn SSF một chút (0,3% so với 0,05–0,2%).
3. **CADE** gốc chạy trên TensorFlow 1.12 (không tương thích Python 3.12) → port kiến trúc, loss và luật phát hiện sang PyTorch, kiểm tra lại trên dataset IDS2018 đi kèm repo để xác nhận port trung thực.
   - **Đã xác nhận 28/09/2026:** port được đánh giá bằng đúng chỉ số của paper CADE (Table 3, `cade/evaluate.py`): các mẫu có điểm > MAD 3,5, xếp theo khoảng cách tới tâm gần nhất, lấy P/R/F1 tại số mẫu kiểm tra tốt nhất.
   - Kết quả: P/R/F1 = Infiltration 1,000 / 1,000 / 1,000; Hulk 0,999 / 0,998 / 0,998; SSH 0,967 / 0,803 / 0,877.
   - Trung bình **0,989 / 0,934 / 0,958**, so với paper 0,98 / 0,93 / 0,96, tức là **tái lập được**.
   - Con số 0,835 trước đây là precision tại ngưỡng MAD 3,5. Đây là luật quyết định mà thực nghiệm của mình dùng, vì triển khai thực tế không có điểm cắt oracle. Nó không phải là chỉ số của paper.
4. Kết quả mock hiện tại chỉ có 1 seed và dữ liệu đã lấy mẫu.
5. **Độ trung thực của SOTA (đã kiểm tra 28/09/2026):**
   - **CADE:** tái lập Table 3 (xem mục 3).
   - **OWAD:** chạy lại demo NIDS của tác giả (Kyoto-2006, 2007 → 2011) bằng đúng đường code của wrapper (`exp/baselines/validate_owad.py`). AUC 0,9473 / 0,8069 / 0,8145 so với notebook của tác giả 0,9472 / 0,8074 / 0,8131, tức là **tái lập được**.
     - Baseline "retraining" trong notebook cho 0,844 so với 0,790. Đây không phải OWAD, chỉ là baseline minh họa của notebook.
   - **SSF:** chạy thêm với cấu hình của tác giả (epoch_1 = 180; cửa sổ 20.000 flow).
   - **MemStream:** chạy thêm bản fixnorm.
   - **Diễn giải cho paper:** các SOTA xếp hạng tốt (AUROC của CADE, Trident, SSF là 0,96–0,99 trên CIC và CIDDS) nhưng hỏng ở điểm vận hành khi có drift. Cụ thể: FPR của CADE 7–21% và Trident 15–28% trên CIC; SSF nhân quả bỏ sót zero-day trên CIC17 (recall 0,31); OWAD có AUROC 0,66–0,68 trên CIC nên gần như không phát hiện gì ở FPR 1%. Đây chính là khoảng trống mà ERNP nhắm tới.


---

## PHẦN F — KẾT QUẢ CUỐI (cập nhật 01/10/2026)
Nguồn số liệu: `exp/reports/criteria.txt`, `criteria_rq1_final.txt`, `rq2lb.txt`, `noise.txt`, `scale.txt`. Một phần RQ2/RQ3 chạy trên server (RTX A5000). Chạy chéo cho thấy lệch rất nhỏ: F1 ≤ 0,003, payload recall và ngày hấp thụ giống hệt.

| RQ | Kết quả | Claim được phép viết |
|---|---|---|
| RQ1 | F1 trung bình 0,867 so với 0,671 (SSF). SOTA-best: 0,907 so với 0,775. Hạng 1 về F1, MCC, recall zero-day. Theo từng dataset: 18/20 cặp có CI 95% > 0; thua SSF trên CIDDS (−0,009) và Trident trên ToN. Friedman mức dataset (n = 4): p = 0,08 | Đứng đầu ổn định, **không** thắng trên mọi dataset. Dùng effect size + CI theo dataset, không dùng p gộp 20 cặp |
| RQ1, quy mô đầy đủ (CIC17, 2,1M flow, 3 seed) | F1: CADE 0,959, ERNP 0,940. Recall zero-day @1% FPR: ERNP 0,859, CADE 0,789. Còn lại ≤ 0,86 | Ngang CADE, rẻ hơn nhiều |
| RQ2 | ERNP không bị hấp thụ; payload giữ ≥ mức sạch. Không SOTA nào bị hấp thụ ở 200 nhãn/ngày. **Với 25 nhãn/ngày**, payload zero-day: ERNP 0,75, SSF 0,63. Đối chứng tự học: 0,74 → 0,07 | Bền vững ở chi phí nhãn thấp; cơ chế bằng chứng tin cậy là nguyên nhân (đối chứng nhân quả) |
| RQ3 | ERNP 25 nhãn/ngày > SSF 200 và 400 nhãn/ngày (CIC18 0,905 so với 0,860/0,873; ToN 0,665 so với 0,378/0,510) | Hiệu quả nhãn gấp khoảng 8 lần |
| RQ4 | Prior có cấu trúc và nguồn bằng chứng tin cậy là then chốt. Dạng rút lại mềm hay cứng không quan trọng | Đóng góp nằm ở prior + bằng chứng tin cậy |
| RQ5 | 104–286k flow/giây trên CPU. Độ nhạy nhỏ trên CIDDS/CIC18, lớn trên ToN | Triển khai được |
| Nhãn sai | ERNP suy giảm khi analyst sai ≥ 10% (CIC18 0,910 → 0,840 so với SSF 0,861). SSF gần như không đổi | **Hạn chế phải ghi rõ**; hướng tương lai: rút lại khi có nhiều xác nhận |

**Còn mở:** dataset held-out (#5); chứng minh P1–P3 (phát biểu lại P1 có nội dung, xét cả trường hợp nhãn sai).
