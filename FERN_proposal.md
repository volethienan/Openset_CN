# FERN: Factorized Evidence-Retracted Novelty
*Đề xuất cơ chế mới trên nhánh `improve-ernp` (01/10/2026). Mọi chẩn đoán dưới đây chỉ dùng dữ liệu phát triển: seed 0–1. Seed 2–4 và CIC17 được giữ lại để kiểm tra cuối.*

> **Cập nhật sau kiểm tra thử (mock) trên seed 0–1, ngày 01/10/2026: KHÔNG ĐẠT, chưa đưa lên server.**
> - `exp/diagnose_coverage.py`: sau khi tính cả bằng chứng đã xác nhận, chỉ 5% báo động giả trên CIC18/CIDDS là "mới về giá trị biên". Vì vậy kênh biên của FERN gần như không có tác dụng ở hai dataset này.
> - Trên ToN, khoảng 90% báo động giả rơi vào những ngày chưa thể có bằng chứng: ngày stream đầu tiên và thứ hai, và ngày 29/04 với một chế độ benign hoàn toàn mới. Mọi cơ chế học từ bằng chứng theo thứ tự nhân quả đều không sửa được những ngày này.
> - Biến thể "rút lại có hình dạng theo bằng chứng" (`--retraction shaped`) cho kết quả trên ToN: F1 0,627 → 0,650, MCC 0,194 → 0,250. Nhưng F1 trên CIC18 giảm 0,904 → 0,893. Không đủ để thắng OWAD về MCC trên ToN (0,357) hay SSF về F1 trên CIDDS (0,989).

> **Hướng B (học trước trôi dạt từ anchor), cũng KHÔNG ĐẠT.**
> - `exp/diagnose_anchor_drift.py`: anchor của ToN có trôi dạt (FPR nửa sau 9,3%), nhưng các flow trôi dạt đó phủ được 0% báo động giả của hai ngày stream đầu.
> - `exp/diagnose_drift_directions.py`: hướng trôi dạt trong anchor ToN không khớp với hướng trôi dạt trong stream (cos 0,38, không trùng đặc trưng nào trong top 5). Hướng trôi dạt anchor lại giống hướng tấn công zero-day hơn (cos 0,60). Học hình dạng trôi dạt từ anchor vì thế sẽ rút lại prior theo hướng tấn công, tức là có hại.
> - Kết luận: báo động giả những ngày đầu trên ToN không suy ra được từ bất kỳ dữ liệu có nhãn nào có sẵn tại thời điểm đó, cả về vị trí lẫn hướng. Cách giảm duy nhất là có nhãn sớm hơn trong ngày (cadence theo giờ: F1 0,729).

---

## 1. ERNP thua ở đâu, và vì sao (chẩn đoán có số liệu)

### 1.1. Bảng thua
| Nơi ERNP không đứng đầu | Đối thủ thắng | Chênh lệch | Metric |
|---|---|---|---|
| NF-ToN-IoT-v3 | Trident (cảnh báo gần như mọi flow) | −0,088 | F1 |
| NF-ToN-IoT-v3 | OWAD / GBM+AL | 0,357 / 0,529 so với 0,176 | MCC |
| CIDDS-001 | SSF | −0,009 | F1 |
| CIDDS-001 | CADE | 0,685 so với 0,657 | Recall zero-day @1% FPR |
| CIC-IDS2017 | kNN | −0,010 | F1 |
| CIC-IDS2017 (đầy đủ) | CADE | −0,019 | F1 |
| CSE-CIC-IDS2018, nhãn sai 10% | SSF | 0,840 so với 0,861 | F1 |

### 1.2. Ba chẩn đoán
**D1 — Trôi dạt benign là trôi dạt biên, không phải trôi dạt phụ thuộc** (`exp/diagnose_drift.py`).
- Cách đo: dùng bộ phân loại hai mẫu để phân biệt benign của ngày *d* với ngày *d+1*, trên (a) dữ liệu thô và (b) dữ liệu đã xáo độc lập từng cột trong mỗi ngày. Cách xáo này giữ nguyên phân phối biên và phá bỏ mọi phụ thuộc giữa các đặc trưng.

| Dataset | AUC thô | AUC khi chỉ giữ biên | AUC tốt nhất của 1 đặc trưng | Tỉ phần do biên |
|---|---|---|---|---|
| ToN | 0,966 | 0,976 | 0,862 | ≈ 1,0 |
| CIDDS | 0,738 | 0,747 | 0,613 | ≈ 0,95 |
| CIC17 | 0,659 | 0,681 | 0,537 | ≈ 1,0 |
| CIC18 | 0,592 | 0,670 | 0,526 | ≈ 1,0 |

→ **Toàn bộ trôi dạt benign giữa các ngày nằm ở phân phối biên.** Cấu trúc phụ thuộc giữa các đặc trưng của benign gần như không đổi. ToN trôi cực mạnh (0,97), và chỉ một đặc trưng đã phân biệt được hai ngày với AUC 0,86.

**D2 — Báo động giả của ERNP là flow có giá trị biên mới** (`exp/diagnose_failures.py`).
Tỉ lệ flow có ít nhất một đặc trưng nằm ngoài dải benign tin cậy (phân vị 0,5–99,5% của anchor):

| Dataset | Trong báo động giả | Trong flow benign không bị báo |
|---|---|---|
| ToN | 81% | 44% |
| CIC17 | 69% | 11% |
| CIC18 | 45% | 16% |
| CIDDS | 19% | 6% |

→ Báo động giả là **tính mới về giá trị** (một đặc trưng mang giá trị lạ), không phải tính mới về tổ hợp.

**D3 — Hai lỗi cấu trúc của ERNP giải thích D1 và D2.**
- **(a) Prior không có vùng "mới về biên".** Prior tráo-từng-phần chỉ dùng giá trị benign đã biết, nên q₀ gần như bằng 0 ngoài miền benign. Ở đó, cả b (benign) và q đều nhỏ, và GBM ngoại suy tùy ý. Hệ quả:
  - benign mang giá trị mới thì bị báo nhầm (ToN, D2);
  - tấn công "xa" benign thì có lúc bị chấm điểm thấp (CIDDS: recall zero-day giảm khi xa bằng chứng, xem hình P3). Đây là lý do kNN, vốn đo khoảng cách, thắng trên CIC17.
- **(b) Phạm vi tác động của nhãn sai hình dạng.** Rút lại theo quả cầu Euclid bán kính ε quanh từng flow được xác nhận. Nhưng theo D1, một chế độ benign mới là một **dịch chuyển biên toàn cục**: mọi flow của chế độ đó mang cùng giá trị mới ở một số đặc trưng, với đủ loại tổ hợp. Muốn phủ hết bằng các quả cầu ε thì cần rất nhiều nhãn. Đó là lý do trên ToN, cập nhật theo ngày cho FPR 42%, và phải chia nhỏ nhãn theo giờ mới đỡ.

**Nguyên nhân chung:** ERNP coi mọi tính mới là một loại, và cho bằng chứng một phạm vi tác động duy nhất (cục bộ). Dữ liệu lại cho thấy tính mới có **hai kiểu với hai cấu trúc khác nhau**:
- tính mới về **biên**: benign trôi dạt và tấn công khối lượng lớn;
- tính mới về **phụ thuộc**: tấn công pha trộn giá trị quen thuộc theo tổ hợp lạ.

Mỗi kiểu cần một phạm vi tác động riêng của bằng chứng.

---

## 2. Nguyên lý mới: phạm vi tác động của bằng chứng phải theo cấu trúc của tính mới

> **Nguyên lý (Factorized Evidence Extent).** Theo định lý Sklar, mật độ của benign tách được thành các phân phối biên và một copula: p(x) = c(F₁(x₁),…,F_d(x_d)) · Π_j f_j(x_j). Vì vậy tính mới cũng tách thành hai phần: tính mới biên (giá trị lạ trên từng đặc trưng) và tính mới phụ thuộc (tổ hợp lạ của các giá trị quen).
> Khi analyst xác nhận một nhóm benign, xác nhận đó chứng minh hai điều khác nhau:
> - **Các giá trị của nhóm là hợp lệ trên từng đặc trưng.** Điều này đúng *cho mọi flow* mang các giá trị đó, nên rút lại tính mới biên **toàn cục theo từng đặc trưng**.
> - **Tổ hợp cụ thể của nhóm là hợp lệ.** Điều này chỉ đúng *quanh chính tổ hợp đó*, nên rút lại tính mới phụ thuộc **cục bộ**, như ERNP.
>
> Không có dữ liệu không nhãn nào đi vào mô hình, nên tính chất P1 của ERNP được giữ nguyên.

Nguyên lý này là **một cơ chế duy nhất**: một toán tử bằng chứng tác động lên prior đã được phân rã. Nó **không phải ERNP ghép thêm module**. ERNP là trường hợp đặc biệt của FERN, khi prior chỉ có thành phần phụ thuộc và bằng chứng chỉ tác động cục bộ. Kiểm tra ablation ở mục 5 được thiết kế để tách riêng đúng thành phần mới này.

---

## 3. Công thức

**Bằng chứng tin cậy:**
- P_t: flow benign đã xác nhận (anchor + nhóm được xác nhận benign);
- A_t^{(k)}: flow đã xác nhận thuộc họ tấn công k, gồm các họ đã biết và các họ mới analyst gán tên;
- E_t ⊆ P_t: phần benign đến từ xác nhận online.

**Biên tin cậy.** Với mỗi đặc trưng j, lấy mật độ biên có làm trơn của các giá trị benign đã xác nhận: f̂_{j,t}(v) = KDE_{h_j}({x_j : x ∈ P_t}). Mật độ này **chỉ thay đổi khi có nhãn**.

**Prior tính mới phân rã:** q_t = ½ q_t^{dep} + ½ q_t^{mar}

- **Phụ thuộc** (như ERNP, trong không gian hạng u_j = F̂_{j,t}(x_j)):
  q₀^{dep}: lấy flow benign x và flow cho x′, rồi tráo mỗi đặc trưng với xác suất ρ.
  Rút lại cục bộ: r_t^{dep}(x) = 1 − exp(−d_u(x, E_t)² / 2ε²).

- **Biên** (thành phần mới): lấy flow benign x, chọn ngẫu nhiên một tập con J gồm |J| ~ Geom(1/2) đặc trưng, rồi thay mỗi x_j, j ∈ J, bằng một giá trị v rút từ
  ν_{j,t}(v) ∝ g_j(v) · (1 − κ_j(v; P_t)),  với κ_j(v; P_t) = min{1, f̂_{j,t}(v) / f̂_{j,0}^{max}}.
  - g_j là mật độ tham chiếu rộng trên thang log của đặc trưng j, phủ \[min − Δ_j, max + Δ_j\] với Δ_j bằng một số lần độ trải của benign.
  - Rút lại **toàn cục theo đặc trưng**: khi benign đã xác nhận có mật độ cao tại giá trị v, khối lượng mới-biên tại v giảm **cho mọi flow**, không phụ thuộc vào các đặc trưng còn lại.

**Bộ nhận diện đa lớp, vừa phân loại họ đã biết vừa phát hiện mới.** Một bộ phân loại xác suất g_t trên các lớp {benign, A^{(1)},…,A^{(K)}, novel}, huấn luyện với các khối lượng
  benign ½ | tấn công đã biết ½ (chia đều giữa các họ) | novel ½ · q_t.

Hai điểm số được lấy ra từ cùng một mô hình:
- điểm mới s_t^{nov}(x) = log p_t(novel | x) / p_t(benign | x);
- lớp đã biết ŷ(x) = argmax_k p_t(A^{(k)} | x).

Quy tắc cảnh báo: cảnh báo khi s^{nov} > τ hoặc khi ŷ là một họ tấn công. τ được đặt một lần ở 1% FPR trên holdout của anchor, **không thêm ngưỡng động nào**.

Vì sao cơ chế này có thể cải thiện **cả phân loại họ đã biết lẫn phát hiện họ mới**: khi tính mới biên đã được mô hình hóa riêng, lớp novel không còn phải hấp thụ phần "giá trị lạ nhưng vô hại". Nhờ đó ranh giới giữa các họ đã biết và benign không bị kéo lệch bởi benign trôi dạt (giả thuyết H6).

---

## 4. Giả thuyết và tính chất cần chứng minh

**Tính chất (sẽ chứng minh, cùng khung với P1–P3 của ERNP):**
- **F1 — Không hấp thụ.** Mô hình chỉ phụ thuộc vào (P_t, A_t, E_t). Lưu lượng không nhãn, với bất kỳ khối lượng nào, không làm mô hình thay đổi.
- **F2 — Độ phức tạp nhãn cho trôi dạt biên.** Nếu một chế độ benign mới chỉ khác chế độ cũ ở phân phối biên của m đặc trưng (đúng như D1 cho thấy), FERN cần O(số giá trị mới trên m đặc trưng) nhóm được xác nhận để FPR của chế độ đó về 0. ERNP cần O(thể tích chế độ / ε^d) nhóm. Đây là một khoảng cách về số mũ theo d.
- **F3 — Bảo toàn zero-day.** Một tấn công có giá trị biên chưa từng được xác nhận benign (thường gặp với tấn công khối lượng lớn), hoặc có tổ hợp xa bằng chứng, vẫn giữ được khối lượng prior. Cận dưới có dạng như P3.
- **F4 — Phạm vi ảnh hưởng của nhãn sai.** Một nhóm tấn công bị xác nhận nhầm là benign làm giảm khối lượng mới-biên tại các giá trị của nó, với mức giảm bị chặn bởi tỉ lệ khối lượng của nhóm. Nó chỉ rút tính mới phụ thuộc trong phạm vi cục bộ. Một tấn công khác chỉ bị bỏ lọt khi đồng thời mang giá trị quen **và** tổ hợp gần nhóm lỗi.

**Giả thuyết kiểm chứng được:**
| Mã | Giả thuyết | Kiểm tra bằng |
|---|---|---|
| H1 | Trôi dạt benign là trôi dạt biên | ✅ **Đã xác nhận** (D1) |
| H4 | FERN giảm FPR trên ToN ở cập nhật theo ngày, mà không cần chia nhỏ chu kỳ | FPR và F1 trên ToN |
| H5 | Prior mới-biên đóng lỗ hổng ngoại suy, giúp recall zero-day khi tấn công xa benign | Recall theo khoảng cách (như hình P3) trên CIDDS và CIC17 |
| H6 | Mô hình đa lớp cải thiện phân loại họ đã biết khi có trôi dạt | Macro-F1 trên các họ đã biết |
| H7 | Phạm vi tác động phân rã giảm nhạy cảm với nhãn sai | Thí nghiệm lật nhãn 5–20% |

Nếu H4 **không** đạt trên dữ liệu phát triển, hướng này dừng lại (quy tắc go/no-go).

---

## 5. Khác biệt với tài liệu (novelty)
| Công trình | Cơ chế | FERN khác ở đâu |
|---|---|---|
| COPOD (Li et al., ICDM 2020), ECOD | Xác suất đuôi của từng đặc trưng, dùng copula thực nghiệm; tĩnh, không nhãn | FERN dùng tính mới biên **như một prior được rút lại bằng bằng chứng theo thời gian**, và mô hình hóa riêng tính mới phụ thuộc |
| MCM (Yin et al., ICLR 2024) | Mô hình hóa phụ thuộc bằng che-và-dự-đoán ô; tĩnh | FERN thích nghi, và tách phạm vi tác động của bằng chứng theo biên và theo phụ thuộc |
| Negative sampling (Sipple, ICML 2020) | Prior đồng nhất, tĩnh | Prior phân rã có cấu trúc, được rút lại |
| AAD (Das et al., ICDM 2016) | Học lại trọng số ensemble từ phản hồi, dữ liệu tĩnh | Cập nhật phân phối tính mới; phạm vi tác động có cấu trúc |
| SSF (INFOCOM 2025), OWAD (NDSS 2023), METANOIA (2025), Shift Detection & Adaptation for NIDS (arXiv 2025) | Cập nhật toàn bộ mô hình bằng mẫu được chọn hoặc gán nhãn giả | FERN không cập nhật mô hình benign bằng dữ liệu không nhãn; mỗi nhãn tác động theo đúng cấu trúc của tính mới |
| CADE (USENIX Sec 2021), Trident (WWW 2024) | Khoảng cách tới lớp đã biết; không thích nghi benign | FERN thích nghi benign mà vẫn giữ tính chất F1 |
| **ERNP (bản trước)** | Prior chỉ có thành phần phụ thuộc; bằng chứng chỉ tác động cục bộ | **FERN tổng quát hóa ERNP**; ERNP là trường hợp đặc biệt |

**Khoảng trống trong tài liệu:** chưa công trình nào trả lời câu hỏi *"một nhãn của analyst nên có hiệu lực tới đâu?"* dựa trên cấu trúc thực nghiệm của trôi dạt. Các công trình hiện có hoặc không thích nghi, hoặc cập nhật toàn bộ mô hình. Kết quả đo D1 (trôi dạt benign trong NIDS là trôi dạt biên) cũng chính là một phát hiện thực nghiệm mới, có thể công bố.

*Cần làm trước khi viết bài:* tìm kiếm tài liệu có hệ thống thêm một vòng, với từ khóa copula + concept drift + active learning + NIDS. Vòng tìm hiện tại chưa thấy công trình nào trùng.

---

## 6. Thiết kế thực nghiệm để chứng minh chính cơ chế tạo ra cải thiện

**Giao thức chống tinh chỉnh trên dữ liệu test:**
- **Phát triển:** seed 0–1 của CIDDS, ToN và CIC18. Mọi lựa chọn thiết kế và mọi go/no-go đều làm ở đây.
- **Kiểm tra (chạy đúng một lần, sau khi chốt):** seed 2–4 của 4 dataset, cộng toàn bộ CIC17 (held-out) và CIC17 đầy đủ.
- **Siêu tham số chung cho mọi dataset:** ρ = 0,3 và ε như ERNP; Δ_j và băng thông KDE chọn bằng quy tắc cố định (quy tắc Silverman), không tinh chỉnh.

**Ablation cô lập cơ chế:**
| Biến thể | Mục đích |
|---|---|
| ERNP (chỉ phụ thuộc, cục bộ) | Mốc |
| + prior mới-biên, **rút lại cục bộ** (sai phạm vi) | Tách tác động của *phạm vi toàn cục theo đặc trưng* khỏi việc *chỉ có thêm prior* |
| + prior mới-biên, **không rút lại** | Tách tác động của rút lại |
| **FERN đầy đủ** (biên toàn cục + phụ thuộc cục bộ) | Cơ chế đề xuất |
| FERN với phạm vi Euclid toàn cục cho cả hai thành phần | Đối chứng: toàn cục sai cấu trúc |
| FERN nhị phân (không đa lớp) | Tách tác động của đầu ra đa lớp lên phân loại họ đã biết |

Tiêu chí: cải thiện phải đến từ biến thể "toàn cục theo đặc trưng", không đến từ việc chỉ thêm prior. Đây là bằng chứng rằng chính cơ chế tạo ra khác biệt.

**Đánh giá:**
- RQ1 như cũ (F1, MCC, recall zero-day @1% FPR), **thêm macro-F1 phân loại họ đã biết**.
- Thí nghiệm đầu độc, số nhãn, nhãn sai và quy mô đầy đủ như cũ.
- Thêm **số nhãn cần để FPR của một chế độ benign mới về dưới 1%** (kiểm chứng F2).

**Tiêu chí thành công, chốt trước khi chạy trên tập kiểm tra:**
- (i) FERN ≥ ERNP trên mọi dataset, trong phạm vi sai số seed;
- (ii) FERN không thua SOTA nào trên dataset nào quá khoảng tin cậy của chênh lệch;
- (iii) FPR trên ToN theo ngày ≤ 15%.

Mình không hứa trước là sẽ thắng mọi SOTA trên mọi dataset. Kết quả sẽ được báo cáo trung thực dù đạt hay không.

---

## 7. Rủi ro
1. **Tấn công khối lượng lớn mang giá trị gần miền benign đã xác nhận.** Khi đó tính mới biên giảm, và chỉ còn tính mới phụ thuộc để bắt. Cần đo riêng DDoS/PortScan trên CIC17 và CIC18.
2. **Nhãn sai giờ có hiệu lực toàn cục trên biên.** F4 chặn mức ảnh hưởng này, nhưng cần kiểm chứng bằng thực nghiệm (H7). Nếu kết quả xấu hơn ERNP, phải báo cáo thẳng.
3. **Mô hình đa lớp cần nhãn họ cho nhóm tấn công.** Analyst gán tên họ là giả định hợp lý trong SOC, nhưng phải nêu rõ.
4. **Novelty:** reviewer có thể xem đây là "COPOD + ERNP". Lập luận phản bác là phần tính mới biên ở đây là một *prior được rút lại theo bằng chứng*, không phải một điểm số cộng thêm; ablation "rút lại cục bộ và toàn cục" chính là bằng chứng cho điều đó.
