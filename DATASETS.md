# Bộ dữ liệu chốt cho QCA (25/09/2026)

Nguồn số liệu: `audit/*.json` (quét toàn bộ file) và `mock/streams/` (kiểm tra benign và tấn công cùng tồn tại theo ngày).

| # | Bộ | Vai trò | Quy mô | Provenance domain | Điểm mạnh | Cảnh báo phải ghi trong paper |
|---|---|---|---|---|---|---|
| 1 | **CIDDS-001** (OpenStack internal, 4 tuần) | **Chính**: minh chứng lõi (chống hấp thụ, drift, provenance) | 31,3M flow, 28 ngày | IP host nội bộ tĩnh (≈ device identity) | Traffic east-west; tấn công xuất phát từ client nội bộ vốn cũng sinh benign (91%); chỉ 0,4% flow tấn công đến từ IP không có benign; benign mỗi ngày; chu kỳ cuối tuần; đã quan sát được hấp thụ tự nhiên (recall DoS 0,50 → 0,30) | Chỉ 4 loại tấn công đơn giản; benign do script sinh; endpoint ngoài bị ẩn danh (`DNS`, `EXT_SERVER`) |
| 2 | **NF-ToN-IoT-v3** | **Phụ**: zero-day theo trình tự + drift mạnh (giao thức có ngân sách gán nhãn) | 27,5M flow, 6 ngày | IPv4 nội bộ 192.168.1.x | 9 họ tấn công xuất hiện lần lượt từng ngày; IP kẻ tấn công cũng có benign (92%); 0% flow tấn công từ IP không có benign | Benign drift cực mạnh (64% benign từ host mới), FPR one-class 60–90% nếu không neo lại; hub top-1 31%; testbed IoT |
| 3 | **CICIDS2017 improved** (Engelen et al.) | **So sánh với literature** | 2,1M flow, 5 ngày | IP; mọi tấn công từ ngoài đi qua NAT `172.16.0.1` (tương đương domain ⊥) | Được trích dẫn nhiều nhất; nhãn đã sửa | Tấn công từ ngoài gom vào 1 domain NAT; hub top-1 62%; chỉ 5 ngày |
| 4 | **CSE-CIC-IDS2018 improved** | **So sánh với literature**, kèm tập con nội bộ (Botnet, Infiltration) | 63,2M flow, 11 ngày | IP (AWS 172.31.x) | Quy mô lớn, 18 loại, nhãn đã sửa | 84% flow tấn công từ IP không có benign → phải báo cáo baseline `new_ip`; hub DNS 41% |
| 5 | **NF-UNSW-NB15-v3** | **Chỉ đối chiếu** (minh họa lộ nhãn qua IP) | 2,4M flow, 3 ngày | IP | Rất phổ biến trong literature | 100% flow tấn công từ subnet 175.45.176.x không có benign; mọi họ nằm cùng ngày → không dùng cho claim về thời gian |

## Đã loại (đã đưa vào Recycle Bin, khôi phục được)
CIC-IDS2017 (bản MachineLearningCVE, không có IP/ts) · CICIoMT2024 · USTC-TFC2016 · UNR-IDD · InSDN · HIKARI2021 · NF-BoT-IoT-v3 · NF-CSE-CIC-IDS2018-v3 (đã thay bằng bản improved) · UWF-ZeekData22 / Fall22 / 24.

## Quy tắc đánh giá áp dụng cho mọi bộ
- Luôn báo cáo baseline `new_ip` để lộ ra hiện tượng lộ nhãn qua IP.
- Anchor = benign trong giai đoạn đầu; mọi ngưỡng được hiệu chỉnh ở cùng FPR 1% trên holdout.
- Claim chính về chống hấp thụ và provenance chỉ dựa trên bộ 1 và 2. Bộ 3–5 chỉ dùng để so sánh với số liệu trong literature.
