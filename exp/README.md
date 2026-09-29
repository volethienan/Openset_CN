# exp/ — code CHÍNH THỨC cho paper ERNP

Chỉ thư mục này được dùng cho các thực nghiệm RQ1–RQ5 (xem `../ERNP_RQs_Experiments.md`).
Code thăm dò cũ (QCA, ADS, DED, tangent, BIR…) nằm ở `../mock/_archive_exploration/` — **không dùng**.

| File | Vai trò |
|---|---|
| `common.py` | Giao thức chung: đọc stream, chia anchor/holdout theo seed, chuẩn hóa, ngưỡng 1% FPR, lưu kết quả |
| `ernp.py` | **Thuật toán đề xuất ERNP** (mặc định = cấu hình chốt; mọi công tắc ablation là tham số dòng lệnh) |
| `baselines/sota.py` | Wrapper gọi **code gốc**: MemStream, OWAD, SSF (+ biến thể nhân quả), Trident; CADE qua `baselines/cade_port.py` |
| `baselines/cade_port.py` | Port PyTorch trung thành của CADE (bản gốc chạy TF 1.12); đã kiểm tra trên split IDS2018 của repo gốc |
| `data/prepare*.py` | Dựng stream (đặc trưng per-flow + ngữ cảnh tính trên toàn bộ dữ liệu), `EXP_SEED` chọn mẫu |
| `data/inject.py` | Kịch bản tiêm tấn công luộc ếch / laundering cho RQ2 (`base payload ramp_days rate`) |
| `evaluate.py` | Chỉ số, gộp seed (mean ± std), Wilcoxon một phía n = dataset × seed, Friedman + Nemenyi |

Stream nằm ở `../mock/streams/` (tên: `<base>` cho seed 0, `<base>_s<k>` cho seed k).
Kết quả: `results/<dataset>__<method>__s<seed>.npz|json`; báo cáo: `reports/`.

## Chạy
```
set EXP_SEED=k                                   # 0..4
python ernp.py CIDDS_ctx_s1                        # ERNP, giao thức nhân quả (mặc định)
python ernp.py CIDDS_ctx_s1 --protocol ssf --name ernp_ssfproto
python ernp.py ToN_ctx_s1 --budget 50 --name ernp_B50                 # RQ3
python ernp.py CIC18_s1 --prior uniform --name abl_prior_uniform      # RQ4
python baselines/sota.py CIC18_s1 ssf_causal                          # SOTA
python evaluate.py --datasets CIDDS_ctx ToN_ctx CIC17 CIC18 --methods ernp ssf_causal owad memstream trident cade --out rq1
```
Lưu ý: tên dataset của seed 0 không có hậu tố `_s0` (trùng với stream mock test).
