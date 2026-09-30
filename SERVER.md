# Chạy tiếp thực nghiệm ERNP trên server

## Nội dung repo
| Thư mục | Nội dung |
|---|---|
| `exp/` | Code (ERNP: `exp/ernp.py`; wrapper SOTA: `exp/baselines/`) |
| `exp/results/` | Kết quả đã chạy (RQ1, RQ4, RQ5, cadence, phần bổ sung RQ1, và một phần RQ2/RQ3) |
| `exp/reports/` | Báo cáo tổng hợp; `criteria_rq1_final.txt` là tổng kết RQ1 |
| `mock/streams/` | Stream đầu vào đã dựng sẵn: dữ liệu thật, đã lấy mẫu 300–400k flow mỗi seed; `CIC17_full` là bản đầy đủ. Tên "mock" chỉ là tên lịch sử |
| `sota/` | Code gốc của SSF, OWAD, MemStream, Trident (đã bỏ dữ liệu demo lớn); dữ liệu IDS2018 của CADE để kiểm tra tái lập |

Không có `dataraw/`, vì không cần cho RQ2/RQ3.

## Cài đặt
```bash
git lfs install
git clone https://github.com/volethienan/Openset_CN.git
cd Openset_CN
git lfs pull
pip install -r requirements.txt
```
Mọi đường dẫn tự suy ra từ vị trí repo. Nếu muốn, có thể ghi đè bằng `export ERNP_ROOT=/duong/dan/Openset_CN/`.

## Bước 1: kiểm tra hai máy cho kết quả nhất quán (bắt buộc)
Chạy lại 3 job đã có sẵn kết quả vào thư mục tạm, rồi so với kết quả trong repo:
```bash
cd exp
mkdir -p /tmp/ernp_check
EXP_SEED=0 EXP_RESULTS=/tmp/ernp_check/ python ernp.py CIC17
EXP_SEED=0 EXP_RESULTS=/tmp/ernp_check/ python baselines/sota.py CIC17 owad
EXP_SEED=0 EXP_RESULTS=/tmp/ernp_check/ python baselines/sota.py CIDDS_ctx_injinit trident
python -c "
import numpy as np
for t in ['CIC17__ernp__s0','CIC17__owad__s0','CIDDS_ctx_injinit__trident__s0']:
    a=np.load('/tmp/ernp_check/'+t+'.npz'); b=np.load('results/'+t+'.npz')
    print(t, 'flags identical:', np.array_equal(a['flag'], b['flag']), 'max score diff:', float(abs(a['score']-b['score']).max()))"
```
Nếu không khớp (thường do khác GPU hoặc phiên bản thư viện), nên chạy lại toàn bộ RQ2/RQ3 trên server để mọi kết quả trong cùng một RQ đến từ cùng một máy.

## Bước 2: chạy RQ2 và RQ3 còn lại
```bash
cd exp
MAX_GPU_PROCS=6 python run_jobs.py rq2+rq3 --seeds 0 1 2 3 4 --gpu 6 --cpu 4 > logs/rq23_server.log 2>&1
```
- Runner tự bỏ qua các job đã có kết quả. Danh sách job còn lại nằm ở `exp/reports/rq23_paused_status.txt`.
- `MAX_GPU_PROCS` là số job GPU tối đa chạy cùng lúc; trên laptop đặt 4 vì giới hạn bộ nhớ. Điều chỉnh theo RAM và GPU của server.
- Job chậm nhất là SSF, khoảng 45–90 phút mỗi lần.

## Bước 3: báo cáo
```bash
bash report_all.sh      # tạo exp/reports/*.txt và criteria.txt (chấm PASS/FAIL từng tiêu chí)
```
Thiết kế và tiêu chí thành công: `ERNP_RQs_Experiments.md`.

## Đợt bổ sung 3 thí nghiệm (30/09/2026)
| Nhóm | Nội dung | Số job |
|---|---|---|
| `rq2lb` | RQ2 với 25 nhãn/ngày cho ERNP, SSF, OWAD (4 kịch bản × 5 seed) | 60 |
| `noise` | Analyst gán nhãn sai: ERNP với 5/10/20%, SSF và OWAD với 10% (RQ1 + RQ2, seed 0–2) | 120 |
| `scale` | CIC17 đầy đủ (2,1 triệu flow), seed 1–2, ERNP + 5 SOTA (seed 0 đã có) | 12 mới |

Cần file LFS cho nhóm `scale`: kiểm tra `ls -lh ../mock/streams/CIC17_full.parquet` nặng khoảng 290 MB; nếu không, chạy `git lfs pull`.

```bash
cd exp && mkdir -p logs
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 MAX_GPU_PROCS=4 python run_jobs.py rq2lb+noise+scale --seeds 0 1 2 3 4 --gpu 4 --cpu 3 > logs/extra3.log 2>&1
```
Xong thì đẩy kết quả: `git add exp/results && git commit -m "extra experiments" && git push`.
