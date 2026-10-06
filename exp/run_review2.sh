#!/usr/bin/env bash
# Experiments for the second external review (06/10/2026). Server limits: GPU 0 only, 24 CPU cores.
#   rq2gate : GBM+AL (no prior) and ERNP without retraction under slow poisoning (4 scenarios) and group camouflage
#             (1 and 3 decoys)                                                                    -> 60 CPU jobs
#   t1      : CADE and Trident with the threshold at 1% FPR on held-out anchor flows (RQ1 datasets) -> 40 GPU jobs
#   diam    : RQ1 ERNP rerun logging each alert group's radius (footprint of Prop. 1(iii))       -> 20 CPU jobs
# usage (inside tmux, from anywhere):  bash exp/run_review2.sh
set -e
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8
mkdir -p logs

# the camouflage streams are git-ignored: build them from the CIDDS base streams if missing (deterministic, rng seed 42)
for s in "" _s1 _s2 _s3 _s4; do
  [ -f "../mock/streams/CIDDS_ctx${s}.parquet" ] || { echo "missing ../mock/streams/CIDDS_ctx${s}.parquet (git lfs pull)"; exit 1; }
  for r in 1 3; do
    [ -f "../mock/streams/CIDDS_ctx${s}_injcamo${r}_bf.parquet" ] || python data/inject_camo.py "CIDDS_ctx${s}" "$r"
  done
  for v in injinit injresp injinit_bf injresp_bf; do
    [ -f "../mock/streams/CIDDS_ctx${s}_${v}.parquet" ] || { echo "missing ../mock/streams/CIDDS_ctx${s}_${v}.parquet (git lfs pull)"; exit 1; }
  done
done
python pending_jobs.py rq2gate:0-4 diam:0-4 t1:0-4

# 4 GPU jobs + 4 CPU jobs, 3 threads each = 24 cores; GPU 0 only
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 MAX_GPU_PROCS=4 taskset -c 0-23 \
  python run_jobs.py rq2gate+diam+t1 --seeds 0 1 2 3 4 --gpu 4 --cpu 4 --threads 3 2>&1 | tee logs/review2.log
bash report_review2.sh
echo REVIEW2_DONE
