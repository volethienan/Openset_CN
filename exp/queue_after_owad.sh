#!/bin/bash
# Queue (05/10/2026): wait for the running OWAD rerun, push its results, then run the MemStream rerun and push.
# Run inside tmux on the server:   bash exp/queue_after_owad.sh > exp/logs/queue.log 2>&1
# GPU 0 only, 24 cores (same limits as the OWAD run).
set -u
cd /workspace/anvlt/Openset_CN
source ~/ernp_env/bin/activate
echo "[$(date '+%F %T')] waiting for the OWAD runner to finish"
while pgrep -f "run_jobs.py rq1" > /dev/null; do sleep 60; done
echo "[$(date '+%F %T')] OWAD runner finished: $(ls exp/results | grep -c '__owad.*json') OWAD results"

push_results() {   # commit only exp/results, merge the latest code, push
  git add exp/results
  git commit -q -m "$1" || echo "nothing to commit"
  git pull --no-rebase --no-edit -q || { echo "PULL FAILED - stopping"; exit 1; }
  git push -q && echo "[$(date '+%F %T')] pushed: $1" || echo "PUSH FAILED"
}
push_results "OWAD rerun (standardised input)"

git checkout -- mock/streams/CIC17_full.parquet 2> /dev/null
git lfs pull --include="mock/streams/CIC17_full.parquet"
ls -lh mock/streams/CIC17_full.parquet

cd exp
echo "[$(date '+%F %T')] pending: $(python pending_jobs.py rq1:0-4 rq2:0-4 x10:0-4 scale:0-2 | tail -1)"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 MAX_GPU_PROCS=6 taskset -c 0-23 \
  python run_jobs.py rq1+rq2+x10+scale --seeds 0 1 2 3 4 --gpu 6 --cpu 1 --threads 3 > logs/memstream_rerun.log 2>&1
echo "[$(date '+%F %T')] MemStream runner finished: $(ls results | grep -c '__memstream__.*json') MemStream results, $(grep -c ERR logs/memstream_rerun.log) ERR"
cd ..
push_results "MemStream rerun (authors' beta = 0.1)"
echo "[$(date '+%F %T')] QUEUE DONE"
