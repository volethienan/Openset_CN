#!/usr/bin/env bash
# RQ2 with a naive analyst (job group `naive`, report exp/naive_report.py): 60 CPU + 20 GPU jobs, CIDDS-001
# protocol-valid poisoning streams, seeds 0-4. Server limits: GPU 1 only, cores 0-23, (gpu + cpu) x threads <= 24.
# Usage on the server, inside tmux, from the repo root:  bash exp/run_naive.sh
# PUSH=0 bash exp/run_naive.sh  -> run and report, but do not commit/push the results.
set -euo pipefail
cd "$(dirname "$0")"; mkdir -p logs
source ~/ernp_env/bin/activate 2>/dev/null || true
echo "== $(date) tests"
python -m pytest -q tests/test_naive_ramp.py
n=$(python run_jobs.py naive --seeds 0 1 2 3 4 --dry | grep -cE " (cpu|gpu) CIDDS_ctx")
[ "$n" -eq 80 ] || { echo "expected 80 jobs, got $n"; exit 1; }
echo "== $(date) jobs (80)"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 MAX_GPU_PROCS=2 taskset -c 0-23 \
  python run_jobs.py naive --seeds 0 1 2 3 4 --gpu 2 --cpu 6 --threads 3 2>&1 | tee -a logs/naive_runner.log
miss=$(python - <<'PY'
import os, run_jobs
miss = [f"{ds}__{n}__s{s}" for s in range(5) for _, ds, n, _ in run_jobs.naive_jobs(s)
        if not os.path.exists(f"results/{ds}__{n}__s{s}.npz")]
print(len(miss)); [print("  missing", m) for m in miss]
PY
)
echo "$miss"; [ "$(echo "$miss" | head -1)" = "0" ] || { echo "some runs failed; see logs/naive_runner.log"; exit 1; }
echo "== $(date) report"
OMP_NUM_THREADS=8 taskset -c 0-23 python naive_report.py | tail -60
if [ "${PUSH:-1}" = "1" ]; then
  cd ..
  git add exp/results/*_injvalid*_bf__*_R50__s*.* exp/results/*_injvalid*_bf__*_R100__s*.* \
          exp/reports/naive_analyst.txt exp/reports/naive_runs.csv exp/reports/naive_channel.csv
  git commit -m "RQ2 naive analyst: 80 runs (group naive) and report"
  git push origin HEAD
fi
echo "== $(date) done"
