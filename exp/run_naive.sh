#!/usr/bin/env bash
# RQ2 with a naive analyst (job group `naive`, report exp/naive_report.py): 80 CPU + 20 GPU jobs, CIDDS-001
# protocol-valid poisoning streams, seeds 0-4. Server limits (27/10/2026): GPU 0 only, cores 0-18,
# (gpu + cpu) x threads <= 19. Override with GPU_ID, CORES, NGPU, NCPU, THREADS if the allocation changes.
# Usage on the server, inside tmux, from the repo root:  bash exp/run_naive.sh
# PUSH=0 bash exp/run_naive.sh  -> run and report, but do not commit/push the results.
# Finished runs are skipped: after the first campaign (80 runs, ac04145) only the 20 representative-only jobs remain.
set -euo pipefail
GPU_ID=${GPU_ID:-0}; CORES=${CORES:-0-18}; NGPU=${NGPU:-2}; NCPU=${NCPU:-7}; THREADS=${THREADS:-2}
[ $(( (NGPU + NCPU) * THREADS )) -le 19 ] || { echo "(NGPU + NCPU) x THREADS exceeds 19 cores"; exit 1; }
cd "$(dirname "$0")"; mkdir -p logs
source ~/ernp_env/bin/activate 2>/dev/null || true
echo "== $(date) tests"
python tests/test_naive_ramp.py          # plain unittest; pytest is not installed on the server
n=$(python run_jobs.py naive --seeds 0 1 2 3 4 --dry | grep -cE " (cpu|gpu) CIDDS_ctx")
[ "$n" -eq 100 ] || { echo "expected 100 jobs, got $n"; exit 1; }
echo "== $(date) jobs (100, finished ones skipped) on GPU $GPU_ID, cores $CORES: --gpu $NGPU --cpu $NCPU --threads $THREADS"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU_ID MAX_GPU_PROCS=$NGPU taskset -c $CORES \
  python run_jobs.py naive --seeds 0 1 2 3 4 --gpu $NGPU --cpu $NCPU --threads $THREADS 2>&1 | tee -a logs/naive_runner.log
miss=$(python - <<'PY'
import os, run_jobs
miss = [f"{ds}__{n}__s{s}" for s in range(5) for _, ds, n, _ in run_jobs.naive_jobs(s)
        if not os.path.exists(f"results/{ds}__{n}__s{s}.npz")]
print(len(miss)); [print("  missing", m) for m in miss]
PY
)
echo "$miss"; [ "$(echo "$miss" | head -1)" = "0" ] || { echo "some runs failed; see logs/naive_runner.log"; exit 1; }
echo "== $(date) report"
OMP_NUM_THREADS=8 taskset -c $CORES python naive_report.py | tail -60
if [ "${PUSH:-1}" = "1" ]; then
  cd ..
  git add exp/results/*_injvalid*_bf__*_R50__s*.* exp/results/*_injvalid*_bf__*_R100__s*.* \
          exp/reports/naive_analyst.txt exp/reports/naive_runs.csv exp/reports/naive_channel.csv
  git commit -m "RQ2 naive analyst: representative-only runs (group naive, 100 runs) and report"
  git push origin HEAD
fi
echo "== $(date) done"
