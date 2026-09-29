#!/usr/bin/env bash
# Resume after the pause of 27/09/2026 with the reviewer-proofing plan (all three tiers, approved 28/09/2026).
# 1) clean cost benchmark + regression check of the instrumented wrappers (idle machine; skipped once done)
# 2) RQ1-related work FIRST (user priority 28/09): cadence rest, extensions (classical refs, MemStream-fixnorm,
#    absorbing counterfactual, SSF with the authors' settings), unsampled CIC17 scale check; then RQ2, RQ3.
#    Runners skip finished runs and runs already executing in another process.
# 3) all reports
cd "$(dirname "$0")"
if [ ! -f reports/bench_cost.csv ]; then
  bash bench.sh > logs/bench.log 2>&1
  grep -q "BENCH OK" logs/bench.log || { echo "BENCH MISMATCH - stopping before the campaign (see logs/bench.log)"; exit 1; }
fi
python run_jobs.py cad+ext+scale+rq2+rq3 --seeds 0 1 2 3 4 --gpu 3 --cpu 2 > logs/main3_runner.log 2>&1
bash report_all.sh > logs/report_all.log 2>&1
echo "CAMPAIGN DONE" | tee -a logs/orchestrate.log
