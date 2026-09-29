#!/usr/bin/env bash
# RQ1 supplements only (28/09/2026; RQ2/RQ3 paused by the user - see reports/rq23_paused_status.txt).
# One runner, at most 4 GPU jobs on the machine at any time (MAX_GPU_PROCS counts jobs left over from earlier
# runners too - keeps the Windows memory commit below its limit). Finished / already-running jobs are skipped.
cd "$(dirname "$0")"
MAX_GPU_PROCS=4 python run_jobs.py scale+ext1 --seeds 0 1 2 3 4 --gpu 4 --cpu 1 > logs/rq1sup_runner.log 2>&1   # long pole first
while powershell -NoProfile -Command "if (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'sota.py|classic.py' }) { exit 0 } else { exit 1 }"; do sleep 120; done
export PYTHONIOENCODING=utf-8; SOTA="ssf_causal owad memstream trident cade"
for s in 0 1 2 3 4; do sfx=$([ $s = 0 ] || echo _s$s)
  for b in CIDDS_ctx$sfx ToN_ctx$sfx CIC17$sfx CIC18$sfx; do [ -f results/${b}__new_ip__s$s.json ] || EXP_SEED=$s python baselines/sanity.py $b > /dev/null 2>&1; done; done
python evaluate.py --out rq1 --datasets CIDDS_ctx ToN_ctx CIC17 CIC18 --methods ernp $SOTA ernp_ssfproto ssf new_ip --compare ernp $SOTA > reports/rq1.txt 2>&1
python evaluate.py --out rq1_ext --datasets CIDDS_ctx ToN_ctx CIC17 CIC18 --methods ernp abl_evid_rep abl_self ernp_B0 $SOTA memstream_fn \
  iforest knn ae_static ssf_causal_E180 ssf_causal_W20000 > reports/rq1_ext.txt 2>&1
python evaluate.py --out scale --seeds 0 --datasets CIC17_full --methods ernp $SOTA > reports/scale.txt 2>&1
python evaluate.py --out cadence --seeds 0 1 2 --datasets CIDDS_ctx ToN_ctx CIC17 CIC18 --methods ernp_P4 ernp_P1 \
  ssf_causal_P4 ssf_causal_P1 owad_P4 owad_P1 > reports/cadence.txt 2>&1
python bench_batch.py > reports/bench_batch.txt 2>&1
python -c "import check_criteria as c; c.rq1(); c.cadence(); c.reviewer_checks()" > reports/criteria_rq1_final.txt 2>&1
echo "RQ1 SUPPLEMENTS DONE $(date)"
