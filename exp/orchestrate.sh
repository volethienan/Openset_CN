#!/usr/bin/env bash
# Full experimental campaign after RQ1 has been started by its own chain (rq1_s{0..4}_runner.log).
cd "$(dirname "$0")"; S=../mock/streams
streams_ready() {   # all base + injection streams of seeds 1..4
  for s in 1 2 3 4; do
    for b in CIDDS_ctx_s$s ToN_ctx_s$s CIC17_s$s CIC18_s$s CIDDS_ctx_s${s}_injinit CIDDS_ctx_s${s}_injresp CIDDS_ctx_s${s}_injinit_bf CIDDS_ctx_s${s}_injresp_bf; do
      [ -f $S/$b.parquet ] || return 1; done; done; return 0; }

# ---------------- CPU track: RQ2 sweep streams, then RQ4 + RQ5 (ERNP only) ----------------
(
  until streams_ready; do sleep 120; done; sleep 60
  for s in 0 1 2; do base=CIDDS_ctx$([ $s = 0 ] || echo _s$s)
    for r in 3 7 14; do for q in 10 30 100; do [ "$r$q" = "730" ] && continue
      [ -f $S/${base}_injinit_bf_r${r}_q${q}.parquet ] || python data/inject.py $base bruteForce $r $q >> logs/sweep_streams.log 2>&1
    done; done; done
  echo "sweep streams done" >> logs/sweep_streams.log
  python run_jobs.py rq4+rq5 --seeds 0 1 2 3 4 --gpu 1 --cpu 2 > logs/rq45_runner.log 2>&1
) &
# ---------------- GPU track: RQ2 + RQ3 after the RQ1 chain ----------------
(
  until grep -q "ALL JOBS DONE" logs/rq1_s4_runner.log 2>/dev/null; do sleep 120; done
  until grep -q "sweep streams done" logs/sweep_streams.log 2>/dev/null; do sleep 60; done
  python run_jobs.py rq2+rq3 --seeds 0 1 2 3 4 --gpu 2 --cpu 1 > logs/rq23_runner.log 2>&1
) &
wait
# ---------------- reports ----------------
export PYTHONIOENCODING=utf-8
mkdir -p reports
python evaluate.py --out rq1 --datasets CIDDS_ctx ToN_ctx CIC17 CIC18 \
  --methods ernp ssf_causal owad memstream trident cade ernp_ssfproto ssf > reports/rq1.txt 2>&1
python evaluate.py --out rq2 --datasets CIDDS_ctx_injinit CIDDS_ctx_injresp CIDDS_ctx_injinit_bf CIDDS_ctx_injresp_bf \
  --methods ernp ernp_B0 ssf_causal owad memstream trident cade > reports/rq2.txt 2>&1
SW=$(for r in 3 7 14; do for q in 10 30 100; do [ "$r$q" = "730" ] && echo -n "CIDDS_ctx_injinit_bf " || echo -n "CIDDS_ctx_injinit_bf_r${r}_q${q} "; done; done)
python evaluate.py --out rq2_sweep --seeds 0 1 2 --datasets $SW --methods ernp ssf_causal owad memstream > reports/rq2_sweep.txt 2>&1
python evaluate.py --out rq3 --seeds 0 1 2 --datasets ToN_ctx CIC18 --methods ernp ernp_B25 ernp_B50 ernp_B100 ernp_B400 \
  ssf_causal ssf_causal_B25 ssf_causal_B50 ssf_causal_B100 ssf_causal_B400 owad owad_B25 owad_B50 owad_B100 owad_B400 > reports/rq3.txt 2>&1
python evaluate.py --out rq4 --datasets CIDDS_ctx ToN_ctx CIC17 CIC18 --methods ernp abl_prior_none abl_prior_uniform \
  abl_prior_marginal abl_retr_none abl_retr_hard abl_evid_rep abl_feat_flow ernp_B0 > reports/rq4.txt 2>&1
python evaluate.py --out rq5 --seeds 0 1 2 --datasets CIDDS_ctx ToN_ctx CIC18 --methods ernp sens_rho0.1 sens_rho0.2 sens_rho0.5 \
  sens_mass0.25 sens_mass0.5 sens_mass2 sens_eps0.5 sens_eps2 sens_clf_rf > reports/rq5.txt 2>&1
echo "CAMPAIGN DONE"
