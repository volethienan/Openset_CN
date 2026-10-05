#!/usr/bin/env bash
# Final reports after the campaign (idempotent; re-run any time). Output: exp/reports/*.txt + *_runs.csv
cd "$(dirname "$0")"; export PYTHONIOENCODING=utf-8; mkdir -p reports
# leakage sanity baseline (seconds per stream)
for s in 0 1 2 3 4; do sfx=$([ $s = 0 ] || echo _s$s)
  for b in CIDDS_ctx$sfx ToN_ctx$sfx CIC17$sfx CIC18$sfx CIDDS_ctx${sfx}_injinit CIDDS_ctx${sfx}_injresp CIDDS_ctx${sfx}_injinit_bf CIDDS_ctx${sfx}_injresp_bf; do
    [ -f results/${b}__new_ip__s$s.json ] || EXP_SEED=$s python baselines/sanity.py $b > /dev/null 2>&1; done; done
SOTA="ssf_causal owad memstream trident cade"
python evaluate.py --out rq1 --datasets CIDDS_ctx ToN_ctx CIC17 CIC18 --methods ernp $SOTA ernp_ssfproto ssf new_ip \
  --compare ernp $SOTA > reports/rq1.txt 2>&1
python evaluate.py --out rq1_ext --datasets CIDDS_ctx ToN_ctx CIC17 CIC18 --methods ernp abl_evid_rep abl_self $SOTA memstream_adapt owad_lr3 \
  iforest knn ae_static ssf_causal_E180 ssf_causal_W20000 > reports/rq1_ext.txt 2>&1
python evaluate.py --out scale --seeds 0 1 2 --datasets CIC17_full --methods ernp $SOTA memstream_adapt > reports/scale.txt 2>&1
INJ="CIDDS_ctx_injinit CIDDS_ctx_injresp CIDDS_ctx_injinit_bf CIDDS_ctx_injresp_bf"
python evaluate.py --out rq2lb --datasets $INJ --methods ernp_B25 ssf_causal_B25 owad_B25 > reports/rq2lb.txt 2>&1
python evaluate.py --out noise --seeds 0 1 2 --datasets CIDDS_ctx ToN_ctx CIC17 CIC18 $INJ --methods ernp_N5 ernp_N10 ernp_N20 \
  ssf_causal_N10 owad_N10 > reports/noise.txt 2>&1
python evaluate.py --out rq2 --datasets CIDDS_ctx_injinit CIDDS_ctx_injresp CIDDS_ctx_injinit_bf CIDDS_ctx_injresp_bf \
  --methods ernp ernp_B0 abl_self $SOTA memstream_adapt new_ip --compare ernp $SOTA > reports/rq2.txt 2>&1
SW=$(for r in 3 7 14; do for q in 10 30 100; do [ "$r$q" = "730" ] && echo -n "CIDDS_ctx_injinit_bf " || echo -n "CIDDS_ctx_injinit_bf_r${r}_q${q} "; done; done)
python evaluate.py --out rq2_sweep --seeds 0 1 2 --datasets $SW --methods ernp ssf_causal owad memstream > reports/rq2_sweep.txt 2>&1
python evaluate.py --out rq3 --seeds 0 1 2 --datasets ToN_ctx CIC18 --methods ernp_B25 ernp_B50 ernp_B100 ernp_B400 \
  ssf_causal_B25 ssf_causal_B50 ssf_causal_B100 ssf_causal_B400 owad_B25 owad_B50 owad_B100 owad_B400 > reports/rq3.txt 2>&1
python evaluate.py --out rq4 --datasets CIDDS_ctx ToN_ctx CIC17 CIC18 --methods ernp abl_prior_none abl_prior_uniform \
  abl_prior_marginal abl_prior_scarf abl_retr_none abl_retr_hard abl_evid_rep abl_feat_flow ernp_B0 > reports/rq4.txt 2>&1
python evaluate.py --out rq5 --seeds 0 1 2 --datasets CIDDS_ctx ToN_ctx CIC18 --methods ernp sens_rho0.1 sens_rho0.2 sens_rho0.5 \
  sens_mass0.25 sens_mass0.5 sens_mass2 sens_eps0.5 sens_eps2 sens_clf_rf > reports/rq5.txt 2>&1
python evaluate.py --out cadence --seeds 0 1 2 3 4 --datasets CIDDS_ctx ToN_ctx CIC17 CIC18 --methods ernp_P4 ernp_P1   ssf_causal_P4 ssf_causal_P1 owad_P4 owad_P1 > reports/cadence.txt 2>&1
python analysis_p23.py --datasets CIDDS_ctx ToN_ctx CIC17 CIC18 > reports/analysis_p23.txt 2>&1
python bench_batch.py > reports/bench_batch.txt 2>&1          # idle machine at this point
python check_criteria.py > reports/criteria.txt 2>&1
echo "REPORTS DONE"
