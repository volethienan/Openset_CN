#!/usr/bin/env bash
# Every report of the manuscript on the K5A view (ERNP = --n_prior 5 --keep A). Build the view first: python exp/k5a_view.py
# Runs inside exp_k5a/ (the view) and copies the reports to exp/reports/, where the paper scripts read them.
# usage: bash exp/report_k5a.sh [stage]    (stage: eval | analysis | all; default all)
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT/exp_k5a" || exit 1
export PYTHONIOENCODING=utf-8 EXP_RESULTS="$ROOT/exp_k5a/results/"; mkdir -p reports logs_k5a
STAGE=${1:-all}
SOTA="ssf_causal owad memstream trident cade"
INJ="CIDDS_ctx_injinit CIDDS_ctx_injresp CIDDS_ctx_injinit_bf CIDDS_ctx_injresp_bf"
SW=$(for r in 3 7 14; do for q in 10 30 100; do [ "$r$q" = "730" ] && echo -n "CIDDS_ctx_injinit_bf " || echo -n "CIDDS_ctx_injinit_bf_r${r}_q${q} "; done; done)
ev() { out=$1; shift; python evaluate.py --out "$out" "$@" > "reports/$out.txt" 2>&1; echo "eval $out exit $?"; }
if [ "$STAGE" = eval ] || [ "$STAGE" = all ]; then
  ev rq1 --datasets CIDDS_ctx ToN_ctx CIC17 CIC18 --methods ernp $SOTA ernp_ssfproto ssf new_ip --compare ernp $SOTA &
  ev rq1_ext --datasets CIDDS_ctx ToN_ctx CIC17 CIC18 --methods ernp abl_evid_rep abl_self $SOTA memstream_adapt owad_lr3 \
    iforest knn ae_static ssf_causal_E180 ssf_causal_W20000 &
  ev scale --seeds 0 1 2 --datasets CIC17_full --methods ernp $SOTA memstream_adapt &
  wait
  ev rq2lb --datasets $INJ --methods ernp_B25 ssf_causal_B25 owad_B25 &
  ev noise --seeds 0 1 2 --datasets CIDDS_ctx ToN_ctx CIC17 CIC18 $INJ --methods ernp_N5 ernp_N10 ernp_N20 ssf_causal_N10 owad_N10 &
  ev rq2 --datasets $INJ --methods ernp ernp_B0 abl_self abl_evid_rep $SOTA memstream_adapt new_ip --compare ernp $SOTA &
  ev camo --datasets CIDDS_ctx_injcamo1_bf CIDDS_ctx_injcamo3_bf --methods ernp abl_evid_rep ernp_B0 ssf_causal &
  wait
  ev rq2gate --datasets $INJ CIDDS_ctx_injcamo1_bf CIDDS_ctx_injcamo3_bf --methods ernp abl_prior_none abl_retr_none abl_evid_rep ssf_causal &
  ev rq2_sweep --seeds 0 1 2 --datasets $SW --methods ernp ssf_causal owad memstream memstream_adapt &
  ev rq3 --seeds 0 1 2 --datasets ToN_ctx CIC18 --methods ernp_B25 ernp_B50 ernp_B100 ernp_B400 \
    ssf_causal_B25 ssf_causal_B50 ssf_causal_B100 ssf_causal_B400 owad_B25 owad_B50 owad_B100 owad_B400 &
  ev rq4 --datasets CIDDS_ctx ToN_ctx CIC17 CIC18 --methods ernp abl_prior_none abl_prior_uniform \
    abl_prior_marginal abl_prior_scarf abl_retr_none abl_retr_hard abl_evid_rep abl_feat_flow ernp_B0 &
  wait
  ev rq5 --seeds 0 1 2 --datasets CIDDS_ctx ToN_ctx CIC18 --methods ernp sens_rho0.1 sens_rho0.2 sens_rho0.5 \
    sens_mass0.25 sens_mass0.5 sens_mass2 sens_eps0.5 sens_eps2 &
  ev cadence --seeds 0 1 2 3 4 --datasets CIDDS_ctx ToN_ctx CIC17 CIC18 --methods ernp_P4 ernp_P1 ssf_causal_P4 ssf_causal_P1 owad_P4 owad_P1 &
  ev t1 --datasets CIDDS_ctx ToN_ctx CIC17 CIC18 --methods ernp cade cade_t1 trident trident_t1 knn &
  ev x10 --seeds 0 1 2 --datasets CIC18_x10 --methods ernp $SOTA memstream_adapt &
  wait
fi
if [ "$STAGE" = analysis ] || [ "$STAGE" = all ]; then
  run() { n=$1; shift; "$@" > "logs_k5a/$n.log" 2>&1; echo "$n exit $?"; }
  run real python real_report.py &
  run e23 python e23_report.py &
  run e45 python e45_report.py &
  wait
  run prefeedback sh -c "python prefeedback_analysis.py && python prefeedback_analysis.py --report" &
  run soc python soc_load.py &
  run appendix env ERNP_AUDIT_RUN=ernp ERNP_DIAM_RUN=ernp python appendix_data.py &
  wait
  run figure python figure_data.py &
  run ton python ton_burst_analysis.py &
  run retr sh -c "python retraction_time_analysis.py && python retraction_time_report.py" &
  run p23 python analysis_p23.py --datasets CIDDS_ctx ToN_ctx CIC17 CIC18 &
  wait
  run x10r python scale_x10_report.py
fi
cp -p reports/* "$ROOT/exp/reports/"
echo REPORT_K5A_DONE
