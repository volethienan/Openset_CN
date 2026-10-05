#!/usr/bin/env bash
# Regenerate the run tables affected by the OWAD/MemStream fixes (05/10/2026); subset of report_all.sh
cd "$(dirname "$0")"; export PYTHONIOENCODING=utf-8; mkdir -p reports
SOTA="ssf_causal owad memstream trident cade"
INJ="CIDDS_ctx_injinit CIDDS_ctx_injresp CIDDS_ctx_injinit_bf CIDDS_ctx_injresp_bf"
python evaluate.py --out rq1 --datasets CIDDS_ctx ToN_ctx CIC17 CIC18 --methods ernp $SOTA ernp_ssfproto ssf new_ip \
  --compare ernp $SOTA > reports/rq1.txt 2>&1; echo rq1
python evaluate.py --out rq1_ext --datasets CIDDS_ctx ToN_ctx CIC17 CIC18 --methods ernp abl_evid_rep abl_self $SOTA memstream_adapt owad_lr3 \
  iforest knn ae_static ssf_causal_E180 ssf_causal_W20000 > reports/rq1_ext.txt 2>&1; echo rq1_ext
python evaluate.py --out scale --seeds 0 1 2 --datasets CIC17_full --methods ernp $SOTA memstream_adapt > reports/scale.txt 2>&1; echo scale
python evaluate.py --out rq2lb --datasets $INJ --methods ernp_B25 ssf_causal_B25 owad_B25 > reports/rq2lb.txt 2>&1; echo rq2lb
python evaluate.py --out noise --seeds 0 1 2 --datasets CIDDS_ctx ToN_ctx CIC17 CIC18 $INJ --methods ernp_N5 ernp_N10 ernp_N20 \
  ssf_causal_N10 owad_N10 > reports/noise.txt 2>&1; echo noise
python evaluate.py --out rq2 --datasets $INJ --methods ernp ernp_B0 abl_self $SOTA memstream_adapt new_ip --compare ernp $SOTA > reports/rq2.txt 2>&1; echo rq2
SW=$(for r in 3 7 14; do for q in 10 30 100; do [ "$r$q" = "730" ] && echo -n "CIDDS_ctx_injinit_bf " || echo -n "CIDDS_ctx_injinit_bf_r${r}_q${q} "; done; done)
python evaluate.py --out rq2_sweep --seeds 0 1 2 --datasets $SW --methods ernp ssf_causal owad memstream memstream_adapt > reports/rq2_sweep.txt 2>&1; echo rq2_sweep
python evaluate.py --out rq3 --seeds 0 1 2 --datasets ToN_ctx CIC18 --methods ernp_B25 ernp_B50 ernp_B100 ernp_B400 \
  ssf_causal_B25 ssf_causal_B50 ssf_causal_B100 ssf_causal_B400 owad_B25 owad_B50 owad_B100 owad_B400 > reports/rq3.txt 2>&1; echo rq3
python evaluate.py --out cadence --seeds 0 1 2 3 4 --datasets CIDDS_ctx ToN_ctx CIC17 CIC18 --methods ernp_P4 ernp_P1 ssf_causal_P4 ssf_causal_P1 owad_P4 owad_P1 > reports/cadence.txt 2>&1; echo cadence
python evaluate.py --out x10 --seeds 0 1 2 --datasets CIC18_x10 --methods ernp $SOTA memstream_adapt > reports/x10.txt 2>&1; echo x10
echo REPORT_UPDATE_DONE
