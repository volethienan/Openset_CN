#!/usr/bin/env bash
# Reports for run_review2.sh (no model is run)
cd "$(dirname "$0")"; export PYTHONIOENCODING=utf-8; mkdir -p reports
INJ="CIDDS_ctx_injinit CIDDS_ctx_injresp CIDDS_ctx_injinit_bf CIDDS_ctx_injresp_bf"
python evaluate.py --out rq2gate --datasets $INJ CIDDS_ctx_injcamo1_bf CIDDS_ctx_injcamo3_bf \
  --methods ernp abl_prior_none abl_retr_none abl_evid_rep ssf_causal > reports/rq2gate.txt 2>&1; echo rq2gate
python evaluate.py --out t1 --datasets CIDDS_ctx ToN_ctx CIC17 CIC18 --methods ernp cade cade_t1 trident trident_t1 knn \
  > reports/t1.txt 2>&1; echo t1
python appendix_data.py > reports/appendix_data.txt 2>&1; echo appendix
python - <<'EOF'
import pandas as pd
r = pd.read_csv("reports/rq2gate_runs.csv")
print("payload recall / FPR, mean over seeds")
print(r.groupby(["dataset", "method"])[["payload_recall", "FPR"]].mean().round(3).unstack("method").to_string())
t = pd.read_csv("reports/t1_runs.csv")
print(t.groupby(["dataset", "method"])[["F1", "MCC", "FPR", "zero_day_recall"]].mean().round(3).to_string())
EOF
