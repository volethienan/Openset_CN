"""RQ2 at a common operating point (no model is run): recall of the final payload when each method's threshold is set
at 1% FPR on the benign flows of the stream (an oracle point, as zd_recall_at1fpr in RQ1). Separates detection from the
decision rule for the evidence-gated variants, whose own rules have very different FPRs.
-> reports/rq2_matched.csv"""
import os, sys, importlib, numpy as np, pandas as pd
EXP = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, EXP)
from evaluate import seeded_name
SC = ["CIDDS_ctx_injinit_bf", "CIDDS_ctx_injresp_bf", "CIDDS_ctx_injcamo1_bf", "CIDDS_ctx_injcamo3_bf"]
M = ["ernp", "abl_evid_rep", "abl_retr_none", "abl_prior_none", "ssf_causal"]
rows = []
for base in SC:
    for s in range(5):
        os.environ["EXP_SEED"] = str(s); import common; importlib.reload(common)
        ds = seeded_name(base, s); df, X, src, dst, tr, ho, si, anc = common.load(ds); del X
        fam = df["fam"].values[si]; ben, pay = fam == "BENIGN", fam == "INJ_attack"
        for m in M:
            p = os.path.join(common.RESULTS, f"{ds}__{m}__s{s}.npz")
            if not os.path.exists(p): continue
            sc = np.load(p)["score"]; thr = np.quantile(sc[ben], 0.99)
            rows.append(dict(dataset=base, method=m, seed=s, payload_at1fpr=float((sc[pay] > thr).mean())))
r = pd.DataFrame(rows); r.to_csv(os.path.join(EXP, "reports", "rq2_matched.csv"), index=False)
print(r.groupby(["dataset", "method"]).payload_at1fpr.mean().unstack().round(3).to_string())
