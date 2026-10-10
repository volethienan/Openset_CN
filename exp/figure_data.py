"""Time-resolved data for the paper figures, from the saved per-flow decisions (no model is run).
  reports/fig_rq2_timeline.csv : RQ2 injection streams, per stream day: alert rate on injected flows (ramp, then payload)
                                 and FPR on benign flows
  reports/fig_ton_timeline.csv : NF-ToN-IoT-v3, per 4 h bin: FPR on benign flows and number of benign flows
"""
import os, sys, importlib, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import seeded_name
REP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")

def runs(base, methods, seeds=range(5)):
    for s in seeds:
        os.environ["EXP_SEED"] = str(s); import common; importlib.reload(common)
        ds = seeded_name(base, s); df, X, src, dst, tr, ho, s_idx, anc = common.load(ds); del X
        for m in methods:
            p = common.RESULTS + f"{ds}__{m}__s{s}.npz"
            if os.path.exists(p): yield s, m, df, s_idx, np.load(p)["flag"].astype(bool)

rows = []
for base in ["CIDDS_ctx_injvalidinit_bf", "CIDDS_ctx_injvalidresp_bf", "CIDDS_ctx_injinit_bf", "CIDDS_ctx_injinit"]:
    for s, m, df, s_idx, fl in runs(base, ["ernp", "abl_self", "ernp_B25", "ernp_B0", "ssf_causal", "ssf_causal_B25", "owad", "memstream", "memstream_adapt", "trident", "cade"]):
        t = df["t"].values[s_idx]; day = ((t - df["t"].min()) // 86400).astype(int); fam = df["fam"].values[s_idx]
        inj, ben = np.isin(fam, ["INJ_ramp", "INJ_attack"]), fam == "BENIGN"
        for d in np.unique(day):
            k = day == d
            rows.append(dict(dataset=base, method=m, seed=s, day=int(d), inj_alert=fl[k & inj].mean() if (k & inj).any() else np.nan,
                             payload_share=(k & (fam == "INJ_attack")).sum() / max(1, (k & inj).sum()), FPR=fl[k & ben].mean()))
pd.DataFrame(rows).to_csv(os.path.join(REP, "fig_rq2_timeline.csv"), index=False); print("rq2 timeline")

rows = []
for s, m, df, s_idx, fl in runs("ToN_ctx", ["ernp", "ernp_P4", "ssf_causal", "ssf_causal_P4", "owad"]):
    t = df["t"].values[s_idx]; b = (t // (4 * 3600)).astype(np.int64); ben = df["fam"].values[s_idx] == "BENIGN"
    burst = (df["src"].values[s_idx] == "192.168.1.1") & (pd.to_datetime(t, unit="s").strftime("%Y-%m-%d %H") == "2019-04-29 12")
    for v in np.unique(b):
        k = (b == v) & ben
        if k.any(): rows.append(dict(method=m, seed=s, bin_start=int(v) * 4 * 3600, n_benign=int(k.sum()), n_burst=int((k & burst).sum()),
                                     FPR=fl[k].mean()))
pd.DataFrame(rows).to_csv(os.path.join(REP, "fig_ton_timeline.csv"), index=False); print("ton timeline")
