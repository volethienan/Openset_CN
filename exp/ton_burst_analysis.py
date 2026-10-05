"""ToN: how much of each method's result comes from one event?
On 2019-04-29 between 12:00 and 13:00 the gateway 192.168.1.1 emits ~20k sampled benign flows (~2M in the full data,
~73k flows per 10 s) with only ~450 distinct timestamps. This one hour is ~95% of that day's benign flows.
Recomputes the ToN metrics of every method with all flows of that host in that hour removed from the evaluation
(the models are unchanged; their saved scores and decisions are re-scored)."""
import os, sys, importlib, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.metrics import roc_auc_score
from evaluate import seeded_name

M = ["ernp", "ssf_causal", "owad", "owad_lr3", "cade", "trident", "memstream", "memstream_adapt", "knn", "iforest", "ae_static", "ernp_P4", "ssf_causal_P4", "owad_P4"]
rows = []
for s in range(5):
    os.environ["EXP_SEED"] = str(s); import common; importlib.reload(common)
    ds = seeded_name("ToN_ctx", s); df, X, src, dst, tr, ho, s_idx, anc = common.load(ds); del X
    t = pd.to_datetime(df["t"].values[s_idx], unit="s")
    burst = (df["src"].values[s_idx] == "192.168.1.1") & (t.strftime("%Y-%m-%d") == "2019-04-29") & (t.hour == 12)
    f = df["fam"].values[s_idx]; y = f != "BENIGN"
    for m in M:
        p = common.RESULTS + f"{ds}__{m}__s{s}.npz"
        if not os.path.exists(p): continue
        r = np.load(p); sc, fl = r["score"], r["flag"].astype(bool)
        for lab, keep in (("all flows", np.ones(len(y), bool)), ("without burst hour", ~burst)):
            yy, ff, ss = y[keep], fl[keep], sc[keep]
            tp, fp = (ff & yy).sum(), (ff & ~yy).sum(); fn, tn = (~ff & yy).sum(), (~ff & ~yy).sum()
            P, R = tp / max(1, tp + fp), tp / max(1, tp + fn); den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            rows.append(dict(method=m, seed=s, eval=lab, burst_flows=int(burst.sum()), F1=2 * P * R / max(1e-9, P + R),
                             MCC=(float(tp) * tn - float(fp) * fn) / den if den else 0, FPR=fp / max(1, fp + tn), AUROC=roc_auc_score(yy, ss)))
r = pd.DataFrame(rows)
r.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports", "ton_burst_analysis.csv"), index=False)
print("burst flows per seed:", r.groupby("seed").burst_flows.first().tolist())
print(r.groupby(["method", "eval"])[["F1", "MCC", "FPR", "AUROC"]].mean().unstack("eval").round(3).to_string())
