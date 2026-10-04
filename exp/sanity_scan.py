"""Sanity scan of all RQ1 results (no model is run). For every method x dataset x seed:
  AUROC_known : known-family attacks vs benign on the stream. Any reasonable detector ranks attacks of families it was
                trained on above benign traffic; < 0.6 points to an inverted score or a configuration problem.
  AUROC_zd    : zero-day attacks vs benign.
  F1, FPR, alert rate.
Flags: AUROC_known < 0.6, AUROC_zd < 0.5, F1 < 0.05, FPR > 0.5, alert rate > 0.9, NaN/inf scores.
"""
import os, sys, importlib, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.metrics import roc_auc_score
from evaluate import seeded_name

M = ["ernp", "ssf_causal", "owad", "cade", "trident", "memstream", "memstream_adapt", "knn", "iforest", "ae_static", "new_ip"]
rows = []
for base in ["CIDDS_ctx", "ToN_ctx", "CIC17", "CIC18"]:
    for s in range(5):
        os.environ["EXP_SEED"] = str(s); import common; importlib.reload(common)
        ds = seeded_name(base, s); df, X, src, dst, tr, ho, s_idx, anc = common.load(ds); del X
        fam = df["fam"].values; f = fam[s_idx]; y = f != "BENIGN"
        known = set(fam[anc & (fam != "BENIGN")]); kn = y & np.isin(f, list(known)); zd = y & ~kn
        for m in M:
            p = common.RESULTS + f"{ds}__{m}__s{s}.npz"
            if not os.path.exists(p): continue
            r = np.load(p); sc, fl = r["score"].astype(float), r["flag"].astype(bool)
            bad = int((~np.isfinite(sc)).sum()); sc = np.nan_to_num(sc, nan=0, posinf=1e30, neginf=-1e30)
            auc = lambda mask: roc_auc_score(np.r_[np.zeros((~y).sum()), np.ones(mask.sum())], np.r_[sc[~y], sc[mask]]) if mask.any() else np.nan
            tp, fp = (fl & y).sum(), (fl & ~y).sum(); P = tp / max(1, tp + fp); R = tp / max(1, y.sum())
            rows.append(dict(dataset=base, seed=s, method=m, AUROC_known=auc(kn), AUROC_zd=auc(zd), F1=2 * P * R / max(1e-9, P + R),
                             FPR=fp / max(1, (~y).sum()), alert_rate=fl.mean(), nonfinite=bad))
r = pd.DataFrame(rows)
r.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports", "sanity_scan.csv"), index=False)
g = r.groupby(["dataset", "method"]).mean(numeric_only=True).drop(columns="seed")
print(g.round(3).to_string())
flags = []
for (d, m), v in g.iterrows():
    if v.AUROC_known < 0.6: flags.append(f"{d} {m}: AUROC on KNOWN families {v.AUROC_known:.2f}")
    if v.AUROC_zd < 0.5: flags.append(f"{d} {m}: zero-day AUROC below chance {v.AUROC_zd:.2f}")
    if v.F1 < 0.05 and m != "new_ip": flags.append(f"{d} {m}: F1 {v.F1:.3f}")
    if v.FPR > 0.5: flags.append(f"{d} {m}: FPR {v.FPR:.2f}")
    if v.nonfinite > 0: flags.append(f"{d} {m}: non-finite scores")
print("\nFLAGS\n" + "\n".join(flags))
