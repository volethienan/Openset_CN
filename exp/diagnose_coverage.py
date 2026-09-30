"""Design check for FERN (development seeds 0-1 only).

For every ERNP false positive on stream day d: is each of its feature values already COVERED by trusted benign
evidence available before day d (anchor benign + benign flows alerted on days < d, i.e. what the analyst confirmed)?
  marginal-novel : at least one feature value never seen in trusted benign  -> FERN's marginal channel can act
  joint-novel    : every value seen, only the combination is new             -> only the dependence channel can act
Coverage is per feature on a 0.05-sd grid (+-1 cell).
"""
import os, sys, importlib, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import seeded_name

G = 0.05; NB = int(20 / G) + 1
def cells(M): return np.clip(np.round((M + 10) / G).astype(int), 0, NB - 1)
def cover(M):
    c = np.zeros((M.shape[1], NB), bool); ci = cells(M)
    for j in range(M.shape[1]): c[j, np.unique(ci[:, j])] = True
    return c | np.roll(c, 1, 1) | np.roll(c, -1, 1)

rows = []
for base in ["CIDDS_ctx", "ToN_ctx", "CIC18"]:
    for s in (0, 1):
        os.environ["EXP_SEED"] = str(s); import common; importlib.reload(common)
        ds = seeded_name(base, s); df, X, src, dst, tr, ho, s_idx, anc = common.load(ds); fam = df["fam"].values[s_idx]
        F = np.load(common.RESULTS + f"{ds}__ernp__s{s}.npz")["flag"].astype(bool)
        day = common.stream_days(df, s_idx); ben = fam == "BENIGN"; Xs = X[s_idx]
        cov = cover(X[tr]); d = np.arange(X.shape[1])
        for dd in dict.fromkeys(day):
            m = day == dd; fp = m & F & ben
            if fp.sum() >= 20:
                unc = ~cov[d[None, :], cells(Xs[fp])]            # uncovered feature values of each FP
                n_unc = unc.sum(1)
                rows.append(dict(dataset=base, seed=s, day=dd, n_fp=int(fp.sum()), fpr=fp.sum() / max(1, (m & ben).sum()),
                                 marginal_novel=(n_unc > 0).mean(), joint_novel=(n_unc == 0).mean(),
                                 mean_uncovered_feats=n_unc.mean()))
            conf = m & F & ben                                    # confirmed benign of day dd enter the evidence
            if conf.any(): cov |= cover(Xs[conf])
r = pd.DataFrame(rows)
r.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports", "diagnostics_coverage_dev.csv"), index=False)
w = r.assign(w_m=r.marginal_novel * r.n_fp, w_u=r.mean_uncovered_feats * r.n_fp).groupby("dataset")[["n_fp", "w_m", "w_u"]].sum()
print(pd.DataFrame({"n_fp": w.n_fp, "share_marginal_novel": (w.w_m / w.n_fp).round(3),
                    "mean_uncovered_feats": (w.w_u / w.n_fp).round(2)}).to_string())
