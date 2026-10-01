"""Where do the false positives on CIDDS-001 come from? (development seeds 0-1)

A false positive is 'collateral' if, within +-10 minutes, an attack flow involves one of its two endpoints.
Compares ERNP with SSF: share of collateral FPs, FPs per day, and how many FPs each method has outside attack windows.
"""
import os, sys, importlib, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import seeded_name

W = 600; rows = []
for s in (0, 1):
    os.environ["EXP_SEED"] = str(s); import common; importlib.reload(common)
    ds = seeded_name("CIDDS_ctx", s); df, X, src, dst, tr, ho, s_idx, anc = common.load(ds)
    t = df["t"].values[s_idx]; fam = df["fam"].values[s_idx]; a = fam != "BENIGN"
    sr, ds_ = src[s_idx], dst[s_idx]
    att_t = {}                                            # endpoint -> sorted times of attack flows touching it
    for e, tt in zip(np.r_[sr[a], ds_[a]], np.r_[t[a], t[a]]): att_t.setdefault(e, []).append(tt)
    att_t = {e: np.sort(v) for e, v in att_t.items()}
    def near(i):
        for e in (sr[i], ds_[i]):
            v = att_t.get(e)
            if v is not None:
                j = np.searchsorted(v, t[i]);
                if (j < len(v) and v[j] - t[i] <= W) or (j > 0 and t[i] - v[j - 1] <= W): return True
        return False
    ben = ~a; coll = np.array([near(i) for i in np.where(ben)[0]]); cmask = np.zeros(len(t), bool); cmask[np.where(ben)[0][coll]] = True
    for m in ["ernp", "ssf", "ssf_causal", "owad", "cade"]:
        f = common.RESULTS + f"{ds}__{m}__s{s}.npz"
        if not os.path.exists(f): continue
        F = np.load(f)["flag"].astype(bool); fp = F & ben
        rows.append(dict(seed=s, method=m, n_fp=int(fp.sum()), collateral_fp=int((fp & cmask).sum()),
                         noncollateral_fp=int((fp & ~cmask).sum()), tp=int((F & a).sum()), fn=int((~F & a).sum()),
                         benign_collateral_total=int(cmask.sum())))
r = pd.DataFrame(rows); print(r.groupby("method").mean(numeric_only=True).drop(columns="seed").round(0).to_string())
r.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports", "diagnostics_collateral_dev.csv"), index=False)
