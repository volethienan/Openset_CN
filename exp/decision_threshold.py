"""Separate decision threshold (alert reported as a detection) from the triage threshold (alerts sent to the analyst).

ERNP learns exactly as before: alerts above tau(1%) are grouped and labelled. A detection is counted only above a
tighter decision threshold tau_dec = (1 - f) quantile of the INITIAL model's holdout scores, f fixed for all datasets.
The initial model is rebuilt with the same seeds; its 1% threshold must equal the stored tau (checked).
f is chosen on development data only (seeds 0-1 of CIDDS, ToN, CIC18) and evaluated on seeds 2-4 and on CIC17.
"""
import os, sys, json, importlib, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.neighbors import NearestNeighbors
from evaluate import seeded_name
import ernp as E

F_GRID = [0.01, 0.005, 0.003, 0.002, 0.001]
rows = []
for base in ["CIDDS_ctx", "ToN_ctx", "CIC18", "CIC17"]:
    for s in range(5):
        os.environ["EXP_SEED"] = str(s); import common; importlib.reload(common); importlib.reload(E)
        ds = seeded_name(base, s); df, X, src, dst, tr, ho, s_idx, anc = common.load(ds); fam = df["fam"].values
        c = dict(E.DEFAULTS); rng = np.random.default_rng(s)               # replicate ernp.run() up to the initial fit
        P = X[rng.choice(tr, min(c["n_pos_max"], len(tr)), replace=False)]
        ka = np.where(anc & (fam != "BENIGN"))[0]
        A = X[rng.choice(ka, min(c["n_known_max"], len(ka)), replace=False)] if len(ka) else np.zeros((0, X.shape[1]), X.dtype)
        m = E.ERNP(c, rng)
        m.eps = c["eps_mult"] * np.quantile(NearestNeighbors(n_neighbors=2).fit(P).kneighbors(P[:5000])[0][:, 1], 0.9)
        m.fit(P, A); h = m.score(X[ho])
        info = json.load(open(common.RESULTS + f"{ds}__ernp__s{s}.json"))
        assert np.isclose(common.calib_upper(h), info["tau"], rtol=1e-6, atol=1e-9), (ds, common.calib_upper(h), info["tau"])
        sc = np.load(common.RESULTS + f"{ds}__ernp__s{s}.npz")["score"]
        f_s = fam[s_idx]; y = f_s != "BENIGN"
        known = set(fam[anc & (fam != "BENIGN")]); zd = y & ~np.isin(f_s, list(known))
        for f in F_GRID:
            F = sc > np.quantile(h, 1 - f)
            tp, fp, fn, tn = (F & y).sum(), (F & ~y).sum(), (~F & y).sum(), (~F & ~y).sum()
            Pr, Rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
            den = np.sqrt(float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn))
            rows.append(dict(dataset=base, seed=s, f=f, F1=2 * Pr * Rc / max(Pr + Rc, 1e-9), MCC=(float(tp) * tn - float(fp) * fn) / den if den else 0.0,
                             FPR=fp / max(fp + tn, 1), zero_day_recall=(F & zd).sum() / max(zd.sum(), 1)))
        print(ds, "tau check OK", flush=True)
r = pd.DataFrame(rows)
r.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports", "decision_threshold_runs.csv"), index=False)
dev = r[(r.seed <= 1) & (r.dataset != "CIC17")]
print("\nDEVELOPMENT (seeds 0-1, CIDDS/ToN/CIC18): mean over datasets\n" + dev.groupby("f")[["F1", "MCC", "zero_day_recall", "FPR"]].mean().round(4).to_string())
