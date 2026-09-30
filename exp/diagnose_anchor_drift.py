"""Feasibility of direction B (development seeds 0-1): does the labelled anchor period already contain the drift that
causes ERNP's false positives on the first stream days?

Split anchor benign chronologically: EARLY (first half) / LATE (second half). Train ERNP on EARLY only (tau at 1% FPR on
a random EARLY holdout) and measure the FPR on LATE benign. A high LATE FPR means benign recombines over time already
inside the anchor, so the anchor can teach the detector that drift before the stream starts.
Also reports whether ERNP's stream false positives (full model) lie near LATE-anchor benign flows that the EARLY model
flagged (anchor "replay" evidence) - i.e. whether replayed anchor evidence would cover them.
"""
import os, sys, importlib, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.neighbors import NearestNeighbors
from evaluate import seeded_name
import ernp as E

rows = []
for base in ["CIDDS_ctx", "ToN_ctx", "CIC18"]:
    for s in (0, 1):
        os.environ["EXP_SEED"] = str(s); import common; importlib.reload(common)
        ds = seeded_name(base, s); df, X, src, dst, tr, ho, s_idx, anc = common.load(ds); fam = df["fam"].values
        t = df["t"].values; rng = np.random.default_rng(s)
        ab = np.where(anc & (fam == "BENIGN"))[0]; ab = ab[np.argsort(t[ab], kind="stable")]
        early, late = ab[: len(ab) // 2], ab[len(ab) // 2:]
        ho_e = rng.choice(early, len(early) // 5, replace=False); ref_e = np.setdiff1d(early, ho_e)
        ka = np.where(anc & (fam != "BENIGN") & (t <= t[early[-1]]))[0]
        c = dict(E.DEFAULTS); m = E.ERNP(c, rng)
        P = X[rng.choice(ref_e, min(40000, len(ref_e)), replace=False)]
        A = X[rng.choice(ka, min(10000, len(ka)), replace=False)] if len(ka) else np.zeros((0, X.shape[1]), X.dtype)
        m.eps = np.quantile(NearestNeighbors(n_neighbors=2).fit(P).kneighbors(P[:5000])[0][:, 1], 0.9)
        m.fit(P, A); tau = common.calib_upper(m.score(X[ho_e]))
        late_flag = m.score(X[late]) > tau
        # stream FPs of the full ERNP run and their distance to replay evidence (LATE benign flagged by the EARLY model)
        F = np.load(common.RESULTS + f"{ds}__ernp__s{s}.npz")["flag"].astype(bool)
        fp = s_idx[F & (fam[s_idx] == "BENIGN")]; day = common.stream_days(df, s_idx)[F & (fam[s_idx] == "BENIGN")]
        first = np.isin(day, list(dict.fromkeys(common.stream_days(df, s_idx)))[:2])
        R = X[late[late_flag]]
        cov = np.nan
        if len(R) and first.any():
            q = fp[first]; q = q[rng.choice(len(q), min(5000, len(q)), replace=False)]
            d = NearestNeighbors(n_neighbors=1).fit(R).kneighbors(X[q])[0][:, 0]; cov = float((d < 2 * m.eps).mean())
        rows.append(dict(dataset=base, seed=s, anchor_hours=(t[ab[-1]] - t[ab[0]]) / 3600, n_late=len(late),
                         late_fpr=float(late_flag.mean()), replay_evidence=int(late_flag.sum()),
                         first2days_fp_covered_by_replay=cov, eps=float(m.eps)))
        print(rows[-1], flush=True)
r = pd.DataFrame(rows)
r.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports", "diagnostics_anchor_drift_dev.csv"), index=False)
print(r.groupby("dataset").mean(numeric_only=True).drop(columns="seed").round(3).to_string())
