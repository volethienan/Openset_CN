"""Does the DIRECTION of benign drift transfer from the anchor to the stream? (development seeds 0-1)

v_anchor : per-feature mean squared displacement of LATE-anchor benign flows flagged by an EARLY-anchor ERNP model,
           from their nearest EARLY benign flow (what the anchor can teach before the stream starts)
v_stream : same for ERNP's false positives on the first two stream days, from their nearest anchor benign flow
Reports the cosine similarity of the two profiles and the overlap of their top-5 features. For comparison, v_attack is the
profile of zero-day attack flows (to check the drift directions are not also the attack directions).
"""
import os, sys, importlib, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.neighbors import NearestNeighbors
from evaluate import seeded_name
import ernp as E

def disp(Q, R, rng, n=4000):
    Q = Q[rng.choice(len(Q), min(n, len(Q)), replace=False)]
    nn = R[NearestNeighbors(n_neighbors=1).fit(R).kneighbors(Q)[1][:, 0]]; return ((Q - nn) ** 2).mean(0)

cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
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
        m = E.ERNP(dict(E.DEFAULTS), rng); P = X[rng.choice(ref_e, min(40000, len(ref_e)), replace=False)]
        A = X[rng.choice(ka, min(10000, len(ka)), replace=False)] if len(ka) else np.zeros((0, X.shape[1]), X.dtype)
        m.eps = np.quantile(NearestNeighbors(n_neighbors=2).fit(P).kneighbors(P[:5000])[0][:, 1], 0.9)
        m.fit(P, A); tau = common.calib_upper(m.score(X[ho_e])); lf = late[m.score(X[late]) > tau]
        Rref = X[rng.choice(tr, min(40000, len(tr)), replace=False)]
        v_anchor = disp(X[lf], P, rng)
        F = np.load(common.RESULTS + f"{ds}__ernp__s{s}.npz")["flag"].astype(bool); day = common.stream_days(df, s_idx)
        first = np.isin(day, list(dict.fromkeys(day))[:2]); fam_s = fam[s_idx]
        v_stream = disp(X[s_idx[F & first & (fam_s == "BENIGN")]], Rref, rng)
        known = set(fam[anc & (fam != "BENIGN")]); zd = (fam_s != "BENIGN") & ~np.isin(fam_s, list(known))
        v_att = disp(X[s_idx[zd]], Rref, rng) if zd.any() else np.full_like(v_stream, np.nan)
        top = lambda v: set(np.argsort(-v)[:5])
        rows.append(dict(dataset=base, seed=s, cos_anchor_stream=cos(v_anchor, v_stream), top5_overlap=len(top(v_anchor) & top(v_stream)),
                         cos_anchor_attack=cos(v_anchor, v_att), cos_stream_attack=cos(v_stream, v_att),
                         top5_anchor=sorted(top(v_anchor)), top5_stream=sorted(top(v_stream))))
        print(rows[-1], flush=True)
r = pd.DataFrame(rows)
r.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports", "diagnostics_drift_directions_dev.csv"), index=False)
print(r.groupby("dataset")[["cos_anchor_stream", "top5_overlap", "cos_anchor_attack", "cos_stream_attack"]].mean().round(3).to_string())
