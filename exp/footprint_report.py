"""Footprint of wrongly confirmed benign groups (Proposition 1(iii)) on the saved ERNP runs. No model is run.

For each run of the final configuration (ernp_K5A, seven datasets, five seeds) the analyst decisions are replayed
from the saved alerts with the grouping of ernp.run (MiniBatchKMeans, representative closest to the centroid). The
replay must reproduce the run's own decision log (info["audit"]: day, label, size, wrong members, family of the
representative, radius) on every day; a day it does not reproduce (clustering of many duplicate alerts
can differ across machines) keeps the replayed groups and is counted in replay_off; above 0.5% of the decisions the
report stops. A benign-labelled group is impure when at least one member is an attack.
Proposition 1(iii): the points whose retained prior fraction drops by more than delta because of an impure group lie
within r_G + R_delta of its representative, with r_G the largest member-to-representative distance and
R_delta = eps * sqrt(2 ln(1/delta)) (the bound). The exact region is smaller: the points within R_delta of some member
of an impure group (retention depends on the distance to the nearest confirmed flow). For every later stream day, the
flows inside each region (groups decided on earlier days) are counted: benign flows, zero-day flows (families absent from the anchor) and their recall
inside and outside. delta = 0.5 (more than half of the prior mass retracted) and delta = 0.1.
Output: reports/footprint.txt, reports/footprint_runs.csv
"""
import os, sys, json, importlib, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.cluster import MiniBatchKMeans
from evaluate import seeded_name

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
BASES = ["CIDDS_ctx", "ToN_ctx", "CIC17", "CIC18", "CTU13", "SIMARGL", "APPRAISE"]
METHOD = "ernp_K5A"
DELTAS = (0.5, 0.1)


def impure_groups(common, ds, df, X, s_idx, flag, s, info):
    """replayed decisions -> list of (day, centre, radius, members) of impure benign groups; checked against info['audit']"""
    fam = df["fam"].values; day = common.stream_days(df, s_idx); B = int(info["cfg"]["budget"]); log, out = [], []
    for d in dict.fromkeys(day):
        idx = np.where(day == d)[0]; al = s_idx[idx][flag[idx]]
        if len(al) < 2: continue
        k = min(B, len(al)); km = MiniBatchKMeans(k, random_state=s, n_init=3).fit(X[al])
        for cl in range(k):
            mem = al[km.labels_ == cl]
            if not len(mem): continue
            rep = mem[np.argmin(np.linalg.norm(X[mem] - km.cluster_centers_[cl], axis=1))]
            is_ben = fam[rep] == "BENIGN"; bad = fam[mem] != "BENIGN" if is_ben else fam[mem] == "BENIGN"
            rad = float(np.linalg.norm(X[mem] - X[rep], axis=1).max())
            log.append((d, int(is_ben), len(mem), int(bad.sum()), fam[rep], rad))
            if is_ben and bad.any(): out.append((d, X[rep], rad, X[mem]))
    ref = info["audit"]; assert len(ref) == int(info["labels_used"]), ds
    # MiniBatchKMeans on many duplicate alerts can give a different set of non-empty clusters on another machine or
    # thread count, so the check counts the run's logged decisions (day, label, size, wrong members, family of the
    # representative, radius) that the replay does not reproduce; they must stay below 0.5% of the decisions.
    key = lambda x: (x[0], int(x[1]), int(x[2]), int(x[3]), str(x[4]), round(float(x[5]), 4))
    from collections import Counter
    off = sum((Counter(key(x) for x in ref) - Counter(key(x) for x in log)).values())
    assert off <= 0.005 * len(ref), (ds, "replay misses", off, "of", len(ref), "logged decisions")
    return out, off


def run_one(base, s):
    os.environ["EXP_SEED"] = str(s); import common; importlib.reload(common)
    ds = seeded_name(base, s); df, X, src, dst, tr, ho, s_idx, anc = common.load(ds)
    z = np.load(common.RESULTS + f"{ds}__{METHOD}__s{s}.npz"); flag = z["flag"].astype(bool)
    info = json.load(open(common.RESULTS + f"{ds}__{METHOD}__s{s}.json")); eps = float(info["eps"])
    fam_all = df["fam"].values; fam = fam_all[s_idx]; ben = fam == "BENIGN"
    zd = ~ben & ~np.isin(fam, list(set(fam_all[anc & (fam_all != "BENIGN")])))
    G, off = impure_groups(common, ds, df, X, s_idx, flag, s, info)
    day = common.stream_days(df, s_idx); order = list(dict.fromkeys(day)); rows = []
    from sklearn.neighbors import NearestNeighbors
    for delta in DELTAS:
        R = eps * np.sqrt(2 * np.log(1 / delta)); scored = np.zeros(len(s_idx), bool)
        bound = np.zeros(len(s_idx), bool); exact = np.zeros(len(s_idx), bool)
        for i, d in enumerate(order):
            prev = [g for g in G if order.index(g[0]) < i]
            idx = np.where(day == d)[0]; scored[idx] = i > 0
            if not prev or not len(idx): continue
            C = np.stack([g[1] for g in prev]); rr = np.array([g[2] for g in prev]) + R
            nn = NearestNeighbors(n_neighbors=1).fit(np.vstack([g[3] for g in prev]))
            for j in range(0, len(idx), 20000):
                sl = idx[j:j + 20000]; q = X[s_idx[sl]]
                d2 = (q * q).sum(1)[:, None] - 2 * q @ C.T + (C * C).sum(1)[None, :]
                bound[sl] = (np.sqrt(np.maximum(d2, 0)) <= rr[None, :]).any(1)
                exact[sl] = nn.kneighbors(q)[0][:, 0] < R
        assert not (exact & ~bound).any(), "exact region must lie inside the bound"
        m = scored; r = dict(dataset=base, seed=s, delta=delta, eps=eps, n_impure=len(G), n_zd=int((m & zd).sum()), replay_off=off)
        for nm, inside in (("bound", bound), ("exact", exact)):
            r.update({f"{nm}_share_benign": float(inside[m & ben].mean()),
                      f"{nm}_share_zd": float(inside[m & zd].mean()) if (m & zd).any() else np.nan,
                      f"{nm}_fpr_in": float(flag[m & ben & inside].mean()) if (m & ben & inside).any() else np.nan,
                      f"{nm}_zd_recall_in": float(flag[m & zd & inside].mean()) if (m & zd & inside).any() else np.nan,
                      f"{nm}_zd_recall_out": float(flag[m & zd & ~inside].mean()) if (m & zd & ~inside).any() else np.nan,
                      f"{nm}_n_zd_in": int((m & zd & inside).sum())})
        rows.append(r)
    print("footprint", ds, len(G), "impure groups", flush=True)
    return rows


if __name__ == "__main__":
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40)
    rows = [r for b in BASES for s in range(5) for r in run_one(b, s)]
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "footprint_runs.csv"), index=False)
    t = df.drop(columns=["seed"]).groupby(["delta", "dataset"]).mean()
    with open(os.path.join(OUT, "footprint.txt"), "w", encoding="utf-8") as fh:
        fh.write(__doc__ + "\nMean over five seeds (shares and rates over stream days after the first update)\n" + t.round(4).to_string() + "\n")
    print(open(os.path.join(OUT, "footprint.txt"), encoding="utf-8").read())
