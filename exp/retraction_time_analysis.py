"""RQ4 follow-up: what does retraction do, where and when? (ERNP vs ERNP without retraction, RQ1 streams, seeds 0-4)

No model is re-trained. Each run's analyst evidence is reconstructed exactly from its saved decisions: every day the
alerted flows are grouped with the same MiniBatchKMeans call as ernp.py (k = min(B, #alerts), random_state = seed) and
each group is labelled by its representative. Sanity gate: the number of reconstructed decisions must equal the run's
labels_used, otherwise the run is reported and skipped.

For every stream day d we then measure, separately for flows near and far from the benign evidence confirmed before d:
  * FPR of benign flows ("repeat false alarms" near the evidence: the analyst already confirmed that region benign),
  * zero-day recall (near: possible collateral damage of retraction; far: preservation, Prop. 3),
  * the share of the day's analyst decisions spent on benign groups.
Two evidence regions: each run's OWN confirmed groups, and ERNP's groups for both runs (a common region).
Distances are in the standardised feature space, in units of the retention bandwidth eps (same for both runs).
"""
import os, sys, json, importlib, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sklearn; sklearn.set_config(working_memory=256)
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from evaluate import seeded_name

BASES = ["CIDDS_ctx", "ToN_ctx", "CIC17", "CIC18"]
VARIANTS = ["ernp", "abl_retr_none"]
B = 200; CAP = 15000                    # daily budget of the RQ1 runs; flows per day and class kept for distances
BINS = [(0, 1), (1, 3), (3, np.inf)]   # distance to evidence in eps: inside the kernel / shoulder / far
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")

def evidence(common, ds, s, X, fam, s_idx, day, order, flag):
    """benign evidence (row idx, day) and per-day decisions, replayed from the saved flags exactly as in ernp.run"""
    ev_idx, ev_day, dec = [], [], []
    for d in order:
        idx = np.where(day == d)[0]; al = s_idx[idx][flag[idx]]
        if len(al) < 2: dec.append((d, 0, 0)); continue
        k = min(B, len(al)); km = MiniBatchKMeans(k, random_state=s, n_init=3).fit(X[al]); nb = n = 0
        for cl in range(k):
            mem = al[km.labels_ == cl]
            if not len(mem): continue
            rep = mem[np.argmin(np.linalg.norm(X[mem] - km.cluster_centers_[cl], axis=1))]; n += 1
            if fam[rep] == "BENIGN": nb += 1; ev_idx.append(mem); ev_day.append(np.full(len(mem), d, object))
        dec.append((d, n, nb))
    ev_idx = np.concatenate(ev_idx) if ev_idx else np.zeros(0, int)
    ev_day = np.concatenate(ev_day) if ev_day else np.zeros(0, object)
    return ev_idx, ev_day, dec

def main():
    rows, drows, gate = [], [], []
    for base in BASES:
        for s in range(5):
            t0 = time.time(); os.environ["EXP_SEED"] = str(s); import common; importlib.reload(common)
            ds = seeded_name(base, s); df, X, src, dst, tr, ho, s_idx, anc = common.load(ds)
            fam_all = df["fam"].values; fam = fam_all[s_idx]; y = fam != "BENIGN"
            known = set(fam_all[anc & (fam_all != "BENIGN")]); zd = y & ~np.isin(fam, list(known))
            day = common.stream_days(df, s_idx); order = list(dict.fromkeys(day))
            rng = np.random.default_rng(s); res = {}
            for v in VARIANTS:
                p = common.RESULTS + f"{ds}__{v}__s{s}"
                flag = np.load(p + ".npz")["flag"].astype(bool); info = json.load(open(p + ".json"))
                ev_idx, ev_day, dec = evidence(common, ds, s, X, fam_all, s_idx, day, order, flag)
                n_dec = sum(n for _, n, _ in dec); ok = n_dec == info["labels_used"]
                gate.append(dict(dataset=base, seed=s, variant=v, reconstructed=n_dec, labels_used=info["labels_used"], ok=ok))
                if not ok: print("GATE FAIL", ds, v, n_dec, info["labels_used"], flush=True)
                res[v] = dict(flag=flag, ev_idx=ev_idx, ev_day=ev_day, dec=dict((d, (n, nb)) for d, n, nb in dec), eps=info["eps"])
            assert abs(res["ernp"]["eps"] - res["abl_retr_none"]["eps"]) < 1e-9, "eps differs between variants"
            eps = res["ernp"]["eps"]
            for i, d in enumerate(order):
                idx = np.where(day == d)[0]
                for v in VARIANTS:
                    fl = res[v]["flag"][idx]; n, nb = res[v]["dec"][d]
                    drows.append(dict(dataset=base, seed=s, variant=v, day_no=i, day=d, n_benign=int((~y[idx]).sum()),
                                      fp=int((fl & ~y[idx]).sum()), n_zd=int(zd[idx].sum()), tp_zd=int((fl & zd[idx]).sum()),
                                      decisions=n, benign_decisions=nb))
                if i == 0: continue                                         # no evidence before the first stream day
                ben = idx[~y[idx]]; zdi = idx[zd[idx]]
                ben = ben[rng.choice(len(ben), min(CAP, len(ben)), replace=False)] if len(ben) else ben
                zdi = zdi[rng.choice(len(zdi), min(CAP, len(zdi)), replace=False)] if len(zdi) else zdi
                Z = X[s_idx[np.r_[ben, zdi]]]; isb = np.r_[np.ones(len(ben), bool), np.zeros(len(zdi), bool)]
                for region in ("own", "ernp"):
                    for v in VARIANTS:
                        r = res[v if region == "own" else "ernp"]; prev = r["ev_idx"][np.isin(r["ev_day"], order[:i])]
                        dist = (NearestNeighbors(n_neighbors=1).fit(X[prev]).kneighbors(Z)[0][:, 0] / eps
                                if len(prev) else np.full(len(Z), np.inf))
                        fl = res[v]["flag"][np.r_[ben, zdi]]
                        for lo, hi in BINS:
                            m = (dist >= lo) & (dist < hi)
                            rows.append(dict(dataset=base, seed=s, variant=v, region=region, day_no=i, bin=f"{lo}-{hi}",
                                             n_benign=int((m & isb).sum()), fp=int((m & isb & fl).sum()),
                                             n_zd=int((m & ~isb).sum()), tp_zd=int((m & ~isb & fl).sum())))
            print(f"done {ds} s{s} ({len(order)} days, {time.time() - t0:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(OUT, "retraction_regions.csv"), index=False)
            pd.DataFrame(drows).to_csv(os.path.join(OUT, "retraction_daily.csv"), index=False)
            pd.DataFrame(gate).to_csv(os.path.join(OUT, "retraction_gate.csv"), index=False)

if __name__ == "__main__":
    main()
