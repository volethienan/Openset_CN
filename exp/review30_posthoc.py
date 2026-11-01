"""Review 30/10 post hoc analyses from saved scores (no retraining). Seven methods x seven datasets x five seeds.

L1 (M8)  deployable thresholds instead of the oracle 1% diagnostic: each stream day d >= 1 is thresholded at the 99th
         percentile of the method's scores on the benign flows of days < d (scores as they were produced on those days;
         assumes the benign status of past flows is known). Reported on days >= 1 next to the oracle threshold set on the
         benign flows of those same days, plus zero-day recall at an oracle 0.1% FPR.
L2 (M7)  per zero-day family at each method's own decision rule: hours from the family's first stream flow to its first
         alert, share of families never alerted, and recall on the family's first 100 stream flows.
L3 (T5)  coverage of the swap prior: zero-day flows with at least one feature outside the range of the anchor benign
         reference flows (values the initial swap prior cannot produce), and ERNP's recall inside and outside.
Output: exp/reports/review30_{l1,l2,l3}.csv, exp/reports/review30_posthoc.txt
usage (from Openset_CN/): python exp/review30_posthoc.py
"""
import os, sys, json, importlib
import numpy as np, pandas as pd
EXP = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, EXP)
import evaluate, common

DS = ["CIDDS_ctx", "ToN_ctx", "CIC17", "CIC18", "CTU13", "SIMARGL", "APPRAISE"]
MS = ["ernp_K5A", "ssf_causal", "cade", "trident", "memstream", "owad", "knn"]
SEEDS = range(5); OUT = os.path.join(EXP, "reports")
L1, L2, L3 = [], [], []

for base in DS:
    for s in SEEDS:
        os.environ["EXP_SEED"] = str(s); importlib.reload(common)
        ds = evaluate.seeded_name(base, s)
        df = evaluate.metric_labels(common.STREAMS + f"{ds}.parquet")
        anc = df.t.to_numpy() < common.anchor_end(ds, df.t.min()); s_idx = np.flatnonzero(~anc)
        fam = df.fam.values[s_idx]; t = df.t.values[s_idx]; y = fam != "BENIGN"
        known = set(df.fam.values[anc & (df.fam.values != "BENIGN")]); unk = y & ~np.isin(fam, list(known))
        days = common.stream_days(df, s_idx); order = sorted(set(days)); di = np.searchsorted(order, days)
        fams = sorted(set(fam[unk]))
        for m in MS:
            f = common.RESULTS + f"{ds}__{m}__s{s}.npz"
            if not os.path.exists(f): print("missing", f); continue
            r = np.load(f); score, flag = r["score"].astype(float), r["flag"].astype(bool)
            assert len(score) == len(s_idx), (ds, m)
            # ---- L1
            later = di >= 1
            fl = np.zeros(len(score), bool)
            for k in range(1, len(order)):
                past = (di < k) & ~y; cur = di == k
                fl[cur] = score[cur] > np.quantile(score[past], .99)
            thr_or = np.quantile(score[later & ~y], .99); thr_01 = np.quantile(score[~y], .999)
            zl = later & unk
            L1.append(dict(dataset=base, seed=s, method=m, n_days=len(order),
                           past_fpr=fl[later & ~y].mean(), past_zd=fl[zl].mean() if zl.any() else np.nan,
                           oracle_zd_later=(score[zl] > thr_or).mean() if zl.any() else np.nan,
                           oracle_zd_01=(score[unk] > thr_01).mean() if unk.any() else np.nan))
            # ---- L2
            for z in fams:
                zi = np.flatnonzero(fam == z); zi = zi[np.argsort(t[zi], kind="stable")]
                hit = zi[flag[zi]]
                L2.append(dict(dataset=base, seed=s, method=m, family=z, n=len(zi),
                               ttd_h=(t[hit].min() - t[zi].min()) / 3600 if len(hit) else np.nan, missed=not len(hit),
                               first100=flag[zi[:100]].mean()))
        # ---- L3 (ERNP; needs the feature matrix of this stream)
        f = common.RESULTS + f"{ds}__ernp_K5A__s{s}.npz"
        if os.path.exists(f) and unk.any():
            _, X, _, _, tr, _, s_idx2, _ = common.load(ds); assert np.array_equal(s_idx, s_idx2)
            lo, hi = X[tr].min(0), X[tr].max(0); Xs = X[s_idx]
            out = ((Xs < lo) | (Xs > hi)).any(1)
            flag = np.load(f)["flag"].astype(bool)
            L3.append(dict(dataset=base, seed=s, n_zd=int(unk.sum()), share_out=out[unk].mean(),
                           rec_out=flag[unk & out].mean() if (unk & out).any() else np.nan,
                           rec_in=flag[unk & ~out].mean() if (unk & ~out).any() else np.nan,
                           benign_share_out=out[~y].mean()))
        print("done", ds, flush=True)

l1, l2, l3 = pd.DataFrame(L1), pd.DataFrame(L2), pd.DataFrame(L3)
for n, d in (("l1", l1), ("l2", l2), ("l3", l3)): d.to_csv(os.path.join(OUT, f"review30_{n}.csv"), index=False)
# dataset means (seeds averaged), then the mean over the seven datasets
a1 = l1.groupby(["method", "dataset"]).mean(numeric_only=True).groupby("method").mean()[["past_fpr", "past_zd", "oracle_zd_later", "oracle_zd_01"]]
fam = l2.groupby(["method", "dataset", "family"]).agg(ttd_h=("ttd_h", "median"), missed=("missed", "mean"), first100=("first100", "mean")).reset_index()
a2 = fam.groupby(["method", "dataset"]).agg(ttd_h=("ttd_h", "median"), missed=("missed", "mean"), first100=("first100", "mean")).groupby("method").mean()
a3 = l3.groupby("dataset").mean(numeric_only=True)
txt = ("L1: past-only daily threshold (days >= 1), mean over 7 datasets\n" + a1.round(3).to_string() +
       "\n\nL2: per family at own rule; median hours to first alert (detected families), share never alerted, recall on first 100 flows\n"
       + a2.round(3).to_string() + "\n\nL3: ERNP, zero-day flows outside the anchor benign range\n" + a3.round(3).to_string() + "\n")
open(os.path.join(OUT, "review30_posthoc.txt"), "w").write(txt); print(txt)
