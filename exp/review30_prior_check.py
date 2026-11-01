"""ARS review 01/11: does the prior's gain survive a past-only threshold recalibrated every day, and are zero-day flows
outside the anchor benign range detected without the prior? Saved scores only. Output: exp/reports/review30_prior.csv
usage (from Openset_CN/): python exp/review30_prior_check.py"""
import os, sys, importlib
import numpy as np, pandas as pd
EXP = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, EXP)
import evaluate, common
DS = ["CIDDS_ctx", "ToN_ctx", "CIC17", "CIC18", "CTU13", "SIMARGL", "APPRAISE"]
MS = ["ernp_K5A", "abl_prior_none_K5A", "ernp_B0_K5A"]
R = []
for base in DS:
    for s in range(5):
        os.environ["EXP_SEED"] = str(s); importlib.reload(common)
        ds = evaluate.seeded_name(base, s)
        df = evaluate.metric_labels(common.STREAMS + f"{ds}.parquet")
        anc = df.t.to_numpy() < common.anchor_end(ds, df.t.min()); s_idx = np.flatnonzero(~anc)
        fam = df.fam.values[s_idx]; y = fam != "BENIGN"
        known = set(df.fam.values[anc & (df.fam.values != "BENIGN")]); unk = y & ~np.isin(fam, list(known))
        days = common.stream_days(df, s_idx); order = sorted(set(days)); di = np.searchsorted(order, days)
        _, X, _, _, tr, _, s2, _ = common.load(ds); assert np.array_equal(s_idx, s2)
        lo, hi = X[tr].min(0), X[tr].max(0); Xs = X[s_idx]; out = ((Xs < lo) | (Xs > hi)).any(1)
        for m in MS:
            f = common.RESULTS + f"{ds}__{m}__s{s}.npz"
            if not os.path.exists(f): print("missing", f); continue
            r = np.load(f); score, flag = r["score"].astype(float), r["flag"].astype(bool)
            later = di >= 1; fl = np.zeros(len(score), bool)
            for k in range(1, len(order)):
                past = (di < k) & ~y; cur = di == k
                fl[cur] = score[cur] > np.quantile(score[past], .99)
            tp = (fl & y & later).sum(); fp = (fl & ~y & later).sum(); fn = (~fl & y & later).sum()
            R.append(dict(dataset=base, seed=s, method=m, past_fpr=fl[later & ~y].mean(), past_zd=fl[later & unk].mean(),
                          past_f1=2 * tp / max(2 * tp + fp + fn, 1),
                          own_fpr=flag[~y].mean(), own_zd=flag[unk].mean(),
                          rec_out=flag[unk & out].mean() if (unk & out).any() else np.nan,
                          rec_in=flag[unk & ~out].mean() if (unk & ~out).any() else np.nan))
        print("done", ds, flush=True)
d = pd.DataFrame(R)
d.to_csv(os.path.join(EXP, "reports", "review30_prior.csv"), index=False)
g = d.groupby(["method", "dataset"]).mean(numeric_only=True).drop(columns="seed")
print(g.round(3).to_string()); print(g.groupby("method").mean().round(3).to_string())
