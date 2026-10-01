"""Feasibility (development seeds 0-1): can analyst labels tell, per deployment, WHICH notion of novelty predicts attacks?

Two novelty hypotheses, taken from saved stream scores: H_ratio = ERNP's density-ratio score (novel relative to trusted
benign evidence), H_struct = OWAD's autoencoder score (off the benign manifold). Causal protocol:
  * scores are turned into percentiles of the pooled scores of PREVIOUS days (day 1: own day, both weights equal);
  * at the end of each day the analyst labels B=200 flows drawn from that day's top-5% of either hypothesis;
  * the hypothesis with the higher cumulative labelled AUROC is used on the next day (posterior mode).
This only checks whether the selection signal exists; a final method must generate both hypotheses in one framework.
"""
import os, sys, importlib, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.metrics import roc_auc_score, average_precision_score
from evaluate import seeded_name

H = ["ernp", "owad"]; rows = []
for base in ["CIDDS_ctx", "ToN_ctx", "CIC18"]:
    for s in (0, 1):
        os.environ["EXP_SEED"] = str(s); import common; importlib.reload(common)
        ds = seeded_name(base, s); df, X, src, dst, tr, ho, s_idx, anc = common.load(ds)
        y = df["fam"].values[s_idx] != "BENIGN"; day = common.stream_days(df, s_idx); rng = np.random.default_rng(s)
        S = {m: np.load(common.RESULTS + f"{ds}__{m}__s{s}.npz")["score"] for m in H}
        out = np.zeros(len(y)); hist = {m: [] for m in H}; lab_idx = []; choice = []
        for d in dict.fromkeys(day):
            k = np.where(day == d)[0]
            pct = {}
            for m in H:
                ref = np.concatenate(hist[m]) if hist[m] else S[m][k]
                pct[m] = np.searchsorted(np.sort(ref), S[m][k]) / len(ref)
            if lab_idx and len(set(y[lab_idx])) == 2:
                auc = {m: roc_auc_score(y[lab_idx], S[m][lab_idx]) for m in H}; pick = max(auc, key=auc.get)
                out[k] = pct[pick]; choice.append(pick)
            else:
                out[k] = 0.5 * (pct["ernp"] + pct["owad"]); choice.append("mix")
            cand = np.unique(np.concatenate([k[S[m][k] >= np.quantile(S[m][k], 0.95)] for m in H]))
            lab_idx += list(rng.choice(cand, min(200, len(cand)), replace=False))
            for m in H: hist[m].append(S[m][k])
        r = dict(dataset=base, seed=s, choices="".join(c[0] for c in choice))
        for name, sc in [("ernp", S["ernp"]), ("owad", S["owad"]), ("selected", out)]:
            r[f"AUROC_{name}"] = roc_auc_score(y, sc); r[f"AUPRC_{name}"] = average_precision_score(y, sc)
        rows.append(r); print(r, flush=True)
r = pd.DataFrame(rows)
r.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports", "diagnostics_hypothesis_selection_dev.csv"), index=False)
print(r.groupby("dataset").mean(numeric_only=True).drop(columns="seed").round(3).to_string())
