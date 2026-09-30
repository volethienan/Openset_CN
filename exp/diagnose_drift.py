"""Corrected H1 test (development seeds 0-1): is benign drift between consecutive days marginal or dependence?

For each consecutive pair of stream days, with EQUAL sample sizes:
  AUC_raw    classifier day d vs day d+1 on the benign flows as they are
  AUC_indep  same after shuffling every column independently within each day (marginals kept exactly, dependence destroyed)
  AUC_single best single-feature AUC (a pure marginal effect)
If AUC_indep ~ AUC_raw, the drift is carried by the marginals; if AUC_indep << AUC_raw, by the dependence structure.
"""
import os, sys, numpy as np, pandas as pd, importlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from evaluate import seeded_name

def cv_auc(A, B):
    X = np.vstack([A, B]); y = np.r_[np.zeros(len(A)), np.ones(len(B))]
    return float(cross_val_score(HistGradientBoostingClassifier(max_iter=150, random_state=0), X, y, cv=3, scoring="roc_auc").mean())

rows = []
for base in ["CIDDS_ctx", "ToN_ctx", "CIC17", "CIC18"]:
    for s in (0, 1):
        os.environ["EXP_SEED"] = str(s); import common; importlib.reload(common)
        df, X, src, dst, tr, ho, s_idx, anc = common.load(seeded_name(base, s)); fam = df["fam"].values
        rng = np.random.default_rng(s); day = common.stream_days(df, s_idx); days = list(dict.fromkeys(day))
        pairs = [(days[0 if base.startswith("CIC17") else i], days[i + 1]) for i in range(min(len(days) - 1, 6))]
        if base == "CIC17":   # two stream days only: compare last anchor day with each stream day
            a_day = pd.to_datetime(df["t"].values[tr], unit="s").strftime("%Y-%m-%d").values
            pairs = [("ANCHOR", d) for d in days]
        for d0, d1 in pairs:
            A = X[tr] if d0 == "ANCHOR" else X[s_idx[(day == d0) & (fam[s_idx] == "BENIGN")]]
            B = X[s_idx[(day == d1) & (fam[s_idx] == "BENIGN")]]
            n = min(len(A), len(B), 5000)
            if n < 500: continue
            A = A[rng.choice(len(A), n, replace=False)]; B = B[rng.choice(len(B), n, replace=False)]
            Ai = np.column_stack([rng.permutation(A[:, j]) for j in range(A.shape[1])])
            Bi = np.column_stack([rng.permutation(B[:, j]) for j in range(B.shape[1])])
            y = np.r_[np.zeros(n), np.ones(n)]
            single = max(max(roc_auc_score(y, np.r_[A[:, j], B[:, j]]), 1 - roc_auc_score(y, np.r_[A[:, j], B[:, j]])) for j in range(A.shape[1]))
            rows.append(dict(dataset=base, seed=s, pair=f"{d0}->{d1}", auc_raw=cv_auc(A, B), auc_indep=cv_auc(Ai, Bi), auc_single=single))
            print(rows[-1], flush=True)
r = pd.DataFrame(rows)
r["marginal_share"] = (r.auc_indep - .5) / (r.auc_raw - .5).clip(lower=1e-6)
r.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports", "diagnostics_drift_dev.csv"), index=False)
print(r.groupby("dataset")[["auc_raw", "auc_indep", "auc_single", "marginal_share"]].mean().round(3).to_string())
