"""Failure diagnostics for ERNP on DEVELOPMENT seeds only (0-1). No model is tuned here.

H1  benign drift between periods is mostly MARGINAL (per-feature distribution) rather than DEPENDENCE (joint structure):
    for each consecutive pair of stream days, a two-sample classifier separates day-d benign from day-(d+1) benign
    (a) on the raw features and (b) after a per-day probability-integral transform (ranks within each day), which
    removes marginal shift and keeps only dependence. Marginal share of drift = 1 - AUC_b-0.5 / AUC_a-0.5.
H2  zero-day attacks differ from benign mostly in DEPENDENCE: benign-vs-zero-day classifier on
    (a) raw features, (b) each feature separately (best single-feature AUC ~ marginal separability),
    (c) independence-shuffled attack flows (marginals kept, dependence destroyed) vs real attack flows.
H3  ERNP's false positives on a drifting day are flows whose MARGINAL values are outside the trusted benign range
    (novel values), not flows with novel combinations of familiar values.
"""
import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from evaluate import seeded_name
import importlib

def auc2(A, B, rng, n=6000):
    A = A[rng.choice(len(A), min(n, len(A)), replace=False)]; B = B[rng.choice(len(B), min(n, len(B)), replace=False)]
    X = np.vstack([A, B]); y = np.r_[np.zeros(len(A)), np.ones(len(B))]
    return float(cross_val_score(HistGradientBoostingClassifier(max_iter=150, random_state=0), X, y, cv=3, scoring="roc_auc").mean())

def ranks(M):
    return np.column_stack([pd.Series(M[:, j]).rank(pct=True).values for j in range(M.shape[1])])

out = []
for base in ["CIDDS_ctx", "ToN_ctx", "CIC17", "CIC18"]:
    for s in (0, 1):
        os.environ["EXP_SEED"] = str(s); import common; importlib.reload(common)
        ds = seeded_name(base, s); df, X, src, dst, tr, ho, s_idx, anc = common.load(ds); fam = df["fam"].values
        rng = np.random.default_rng(s); day = common.stream_days(df, s_idx); days = list(dict.fromkeys(day))
        # H1: consecutive-day benign drift, raw vs rank-transformed (marginals removed)
        h1 = []
        for d0, d1 in zip(days[:-1], days[1:]):
            A = X[s_idx[(day == d0) & (fam[s_idx] == "BENIGN")]]; B = X[s_idx[(day == d1) & (fam[s_idx] == "BENIGN")]]
            if len(A) < 500 or len(B) < 500: continue
            a_raw = auc2(A, B, rng); a_dep = auc2(ranks(A), ranks(B), rng)
            h1.append(dict(raw=a_raw, dep=a_dep))
        h1 = pd.DataFrame(h1)
        # H2: benign vs zero-day
        known = set(fam[anc & (fam != "BENIGN")]); zd_mask = (fam[s_idx] != "BENIGN") & ~np.isin(fam[s_idx], list(known))
        Bn = X[s_idx[fam[s_idx] == "BENIGN"]]; Z = X[s_idx[zd_mask]]
        if len(Z) >= 300:
            a_raw = auc2(Bn, Z, rng)
            Zs = Z[rng.choice(len(Z), min(6000, len(Z)), replace=False)]
            Zind = np.column_stack([rng.permutation(Zs[:, j]) for j in range(Zs.shape[1])])   # attack marginals, no dependence
            a_marg = auc2(Bn, Zind, rng)              # separability explained by marginals alone
            a_dep = auc2(Zind, Zs, rng)               # how much the attack's own dependence structure is distinctive
        else: a_raw = a_marg = a_dep = np.nan
        # H3: ERNP false positives - are their feature values inside the trusted benign range?
        F = np.load(common.RESULTS + f"{ds}__ernp__s{s}.npz")["flag"].astype(bool)
        lo, hi = np.quantile(X[tr], 0.005, axis=0), np.quantile(X[tr], 0.995, axis=0)
        ben = fam[s_idx] == "BENIGN"; fp = F & ben; tn = ~F & ben
        out_of_range = lambda M: ((M < lo) | (M > hi)).any(1).mean() if len(M) else np.nan
        out.append(dict(dataset=base, seed=s, days=len(h1),
                        drift_auc_raw=h1.raw.mean() if len(h1) else np.nan, drift_auc_dep=h1.dep.mean() if len(h1) else np.nan,
                        zd_auc_raw=a_raw, zd_auc_marginals_only=a_marg, zd_dependence_signature=a_dep,
                        fp_rate=fp.mean() / max(ben.mean(), 1e-9), fp_outside_benign_range=out_of_range(X[s_idx[fp]]),
                        tn_outside_benign_range=out_of_range(X[s_idx[tn]])))
        print(out[-1], flush=True)
r = pd.DataFrame(out); r.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports", "diagnostics_dev.csv"), index=False)
print(r.groupby("dataset").mean(numeric_only=True).drop(columns="seed").round(3).to_string())
