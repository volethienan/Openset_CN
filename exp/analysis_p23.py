"""RQ4 checks of P2 (local convergence) and P3 (zero-day preservation) for finished ERNP runs.

The trusted evidence E_t is replayed from the saved alerts of a causal ERNP run with exactly the clustering of
ernp.run (same features, MiniBatchKMeans(k, random_state=seed, n_init=3), representative = member closest to the
centre); the replay is validated by reproducing the run's labels_used. No model is refitted.

P2: FPR of benign flows lying within eps of E_{t-1} (the region the analyst has vouched for), per day, against the
    cumulative number of labels, next to the SAME flows' FPR under ERNP without online labels (ernp_B0): the
    counterfactual shows how much of the local false-alarm mass the retraction removed.
Labels: representatives asked, flows that inherit a label through their alert group, and the label noise of that
    propagation (attack flows inside groups labelled benign, and vice versa).
P3: recall of zero-day flows binned by d(x, E_{t-1}) / eps, next to the same flows' recall under ERNP without online
    labels (ernp_B0) and the retained prior fraction 1 - exp(-d^2 / 2 eps^2) at the bin centre -> far from benign
    evidence the detection of ERNP must stay at (or above) the no-label level.
usage: python analysis_p23.py --datasets CIDDS_ctx ToN_ctx CIC17 CIC18 --seeds 0 1 2 3 4
"""
import os, sys, json, argparse, importlib, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import MiniBatchKMeans
from evaluate import seeded_name

BINS = [0, 0.5, 1, 2, 4, np.inf]

def replay(base, s, budget=200):
    os.environ["EXP_SEED"] = str(s); import common; importlib.reload(common)
    ds = seeded_name(base, s); R = common.RESULTS
    if not os.path.exists(R + f"{ds}__ernp__s{s}.npz"): return None
    df, X, src, dst, tr, ho, s_idx, anc = common.load(ds); fam = df["fam"].values
    F = np.load(R + f"{ds}__ernp__s{s}.npz")["flag"].astype(bool); info = json.load(open(R + f"{ds}__ernp__s{s}.json"))
    b0 = R + f"{ds}__ernp_B0__s{s}.npz"; F0 = np.load(b0)["flag"].astype(bool) if os.path.exists(b0) else None
    eps = info["eps"]; known = set(fam[anc & (fam != "BENIGN")])
    day = common.stream_days(df, s_idx); E = None; n_lab = 0; p2, p3 = [], []
    lab = dict(flows_benign_groups=0, attack_in_benign_groups=0, flows_attack_groups=0, benign_in_attack_groups=0)
    for d in dict.fromkeys(day):
        idx = np.where(day == d)[0]; fl = s_idx[idx]; fm = fam[fl]
        ben, zd = fm == "BENIGN", (fm != "BENIGN") & ~np.isin(fm, list(known))
        if E is not None:
            nn = NearestNeighbors(n_neighbors=1).fit(E)
            if ben.any():
                db = nn.kneighbors(X[fl[ben]])[0][:, 0] / eps; near = db < 1
                p2.append(dict(day=d, labels_before=n_lab, n_near=int(near.sum()),
                               fpr_near=float(F[idx][ben][near].mean()) if near.any() else np.nan,
                               fpr_near_B0=float(F0[idx][ben][near].mean()) if near.any() and F0 is not None else np.nan,
                               fpr_far=float(F[idx][ben][~near].mean()) if (~near).any() else np.nan))
            if zd.any():
                dz = nn.kneighbors(X[fl[zd]])[0][:, 0] / eps
                p3.append(pd.DataFrame(dict(dist=dz, hit=F[idx][zd], hit_B0=F0[idx][zd] if F0 is not None else np.nan)))
        al = fl[F[idx]]                                       # --- replay of ernp.run's evidence update ---
        if len(al) < 2: continue
        k = min(budget, len(al)); km = MiniBatchKMeans(k, random_state=s, n_init=3).fit(X[al]); addP = []
        for cl in range(k):
            mem = al[km.labels_ == cl]
            if not len(mem): continue
            rep = mem[np.argmin(np.linalg.norm(X[mem] - km.cluster_centers_[cl], axis=1))]; n_lab += 1
            if fam[rep] == "BENIGN":
                addP.append(X[mem]); lab["flows_benign_groups"] += len(mem); lab["attack_in_benign_groups"] += int((fam[mem] != "BENIGN").sum())
            else:
                lab["flows_attack_groups"] += len(mem); lab["benign_in_attack_groups"] += int((fam[mem] == "BENIGN").sum())
        if addP: E = np.vstack(([E] if E is not None else []) + addP)
    ok = n_lab == info["labels_used"]
    p3 = pd.concat(p3) if p3 else pd.DataFrame(columns=["dist", "hit", "hit_B0"])
    p3["bin"] = pd.cut(p3["dist"], BINS, right=False)
    t3 = p3.groupby("bin", observed=False).agg(n=("hit", "size"), recall=("hit", "mean"), recall_B0=("hit_B0", "mean"))
    mid = np.array([0.25, 0.75, 1.5, 3.0, 6.0]); t3["retained_prior_at_mid"] = 1 - np.exp(-mid ** 2 / 2)
    lab.update(dataset=base, seed=s, representatives=n_lab)
    return ok, n_lab, info["labels_used"], pd.DataFrame(p2), t3, lab

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4]); a = ap.parse_args()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports"); os.makedirs(out, exist_ok=True)
    LAB = []
    for base in a.datasets:
        P2, P3 = [], []
        for s in a.seeds:
            r = replay(base, s)
            if r is None: continue
            ok, n, n_ref, p2, t3, lab = r; LAB.append(lab)
            print(f"{base} s{s}: replay labels {n} vs run {n_ref} -> {'OK' if ok else 'MISMATCH (excluded)'}", flush=True)
            if not ok: continue
            p2["seed"] = s; P2.append(p2); t3["seed"] = s; P3.append(t3.reset_index())
        if not P2: continue
        P2, P3 = pd.concat(P2), pd.concat(P3)
        P2.to_csv(f"{out}/p2_{base}.csv", index=False); P3.to_csv(f"{out}/p3_{base}.csv", index=False)
        q = P2.groupby("seed").apply(lambda g: np.corrcoef(g["labels_before"], g["fpr_near"].fillna(g["fpr_near"].mean()))[0, 1]
                                     if len(g) > 2 else np.nan, include_groups=False)
        print(f"== P2 {base}: FPR near evidence (first -> last day, mean over seeds): "
              f"{P2.groupby('seed')['fpr_near'].first().mean():.4f} -> {P2.groupby('seed')['fpr_near'].last().mean():.4f}; "
              f"same flows without online labels (B0): {P2['fpr_near_B0'].mean():.4f} vs ERNP {P2['fpr_near'].mean():.4f}; "
              f"corr(labels, FPR_near) per seed = {q.round(2).tolist()}")
        g = P3.groupby("bin", observed=False)[["n", "recall", "recall_B0", "retained_prior_at_mid"]].mean()
        print(f"== P3 {base}: zero-day recall by distance to benign evidence (units of eps), mean over seeds\n{g.round(4).to_string()}\n")
    if LAB:
        L = pd.DataFrame(LAB); L["flows_labelled"] = L.flows_benign_groups + L.flows_attack_groups
        L["noise_attack_in_benign"] = L.attack_in_benign_groups / L.flows_benign_groups.clip(lower=1)
        L["noise_benign_in_attack"] = L.benign_in_attack_groups / L.flows_attack_groups.clip(lower=1)
        L.to_csv(f"{out}/labels_group_propagation.csv", index=False)
        print("== Label semantics of ERNP (mean over seeds): representatives asked vs flows inheriting a label\n" +
              L.groupby("dataset")[["representatives", "flows_labelled", "noise_attack_in_benign", "noise_benign_in_attack"]]
              .mean().round(4).to_string())
