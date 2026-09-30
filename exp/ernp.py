"""ERNP - Evidence-Retracted Novelty Prior (the proposed method).

Detector: density-ratio classifier g_t between benign evidence P_t and {attack evidence A_t, novelty prior q_t}
          s_t(x) = logit g_t(x); alert iff s_t(x) > tau (tau: 1% FPR on the anchor holdout, fixed).
Novelty prior q_0: benign flows whose features are swapped, each with probability rho, with the same feature of another
          benign flow (benign marginals kept, joint structure broken).
Adaptation (the ONLY channel): trusted evidence retracts prior mass
          q_{t+1}(x) = q_t(x) * (1 - kappa(x; E_t)),  kappa = exp(-d(x, E_t)^2 / (2 eps^2)),
          E_t = analyst-confirmed benign alert groups, eps = 90th pct 1-NN spacing of the benign reference.
Labels: each day the alerts are grouped into <= B clusters, the analyst labels one representative per cluster;
          the label applies to the whole alert group (SOC alert-group triage).
Mass: benign 1/2 | attack evidence 1/2 | prior 1/2 * (retained fraction)   -- retraction removes mass, never renormalises.

Ablation switches (RQ4) are explicit arguments; defaults = the final ERNP configuration.
"""
import sys, time, argparse, numpy as np
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, calib_upper, stream_days, stream_periods, period_budget, save, seed, N_CTX
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import MiniBatchKMeans

DEFAULTS = dict(rho=0.3, prior_mass=1.0, budget=200, prior="partial", retraction="soft", evidence="cluster",
                features="ctx", protocol="causal", n_pos_max=40000, n_known_max=10000, gbm_iter=300, gbm_lr=0.1,
                eps_mult=1.0, clf="gbm", period=24,          # period: update cadence in hours (24 = daily)
                evidence_source="analyst", n_self=5000,
                label_noise=0.0)                             # analyst flips each representative label w.p. label_noise      # "self": counterfactual absorbing channel (RQ2/RQ4)

class ERNP:
    def __init__(self, cfg, rng):
        self.c, self.rng = cfg, rng
        self.conf = None                       # trusted benign evidence E_t (retraction centres)
        self.eps = None
        self.clf = None

    # ---------------- novelty prior q_0 ----------------
    def sample_prior(self, P):
        n = int(self.c["prior_mass"] * len(P)); rng = self.rng
        if self.c["prior"] == "none" or n == 0: return np.zeros((0, P.shape[1]), P.dtype)
        base = P[rng.choice(len(P), n)]
        if self.c["prior"] == "partial":       # ERNP: row-wise crossover with one donor, each feature w.p. rho
            donor = base[rng.integers(0, len(base), len(base))]; mask = rng.random(base.shape) < self.c["rho"]
            return np.where(mask, donor, base)
        if self.c["prior"] == "marginal":      # full product of marginals (independent donor per feature)
            return np.stack([base[rng.integers(0, n, n), j] for j in range(P.shape[1])], 1)
        if self.c["prior"] == "uniform":       # Sipple (ICML'20): uniform in the benign bounding box
            lo, hi = P.min(0), P.max(0); return (lo + rng.random((n, P.shape[1])) * (hi - lo)).astype(P.dtype)
        raise ValueError(self.c["prior"])

    # ---------------- retraction kappa ----------------
    def retained(self, Q):
        if self.conf is None or not len(self.conf) or self.c["retraction"] == "none" or not len(Q):
            return np.ones(len(Q))
        d = NearestNeighbors(n_neighbors=1).fit(self.conf).kneighbors(Q)[0][:, 0]
        if self.c["retraction"] == "soft": return 1.0 - np.exp(-d ** 2 / (2 * self.eps ** 2))
        if self.c["retraction"] == "hard": return (d > self.eps).astype(float)
        raise ValueError(self.c["retraction"])

    def fit(self, P, A):
        Q = self.sample_prior(P); n_gen = max(len(Q), 1)
        w_q = self.retained(Q); keep = w_q > 1e-3; Q, w_q = Q[keep], w_q[keep]
        X = np.vstack([P, Q, A]); y = np.r_[np.zeros(len(P)), np.ones(len(Q) + len(A))]
        w = np.r_[np.full(len(P), 0.5 / len(P)), 0.5 * w_q / n_gen, np.full(len(A), 0.5 / max(len(A), 1))]
        w = w / w.mean()
        if self.c["clf"] == "gbm":
            self.clf = HistGradientBoostingClassifier(max_iter=self.c["gbm_iter"], learning_rate=self.c["gbm_lr"],
                                                      random_state=0).fit(X, y, sample_weight=w)
        elif self.c["clf"] == "rf":                                   # RQ5: classifier family sensitivity
            from sklearn.ensemble import RandomForestClassifier
            self.clf = RandomForestClassifier(300, min_samples_leaf=5, n_jobs=4, random_state=0).fit(X, y, sample_weight=w)
        else: raise ValueError(self.c["clf"])
        return self

    def score(self, Z):
        if self.c["clf"] == "gbm": return self.clf.decision_function(Z)
        p = np.clip(self.clf.predict_proba(Z)[:, 1], 1e-6, 1 - 1e-6); return np.log(p / (1 - p))

def run(ds, **cfg):
    c = dict(DEFAULTS); c.update(cfg)
    df, X, src, dst, tr, ho, s_idx, anc = load(ds); fam = df["fam"].values
    if c["features"] == "flow": X = X[:, :-N_CTX]               # ablation: per-flow features only
    rng = np.random.default_rng(seed()); t_fit, t_score, n_scored = [], 0.0, 0
    P = X[rng.choice(tr, min(c["n_pos_max"], len(tr)), replace=False)]
    ka = np.where(anc & (fam != "BENIGN"))[0]
    A = X[rng.choice(ka, min(c["n_known_max"], len(ka)), replace=False)] if len(ka) else np.zeros((0, X.shape[1]), X.dtype)
    m = ERNP(c, rng); nrng = np.random.default_rng(10_000 + seed())   # noise rng: separate, only used if label_noise > 0
    m.eps = c["eps_mult"] * np.quantile(NearestNeighbors(n_neighbors=2).fit(P).kneighbors(P[:5000])[0][:, 1], 0.9)
    t0 = time.time(); m.fit(P, A); t_fit.append(time.time() - t0)
    tau = calib_upper(m.score(X[ho]))
    day = stream_periods(df, s_idx, c["period"]); s = np.zeros(len(s_idx)); F = np.zeros(len(s_idx), bool); n_lab = 0
    for d in dict.fromkeys(day):
        idx = np.where(day == d)[0]
        t0 = time.time(); s[idx] = m.score(X[s_idx[idx]]); t_score += time.time() - t0; n_scored += len(idx)
        F[idx] = s[idx] > tau
        if c["evidence_source"] == "self":     # counterfactual: the detector's OWN verdicts (non-alerted flows) are
            nb = s_idx[idx][~F[idx]]           # trusted as benign evidence -> the usual self-updating channel that
            if len(nb):                        # frog-boiling exploits; no analyst labels are used
                nb = nb[rng.choice(len(nb), min(c["n_self"], len(nb)), replace=False)]
                P = np.vstack([P, X[nb]]); m.conf = np.vstack(([m.conf] if m.conf is not None else []) + [X[nb]])
                t0 = time.time(); m.fit(P[-2 * c["n_pos_max"]:], A[-4 * c["n_known_max"]:]); t_fit.append(time.time() - t0)
            continue
        if c["budget"] <= 0: continue                              # no online labels -> q never changes (P1)
        al = s_idx[idx][s[idx] > tau]; Bk = period_budget(d, c["period"], c["budget"])   # = budget when daily
        if len(al) < 2 or Bk <= 0: continue
        k = min(Bk, len(al)); km = MiniBatchKMeans(k, random_state=seed(), n_init=3).fit(X[al])
        addP, addA = [], []
        for cl in range(k):
            mem = al[km.labels_ == cl]
            if not len(mem): continue
            rep = mem[np.argmin(np.linalg.norm(X[mem] - km.cluster_centers_[cl], axis=1))]; n_lab += 1
            grp = X[mem] if c["evidence"] == "cluster" else X[[rep]]
            is_ben = fam[rep] == "BENIGN"
            if c["label_noise"] > 0 and nrng.random() < c["label_noise"]: is_ben = not is_ben   # analyst error
            (addP if is_ben else addA).append(grp)
        if addP:
            P = np.vstack([P] + addP); m.conf = np.vstack(([m.conf] if m.conf is not None else []) + addP)
        if addA: A = np.vstack([A] + addA)
        t0 = time.time(); m.fit(P[-2 * c["n_pos_max"]:], A[-4 * c["n_known_max"]:]); t_fit.append(time.time() - t0)
        if c["protocol"] == "ssf":                                 # SSF's native protocol: predict after learning the day
            s[idx] = m.score(X[s_idx[idx]]); F[idx] = s[idx] > tau
    info = dict(cfg={k: v for k, v in c.items()}, tau=tau, eps=m.eps, labels_used=n_lab,
                timing=dict(fit_mean_s=float(np.mean(t_fit)), fit_max_s=float(np.max(t_fit)),
                            score_flows_per_s=n_scored / max(t_score, 1e-9)))
    return s, F, info

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ds"); ap.add_argument("--name", default="ernp")
    for k, v in DEFAULTS.items(): ap.add_argument("--" + k, type=type(v), default=v)
    a = vars(ap.parse_args()); ds, name = a.pop("ds"), a.pop("name")
    s, F, info = run(ds, **a); save(ds, name, s, F, info)
