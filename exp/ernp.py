"""ERNP - Evidence-Retracted Novelty Prior (the proposed method).

Detector: density-ratio classifier g_t between benign evidence P_t and {attack evidence A_t, novelty prior q_t}
          s_t(x) = logit g_t(x); alert iff s_t(x) > tau (tau: 1% FPR on the anchor holdout, fixed).
Novelty prior q_0: benign flows whose features are swapped, each with probability rho, with the same feature of another
          benign flow (benign marginals kept, joint structure broken).
Adaptation: analyst group decisions extend P_t/A_t; benign confirmations additionally retract prior mass.
          Each refit samples a FRESH q(P_t) and weights it by r_t(x) = 1 - exp(-d(x,E_t)^2/(2 eps^2)).
          This is not recursive multiplication of all historical kernels. E_t contains flows in groups whose
          representative was confirmed benign; group-label propagation can be wrong. eps is fixed from the anchor.
Labels: each day the alerts are grouped into <= B clusters, the analyst labels one representative per cluster;
          the label applies to the whole alert group (SOC alert-group triage).
Mass: benign 1/2 | attack evidence 1/2 | prior 1/2 * (retained fraction)   -- retraction removes mass, never renormalises.

Ablation switches are explicit. The final configuration (K5A) is --n_prior 5 --keep A: five prior draws per fit and
the anchor's known attacks kept in the attack evidence; the CLI defaults (n_prior=1, keep="fifo") stay the historical
configuration so that earlier runs remain reproducible.
Retraction is local in prior weights, not a guarantee that refitted GBM scores are unchanged elsewhere.
The causal protocol concerns feedback order on retrospective complete-flow records, not packet-time causality.
"""
import sys, time, argparse, numpy as np
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, calib_upper, stream_days, stream_periods, period_budget, save, seed, N_CTX, ROOT, naive_ramp_mask
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import MiniBatchKMeans

DEFAULTS = dict(rho=0.3, prior_mass=1.0, budget=200, prior="partial", retraction="soft", evidence="cluster",
                features="ctx", protocol="causal", n_pos_max=40000, n_known_max=10000, gbm_iter=300, gbm_lr=0.1,
                eps_mult=1.0, clf="gbm", period=24,          # period: update cadence in hours (24 = daily)
                evidence_source="analyst", n_self=5000,      # "self": counterfactual absorbing channel (RQ2/RQ4)
                label_noise=0.0, fpr=0.01,                   # analyst flips each representative label w.p. label_noise
                analytic=0, refit=24, knn=32,                # analytic: 3-class model, retraction applied at inference; refit: hours between refits
                replay="", flow_cap=0, delay=0,              # replay: take the alerts that form the groups from a saved run
                n_prior=1, rule="q1", keep="fifo", att_cap=0, mut=0.0,
                naive_ramp=0.0)                              # RQ2 analyst model: ramp flows with fraction a < naive_ramp are answered BENIGN
# (fixed feedback schedule, E3/mechanism control); flow_cap: max flows entering the evidence per period (E2);
# delay: periods between an analyst decision and the refit that uses it (E5);
# n_prior: models per fit, each trained against its own draw of the prior, scores averaged (Monte-Carlo variance of the
# prior sample); rule: "q1" = threshold at the 1% holdout quantile, "post" = the model's own Bayes rule (score > 0);
# keep: which evidence a fit sees once it exceeds its size limit: "fifo" = the most recent flows (the anchor can be
# evicted), "A" / "P" / "AP" = the anchor attacks / anchor benign flows always stay and the online evidence is sampled
# uniformly; att_cap: max flows of one attack-confirmed group entering the attack evidence (0 = the whole group);
# mut: fraction of prior rows in which 1-2 features take a value drawn uniformly from the standardised box [-10, 10]
# (a value benign need not have shown; the crossover alone only recombines values already seen in benign)

class ERNP:
    def __init__(self, cfg, rng):
        self.c, self.rng = cfg, rng
        self.conf = None                       # trusted benign evidence E_t (retraction centres)
        self.eps = None
        self.clf = None
        self.dsum, self.dn = None, 0            # evidence-shaped retraction: accumulated squared displacements

    # ---------------- novelty prior q_0 ----------------
    def sample_prior(self, P):
        n = int(self.c["prior_mass"] * len(P)); rng = self.rng
        if self.c["prior"] == "none" or n == 0: return np.zeros((0, P.shape[1]), P.dtype)
        base = P[rng.choice(len(P), n)]
        if self.c["prior"] == "partial":       # ERNP: row-wise crossover with one donor, each feature w.p. rho
            donor = base[rng.integers(0, len(base), len(base))]; mask = rng.random(base.shape) < self.c["rho"]
            Q = np.where(mask, donor, base)
            if self.c["mut"] > 0:                 # mutation: values outside what benign has shown
                m = int(self.c["mut"] * n); i = rng.choice(n, m, replace=False); j = rng.integers(0, P.shape[1], (m, 2))
                Q[i, j[:, 0]] = rng.uniform(-10, 10, m); two = rng.random(m) < 0.5; Q[i[two], j[two, 1]] = rng.uniform(-10, 10, two.sum())
            return Q
        if self.c["prior"] == "scarf":         # SCARF/VIME corruption: each feature w.p. rho from its own random donor
            mask = rng.random(base.shape) < self.c["rho"]
            repl = np.stack([base[rng.integers(0, n, n), j] for j in range(P.shape[1])], 1)
            return np.where(mask, repl, base)
        if self.c["prior"] == "marginal":      # full product of marginals (independent donor per feature)
            return np.stack([base[rng.integers(0, n, n), j] for j in range(P.shape[1])], 1)
        if self.c["prior"] == "uniform":       # Sipple (ICML'20): uniform in the benign bounding box
            lo, hi = P.min(0), P.max(0); return (lo + rng.random((n, P.shape[1])) * (hi - lo)).astype(P.dtype)
        raise ValueError(self.c["prior"])

    # ---------------- retraction kappa ----------------
    def retained(self, Q):
        if self.conf is None or not len(self.conf) or self.c["retraction"] == "none" or not len(Q):
            return np.ones(len(Q))
        if self.c["retraction"] == "shaped":   # kernel stretched along the drift directions the evidence revealed
            w = self.shape_w(); d = NearestNeighbors(n_neighbors=1).fit(self.conf * w).kneighbors(Q * w)[0][:, 0]
            return 1.0 - np.exp(-d ** 2 / (2 * self.eps ** 2))
        if self.c["retraction"] == "prod":     # independent thinning by EVERY evidence point: r = prod_e (1 - k(x, e))
            k = min(self.c["knn"], len(self.conf))
            d = NearestNeighbors(n_neighbors=k).fit(self.conf).kneighbors(Q)[0]
            kern = np.minimum(np.exp(-d ** 2 / (2 * self.eps ** 2)), 1 - 1e-12)
            return np.exp(np.log1p(-kern).sum(1))
        d = NearestNeighbors(n_neighbors=1).fit(self.conf).kneighbors(Q)[0][:, 0]
        if self.c["retraction"] == "soft": return 1.0 - np.exp(-d ** 2 / (2 * self.eps ** 2))
        if self.c["retraction"] == "hard": return (d > self.eps).astype(float)
        raise ValueError(self.c["retraction"])

    def shape_w(self):
        """diagonal metric Sigma_t = eps^2 I + diag(E[delta^2]); returns per-feature scaling eps / sqrt(eps^2 + v_j)"""
        if not self.dn: return np.ones(self.conf.shape[1], np.float32)
        v = self.dsum / self.dn; return (self.eps / np.sqrt(self.eps ** 2 + v)).astype(np.float32)

    def observe(self, G, P):
        """displacement of newly confirmed benign flows G from the trusted benign evidence P they were confirmed against"""
        if self.c["retraction"] != "shaped" or not len(G): return
        G = G[self.rng.choice(len(G), min(2000, len(G)), replace=False)]
        nn = P[NearestNeighbors(n_neighbors=1).fit(P).kneighbors(G)[1][:, 0]]
        d2 = ((G - nn) ** 2).astype(np.float64)
        self.dsum = d2.sum(0) if self.dsum is None else self.dsum + d2.sum(0); self.dn += len(G)

    def fit(self, P, A):
        if self.c["analytic"]: return self.fit3(P, A)
        if self.c["n_prior"] <= 1: return self.fit1(P, A)
        assert self.c["clf"] == "gbm"
        self.clf = [self.fit1(P, A).clf for _ in range(self.c["n_prior"])]   # one prior draw per model
        return self

    def fit1(self, P, A):
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
            self.clf = RandomForestClassifier(300, min_samples_leaf=5, n_jobs=int(os.environ.get("OMP_NUM_THREADS", 4)), random_state=0).fit(X, y, sample_weight=w)
        elif self.c["clf"] == "mlp":                                  # diagnostic: smooth, relational ratio estimator
            self.clf = _mlp_fit(X, y, w)
        else: raise ValueError(self.c["clf"])
        return self

    def fit3(self, P, A):
        """analytic variant: one 3-class model {benign, attack, prior} on the UNRETRACTED prior. In the population,
        p_c(x) ~ w_c f_c(x), so log((p_A + r_t p_Q) / p_B) equals the score of a model trained on the retracted prior:
        evidence then acts at inference time, without refitting."""
        Q = self.sample_prior(P)
        X = np.vstack([P, A, Q]); y = np.r_[np.zeros(len(P)), np.ones(len(A)), np.full(len(Q), 2)]
        w = np.r_[np.full(len(P), 0.5 / len(P)), np.full(len(A), 0.5 / max(len(A), 1)), np.full(len(Q), 0.5 / max(len(Q), 1))]
        self.clf = HistGradientBoostingClassifier(max_iter=self.c["gbm_iter"], learning_rate=self.c["gbm_lr"],
                                                  random_state=0).fit(X, y, sample_weight=w / w.mean())
        return self

    def score(self, Z):
        if self.c["analytic"]:
            p = self.clf.predict_proba(Z); pa = p[:, 1] if p.shape[1] == 3 else 0.0
            return np.log(pa + self.retained(Z) * p[:, -1] + 1e-9) - np.log(p[:, 0] + 1e-9)
        if isinstance(self.clf, list): return np.mean([m.decision_function(Z) for m in self.clf], 0)
        if self.c["clf"] == "gbm": return self.clf.decision_function(Z)
        if self.c["clf"] == "mlp": return _mlp_score(self.clf, Z)
        p = np.clip(self.clf.predict_proba(Z)[:, 1], 1e-6, 1 - 1e-6); return np.log(p / (1 - p))

def _mlp_fit(X, y, w, epochs=20, bs=1024):
    """weighted logistic MLP (2 x 256, ReLU) - same objective as the GBM density-ratio classifier"""
    import torch
    torch.manual_seed(seed()); dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 256), torch.nn.ReLU(), torch.nn.Linear(256, 256), torch.nn.ReLU(),
                              torch.nn.Linear(256, 1)).to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-5)
    Xt, yt, wt = (torch.tensor(a, dtype=torch.float32, device=dev) for a in (X, y, w))
    g = torch.Generator(device="cpu").manual_seed(seed())
    for _ in range(epochs):
        for b in torch.randperm(len(Xt), generator=g).split(bs):
            b = b.to(dev); loss = (torch.nn.functional.binary_cross_entropy_with_logits(net(Xt[b])[:, 0], yt[b], reduction="none") * wt[b]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return net.eval()

def _mlp_score(net, Z, bs=65536):
    import torch
    dev = next(net.parameters()).device
    with torch.no_grad():
        return np.concatenate([net(torch.tensor(Z[i:i + bs], dtype=torch.float32, device=dev))[:, 0].cpu().numpy() for i in range(0, len(Z), bs)])

def run(ds, **cfg):
    c = dict(DEFAULTS); c.update(cfg)
    df, X, src, dst, tr, ho, s_idx, anc = load(ds); fam = df["fam"].values
    naive = naive_ramp_mask(ds, df, c["naive_ramp"]); n_naive = 0      # all False unless --naive_ramp > 0 (RQ2 analyst model)
    if c["features"] == "flow": X = X[:, :-N_CTX]               # ablation: per-flow features only
    rng = np.random.default_rng(seed()); t_fit, t_score, n_scored = [], 0.0, 0
    P = X[rng.choice(tr, min(c["n_pos_max"], len(tr)), replace=False)]
    ka = np.where(anc & (fam != "BENIGN"))[0]
    A = X[rng.choice(ka, min(c["n_known_max"], len(ka)), replace=False)] if len(ka) else np.zeros((0, X.shape[1]), X.dtype)
    m = ERNP(c, rng); nrng = np.random.default_rng(10_000 + seed())   # noise rng: separate, only used if label_noise > 0
    crng = np.random.default_rng(20_000 + seed())                      # flow-cap rng: separate, only used if flow_cap > 0
    wrng = np.random.default_rng(30_000 + seed()); nP0, nA0 = len(P), len(A)   # evidence-window rng: only used if keep != "fifo" or att_cap > 0
    def fitset(Z, n0, cap, kept):                                  # evidence passed to a fit
        if not kept or len(Z) <= cap: return Z[-cap:]              # FIFO: the most recent flows
        on = Z[n0:]; return np.vstack([Z[:n0], on[np.sort(wrng.choice(len(on), cap - n0, replace=False))]])   # anchor + uniform sample of the online evidence
    fit_args = lambda: (fitset(P, nP0, 2 * c["n_pos_max"], "P" in c["keep"]), fitset(A, nA0, 4 * c["n_known_max"], "A" in c["keep"]))
    F_ref = np.load(ROOT + f"exp/results/{ds}__{c['replay']}__s{seed()}.npz")["flag"].astype(bool) if c["replay"] else None
    m.eps = max(c["eps_mult"] * np.quantile(NearestNeighbors(n_neighbors=2).fit(P).kneighbors(P[:5000])[0][:, 1], 0.9), 1e-6)   # floor: duplicate-heavy anchors
    t0 = time.time(); m.fit(P, A); t_fit.append(time.time() - t0)
    tau = calib_upper(m.score(X[ho])) if c["fpr"] == 0.01 else float(np.quantile(m.score(X[ho]), 1 - c["fpr"]))   # operating point
    if c["rule"] == "post": tau = 0.0                              # posterior of the trained mixture > 1/2; no holdout quantile
    day = stream_periods(df, s_idx, c["period"]); s = np.zeros(len(s_idx)); F = np.zeros(len(s_idx), bool); n_lab = 0; n_flow = 0; audit = []
    refit_at = stream_periods(df, s_idx, c["refit"])               # analytic mode: refit only when this block ends
    order = list(dict.fromkeys(day)); last_of_block = {}
    for i, d in enumerate(order):
        i0 = np.argmax(day == d); last_of_block[d] = i == len(order) - 1 or refit_at[np.argmax(day == order[i + 1])] != refit_at[i0]
    pend = []                                                      # delayed decisions: (release period, addP, addA)
    for i_d, d in enumerate(order):
        idx = np.where(day == d)[0]
        t0 = time.time(); s[idx] = m.score(X[s_idx[idx]]); t_score += time.time() - t0; n_scored += len(idx)
        F[idx] = s[idx] > tau
        if c["evidence_source"] == "self":     # counterfactual: the detector's OWN verdicts (non-alerted flows) are
            nb = s_idx[idx][~F[idx]]           # trusted as benign evidence -> the usual self-updating channel that
            if len(nb):                        # frog-boiling exploits; no analyst labels are used
                nb = nb[rng.choice(len(nb), min(c["n_self"], len(nb)), replace=False)]
                P = np.vstack([P, X[nb]]); m.conf = np.vstack(([m.conf] if m.conf is not None else []) + [X[nb]])
                t0 = time.time(); m.fit(*fit_args()); t_fit.append(time.time() - t0)
            continue
        if c["budget"] <= 0: continue                              # no online labels -> q never changes (P1)
        al = s_idx[idx][F_ref[idx] if F_ref is not None else s[idx] > tau]; Bk = period_budget(d, c["period"], c["budget"])   # = budget when daily
        if len(al) < 2 or Bk <= 0:
            if c["analytic"] and last_of_block[d]:                 # block ends without new labels: refit on what accrued
                t0 = time.time(); m.fit(*fit_args()); t_fit.append(time.time() - t0)
            if not c["delay"]: continue
            k = 0                                                  # no decision now; delayed decisions may still arrive
        else: k = min(Bk, len(al)); km = MiniBatchKMeans(k, random_state=seed(), n_init=3).fit(X[al])
        addP, addA = [], []; per = max(1, c["flow_cap"] // k) if c["flow_cap"] > 0 and k else 0   # flows per group under the cap
        for cl in range(k):
            mem = al[km.labels_ == cl]
            if not len(mem): continue
            rep = mem[np.argmin(np.linalg.norm(X[mem] - km.cluster_centers_[cl], axis=1))]; n_lab += 1
            grp = X[mem] if c["evidence"] == "cluster" else X[[rep]]
            if per and len(mem) > per:          # E2: representative + random members, at most `per` flows per group
                grp = X[np.r_[rep, crng.choice(mem[mem != rep], per - 1, replace=False)]]
            if c["evidence"] == "oracle_pure":   # DIAGNOSTIC upper bound only: perfectly pure groups (uses ground truth)
                grp = X[mem[(fam[mem] == "BENIGN") == (fam[rep] == "BENIGN")]]
            is_ben = fam[rep] == "BENIGN" or naive[rep]   # the analyst's answer; naive[] is all False by default
            n_naive += int(bool(naive[rep]))
            if c["label_noise"] > 0 and nrng.random() < c["label_noise"]: is_ben = not is_ben   # analyst error
            if c["att_cap"] and not is_ben and len(grp) > c["att_cap"]:   # one attack decision adds a bounded number of flows
                grp = X[np.r_[rep, wrng.choice(mem[mem != rep], c["att_cap"] - 1, replace=False)]]
            (addP if is_ben else addA).append(grp); n_flow += len(grp)
            if c["evidence"] == "cluster":      # logging only (ground truth never used for decisions): group purity
                bad = fam[mem] != "BENIGN" if is_ben else fam[mem] == "BENIGN"
                rad = float(np.linalg.norm(X[mem] - X[rep], axis=1).max())   # group radius around the representative
                audit.append((d, int(is_ben), len(mem), int(bad.sum()), fam[rep], rad))
        if c["delay"]:                         # E5: decisions on period i reach the model at the end of period i + delay
            pend.append((i_d + c["delay"], addP, addA)); rel = [q for q in pend if q[0] <= i_d]; pend = [q for q in pend if q[0] > i_d]
            addP = [g for q in rel for g in q[1]]; addA = [g for q in rel for g in q[2]]
            if not addP and not addA: continue
        if addP:
            m.observe(np.vstack(addP), P[-2 * c["n_pos_max"]:])
            P = np.vstack([P] + addP); m.conf = np.vstack(([m.conf] if m.conf is not None else []) + addP)
        if addA: A = np.vstack([A] + addA)
        if c["analytic"] and not last_of_block[d]: continue        # evidence already acts through r_t at inference
        t0 = time.time(); m.fit(*fit_args()); t_fit.append(time.time() - t0)
        if c["protocol"] == "ssf":                                 # SSF's native protocol: predict after learning the day
            s[idx] = m.score(X[s_idx[idx]]); F[idx] = s[idx] > tau
    info = dict(cfg={k: v for k, v in c.items()}, tau=tau, eps=m.eps, labels_used=n_lab, audit=audit, flows_in_evidence=n_flow,
                naive_ramp_answers=n_naive, naive_ramp_flows=int(naive.sum()),
                timing=dict(fit_mean_s=float(np.mean(t_fit)), fit_max_s=float(np.max(t_fit)),
                            score_flows_per_s=n_scored / max(t_score, 1e-9)))
    return s, F, info

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ds"); ap.add_argument("--name", default="ernp")
    for k, v in DEFAULTS.items(): ap.add_argument("--" + k, type=type(v), default=v)
    a = vars(ap.parse_args()); ds, name = a.pop("ds"), a.pop("name")
    s, F, info = run(ds, **a); save(ds, name, s, F, info)
