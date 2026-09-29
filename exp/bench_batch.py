"""ERNP inference throughput on a fixed large batch (RQ5, criterion >= 100k flows/s on CPU).

The campaign measures scoring per daily batch, which on streams with many small days is dominated by per-call
overhead. Here the detector is fitted exactly as at the start of ernp.run (seed 0) and then scores one batch of
100,000 stream flows, 3 repetitions, best time reported. Run on an idle machine (report_all.sh runs it last).
"""
import os, sys, time, numpy as np
os.environ.setdefault("EXP_SEED", "0"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, seed
from sklearn.neighbors import NearestNeighbors
import ernp

N = 100_000
for ds in ("CIDDS_ctx", "ToN_ctx", "CIC18"):
    c = dict(ernp.DEFAULTS); df, X, src, dst, tr, ho, s_idx, anc = load(ds); fam = df["fam"].values
    rng = np.random.default_rng(seed())
    P = X[rng.choice(tr, min(c["n_pos_max"], len(tr)), replace=False)]
    ka = np.where(anc & (fam != "BENIGN"))[0]
    A = X[rng.choice(ka, min(c["n_known_max"], len(ka)), replace=False)] if len(ka) else np.zeros((0, X.shape[1]), X.dtype)
    m = ernp.ERNP(c, rng)
    m.eps = c["eps_mult"] * np.quantile(NearestNeighbors(n_neighbors=2).fit(P).kneighbors(P[:5000])[0][:, 1], 0.9)
    t0 = time.time(); m.fit(P, A); t_fit = time.time() - t0
    Z = X[s_idx[:N]]; best = np.inf
    for _ in range(3):
        t0 = time.time(); m.score(Z); best = min(best, time.time() - t0)
    print(f"{ds}: fit {t_fit:.1f}s | scoring {len(Z)} flows in {best:.3f}s -> {len(Z) / best:,.0f} flows/s (CPU, {os.cpu_count()} threads)", flush=True)
