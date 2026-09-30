"""Classical reference detectors (not SOTA; static, no labels, no adaptation), same protocol as every method:
fit on the anchor benign reference, threshold at 1% FPR on the anchor holdout, score the stream.

iforest   : IsolationForest (100 trees), score = -score_samples
knn       : distance to the 5th nearest reference flow
ae_static : MLP autoencoder d-64-32-64-d (Adam 1e-3, 20 epochs, batch 256), score = reconstruction MSE
usage: EXP_SEED=s python baselines/classic.py DS METHOD
"""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load, calib_upper, save, seed

N_REF = 40000

def run(ds, method):
    df, X, src, dst, tr, ho, s_idx, anc = load(ds); rng = np.random.default_rng(seed())
    ref = X[rng.choice(tr, min(N_REF, len(tr)), replace=False)]; t0 = time.time()
    if method == "iforest":
        from sklearn.ensemble import IsolationForest
        m = IsolationForest(n_estimators=100, random_state=seed(), n_jobs=int(__import__("os").environ.get("OMP_NUM_THREADS", 4))).fit(ref); f = lambda Z: -m.score_samples(Z)
    elif method == "knn":
        from sklearn.neighbors import NearestNeighbors
        nn_ = NearestNeighbors(n_neighbors=5).fit(ref); f = lambda Z: nn_.kneighbors(Z)[0][:, -1]
    elif method == "ae_static":
        import torch, torch.nn as nn
        dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); torch.manual_seed(seed())
        d = X.shape[1]; ae = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(),
                                           nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, d)).to(dev)
        opt = torch.optim.Adam(ae.parameters(), 1e-3); R = torch.tensor(ref, device=dev)
        for ep in range(20):
            for b in torch.randperm(len(R), device=dev).split(256):
                loss = ((ae(R[b]) - R[b]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        ae.eval()
        def f(Z):
            with torch.no_grad():
                out = []
                for c in np.array_split(np.arange(len(Z)), max(1, len(Z) // 50000)):
                    z = torch.tensor(Z[c], device=dev); out.append(((ae(z) - z) ** 2).mean(1).cpu().numpy())
                return np.concatenate(out)
    else: raise ValueError(method)
    t_init = time.time() - t0
    tau = calib_upper(f(X[ho])); t0 = time.time(); s = f(X[s_idx]); t_s = time.time() - t0
    return s, s > tau, dict(tau=tau, labels_used=0, timing=dict(init_s=t_init, score_flows_per_s=len(s_idx) / max(t_s, 1e-9)))

if __name__ == "__main__":
    t0 = time.time(); s, f, info = run(sys.argv[1], sys.argv[2]); info["seconds"] = round(time.time() - t0)
    save(sys.argv[1], sys.argv[2], s, f, info)
