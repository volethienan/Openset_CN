"""Faithful PyTorch port of CADE (Yang et al., USENIX Security 2021; github.com/whyisyoung/CADE, TF 1.12 original).

Kept identical to the original:
  - encoder dims [d, *hidden, n_classes], ReLU on hidden layers, linear latent and linear output, glorot-uniform init
  - batches: half_size real samples + half_size partners; the first `similar_ratio*batch` partners share the label,
    the rest have a different label (data.epoch_batches)
  - loss = MSE(x, x_hat) + lambda_1 * mean( y_ij * d_ij + (1 - y_ij) * relu(margin - d_ij) ),  d_ij = ||z_i - z_j|| + tiny
  - Adam, best-loss checkpoint over epochs
  - detection: per-class centroid in latent space, MAD of distances (x 1.4826), anomaly = min_i |d_i - median_i| / MAD_i,
    drift if anomaly > mad_threshold (3.5)
"""
import numpy as np, torch, torch.nn as nn

class CAE(nn.Module):
    def __init__(self, dims):
        super().__init__()
        enc, dec = [], []
        for i in range(len(dims) - 1):
            enc.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2: enc.append(nn.ReLU())
        for i in range(len(dims) - 1, 0, -1):
            dec.append(nn.Linear(dims[i], dims[i - 1]))
            if i > 1: dec.append(nn.ReLU())
        self.enc, self.dec = nn.Sequential(*enc), nn.Sequential(*dec)
        for m in self.modules():
            if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, x):
        z = self.enc(x); return z, self.dec(z)

def epoch_batches(y, batch_size, similar_ratio, rng):
    half = batch_size // 2; n_b = len(y) // half; num_sim = int(batch_size * similar_ratio)
    idx = rng.permutation(len(y))[: n_b * half].reshape(n_b, half)
    by_cls = {c: np.where(y == c)[0] for c in np.unique(y)}; not_cls = {c: np.where(y != c)[0] for c in np.unique(y)}
    partner = np.empty_like(idx)
    # vectorised but distribution-identical to the original per-sample loop: slot m < num_sim draws a partner of the
    # same class, the other slots draw a partner of a different class, uniformly at random with replacement
    lab = y[idx]; sim = np.zeros_like(idx, bool); sim[:, :num_sim] = True
    for c in by_cls:
        for is_sim, pool in ((True, by_cls[c]), (False, not_cls[c])):
            sel = (lab == c) & (sim == is_sim)
            if sel.any(): partner[sel] = pool[rng.integers(0, len(pool), sel.sum())]
    return idx, partner

def train_cade(X, y, hidden=(64, 32, 16), lr=1e-4, batch_size=512, epochs=250, lambda_1=0.1, margin=10.0,
               similar_ratio=0.25, device="cuda", seed=0, verbose=False):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    classes = np.unique(y); ymap = {c: i for i, c in enumerate(classes)}; yi = np.array([ymap[c] for c in y])
    model = CAE([X.shape[1], *hidden, len(classes)]).to(device); opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt = torch.tensor(X, dtype=torch.float32, device=device); best, best_state = np.inf, None
    for ep in range(epochs):
        idx, partner = epoch_batches(yi, batch_size, similar_ratio, rng); losses = []
        for b in range(len(idx)):
            L, Rr = torch.tensor(idx[b], device=device), torch.tensor(partner[b], device=device)
            xb = torch.cat([Xt[L], Xt[Rr]]); z, xh = model(xb)
            same = torch.tensor((yi[idx[b]] == yi[partner[b]]).astype(np.float32), device=device)
            half = len(idx[b]); d = torch.sqrt(((z[:half] - z[half:]) ** 2).sum(1) + 1e-10)
            con = (same * d + (1 - same) * torch.relu(margin - d)).mean()
            loss = ((xh - xb) ** 2).mean() + lambda_1 * con
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.detach())
        cur = float(torch.stack(losses).mean())                     # one GPU sync per epoch (was: per step)
        if cur < best: best, best_state = cur, {k: v.detach().clone() for k, v in model.state_dict().items()}
        if verbose and ep % 25 == 0: print("  cade epoch", ep, round(cur, 4), flush=True)
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad(): Z = model(Xt)[0].cpu().numpy()
    cents, med, mad = [], [], []
    for c in range(len(classes)):
        zc = Z[yi == c]; ce = zc.mean(0); dist = np.linalg.norm(zc - ce, axis=1)
        m_ = np.median(dist); cents.append(ce); med.append(m_); mad.append(1.4826 * np.median(np.abs(dist - m_)) + 1e-12)
    return dict(model=model, classes=classes, cents=np.array(cents), med=np.array(med), mad=np.array(mad), device=device)

def detect(st, X, mad_threshold=3.5):
    with torch.no_grad():
        Z = st["model"](torch.tensor(X, dtype=torch.float32, device=st["device"]))[0].cpu().numpy()
    D = np.linalg.norm(Z[:, None, :] - st["cents"][None], axis=2)            # (n, classes)
    A = np.abs(D - st["med"][None]) / st["mad"][None]
    closest = st["classes"][D.argmin(1)]; amin = A.min(1)
    return closest, amin, amin > mad_threshold, A

if __name__ == "__main__":   # validation on the IDS2018 split shipped with the original repo
    import sys
    from sklearn.metrics import precision_recall_fscore_support
    f = sys.argv[1] if len(sys.argv) > 1 else __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))), "sota/CADE/data/IDS_new_Infilteration.npz").replace("\\", "/")
    d = np.load(f)
    new = {"IDS_new_Infilteration": 3, "IDS_new_Hulk": 2, "IDS_new_SSH": 1}[f.split("/")[-1].replace(".npz", "")]  # as in run_ids_cade.sh
    st = train_cade(d["X_train"].astype(np.float32), d["y_train"], verbose=True)
    _, _, drift, _ = detect(st, d["X_test"].astype(np.float32))
    y_true = d["y_test"] == new
    P, R, F, _ = precision_recall_fscore_support(y_true, drift, average="binary")
    print(f"{f.split('/')[-1]}: at the MAD threshold (used in our experiments)  precision={P:.3f} recall={R:.3f} F1={F:.3f}")
    # the metric of the CADE paper (Table 3; cade/evaluate.py): samples with score > MAD threshold, ranked by the
    # distance to the closest centroid (descending), P/R/F1 at the BEST inspection count (oracle cut-off)
    Zt = st["model"](torch.tensor(d["X_test"].astype(np.float32), device=st["device"]))[0].detach().cpu().numpy()
    min_dis = np.linalg.norm(Zt[:, None, :] - st["cents"][None], axis=2).min(1)
    cand = np.where(drift)[0]; order = cand[np.argsort(-min_dis[cand], kind="stable")]
    tp = np.cumsum(y_true[order]); k = np.arange(1, len(order) + 1); prec = tp / k; rec = tp / y_true.sum()
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0); b = int(np.argmax(f1))
    print(f"{f.split('/')[-1]}: CADE-paper metric (best inspection count {b + 1})  precision={prec[b]:.3f} recall={rec[b]:.3f} F1={f1[b]:.3f}")
