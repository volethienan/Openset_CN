"""MemStream memory-admission threshold beta (diagnostic, seed 0).

Official README: beta = 0.1 for the intrusion datasets (NSL-KDD, UNSW-NB15, CICIDS-DoS), memlen 2048. Our wrapper used
beta = the alert threshold (99th percentile of holdout scores), 100-900x larger on our feature scale. Here the alert
threshold stays at 1% holdout FPR and only the memory-admission beta changes:
  official : beta = 0.1 (the authors' value)
  q50      : beta = median holdout score (scale-free, strict)
  q99      : beta = 99th percentile (= previous wrapper)
"""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, HERE + "/baselines")
os.environ.setdefault("EXP_SEED", "0")
import torch, common, sota as S
from sklearn.metrics import roc_auc_score, f1_score

def build(X, tr):
    code = open(S.SOTA + "MemStream/code/memstream.py").read()
    cls = code[code.index("class MemStream"): code.index("if args.dataset in")]
    import torch.nn as nn, torch.nn.functional as F
    from torch.autograd import Variable
    ns = dict(torch=torch, nn=nn, F=F, Variable=Variable, device=S.dev, np=np); exec(cls, ns); torch.manual_seed(0)
    Xt = torch.tensor(X, dtype=torch.float32, device=S.dev); init = Xt[tr[:2048]]
    m = ns["MemStream"](X.shape[1], {"beta": 0.0, "memory_len": len(init), "batch_size": 1, "lr": 1e-2}).to(S.dev)
    ns["model"] = m; m.mem_data = init; torch.set_grad_enabled(True); m.train_autoencoder(init, epochs=5000)
    torch.set_grad_enabled(False); m.initialize_memory(init); return m, Xt

for ds in sys.argv[1:] or ["CIC18", "CIDDS_ctx"]:
    df, X, src, dst, tr, ho, s_idx, anc = common.load(ds); y = df["fam"].values[s_idx] != "BENIGN"
    for mode in ("official", "q50", "q99"):
        m, Xt = build(X, tr)
        new = (Xt[ho] - m.mean) / m.std; new[:, m.std == 0] = 0
        h = torch.cdist(m.encoder(new), m.memory, p=1).min(1).values.cpu().numpy()
        alert = common.calib_upper(h)
        beta = {"official": 0.1, "q50": float(np.median(h)), "q99": alert}[mode]
        m.max_thres = torch.tensor(beta).to(S.dev)
        s = np.array([float(m(Xt[g: g + 1])) for g in s_idx])
        f = s > alert
        print(f"{ds:10s} {mode:8s} beta={beta:8.3f}  AUROC {roc_auc_score(y, s):.3f}  F1 {f1_score(y, f):.3f}  "
              f"FPR {f[~y].mean():.4f}  memory updates {int(m.count)}", flush=True)
