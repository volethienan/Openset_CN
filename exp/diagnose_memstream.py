"""Why does MemStream rank attacks BELOW benign (AUROC < 0.5) from the first stream day? (diagnostic, seed 0)

Compares, with the official MemStream class and our wrapper's settings:
  static : the initial model scores the stream with its initial memory, no memory update
  stream : the official streaming mode, memory updated with every flow scored below beta (= our 1%-FPR threshold)
If static ranks well and stream inverts, the inversion is absorption through the memory update, not a broken model.
Also reports the fraction of the final memory filled by attack flows.
"""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, HERE + "/baselines")
os.environ.setdefault("EXP_SEED", "0")
import torch, common, sota as S
from sklearn.metrics import roc_auc_score

for ds in sys.argv[1:] or ["CIC18", "CIDDS_ctx"]:
    df, X, src, dst, tr, ho, s_idx, anc = common.load(ds); fam = df["fam"].values; y = fam[s_idx] != "BENIGN"
    code = open(S.SOTA + "MemStream/code/memstream.py").read()
    cls = code[code.index("class MemStream"): code.index("if args.dataset in")]
    import torch.nn as nn, torch.nn.functional as F
    from torch.autograd import Variable
    ns = dict(torch=torch, nn=nn, F=F, Variable=Variable, device=S.dev, np=np); exec(cls, ns); torch.manual_seed(0)
    Xt = torch.tensor(X, dtype=torch.float32, device=S.dev); init = Xt[tr[:2048]]
    model = ns["MemStream"](X.shape[1], {"beta": 0.0, "memory_len": len(init), "batch_size": 1, "lr": 1e-2}).to(S.dev)
    ns["model"] = model; model.mem_data = init; torch.set_grad_enabled(True); model.train_autoencoder(init, epochs=5000)
    torch.set_grad_enabled(False); model.initialize_memory(init)
    def score_only(Z):
        new = (Z - model.mean) / model.std; new[:, model.std == 0] = 0
        return torch.cdist(model.encoder(new), model.memory, p=1).min(1).values.cpu().numpy()
    static = np.concatenate([score_only(Xt[s_idx[i:i + 65536]]) for i in range(0, len(s_idx), 65536)])
    thr = common.calib_upper(score_only(Xt[ho])); model.max_thres = torch.tensor(thr).to(S.dev)
    span = (fam[tr[:2048]] == "BENIGN").mean()
    print(f"{ds}: initial memory = first 2048 reference flows (benign share {span:.2f}); static AUROC {roc_auc_score(y, static):.3f}", flush=True)
    stream = np.zeros(len(s_idx)); admitted = np.zeros(len(s_idx), bool)
    for j, g in enumerate(s_idx):
        stream[j] = float(model(Xt[g: g + 1])); admitted[j] = stream[j] <= thr
    print(f"{ds}: streaming AUROC {roc_auc_score(y, stream):.3f}; attack flows admitted to memory: {admitted[y].mean():.3f} of attacks; "
          f"attack share of admitted flows: {y[admitted].mean():.3f}", flush=True)
