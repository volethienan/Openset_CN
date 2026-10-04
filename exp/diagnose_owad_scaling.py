"""Is OWAD's poor ranking on the CIC datasets caused by input scaling? (diagnostic, seed 0, initial model only)

The OWAD wrapper min-max scales features that are already standardised and clipped at +-10 sd; heavy tails then squeeze
most flows into a narrow band of [0, 1]. Trains OWAD's own AE (official code and parameters) on the anchor reference
under three input scalings and reports the stream AUROC of the initial model (before any OWAD adaptation step).
"""
import os, sys, importlib, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ.setdefault("EXP_SEED", "0")
import common
SOTA = common.ROOT + "sota/"
for p in ("OWAD", "OWAD/admodels", "OWAD/moudles"): sys.path.insert(0, SOTA + p)
import yaml
_load = yaml.load; yaml.load = lambda f, Loader=yaml.SafeLoader: _load(f, Loader=Loader)
cwd = os.getcwd(); os.chdir(SOTA + "OWAD")
import myutils as U, AE
os.chdir(cwd)
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score

for ds in sys.argv[1:] or ["CIC17", "CIC18", "CIDDS_ctx"]:
    df, X, src, dst, tr, ho, s_idx, anc = common.load(ds); y = df["fam"].values[s_idx] != "BENIGN"
    lo, hi = np.quantile(X[tr], 0.001, axis=0), np.quantile(X[tr], 0.999, axis=0)
    variants = {"minmax (current wrapper)": MinMaxScaler().fit(X[tr]).transform(X),
                "standardised (as ae_static)": X,
                "robust minmax (0.1-99.9 pct, clipped)": np.clip((X - lo) / np.maximum(hi - lo, 1e-6), -1, 2)}
    for name, Z in variants.items():
        Z = Z.astype(np.float32); U.set_random_seed()
        model, _ = AE.train(Z[tr], Z.shape[1])
        _, s = AE.test(model, 0, Z[s_idx])
        print(f"{ds:10s} {name:40s} stream AUROC {roc_auc_score(y, s):.3f}", flush=True)
