"""Leakage sanity baseline (not a SOTA): new_ip flags every stream flow whose source domain never appeared in the anchor.

If new_ip scores well on a dataset, attacks there are separable by IP novelty alone; the paper reports it next to RQ1/RQ2
so readers can see that ERNP's gains do not come from IP identity (IPs are never used as features).
usage: EXP_SEED=s python baselines/sanity.py DS
"""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load, save

def run_new_ip(ds):
    df, X, src, dst, tr, ho, s_idx, anc = load(ds)
    seen = set(src[anc]) | set(dst[anc]); f = np.array([s not in seen for s in src[s_idx]])
    return f.astype(float), f, dict(labels_used=0)

if __name__ == "__main__":
    t0 = time.time(); s, f, info = run_new_ip(sys.argv[1]); info["seconds"] = round(time.time() - t0)
    save(sys.argv[1], "new_ip", s, f, info)
