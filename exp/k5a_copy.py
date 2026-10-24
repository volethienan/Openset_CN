"""Final configuration K5A = K5 + anchored attack evidence (--keep A). With --keep A, a fit sees exactly the same attack
evidence as with the FIFO window as long as the attack evidence holds at most 4 * n_known_max flows, so such a run is
identical to its K5 counterpart. This script finds, for every job of run_jobs group `k5a`, whether the K5 counterpart
ever exceeded that size after a period whose refit was used for scoring (from the run's audit of analyst decisions,
counting whole groups, which is conservative), and copies the unaffected K5 runs to the K5A name (scores and flags
unchanged; cfg.keep set to "A", field k5a_copied_from added). The affected runs are left for run_jobs.py k5a.
usage: python k5a_copy.py [--dry] [seeds ...]      (default seeds 0-4)
"""
import os, sys, json, shutil, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common, run_jobs

CAP = 4 * 10000                                   # 4 * n_known_max, the size of the attack evidence a fit sees
_periods, _nA0 = {}, {}

def stream_info(ds, hours):
    if (ds, hours) not in _periods:
        df = pd.read_parquet(common.STREAMS + f"{ds}.parquet", columns=["t", "fam"])
        anc = (df["t"] < common.anchor_end(ds, df["t"].min())).values
        _nA0[ds] = int(min(10000, (anc & (df["fam"].values != "BENIGN")).sum()))
        t = df["t"].values[~anc]
        p = pd.to_datetime(t, unit="s").strftime("%Y-%m-%d").values if hours == 24 else (t // (3600 * hours)).astype(np.int64)
        _periods[(ds, hours)] = list(dict.fromkeys(p))
    return _periods[(ds, hours)], _nA0[ds]

def affected(ds, info):
    c = info.get("cfg", {}); audit = info.get("audit") or []
    if c.get("evidence_source") == "self" or c.get("budget", 200) <= 0 or not audit: return False
    hours = int(c.get("period", 24)); periods, nA0 = stream_info(ds, hours)
    last = periods[-1]; key = (lambda d: str(d)) if hours == 24 else (lambda d: int(d))
    add = {}
    for d, is_ben, n, *_ in audit:
        if not is_ben: add[key(d)] = add.get(key(d), 0) + int(n)
    size = nA0
    for p in periods:
        size += add.get(key(p), 0)
        if size > CAP and p != last: return True        # an overflowing evidence set was used by a later refit
    return False

if __name__ == "__main__":
    dry = "--dry" in sys.argv; seeds = [int(a) for a in sys.argv[1:] if a != "--dry"] or [0, 1, 2, 3, 4]
    R = common.RESULTS; stats = dict(copied=0, already=0, affected=0, missing_k5=0)
    for s in seeds:
        for pool, ds, nm, args in run_jobs.k5a_jobs(s):
            dst = R + f"{ds}__{nm}__s{s}"; src = R + f"{ds}__{nm[:-4]}_K5__s{s}"
            if os.path.exists(dst + ".json"): stats["already"] += 1; continue
            if not os.path.exists(src + ".json"): stats["missing_k5"] += 1; continue
            info = json.load(open(src + ".json"))
            if affected(ds, info): stats["affected"] += 1; continue
            stats["copied"] += 1
            if dry: continue
            shutil.copyfile(src + ".npz", dst + ".npz")
            info.setdefault("cfg", {})["keep"] = "A"; info["k5a_copied_from"] = f"{nm[:-4]}_K5"
            json.dump(info, open(dst + ".json", "w"), default=str)
    print(stats)
