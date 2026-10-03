"""Streams WITH temporal-context features (Kitsune/AfterImage-style), computed on the FULL data before sampling
(so upsampling rare attack families cannot leak into rate features). Context uses only past/current flows.

ctx features per flow, for W in {10 s, 120 s}:
  n_src, n_dst, n_pair  : number of flows with the same src / dst / (src,dst) pair in (t-W, t]
  b_src                 : bytes sent by the same src in (t-W, t]
  fan = n_src / n_pair  : fan-out proxy
"""
import pandas as pd, numpy as np, glob, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prepare as P

ROOT, OUT = P.ROOT, P.OUT
WINDOWS_MS = (10_000, 120_000)

def window_stats(key, t_ms, val=None):
    """For each i: count (and sum of val) of rows with same key and t in (t_i - W, t_i]. Exact, vectorized."""
    assert key.max() < 2 ** 30, "key too large for (key << 32) packing"
    order = np.lexsort((t_ms, key)); k_hi = key[order].astype(np.int64) << 32
    trel = (t_ms[order] - t_ms.min()).astype(np.int64); comp = k_hi | trel
    res = []
    for W in WINDOWS_MS:
        # first row of the same key with t' > t - W (integer ms), never crossing into the previous key
        left = np.searchsorted(comp, k_hi + np.maximum(trel - W + 1, 0), side="left")
        right = np.searchsorted(comp, comp, side="right")          # include ties at the same ms
        cnt = np.empty(len(key)); cnt[order] = right - left
        out = [cnt]
        if val is not None:
            cs = np.concatenate([[0.0], np.cumsum(val[order])]); s = np.empty(len(key)); s[order] = cs[right] - cs[left]; out.append(s)
        res.append(out)
    return res

def ctx_features(t_ms, src, dst, byts):
    pair = pd.factorize(src.astype(np.int64) * 1_048_576 + dst)[0].astype(np.int64)  # compact ids: key << 32 must not overflow
    S = window_stats(src, t_ms, byts); D = window_stats(dst, t_ms); Pp = window_stats(pair, t_ms)
    cols = []
    for w in range(len(WINDOWS_MS)):
        n_src, b_src = S[w]; n_dst = D[w][0]; n_pair = Pp[w][0]
        cols += [np.log1p(n_src), np.log1p(n_dst), np.log1p(n_pair), np.log1p(b_src), np.log1p(n_src / n_pair)]
    return np.column_stack(cols).astype(np.float32)

def build(name):
    rng = np.random.default_rng(P.SEED)
    if name == "CIDDS":
        files = sorted(glob.glob(ROOT + "CIDDS-001/traffic/OpenStack/*.csv"))
        cols = ["Date first seen", "Src IP Addr", "Dst IP Addr", "Bytes", "attackType"]
        rd = lambda f, **kw: pd.read_csv(f, low_memory=False, **kw)
        def pass1(ch):
            t = pd.to_datetime(ch["Date first seen"], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce").astype("int64").values // 1_000_000
            return t, ch["Src IP Addr"].astype(str).str.strip().values, ch["Dst IP Addr"].astype(str).str.strip().values, \
                   P.parse_bytes(ch["Bytes"]), ch["attackType"].astype(str).str.strip().replace({"---": "BENIGN"}).values
    else:  # ToN
        files = [ROOT + "NF-ToN-IoT-v3/NF-ToN-IoT-v3.csv"]
        cols = ["FLOW_START_MILLISECONDS", "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "IN_BYTES", "OUT_BYTES", "Attack"]
        rd = lambda f, **kw: pd.read_csv(f, low_memory=False, **kw)
        def pass1(ch):
            return ch["FLOW_START_MILLISECONDS"].astype("int64").values, ch["IPV4_SRC_ADDR"].astype(str).values, ch["IPV4_DST_ADDR"].astype(str).values, \
                   (ch["IN_BYTES"].astype("float64") + ch["OUT_BYTES"].astype("float64")).values, ch["Attack"].astype(str).str.strip().replace({"Benign": "BENIGN"}).values
    # ---- pass 1: full data, key columns only
    T, S, D, BY, F = [], [], [], [], []
    for f in files:
        for ch in rd(f, usecols=cols, chunksize=3_000_000):
            t, s, d, b, fam = pass1(ch); T.append(t); S.append(s); D.append(d); BY.append(b); F.append(fam)
        print("  pass1", name, f, flush=True)
    t = np.concatenate(T); fam = np.concatenate(F); byts = np.concatenate(BY)
    t = np.where(t > 0, t, np.median(t[t > 0])).astype(np.int64)  # guard against unparsable timestamps
    ids, uniq = pd.factorize(np.concatenate([np.concatenate(S), np.concatenate(D)]))
    src, dst = ids[: len(t)].astype(np.int64), ids[len(t):].astype(np.int64)
    del S, D, T, F, BY
    ctx = ctx_features(t, src, dst, byts); print("  ctx done", ctx.shape, flush=True)
    fc = pd.Series(fam).value_counts().to_dict(); r = P.TARGET / len(t)
    rate = np.where(fam == "BENIGN", r, np.array([max(r, min(1.0, P.MIN_FAM / fc[x])) for x in fam]))
    keep = rng.random(len(t)) < rate; keep_idx = np.where(keep)[0]; print("  sampled", len(keep_idx), flush=True)
    # ---- pass 2: base features (same code as prepare.py) for the sampled rows only
    parts, off = [], 0
    for f in files:
        for ch in rd(f, chunksize=2_000_000):
            n = len(ch); sel = keep_idx[(keep_idx >= off) & (keep_idx < off + n)] - off
            if len(sel):
                sub = ch.iloc[sel]
                df = P.cidds_chunk(sub) if name == "CIDDS" else P.nf_chunk(sub)
                df.index = sel + off; parts.append(df)
            off += n
        print("  pass2", name, f, flush=True)
    df = pd.concat(parts)
    C = ctx[df.index.values]; base = len([c for c in df.columns if c.startswith("f_")])
    for j in range(C.shape[1]): df[f"f_{base + j}"] = C[:, j]
    df = df.dropna(subset=["t"]).sort_values("t", kind="stable").reset_index(drop=True)
    sfx = "" if P.SEED == 0 else f"_s{P.SEED}"
    df.to_parquet(OUT + f"{name}_ctx{sfx}{P.EXTRA}.parquet"); print(name, sfx, df.shape, df["fam"].value_counts().to_dict(), flush=True)

if __name__ == "__main__":
    build(sys.argv[1])
