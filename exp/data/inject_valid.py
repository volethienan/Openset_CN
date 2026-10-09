"""E4: slow poisoning with protocol-valid flow records, re-extracted by the stream pipeline (CIDDS-001).

Same threat model and schedule as inject.py (compromised host A, ramp from day 4 for 7 days, then 2 days of the zero-day
bruteForce payload, 30 flows per hour), but the injected traffic is generated as CIDDS-001 flow RECORDS, not as
interpolated feature vectors:
  * templates are real records: A's own benign TCP flows and the real bruteForce flows of the dataset;
  * forward / reply direction follows the bruteForce template (port 22 on the victim side), roles as in inject.py:
    'init' A initiates towards its usual partners, 'resp' A is a server that honest clients connect to;
  * ramp fraction a: continuous fields interpolate between the two records and are projected onto valid values
    (packets integer >= 1, bytes integer in [40, 1500] x packets, duration >= 0); discrete fields (ports, TCP flags,
    ToS) are those of the benign template while a < 0.5 and of the attack template afterwards; protocol TCP throughout;
  * per-flow features: prepare.cidds_chunk on the generated records (the stream code itself);
  * context features: recomputed exactly over the FULL raw data plus the injected records (prepare_ctx window
    statistics), for the injected flows AND for every sampled stream flow that shares a source, destination or pair
    with them inside a window.
The original sampled flows are those of the clean stream (a rebuild with the injected rows appended last keeps the same
sampling draws), so clean and poisoned streams differ only by the attacker's traffic and its context effect.
Partners: the 20 most frequent partners of A in the seed-0 stream. Labels: 'INJ_ramp' (ramp), 'INJ_attack' (payload).
Gate: before injection, the recomputed context of sampled flows that touch these hosts equals the clean stream's.

usage: ERNP_RAW=/path/to/dataraw python inject_valid.py [seeds...]   (needs CIDDS-001; one raw scan, cached in mock/streams/_raw_cache_injvalid.pkl)
"""
import os, sys, glob, pickle, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prepare as P
from prepare_ctx import window_stats, WINDOWS_MS

STREAMS = P.OUT; RAW = os.environ.get("ERNP_RAW", P.ROOT).rstrip("/") + "/CIDDS-001/traffic/OpenStack/"   # ERNP_RAW: folder with the raw datasets
START, RAMP, ATTACK, RATE_PER_H, N_PART = 4.0, 7.0, 2.0, 30, 20
CACHE = STREAMS + "_raw_cache_injvalid.pkl"
N_FLOW = 20                                                           # per-flow feature columns f_0..f_19; f_20..f_29 = context

def hosts():
    """A and its partners for both roles, chosen exactly as inject.py does, on the seed-0 stream"""
    df = pd.read_parquet(STREAMS + "CIDDS_ctx.parquet"); ben = df[df["fam"] == "BENIGN"]
    attackers = set(df.loc[df["fam"] != "BENIGN", "src"])
    cand = ben["src"].value_counts(); cand = cand[[s.startswith("192.168.2") and s not in attackers for s in cand.index]]
    A_i = cand.index[0]; p_i = ben.loc[ben["src"] == A_i, "dst"].value_counts().index[:N_PART].tolist()
    internal = ben[ben["src"].str.startswith("192.168.2")]
    cand = internal.groupby("dst")["src"].nunique().sort_values(ascending=False)
    cand = cand[[s.startswith("192.168.100.") and not s.endswith(".255") for s in cand.index]]
    A_r = cand.index[0]; p_r = internal.loc[internal["dst"] == A_r, "src"].value_counts().index[:N_PART].tolist()
    return dict(init=(A_i, p_i), resp=(A_r, p_r))

def scan(H):
    """one pass over the raw data: templates (A's benign TCP records, all bruteForce records) and the context events
    (t_ms, src, dst, bytes) of every flow that touches A or a partner"""
    if os.path.exists(CACHE): return pickle.load(open(CACHE, "rb"))
    K = set(); [K.update([A, *p]) for A, p in H.values()]; As = {A for A, _ in H.values()}
    tmpl, ev = [], []
    for f in sorted(glob.glob(RAW + "*.csv")):
        for ch in pd.read_csv(f, chunksize=2_000_000, low_memory=False):
            src = ch["Src IP Addr"].astype(str).str.strip(); dst = ch["Dst IP Addr"].astype(str).str.strip()
            fam = ch["attackType"].astype(str).str.strip()
            touch = src.isin(K) | dst.isin(K)
            t = pd.to_datetime(ch.loc[touch, "Date first seen"], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce").astype("int64").values // 1_000_000
            ev.append(pd.DataFrame(dict(t=t, src=src[touch].values, dst=dst[touch].values, b=P.parse_bytes(ch.loc[touch, "Bytes"]))))
            keep = (fam == "bruteForce") | ((src.isin(As) | dst.isin(As)) & (fam == "---") & (ch["Proto"].astype(str).str.strip() == "TCP"))
            if keep.any():
                s = ch[keep].copy()
                for c in s.columns[s.dtypes == object]: s[c] = s[c].astype(str).str.strip()
                tmpl.append(s)
            print("  scan", os.path.basename(f), len(ev[-1]), flush=True)
    out = dict(tmpl=pd.concat(tmpl, ignore_index=True), ev=pd.concat(ev, ignore_index=True)); pickle.dump(out, open(CACHE, "wb"))
    return out

def num(s): return pd.to_numeric(s, errors="coerce").fillna(0).values

def generate(variant, A, partners, tmpl, t0, rng):
    """injected flow records in the raw CIDDS-001 format"""
    bf = tmpl[tmpl["attackType"] == "bruteForce"]; bf = bf[(num(bf["Dst Pt"]) == 22) | (num(bf["Src Pt"]) == 22)]
    ben = tmpl[tmpl["attackType"] == "---"]
    a_src, a_dst = ben[ben["Src IP Addr"] == A], ben[ben["Dst IP Addr"] == A]
    rows = []
    for t in np.arange(t0 + START * 86400, t0 + (START + RAMP + ATTACK) * 86400, 3600 / RATE_PER_H):
        a = min(1.0, (t - t0 - START * 86400) / (RAMP * 86400))
        d = bf.iloc[rng.integers(len(bf))]; fwd = num(pd.Series([d["Dst Pt"]]))[0] == 22   # attacker -> victim:22
        p = partners[rng.integers(len(partners))]; eph = int(rng.integers(32768, 61000))
        # victim side: 'init' -> the partner, 'resp' -> A; the attacker side is the other host
        att, vic = (A, p) if variant == "init" else (p, A)
        src, dst, sp, dp = (att, vic, eph, 22) if fwd else (vic, att, 22, eph)
        pool = a_src if src == A else a_dst                           # A's benign records in the same role
        b = pool.iloc[rng.integers(len(pool))]
        dur = max(0.0, (1 - a) * num(pd.Series([b["Duration"]]))[0] + a * num(pd.Series([d["Duration"]]))[0])
        pk = max(1, int(round((1 - a) * num(pd.Series([b["Packets"]]))[0] + a * num(pd.Series([d["Packets"]]))[0])))
        by = (1 - a) * P.parse_bytes(pd.Series([b["Bytes"]]))[0] + a * P.parse_bytes(pd.Series([d["Bytes"]]))[0]
        by = int(np.clip(round(by), 40 * pk, 1500 * pk))
        disc = b if a < 0.5 else d
        if a < 0.5:                                                   # ports of the benign template, ephemeral side redrawn
            bs, bd = int(num(pd.Series([b["Src Pt"]]))[0]), int(num(pd.Series([b["Dst Pt"]]))[0])
            sp, dp = (eph, bd) if bs >= 1024 and bd < 1024 else (bs, eph) if bd >= 1024 and bs < 1024 else (bs, bd)
        rows.append({"Date first seen": pd.Timestamp(int(round(t * 1000)), unit="ms").strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                     "Duration": round(dur, 3), "Proto": "TCP", "Src IP Addr": src, "Src Pt": sp, "Dst IP Addr": dst, "Dst Pt": dp,
                     "Packets": pk, "Bytes": by, "Flags": disc["Flags"], "Tos": int(num(pd.Series([disc["Tos"]]))[0]),
                     "attackType": "INJ_attack" if a >= 1.0 else "INJ_ramp"})
    g = pd.DataFrame(rows)
    assert (g["Packets"] >= 1).all() and (g["Bytes"] >= 40 * g["Packets"]).all() and (g["Bytes"] <= 1500 * g["Packets"]).all()
    assert (g["Duration"] >= 0).all() and g[["Src Pt", "Dst Pt"]].apply(lambda c: c.between(0, 65535)).all().all()
    return g

def counts_of(q_t, q_src, q_dst, e_t, e_src, e_dst, e_b):
    """window statistics of prepare_ctx (per window: n_src, n_dst, n_pair, b_src) of query flows q against the event set e
    (which contains q). Exact for every key whose events are all in e."""
    codes, _ = pd.factorize(np.r_[e_src, e_dst, q_src, q_dst]); n = len(e_t)
    es, ed = codes[:n], codes[n:2 * n]; qs, qd = codes[2 * n:2 * n + len(q_t)], codes[2 * n + len(q_t):]
    out = []
    for key_e, key_q, val in ((es, qs, e_b), (ed, qd, None), (es * 1_048_576 + ed, qs * 1_048_576 + qd, None)):
        kk, _ = pd.factorize(np.r_[key_e, key_q]); ke, kq = kk[:n], kk[n:]
        o = np.lexsort((e_t, ke)); comp_e = (ke[o].astype(np.int64) << 32) | (e_t[o] - e_t.min())
        cs = np.concatenate([[0.0], np.cumsum(val[o])]) if val is not None else None
        hi = (kq.astype(np.int64) << 32) | (q_t - e_t.min()); r = []
        for W in WINDOWS_MS:
            left = np.searchsorted(comp_e, (kq.astype(np.int64) << 32) + np.maximum(q_t - e_t.min() - W + 1, 0), "left")
            right = np.searchsorted(comp_e, hi, "right"); r.append((right - left, cs[right] - cs[left] if cs is not None else None))
        out.append(r)
    return np.column_stack([c for w in range(len(WINDOWS_MS)) for c in (out[0][w][0], out[1][w][0], out[2][w][0], out[0][w][1])]).astype(np.float64)

def features(C):
    """counts (per window: n_src, n_dst, n_pair, b_src) -> the 10 context features of prepare_ctx.ctx_features"""
    cols = []
    for w in range(len(WINDOWS_MS)):
        n_src, n_dst, n_pair, b_src = C[:, 4 * w:4 * w + 4].T
        cols += [np.log1p(n_src), np.log1p(n_dst), np.log1p(n_pair), np.log1p(b_src), np.log1p(n_src / n_pair)]
    return np.column_stack(cols).astype(np.float32)

def counts_from_features(F):
    """inverse of features() for the clean stream's context columns (counts are integers)"""
    C = []
    for w in range(len(WINDOWS_MS)):
        x = np.expm1(F[:, 5 * w:5 * w + 4].astype(np.float64)); x[:, :3] = np.round(x[:, :3]); C.append(x)
    return np.hstack(C)

def build(seed, H, cache):
    sfx = "" if seed == 0 else f"_s{seed}"; df = pd.read_parquet(STREAMS + f"CIDDS_ctx{sfx}.parquet")
    f = [c for c in df.columns if c.startswith("f_")]; assert len(f) == N_FLOW + 10
    ev = cache["ev"]; e_t, e_src, e_dst, e_b = ev["t"].values.astype(np.int64), ev["src"].values, ev["dst"].values, ev["b"].values
    t_ms = np.round(df["t"].values * 1000).astype(np.int64)
    # gate: the recomputed context of sampled flows touching these hosts equals the clean stream's
    K = set(); [K.update([A, *p]) for A, p in H.values()]
    tq = np.where(df["src"].isin(K).values & df["dst"].isin(K).values)[0]
    tq = tq[np.random.default_rng(seed).choice(len(tq), min(20000, len(tq)), replace=False)]
    C0 = features(counts_of(t_ms[tq], df["src"].values[tq], df["dst"].values[tq], e_t, e_src, e_dst, e_b))
    err = np.abs(C0 - df[f[N_FLOW:]].values[tq]).max(); print(f"  gate s{seed}: {len(tq)} flows, max |ctx diff| = {err:.2e}", flush=True)
    assert err < 1e-3, "context recomputation does not match the clean stream"
    for variant, (A, partners) in H.items():
        g = generate(variant, A, partners, cache["tmpl"], df["t"].min(), np.random.default_rng(42))
        inj = P.cidds_chunk(g)
        g_t = (inj["t"].values * 1000).round().astype(np.int64); g_b = P.parse_bytes(g["Bytes"].astype(str))
        a_t, a_src, a_dst, a_b = np.r_[e_t, g_t], np.r_[e_src, inj["src"].values], np.r_[e_dst, inj["dst"].values], np.r_[e_b, g_b]
        inj[f[N_FLOW:]] = features(counts_of(g_t, inj["src"].values, inj["dst"].values, a_t, a_src, a_dst, a_b))
        hk = set(inj["src"]) | set(inj["dst"]); out = df.copy()
        hit = np.where(out["src"].isin(hk).values | out["dst"].isin(hk).values)[0]
        # sampled flows sharing a host with the injection: clean counts + the injected flows inside their windows
        # (the difference new - old is exact for every key: injected flows only touch keys of hosts in K)
        qh = (t_ms[hit], out["src"].values[hit], out["dst"].values[hit])
        dlt = counts_of(*qh, a_t, a_src, a_dst, a_b) - counts_of(*qh, e_t, e_src, e_dst, e_b)
        out.loc[out.index[hit], f[N_FLOW:]] = features(counts_from_features(out[f[N_FLOW:]].values[hit]) + dlt)
        inj = inj[f + ["t", "src", "dst", "fam"]]
        res = pd.concat([out, inj]).sort_values("t", kind="stable").reset_index(drop=True)
        name = f"CIDDS_ctx{sfx}_injvalid{variant}_bf"; res.to_parquet(STREAMS + name + ".parquet")
        print(" ", name, "A =", A, "injected", len(inj), inj["fam"].value_counts().to_dict(), "context-updated flows", len(hit), flush=True)

if __name__ == "__main__":
    H = hosts(); print("hosts", {k: (v[0], len(v[1])) for k, v in H.items()}, flush=True)
    cache = scan(H)
    for s in [int(x) for x in sys.argv[1:]] or range(5): build(s, H, cache)
