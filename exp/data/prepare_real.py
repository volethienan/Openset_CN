"""Streams from real-network datasets, in the schema of the other streams:
   t (unix s), src, dst, fam (attack family or 'BENIGN'), f_* numeric features (per-flow features, then 10 context features).

  CTU13   : CTU-13 (CTU University, Czech Republic, August 2011). Real background traffic of the university network with
            13 captures of real botnets (Neris, Rbot, Virut, Menti, Sogou, Murlo, NSIS.ay). Bidirectional Argus flows
            (*.binetflow). Attack = flows labelled 'From-Botnet', family = the botnet of the capture. Background flows
            are unverified and are treated as benign, as is usual; column `lab` keeps botnet / normal / background, so
            that the false-positive rate can also be reported on the verified 'Normal' flows alone.
  SIMARGL : RoEduNet-SIMARGL2021 (Romanian education network, March-May 2021), NetFlow v9 (nProbe), five capture days,
            seven attack types run against servers of the network (SYN/UDP/NULL/XMAS/FIN scans, Slowloris, R-U-Dead-Yet).
  APPRAISE: APPRAISE H2020 real labelled NetFlow dataset (network of the BTC complex in Ljubljana, a commercial and
            building-management network; 18 January - 12 April 2023, 76 consecutive days), NetFlow v9, 15.1M flows,
            three attack categories run in the network (reconnaissance from the first day, DDoS, brute force).

Protocol identical to prepare_ctx.py: the 10 temporal-context features (counts and bytes per source, destination and
pair in a short and a long window, and fan-out) are computed retrospectively from COMPLETE records on the FULL data before sampling, over flows that started
at or before the flow; sampling is uniform with rate r = TARGET / N, rare attack families kept up to MIN_FAM flows.
Addresses are never features. The windows are 10 s and 120 s; time is counted in milliseconds for CTU13, in 10 ms
ticks for SIMARGL, whose capture spans 67 days, and in seconds for APPRAISE, whose timestamps have a resolution of
one second (flows of the same second are included in each other's windows).
This replay convention does NOT verify feature availability at flow start or exporter arrival order.
(LITNET-2020 was examined with this builder and is not used, see exp/reports/litnet_check.txt.)

usage: ERNP_RAW=/path/to/dataraw python exp/data/prepare_real.py CTU13|SIMARGL|SIMARGL_all|APPRAISE [seeds ...]   (default: EXP_SEED)
"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prepare as P

RAW = os.environ.get("ERNP_RAW", P.ROOT).rstrip("/") + "/"
CHUNK = 2_000_000
EXCLUDED = "EXCLUDED"                        # family name of flows that are counted in the context but left out of the stream

# ---------------------------------------------------------------- context features (prepare_ctx, parametric time unit)
def window_stats(key, t, windows, val=None):
    """for each i: count (and sum of val) of rows with the same key and t in (t_i - W, t_i]; exact, vectorised"""
    assert key.max() < 2 ** 31 and (t.max() - t.min()) < 2 ** 32, "key or time range too large for (key << 32 | t) packing"
    order = np.lexsort((t, key)); k_hi = key[order].astype(np.int64) << 32
    trel = (t[order] - t.min()).astype(np.int64); comp = k_hi | trel
    res = []
    for W in windows:
        left = np.searchsorted(comp, k_hi + np.maximum(trel - W + 1, 0), side="left")
        right = np.searchsorted(comp, comp, side="right")          # ties at the same instant are included
        cnt = np.empty(len(key)); cnt[order] = right - left; out = [cnt]
        if val is not None:
            cs = np.concatenate([[0.0], np.cumsum(val[order])]); s = np.empty(len(key)); s[order] = cs[right] - cs[left]; out.append(s)
        res.append(out)
    return res

def ctx_features(t, src, dst, byts, windows):
    pair = pd.factorize((src.astype(np.int64) << 32) | dst.astype(np.int64))[0].astype(np.int64)
    S = window_stats(src.astype(np.int64), t, windows, byts); D = window_stats(dst.astype(np.int64), t, windows); Pp = window_stats(pair, t, windows)
    cols = []
    for w in range(len(windows)):
        n_src, b_src = S[w]; n_dst = D[w][0]; n_pair = Pp[w][0]
        cols += [np.log1p(n_src), np.log1p(n_dst), np.log1p(n_pair), np.log1p(b_src), np.log1p(n_src / n_pair)]
    return np.column_stack(cols).astype(np.float32)

class Ids:
    """address -> integer id, built chunk by chunk (keeps memory low on tens of millions of flows)"""
    def __init__(self): self.d = {}
    def __call__(self, a):
        codes, uniq = pd.factorize(a); lut = np.fromiter((self.d.setdefault(u, len(self.d)) for u in uniq), np.int64, len(uniq))
        return lut[codes].astype(np.int32)

num = lambda s: pd.to_numeric(s, errors="coerce").fillna(0).values

# ---------------------------------------------------------------- generic two-pass builder
def build(name, chunks, pass1, feats, windows, seeds):
    """chunks(): raw chunks in a fixed order (called twice); pass1(ch) -> t (ticks, int64), src, dst (str), bytes, fam
    (rows with t < 0 are dropped); feats(ch) -> DataFrame with f_*, t (s), src, dst, fam [, lab]; windows in ticks.
    The context features do not depend on the seed, so several seeds are built from one reading of the raw data;
    each seed draws its sample with its own generator, exactly as a single-seed run does."""
    ids = Ids(); T, S, D, B, F, n = [], [], [], [], [], 0
    for ch in chunks():
        t, s, d, b, fam = pass1(ch); T.append(t); S.append(ids(s)); D.append(ids(d)); B.append(b.astype(np.float32)); F.append(pd.Categorical(fam))
        n += len(t); print("  pass1", name, n, flush=True)
    t = np.concatenate(T); src = np.concatenate(S); dst = np.concatenate(D); byts = np.concatenate(B).astype(np.float64)
    fam = np.asarray(pd.concat([pd.Series(f) for f in F], ignore_index=True).astype(str)); del T, S, D, B, F
    ok = t >= 0; print("  flows", len(t), "valid time", int(ok.sum()), "hosts", len(ids.d), flush=True)
    ctx = np.zeros((len(t), 2 * 5), np.float32)
    ctx[ok] = ctx_features(t[ok], src[ok], dst[ok], byts[ok], windows); del src, dst, byts
    use = ok & (fam != EXCLUDED)             # excluded flows stay in the context windows but never enter a stream
    fc = pd.Series(fam[use]).value_counts().to_dict(); r = P.TARGET / use.sum()
    print("  flows used", int(use.sum()), "excluded", int((ok & ~use).sum()), flush=True)
    fam_rate = {f: (r if f == "BENIGN" else max(r, min(1.0, P.MIN_FAM / c))) for f, c in fc.items()}
    rate = pd.Series(fam).map(fam_rate).fillna(0.0).values * ok
    keep = {s: np.where(np.random.default_rng(s).random(len(t)) < rate)[0] for s in seeds}
    union = np.unique(np.concatenate(list(keep.values()))); print("  sampled", {s: len(k) for s, k in keep.items()}, flush=True)
    parts, off = [], 0
    for ch in chunks():
        m = len(ch); sel = union[(union >= off) & (union < off + m)] - off
        if len(sel): df = feats(ch.iloc[sel]); df.index = sel + off; parts.append(df)
        off += m
    assert off == len(t), "the two passes read a different number of rows"
    df = pd.concat(parts); C = ctx[df.index.values]; base = len([c for c in df.columns if c.startswith("f_")])
    assert (df["fam"].values == fam[df.index.values]).all(), "labels differ between the two passes"
    for j in range(C.shape[1]): df[f"f_{base + j}"] = C[:, j]
    for s in seeds:
        d = df.loc[keep[s]].dropna(subset=["t"]).sort_values("t", kind="stable").reset_index(drop=True)
        sfx = "" if s == 0 else f"_s{s}"
        d.to_parquet(P.OUT + f"{name}{sfx}{P.EXTRA}.parquet"); print(name, sfx, d.shape, d["fam"].value_counts().to_dict(), flush=True)

# ---------------------------------------------------------------- CTU-13
CTU = [("capture20110810", "Neris"), ("capture20110811", "Neris"), ("capture20110812", "Rbot"), ("capture20110815", "Rbot"),
       ("capture20110815-2", "Virut"), ("capture20110815-3", "Virut"), ("capture20110816", "Menti"), ("capture20110816-2", "Sogou"),
       ("capture20110816-3", "Murlo"), ("capture20110817", "Neris"), ("capture20110818", "Rbot"), ("capture20110818-2", "Rbot"),
       ("capture20110819", "NSIS.ay")]          # captures 42-54 of the Malware Capture Facility, in time order

def ctu_chunks():
    for cap, bot in CTU:
        for ch in pd.read_csv(RAW + f"CTU-13/{cap}.binetflow", chunksize=CHUNK, low_memory=False):
            ch["_bot"] = bot; yield ch

def ctu_fam(ch):
    lab = ch["Label"].astype(str)
    return np.where(lab.str.contains("From-Botnet", regex=False), ch["_bot"].values, "BENIGN"), lab

def ctu_time(ch): return pd.to_datetime(ch["StartTime"], format="%Y/%m/%d %H:%M:%S.%f", errors="coerce")

def ctu_pass1(ch):
    ts = ctu_time(ch); t = np.where(ts.isna(), -1, ts.astype("int64").values // 1_000_000)
    return t.astype(np.int64), ch["SrcAddr"].astype(str).values, ch["DstAddr"].astype(str).values, num(ch["TotBytes"]), ctu_fam(ch)[0]

def ctu_feats(ch):
    fam, lab = ctu_fam(ch); st = ch["State"].astype(str); s_fl = st.str.split("_").str[0]; d_fl = st.str.split("_").str[1].fillna("")
    proto = ch["Proto"].astype(str).str.strip(); pk = num(ch["TotPkts"]); by = num(ch["TotBytes"]); sb = num(ch["SrcBytes"])
    tcp = (proto == "tcp").values
    X = np.column_stack([np.log1p(num(ch["Dur"])), np.log1p(pk), np.log1p(by), np.log1p(by / np.maximum(pk, 1)), np.log1p(sb),
                         sb / np.maximum(by, 1),
                         *[(s_fl.str.contains(c, regex=False).values & tcp).astype(float) for c in "FSRPAU"],
                         *[(d_fl.str.contains(c, regex=False).values & tcp).astype(float) for c in "FSRPAU"],
                         (st == "CON").values, (st == "INT").values,
                         tcp, (proto == "udp").values, (proto == "icmp").values,
                         ch["Dir"].astype(str).str.contains("<", regex=False).values,          # traffic seen in both directions
                         P.port_bucket(ch["Sport"]), P.port_bucket(ch["Dport"]), num(ch["sTos"]) > 0]).astype(np.float32)
    df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(X.shape[1])])
    df["t"] = ctu_time(ch).astype("int64").values / 1e9
    df["src"] = ch["SrcAddr"].astype(str).values; df["dst"] = ch["DstAddr"].astype(str).values; df["fam"] = fam
    df["lab"] = np.where(fam != "BENIGN", "botnet", np.where(lab.str.contains("Normal", regex=False), "normal", "background"))
    return df

# ---------------------------------------------------------------- RoEduNet-SIMARGL2021
# One parquet file with every flow of the five capture days. Counters and export-time fields (FLOW_ID, TOTAL_FLOWS_EXP,
# FIRST/LAST_SWITCHED, timeouts) are not features: they encode the position of a flow in the capture.
SIM_KEY = ["FLOW_START_MILLISECONDS", "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "IN_BYTES", "OUT_BYTES", "LABEL"]
SIM_NUM = ["FLOW_DURATION_MILLISECONDS", "IN_BYTES", "IN_PKTS", "OUT_BYTES", "OUT_PKTS", "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT", "TCP_WIN_MIN_IN",
           "TCP_WIN_MIN_OUT", "TCP_WIN_MSS_IN", "TCP_WIN_MSS_OUT", "TCP_WIN_SCALE_IN", "TCP_WIN_SCALE_OUT", "RETRANSMITTED_IN_PKTS",
           "RETRANSMITTED_OUT_PKTS", "OOORDER_IN_PKTS", "OOORDER_OUT_PKTS"]
SIM_FAM = {"Normal flow": "BENIGN", "SYN Scan - aggressive": "SYN Scan", "Denial of Service R-U-Dead-Yet": "DoS RUDY",
           "Denial of Service Slowloris": "DoS Slowloris"}
_sim_pass = [0]

def sim_chunks():
    """first call: key columns only (pass 1); second call: feature columns (pass 2) - keeps the 40M-flow table small"""
    import pyarrow.parquet as pq
    _sim_pass[0] += 1
    cols = SIM_KEY if _sim_pass[0] == 1 else sorted(set(SIM_KEY + SIM_NUM + ["TCP_FLAGS", "PROTOCOL_MAP", "L4_SRC_PORT", "L4_DST_PORT", "SRC_TOS", "DST_TOS"]))
    tab = pq.read_table(RAW + "SIMARGL2021/simargl-2021-combined.parquet", columns=cols)
    for i in range(0, tab.num_rows, CHUNK): yield tab.slice(i, CHUNK).to_pandas()

# Suspected label anomaly, NOT independently verified attack ground truth. Raw-field audit (2026-10-23) found
# 24,418,933 published Normal flow records from this host in 04:58:17.290--06:56:05.513 on 2021-05-31,
# covering 65,535 destination ports and 32 destinations. TCP/flags=21: 24,412,910 records with IN/OUT_PKTS=1/1,
# 5,705 with 2/2; TCP/flags=1: 318 records with 1/0. Do not describe every record as a single-packet attack.
# The bounded original-label/time mask below is identical on ALL 40,263,811 raw records to the old host/whole-day
# rule (exp/reports/access/simargl_raw_audit.json). No old stream or predictions need replacement for this change.
# Both the filtered cohort and SIMARGL_all (published labels) MUST be reported; excluded records still affect context.
SIM_SCAN_HOST = "10.9.3.238"
SIM_SCAN_FROM_MS, SIM_SCAN_TO_MS = 1622437097000, 1622444166000  # [04:58:17, 06:56:06) in dataset epoch time
SIM_KEEP_SCAN = [False]

def sim_fam(ch):
    lab = ch["LABEL"].astype(str).str.strip(); fam = lab.map(SIM_FAM).fillna(lab).values
    if SIM_KEEP_SCAN[0]: return fam
    t = ch['FLOW_START_MILLISECONDS'].to_numpy()
    scan = ((ch['IPV4_SRC_ADDR'].astype(str).to_numpy() == SIM_SCAN_HOST)
            & (lab.to_numpy() == 'Normal flow') & (t >= SIM_SCAN_FROM_MS) & (t < SIM_SCAN_TO_MS))
    return np.where(scan, EXCLUDED, fam)

def sim_pass1(ch):
    t = ch["FLOW_START_MILLISECONDS"].values.astype(np.int64) // 10          # 10 ms ticks: the capture spans 67 days
    return np.where(t > 0, t, -1), ch["IPV4_SRC_ADDR"].astype(str).values, ch["IPV4_DST_ADDR"].astype(str).values, \
           num(ch["IN_BYTES"]) + num(ch["OUT_BYTES"]), sim_fam(ch)

def sim_feats(ch):
    v = {c: np.maximum(num(ch[c]), 0) for c in SIM_NUM}; fl = num(ch["TCP_FLAGS"]).astype(np.int64); proto = ch["PROTOCOL_MAP"].astype(str).str.lower()
    X = np.column_stack([*[np.log1p(v[c]) for c in SIM_NUM],
                         np.log1p(v["IN_BYTES"] / np.maximum(v["IN_PKTS"], 1)), np.log1p(v["OUT_BYTES"] / np.maximum(v["OUT_PKTS"], 1)),
                         *[((fl >> b) & 1).astype(float) for b in range(6)],                    # FIN SYN RST PSH ACK URG
                         (proto == "tcp").values, (proto == "udp").values, (proto == "icmp").values,
                         P.port_bucket(ch["L4_SRC_PORT"]), P.port_bucket(ch["L4_DST_PORT"]),
                         num(ch["SRC_TOS"]) > 0, num(ch["DST_TOS"]) > 0]).astype(np.float32)
    df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(X.shape[1])])
    df["t"] = ch["FLOW_START_MILLISECONDS"].values.astype(np.float64) / 1e3
    df["src"] = ch["IPV4_SRC_ADDR"].astype(str).values; df["dst"] = ch["IPV4_DST_ADDR"].astype(str).values; df["fam"] = sim_fam(ch)
    return df

# ---------------------------------------------------------------- APPRAISE H2020
# FLOW_ID and TOTAL_FLOWS_EXP are counters of the export and are not features; L7_PROTO is not used either.
APP_FILE = os.environ.get("ERNP_APPRAISE", "APPRAISE/APPRAISE_NETFLOW.csv")      # path under ERNP_RAW

def app_chunks():
    yield from pd.read_csv(RAW + APP_FILE, chunksize=CHUNK, low_memory=False)

def app_fam(ch):
    c = ch["ANOMALY_CATEGORY"].astype(str).str.strip()
    return np.where((num(ch["ANOMALY"]) == 1) & (c != "-"), c, "BENIGN")

def app_pass1(ch):
    t = num(ch["FIRST_SWITCHED"]).astype(np.int64)                                # seconds
    return np.where(t > 0, t, -1), ch["IPV4_SRC_ADDR"].astype(str).values, ch["IPV4_DST_ADDR"].astype(str).values, \
           num(ch["IN_BYTES"]) + num(ch["OUT_BYTES"]), app_fam(ch)

def app_feats(ch):
    ip, ib, op, ob = (np.maximum(num(ch[c]), 0) for c in ("IN_PKTS", "IN_BYTES", "OUT_PKTS", "OUT_BYTES"))
    fl = num(ch["TCP_FLAGS"]).astype(np.int64); proto = ch["PROTOCOL_MAP"].astype(str).str.lower()
    X = np.column_stack([np.log1p(np.maximum(num(ch["LAST_SWITCHED"]) - num(ch["FIRST_SWITCHED"]), 0)), np.log1p(ip), np.log1p(ib),
                         np.log1p(op), np.log1p(ob), np.log1p(ib / np.maximum(ip, 1)), np.log1p(ob / np.maximum(op, 1)),
                         *[((fl >> b) & 1).astype(float) for b in range(6)],                    # FIN SYN RST PSH ACK URG
                         (proto == "tcp").values, (proto == "udp").values, (proto == "icmp").values,
                         P.port_bucket(ch["L4_SRC_PORT"]), P.port_bucket(ch["L4_DST_PORT"])]).astype(np.float32)
    df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(X.shape[1])])
    df["t"] = num(ch["FIRST_SWITCHED"]).astype(np.float64)
    df["src"] = ch["IPV4_SRC_ADDR"].astype(str).values; df["dst"] = ch["IPV4_DST_ADDR"].astype(str).values; df["fam"] = app_fam(ch)
    return df

if __name__ == "__main__":
    name = sys.argv[1]; P.TARGET = int(os.environ.get("ERNP_TARGET", 400_000))      # 1-2% of each trace, as for CSE-CIC-IDS2018
    seeds = [int(x) for x in sys.argv[2:]] or [P.SEED]                              # e.g. `prepare_real.py CTU13 0 1 2 3 4`
    if name == "CTU13": build("CTU13", ctu_chunks, ctu_pass1, ctu_feats, (10_000, 120_000), seeds)        # ms
    elif name in ("SIMARGL", "SIMARGL_all"):                                                              # 10 ms ticks
        SIM_KEEP_SCAN[0] = name == "SIMARGL_all"; build(name, sim_chunks, sim_pass1, sim_feats, (1_000, 12_000), seeds)
    elif name == "APPRAISE": build("APPRAISE", app_chunks, app_pass1, app_feats, (10, 120), seeds)        # s (timestamp resolution)
    else: raise SystemExit("unknown dataset " + name)
