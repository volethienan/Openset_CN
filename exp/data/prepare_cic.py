"""Streams for CICIDS2017_improved / CSECICIDS2018_improved (Distrinet), same schema as the other streams.
- 'X - Attempted' labels -> BENIGN (one of the two options recommended by the Distrinet authors; never a separate class)
- context features (prepare_ctx.ctx_features) computed per day on the FULL data before sampling
- sampling: uniform rate r, rare families kept up to >= MIN_FAM flows (context unaffected)
"""
import zipfile, sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prepare as P, prepare_ctx as C

DROP = {"id", "Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol", "Timestamp", "Label", "Attempted Category"}
CFG = {"CIC17": ("CICIDS2017_improved/CICIDS2017_improved.zip", 300_000),
       "CIC18": ("CSECICIDS2018_improved/CSECICIDS2018_improved.zip", 400_000),
       "CIC17_full": ("CICIDS2017_improved/CICIDS2017_improved.zip", None)}   # scale check: no sampling (2.1M flows)

def norm_label(s):
    s = s.astype(str).str.strip()
    return np.where(s.str.contains("Attempted", regex=False) | (s.str.upper() == "BENIGN"), "BENIGN", s)

def featurize(ch):
    num = [c for c in ch.columns if c not in DROP]
    V = ch[num].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float64)
    V = np.sign(V) * np.log1p(np.abs(V))
    proto = pd.to_numeric(ch["Protocol"], errors="coerce").fillna(0).values
    return np.column_stack([V, proto == 6, proto == 17, proto == 1, P.port_bucket(ch["Src Port"]), P.port_bucket(ch["Dst Port"])]).astype(np.float32)

def build(name):
    path, target = CFG[name]; zf = zipfile.ZipFile(P.ROOT + path)
    days = sorted(n for n in zf.namelist() if n.endswith(".csv"))
    # ---- pass 1: labels only, to get family counts for the sampling rates
    fam_all = []
    for d in days:
        for ch in pd.read_csv(zf.open(d), usecols=["Label"], chunksize=3_000_000): fam_all.append(norm_label(ch["Label"]))
    fam_all = np.concatenate(fam_all); fc = pd.Series(fam_all).value_counts().to_dict(); N = len(fam_all); r = target / N if target else 1.0
    print(name, "rows", N, "families", fc, flush=True)
    rng = np.random.default_rng(P.SEED); parts = []
    for d in days:
        key = pd.read_csv(zf.open(d), usecols=["Timestamp", "Src IP", "Dst IP", "Total Length of Fwd Packet", "Total Length of Bwd Packet", "Label"])
        t = pd.to_datetime(key["Timestamp"], errors="coerce", format="mixed").astype("int64").values // 1_000_000
        t = np.where(t > 0, t, np.median(t[t > 0])).astype(np.int64)
        ids, _ = pd.factorize(np.concatenate([key["Src IP"].astype(str).values, key["Dst IP"].astype(str).values]))
        n = len(key); src, dst = ids[:n].astype(np.int64), ids[n:].astype(np.int64)
        byts = pd.to_numeric(key["Total Length of Fwd Packet"], errors="coerce").fillna(0).values + pd.to_numeric(key["Total Length of Bwd Packet"], errors="coerce").fillna(0).values
        ctx = C.ctx_features(t, src, dst, byts)
        fam = norm_label(key["Label"]); rate = np.where(fam == "BENIGN", r, np.array([max(r, min(1.0, P.MIN_FAM / fc[x])) for x in fam]))
        keep = np.where(rng.random(n) < rate)[0]; del key
        off = 0
        for ch in pd.read_csv(zf.open(d), chunksize=1_000_000, low_memory=False):
            sel = keep[(keep >= off) & (keep < off + len(ch))] - off
            if len(sel):
                sub = ch.iloc[sel]; X = featurize(sub); Cx = ctx[sel + off]
                df = pd.DataFrame(np.column_stack([X, Cx]), columns=[f"f_{i}" for i in range(X.shape[1] + Cx.shape[1])])
                df["t"] = pd.to_datetime(sub["Timestamp"], errors="coerce", format="mixed").astype("int64").values / 1e9
                df["src"] = sub["Src IP"].astype(str).values; df["dst"] = sub["Dst IP"].astype(str).values
                df["fam"] = norm_label(sub["Label"]); parts.append(df)
            off += len(ch)
        print("  day", d, "kept", len(keep), flush=True)
    df = pd.concat(parts).dropna(subset=["t"]).sort_values("t", kind="stable").reset_index(drop=True)
    sfx = "" if P.SEED == 0 else f"_s{P.SEED}"
    df.to_parquet(P.OUT + f"{name}{sfx}.parquet"); print(name, sfx, df.shape, df["fam"].value_counts().to_dict(), flush=True)

if __name__ == "__main__":
    build(sys.argv[1])
