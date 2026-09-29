"""Build time-ordered, subsampled event streams with a common schema:
   t (unix s), src, dst, fam (attack family or 'BENIGN'), f_* numeric features.
Sampling: uniform rate r for everything; rare attack families are kept up to >= MIN_FAM flows
(documented: this raises attack prevalence slightly for rare families only)."""
import pandas as pd, numpy as np, glob, sys, os
_PROJ = os.environ.get("ERNP_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))).replace("\\", "/").rstrip("/") + "/"
ROOT = _PROJ + "dataraw/"
OUT = _PROJ + "mock/streams/"; os.makedirs(OUT, exist_ok=True)
TARGET, MIN_FAM = 300_000, 600
SEED = int(os.environ.get("EXP_SEED", 0))   # seed 0 reproduces the mock-test streams
rng = np.random.default_rng(SEED)

def port_bucket(p):
    p = pd.to_numeric(p, errors="coerce").fillna(-1).values
    return np.stack([(p >= 0) & (p <= 1023), (p >= 1024) & (p <= 49151), p >= 49152], 1).astype(np.float32)

def sample_mask(fam, n_total, fam_counts):
    r = TARGET / n_total
    rate = np.array([max(r, min(1.0, MIN_FAM / fam_counts[f])) if f != "BENIGN" else r for f in fam])
    return rng.random(len(fam)) < rate

# ---------------- CIDDS-001 internal ----------------
def parse_bytes(s):
    s = s.astype(str).str.strip()
    mult = np.where(s.str.endswith("M"), 1e6, np.where(s.str.endswith("K"), 1e3, np.where(s.str.endswith("G"), 1e9, 1.0)))
    return pd.to_numeric(s.str.replace(r"[MKG]", "", regex=True).str.strip(), errors="coerce").fillna(0).values * mult

def cidds_chunk(ch):
    fam = ch["attackType"].astype(str).str.strip().replace({"---": "BENIGN"}).values
    flags = ch["Flags"].astype(str).str.strip()
    proto = ch["Proto"].astype(str).str.strip()
    pk = pd.to_numeric(ch["Packets"], errors="coerce").fillna(0).values; by = parse_bytes(ch["Bytes"])
    X = np.column_stack([np.log1p(pd.to_numeric(ch["Duration"], errors="coerce").fillna(0).values), np.log1p(pk), np.log1p(by),
                         np.log1p(by / np.maximum(pk, 1)),
                         *[flags.str.contains(c, regex=False).values.astype(float) for c in "UAPRSF"],
                         (proto == "TCP").values, (proto == "UDP").values, (proto == "ICMP").values,
                         port_bucket(ch["Src Pt"]), port_bucket(ch["Dst Pt"]),
                         (pd.to_numeric(ch["Tos"], errors="coerce").fillna(0).values > 0)]).astype(np.float32)
    df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(X.shape[1])])
    df["t"] = pd.to_datetime(ch["Date first seen"], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce").astype("int64").values / 1e9
    df["src"] = ch["Src IP Addr"].astype(str).str.strip().values; df["dst"] = ch["Dst IP Addr"].astype(str).str.strip().values
    df["fam"] = fam
    return df

def cidds():
    files = sorted(glob.glob(ROOT + "CIDDS-001/traffic/OpenStack/*.csv"))
    fam_counts = {"dos": 2959027, "portScan": 265918, "pingScan": 6090, "bruteForce": 4992}
    n_total = 31_287_933; parts = []
    for f in files:
        for ch in pd.read_csv(f, chunksize=2_000_000, low_memory=False):
            fam = ch["attackType"].astype(str).str.strip().replace({"---": "BENIGN"}).values
            ch = ch[sample_mask(fam, n_total, fam_counts)]
            if len(ch): parts.append(cidds_chunk(ch))
        print("  cidds", f, flush=True)
    return pd.concat(parts)

# ---------------- NF-ToN-IoT-v3 ----------------
NF_DROP = {"FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS", "IPV4_SRC_ADDR", "L4_SRC_PORT", "IPV4_DST_ADDR", "L4_DST_PORT",
           "Label", "Attack", "DNS_QUERY_ID", "PROTOCOL", "L7_PROTO", "ICMP_TYPE", "ICMP_IPV4_TYPE", "TCP_FLAGS", "CLIENT_TCP_FLAGS",
           "SERVER_TCP_FLAGS", "DNS_QUERY_TYPE", "FTP_COMMAND_RET_CODE"}
def nf(path, n_total, fam_counts):
    parts = []
    for ch in pd.read_csv(ROOT + path, chunksize=2_000_000, low_memory=False):
        fam = ch["Attack"].astype(str).str.strip().replace({"Benign": "BENIGN"}).values
        ch = ch[sample_mask(fam, n_total, fam_counts)]
        parts.append(nf_chunk(ch))
    return pd.concat(parts)

def nf_chunk(ch):
    fam = ch["Attack"].astype(str).str.strip().replace({"Benign": "BENIGN"}).values
    num = [c for c in ch.columns if c not in NF_DROP]
    X = np.log1p(np.clip(ch[num].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float64), 0, None))
    proto = pd.to_numeric(ch["PROTOCOL"], errors="coerce").fillna(0).values
    fl = pd.to_numeric(ch["TCP_FLAGS"], errors="coerce").fillna(0).astype(int).values
    X = np.column_stack([X, proto == 6, proto == 17, proto == 1, *[(fl >> b) & 1 for b in range(6)],
                         port_bucket(ch["L4_SRC_PORT"]), port_bucket(ch["L4_DST_PORT"])]).astype(np.float32)
    df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(X.shape[1])])
    df["t"] = ch["FLOW_START_MILLISECONDS"].astype("float64").values / 1000
    df["src"] = ch["IPV4_SRC_ADDR"].astype(str).values; df["dst"] = ch["IPV4_DST_ADDR"].astype(str).values; df["fam"] = fam
    return df

# ---------------- UWF-ZeekData24 ----------------
def uwf24():
    files = sorted(f for f in glob.glob(ROOT + "UWF-ZeekData24/parquet/**/*.parquet", recursive=True) if "_metrics_" not in f)
    d = pd.concat(pd.read_parquet(f) for f in files)
    d["fam"] = d["label_tactic"].astype(str).str.strip().replace({"none": "BENIGN"})
    fc = d["fam"].value_counts().to_dict()
    d = d[sample_mask(d["fam"].values, len(d), fc)]
    num = ["duration", "orig_bytes", "resp_bytes", "orig_pkts", "resp_pkts", "orig_ip_bytes", "resp_ip_bytes", "missed_bytes"]
    X = [np.log1p(np.clip(d[num].apply(pd.to_numeric, errors="coerce").fillna(0).values, 0, None))]
    for col, cats in [("proto", ["tcp", "udp", "icmp"]), ("conn_state", ["S0", "SF", "REJ", "RSTO", "RSTR", "OTH", "SH", "S1", "RSTOS0"]),
                      ("service", ["dns", "http", "ssl", "ssh", "dhcp", "ntp", "smb", "krb", "dce_rpc"])]:
        v = d[col].astype(str).values
        X.append(np.stack([np.char.find(v.astype(str), c) >= 0 for c in cats], 1))
    X.append(d["history"].astype(str).str.len().values[:, None])
    X += [port_bucket(d["src_port_zeek"]), port_bucket(d["dest_port_zeek"])]
    X = np.column_stack(X).astype(np.float32)
    df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(X.shape[1])])
    df["t"] = d["ts"].astype("float64").values; df["src"] = d["src_ip_zeek"].astype(str).values
    df["dst"] = d["dest_ip_zeek"].astype(str).values; df["fam"] = d["fam"].values
    return df

if __name__ == "__main__":
    for name in sys.argv[1:]:
        if name == "CIDDS": df = cidds()
        elif name == "ToN":
            fc = {"ddos": 4141256, "xss": 2834435, "password": 1594777, "scanning": 1358977, "injection": 381777, "dos": 203456,
                  "Backdoor": 203384, "mitm": 6013, "ransomware": 3971}
            df = nf("NF-ToN-IoT-v3/NF-ToN-IoT-v3.csv", 27_520_260, fc)
        elif name == "UWF24": df = uwf24()
        df = df.dropna(subset=["t"]).sort_values("t", kind="stable").reset_index(drop=True)
        df.to_parquet(OUT + f"{name}.parquet")
        print(name, df.shape, df["fam"].value_counts().to_dict(), flush=True)
