"""Shared protocol for all experiments (ERNP and baselines): stream loading, anchor split, calibration.

Anchor period (known attacks) / stream (zero-day evaluation):
  CIDDS*  : first 2 days          ToN*   : first day
  CIC17*  : before 2017-07-06     CIC18* : before 2018-02-17
  CTU13*  : before 2011-08-12     SIMARGL*: before 2021-04-12   (real-network datasets)
Anchor BENIGN flows -> 80% reference / 20% holdout (seeded by EXP_SEED). Features are standardised with reference stats.
Every score-based threshold is set at FPR_TARGET on the holdout.
"""
import os, numpy as np, pandas as pd

ROOT = os.environ.get("ERNP_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/") + "/")   # project root (parent of exp/); override with ERNP_ROOT
STREAMS = ROOT + "mock/streams/"          # stream parquet files (built by exp/data/*)
RESULTS = os.environ.get("EXP_RESULTS", ROOT + "exp/results/")   # EXP_RESULTS: separate folder (e.g. clean benchmark runs)
FPR_TARGET = 0.01
N_CTX = 10                                # last 10 feature columns are the temporal-context features

def seed():
    return int(os.environ.get("EXP_SEED", 0))

def anchor_end(ds, t0):
    if ds.startswith("CIC17"): return pd.Timestamp("2017-07-06").value / 1e9
    if ds.startswith("CIC18"): return pd.Timestamp("2018-02-17").value / 1e9
    if ds.startswith("CTU13"): return pd.Timestamp("2011-08-12").value / 1e9      # captures of 10-11 August (Neris)
    if ds.startswith("SIMARGL"): return pd.Timestamp("2021-04-12").value / 1e9    # capture days 2021-03-25 and 2021-04-11
    return t0 + (2.0 if ds.startswith("CIDDS") else 1.0) * 86400

def load(ds):
    """returns df, X (standardised), src, dst (domain ids), tr (reference idx), ho (holdout idx), s_idx (stream idx), anc (bool)"""
    df = pd.read_parquet(STREAMS + f"{ds}.parquet")
    fcols = [c for c in df.columns if c.startswith("f_")]
    anc = (df["t"] < anchor_end(ds, df["t"].min())).values; ben = (df["fam"] == "BENIGN").values
    rng = np.random.default_rng(seed())
    a_idx = np.where(anc & ben)[0]; rng.shuffle(a_idx)
    n_tr = int(0.8 * len(a_idx)); tr, ho = np.sort(a_idx[:n_tr]), np.sort(a_idx[n_tr:])
    X = df[fcols].values.astype(np.float64)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    X = np.clip((X - mu) / sd, -10, 10).astype(np.float32)
    doms, _ = pd.factorize(pd.concat([df["src"], df["dst"]]))
    src, dst = doms[: len(df)], doms[len(df):]
    s_idx = np.where(~anc)[0]
    return df, X, src, dst, tr, ho, s_idx, anc

def calib_upper(scores_holdout):
    """flag if score > threshold; threshold at (1 - FPR_TARGET) quantile of holdout benign scores"""
    return float(np.quantile(scores_holdout, 1 - FPR_TARGET))

def stream_days(df, s_idx):
    return pd.to_datetime(df["t"].values[s_idx], unit="s").strftime("%Y-%m-%d").values

def stream_periods(df, s_idx, hours=24):
    """update batches of `hours` (cadence study); hours = 24 -> calendar days, identical to stream_days"""
    if hours == 24: return stream_days(df, s_idx)
    return (df["t"].values[s_idx] // (3600 * hours)).astype(np.int64)

def period_budget(p, hours, B):
    """labels available in update period p (absolute period number) so that every calendar day gets exactly B labels"""
    if hours == 24: return B
    p = int(p); return (B * hours * (p + 1)) // 24 - (B * hours * p) // 24

def save(ds, method, score, flag, info):
    import json
    os.makedirs(RESULTS, exist_ok=True); tag = f"{ds}__{method}__s{seed()}"
    try:                                  # peak resident memory of the whole run (cost reporting, RQ5)
        import psutil; mi = psutil.Process().memory_info()
        info["peak_mem_mb"] = round(getattr(mi, "peak_wset", mi.rss) / 2 ** 20)
    except Exception: pass
    np.savez(RESULTS + tag + ".npz", score=score, flag=flag)
    json.dump(info, open(RESULTS + tag + ".json", "w"), default=str)
    print("DONE", tag, {k: v for k, v in info.items() if k != "timing"}, flush=True)
