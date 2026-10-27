"""Shared protocol for all experiments (ERNP and baselines): stream loading, anchor split, calibration.

Anchor period (known attacks) / stream (zero-day evaluation):
  CIDDS*  : first 2 days          ToN*   : first day
  CIC17*  : before 2017-07-06     CIC18* : before 2018-02-17
  CTU13*  : before 2011-08-12     SIMARGL*: before 2021-04-12     APPRAISE*: before 2023-01-25   (real-network datasets)
Anchor BENIGN flows -> 80% reference / 20% holdout (seeded by EXP_SEED). Features are standardised with reference stats.
ERNP and calibrated baseline thresholds use anchor holdout; native baseline rules can differ.
The existing streams replay complete flow records ordered by flow START time. Completed bytes/duration and
same-timestamp context are retrospective features. Causal model feedback does not imply packet-time causality.
"""
import os, numpy as np, pandas as pd

ROOT = os.environ.get("ERNP_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/") + "/")   # project root (parent of exp/); override with ERNP_ROOT
STREAMS = ROOT + "mock/streams/"          # stream parquet files (built by exp/data/*)
RESULTS = os.environ.get("EXP_RESULTS", ROOT + "exp/results/")   # EXP_RESULTS: separate folder (e.g. clean benchmark runs)
FPR_TARGET = 0.01
N_CTX = 10                                # last 10 feature columns are the temporal-context features

def seed():
    return int(os.environ.get("EXP_SEED", 0))

# Schedule of the protocol-valid poisoning streams (data/inject_valid.py: START, RAMP, ATTACK, RATE_PER_H); the
# test tests/test_naive_ramp.py checks that these constants equal the generator's.
INJ_START_D, INJ_RAMP_D, INJ_ATTACK_D, INJ_RATE_H = 4.0, 7.0, 2.0, 30

def naive_ramp_mask(ds, df, a_max):
    """Analyst model for RQ2 (`naive analyst`): True for the injected ramp flows whose ramp fraction a is below a_max.
    Ground truth marks every ramp flow INJ_ramp (an attack), even at a ~ 0 where the record is a copy of the host's
    benign behaviour; a naive analyst answers BENIGN for such flows. a is recovered from the flow start time exactly as
    data/inject_valid.py generates it, a = (t - t0 - START) / RAMP with t0 = the stream's first time stamp; for
    a < 0.5 the generator also keeps the ports, flags and ToS of the benign template. Used ONLY for the analyst's
    answer; evaluation keeps the ground truth. a_max <= 0 -> all False."""
    fam = df["fam"].values; m = np.zeros(len(df), bool)
    if a_max <= 0: return m
    assert "_injvalid" in ds, f"naive analyst is defined for the protocol-valid poisoning streams only, not {ds}"
    t = df["t"].values; t0 = t.min(); inj = np.char.startswith(fam.astype(str), "INJ_")
    ramp, pay = fam == "INJ_ramp", fam == "INJ_attack"
    k = (t[inj] - t0 - INJ_START_D * 86400) / (3600 / INJ_RATE_H)       # injection slot index: integer by construction
    assert np.abs(k - np.round(k)).max() < 1e-6, "injected flows are off the generator's time grid (t0 differs?)"
    assert ramp.sum() == INJ_RAMP_D * 24 * INJ_RATE_H and pay.sum() == INJ_ATTACK_D * 24 * INJ_RATE_H, (ramp.sum(), pay.sum())
    a = (t - t0 - INJ_START_D * 86400) / (INJ_RAMP_D * 86400)
    assert a[ramp].min() > -1e-9 and a[ramp].max() < 1 and a[pay].min() > 1 - 1e-9, "ramp/payload labels do not match the schedule"
    return ramp & (a < a_max - 1e-7)                 # flows exactly at a = a_max are not naive (the generator uses a < 0.5)

def anchor_end(ds, t0):
    if ds.startswith("CIC17"): return pd.Timestamp("2017-07-06").value / 1e9
    if ds.startswith("CIC18"): return pd.Timestamp("2018-02-17").value / 1e9
    if ds.startswith("CTU13"): return pd.Timestamp("2011-08-12").value / 1e9      # captures of 10-11 August (Neris)
    if ds.startswith("SIMARGL"): return pd.Timestamp("2021-04-12").value / 1e9    # capture days 2021-03-25 and 2021-04-11
    if ds.startswith("APPRAISE"): return pd.Timestamp("2023-01-25").value / 1e9   # first week, 18-24 January 2023
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
    info.setdefault('protocol_scope', dict(record_order='flow_start_timestamp',
                    features='complete_flow_records_retrospective_context_with_timestamp_ties',
                    feedback='predict_then_update_if_cfg_protocol_causal',
                    packet_causal_verified=False,
                    roc_at_1fpr='test_label_diagnostic_not_a_deployment_threshold'))
    os.makedirs(RESULTS, exist_ok=True); tag = f"{ds}__{method}__s{seed()}"
    try:                                  # peak resident memory of the whole run (cost reporting, RQ5)
        import psutil; mi = psutil.Process().memory_info()
        info["peak_mem_mb"] = round(getattr(mi, "peak_wset", mi.rss) / 2 ** 20)
    except Exception: pass
    np.savez(RESULTS + tag + ".npz", score=score, flag=flag)
    json.dump(info, open(RESULTS + tag + ".json", "w"), default=str)
    print("DONE", tag, {k: v for k, v in info.items() if k != "timing"}, flush=True)
