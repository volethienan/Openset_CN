"""Summarises retraction_time_analysis.py (reports/retraction_*.csv): mean over seeds of per-seed rates."""
import os, numpy as np, pandas as pd
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
g = pd.read_csv(os.path.join(R, "retraction_gate.csv")); reg = pd.read_csv(os.path.join(R, "retraction_regions.csv"))
dl = pd.read_csv(os.path.join(R, "retraction_daily.csv"))
print("gate: %d/%d runs reconstructed exactly" % (g.ok.sum(), len(g)))
V = {"ernp": "ERNP", "abl_retr_none": "no-retr"}

def rate(df, num, den, keys):
    s = df.groupby(keys + ["seed"])[[num, den]].sum(); return (s[num] / s[den].replace(0, np.nan)).groupby(keys).mean()

# 1) stream-level totals and analyst load (all days)
t = dl.groupby(["dataset", "variant", "seed"])[["fp", "n_benign", "tp_zd", "n_zd", "decisions", "benign_decisions"]].sum()
t["FPR%"] = 100 * t.fp / t.n_benign; t["ZDrec"] = t.tp_zd / t.n_zd.replace(0, np.nan)
t["benign_dec%"] = 100 * t.benign_decisions / t.decisions.replace(0, np.nan); t["benign_alerts/day"] = t.fp
nd = dl.groupby(["dataset", "variant", "seed"]).day_no.nunique(); t["benign_alerts/day"] = t.fp / nd
print("\n== 1. Stream totals (mean over seeds)")
print(t.groupby(["dataset", "variant"])[["FPR%", "ZDrec", "decisions", "benign_dec%", "benign_alerts/day"]].mean().round(3).unstack("variant").to_string())

# 2) FPR and zero-day recall by distance to confirmed benign evidence (days >= 1)
for region in ("ernp", "own"):
    r = reg[reg.region == region]
    f = (100 * rate(r, "fp", "n_benign", ["dataset", "variant", "bin"])).unstack("variant").round(2)
    n = r.groupby(["dataset", "variant", "bin", "seed"]).n_benign.sum().groupby(["dataset", "variant", "bin"]).mean().unstack("variant").round(0)
    z = rate(r, "tp_zd", "n_zd", ["dataset", "variant", "bin"]).unstack("variant").round(3)
    nz = r.groupby(["dataset", "variant", "bin", "seed"]).n_zd.sum().groupby(["dataset", "variant", "bin"]).mean().unstack("variant").round(0)
    print(f"\n== 2. Region = {region} evidence | benign FPR % (n benign sampled) | zero-day recall (n zd sampled); bins in eps")
    out = pd.concat({"FPR%": f, "n_ben": n, "ZDrec": z, "n_zd": nz}, axis=1); print(out.to_string())

# 3) per-day FPR and benign decisions, both variants (mean over seeds)
d = dl.copy(); d["FPR%"] = 100 * d.fp / d.n_benign.replace(0, np.nan)
d["ZDrec"] = d.tp_zd / d.n_zd.replace(0, np.nan)
print("\n== 3. Per-day FPR % / benign decisions (mean over seeds)")
for ds_ in d.dataset.unique():
    x = d[d.dataset == ds_].groupby(["day_no", "variant"])[["FPR%", "ZDrec", "benign_decisions", "fp"]].mean().unstack("variant").round(2)
    print("\n--", ds_); print(x.to_string())
