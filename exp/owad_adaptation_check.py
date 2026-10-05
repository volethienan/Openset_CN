"""How much does OWAD's adaptation change its detector? (no model is run)
1) adaptations / labels used per RQ1 run (from the run records);
2) score change: the RQ1 stream scores of OWAD (adapting daily) against those of its initial model, i.e. the same
   wrapper with adaptation disabled (owad_P0 runs, if present), else against the 24 h vs 1 h cadence runs.
"""
import os, sys, json, glob, numpy as np, pandas as pd
EXP = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(EXP, "results")
rows = []
for f in sorted(glob.glob(os.path.join(RES, "*__owad__s*.json"))):
    ds = os.path.basename(f).split("__")[0]
    if "_inj" in ds or "_x10" in ds or "_full" in ds: continue
    i = json.load(open(f)); base = ds.split("_s")[0] if not ds.startswith(("CIDDS_ctx", "ToN_ctx")) else "_".join(ds.split("_")[:2])
    r = dict(dataset=base, stream=ds, adaptations=i.get("adaptations"), labels=i.get("labels_used"))
    p1 = f.replace("__owad__", "__owad_P1__").replace(".json", ".npz")
    if os.path.exists(p1):     # same seed, hourly instead of daily adaptation: relative score change
        a, b = np.load(f.replace(".json", ".npz"))["score"], np.load(p1)["score"]
        r["rel_score_change_24h_vs_1h"] = float(np.median(np.abs(a - b) / (np.abs(a) + 1e-12)))
        r["flag_agreement_24h_vs_1h"] = float((np.load(f.replace(".json", ".npz"))["flag"] == np.load(p1)["flag"]).mean())
    rows.append(r)
d = pd.DataFrame(rows); d.to_csv(os.path.join(EXP, "reports", "owad_adaptation_check.csv"), index=False)
print(d.groupby("dataset").mean(numeric_only=True).round(4).to_string())
