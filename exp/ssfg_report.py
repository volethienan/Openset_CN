"""Review 30/10 (M5): SSF with ERNP's group feedback channel (ssf_causal_G, ssf_causal_B25_G; run_jobs group `ssfg`)
against ERNP and SSF with flow labels on the RQ3 datasets.

Rows for ERNP and flow-label SSF come from the K5A RQ3 report (exp_k5a/reports/e2_runs.csv); the _G runs are scored
here with evaluate.run_metrics, and their decisions and labelled flows per day are read from the run's json
(group_decisions, group_flows_used: flows that entered SSF's memory; group_labeled_flows: before the memory cap).
Output: exp/reports/ssfg_runs.csv and exp/reports/ssfg.txt
usage (from Openset_CN/): python exp/ssfg_report.py
"""
import os, sys, json
import numpy as np, pandas as pd
EXP = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(EXP); sys.path.insert(0, EXP)
import evaluate, common

DS = ("ToN_ctx", "CIC18"); SEEDS = range(5); G = ("ssf_causal_G", "ssf_causal_B25_G")
rows = []
for base in DS:
    for s in SEEDS:
        ds = evaluate.seeded_name(base, s)
        df = evaluate.metric_labels(common.STREAMS + f"{ds}.parquet")
        os.environ["EXP_SEED"] = str(s)
        t = df.t.to_numpy(); n_days = len(np.unique((t[t >= common.anchor_end(ds, t.min())]) // 86400))
        for m in G:
            r = evaluate.run_metrics(base, m, s)
            if r is None: continue
            info = json.load(open(os.path.join(common.RESULTS, f"{ds}__{m}__s{s}.json")))
            r.update(decisions_per_day=info["group_decisions"] / n_days, flows_per_day=info["group_flows_used"] / n_days,
                     propagated_flows_per_day=info["group_labeled_flows"] / n_days, periods_without_decision=info["periods_without_decision"])
            rows.append(r)
g = pd.DataFrame(rows)
if g.empty: sys.exit("no ssf_causal_G runs found")
e2 = pd.read_csv(os.path.join(ROOT, "exp_k5a", "reports", "e2_runs.csv"))
ref = e2[e2.method.isin(["ernp", "ernp_B25", "ernp_B25_cap200", "abl_evid_rep_B25", "ssf_causal", "ssf_causal_B25"])]
allr = pd.concat([ref, g], ignore_index=True)
os.makedirs(os.path.join(EXP, "reports"), exist_ok=True)
allr.to_csv(os.path.join(EXP, "reports", "ssfg_runs.csv"), index=False)
cols = ["F1", "MCC", "zero_day_recall", "FPR", "decisions_per_day", "flows_per_day"]
summ = allr.groupby(["dataset", "method"])[cols].mean().round(3)
summ["seeds"] = allr.groupby(["dataset", "method"]).seed.nunique()
txt = summ.to_string()
open(os.path.join(EXP, "reports", "ssfg.txt"), "w").write(txt + "\n")
print(txt)
