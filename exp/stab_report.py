"""Stability study (run_jobs group `stab`): ERNP with the anchored attack evidence (keepA), K prior draws per fit (K5),
the posterior decision rule (post) and their combinations, against the saved ERNP run and the references.

Per run: the metrics of evaluate.run_metrics. Tables: mean over the available seeds per dataset and method (the run-count
table shows how many; means over different seed sets are not comparable), recall per family on SIMARGL and CTU-13, and
per variant the number of datasets on which it beats SSF and ERNP (F1 and zero-day recall), paired by dataset and seed.
Output: reports/stab_runs.csv, reports/stab.txt
usage: python stab_report.py [seeds ...]     (default: 0 1)
"""
import os, sys, io, contextlib, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import run_metrics
from common import RESULTS
from scipy.stats import wilcoxon

DS = ["CIDDS_ctx", "ToN_ctx", "CIC17", "CIC18", "CTU13", "SIMARGL", "APPRAISE"]
VAR = ["ernp_keepA", "ernp_post", "ernp_keepA_post", "ernp_keepAP", "ernp_K5", "ernp_keepA_K5", "ernp_K5_post", "ernp_keepA_K5_post",
       "ernp_K5_mut", "ernp_K5_keepA0", "ernp_K5_keepA1k", "ernp_K5_mut_keepA0"]
REF = ["ernp", "ernp_B0", "abl_evid_rep", "ssf_causal", "knn"]
COLS = ["F1", "MCC", "FPR", "zero_day_recall", "zd_recall_at1fpr", "AUROC"]

if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1:]] or [0, 1]
    rows = [r for d in DS for m in REF + VAR for s in seeds if (r := run_metrics(d, m, s)) is not None]
    df = pd.DataFrame(rows); df["method"] = pd.Categorical(df["method"], REF + VAR, ordered=True)
    rep = os.path.join(RESULTS, "..", "reports"); os.makedirs(rep, exist_ok=True); df.to_csv(os.path.join(rep, "stab_runs.csv"), index=False)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), pd.option_context("display.width", 250, "display.max_columns", 60, "display.max_rows", 400):
        print(__doc__.strip().split("\n\n")[0]); print(f"\nseeds {seeds}; runs per dataset and method")
        print(df.groupby(["dataset", "method"], observed=True).size().unstack(0).fillna(0).astype(int).to_string())
        print("\nMean over seeds"); print(df.groupby(["dataset", "method"], observed=True)[COLS].mean().round(3).to_string())
        for d in ("SIMARGL", "CTU13"):
            x = df[df.dataset == d]; fc = [c for c in x.columns if c.startswith("rec::") and x[c].notna().any()]
            if len(x): print(f"\n{d}: recall per family (own rule)"); print(x.groupby("method", observed=True)[fc].mean().round(2).to_string())
        print("\nPer variant, paired with the reference on the same dataset and seed, then averaged per dataset: datasets won, mean difference, one-sided Wilcoxon p")
        for v in VAR:
            for ref in ("ssf_causal", "ernp"):
                for col in ("F1", "zero_day_recall"):
                    piv = df.pivot_table(index=["dataset", "seed"], columns="method", values=col, observed=True)
                    if v not in piv or ref not in piv: continue
                    d = (piv[v] - piv[ref]).dropna().groupby("dataset").mean()
                    if not len(d): continue
                    p = wilcoxon(d, alternative="greater").pvalue if (d != 0).any() else 1.0
                    print(f"  {v:20s} vs {ref:10s} {col:16s}: wins {int((d > 0).sum())}/{len(d)}, mean diff {d.mean():+.3f}, p = {p:.4f}")
    open(os.path.join(rep, "stab.txt"), "w").write(buf.getvalue()); print(buf.getvalue())
