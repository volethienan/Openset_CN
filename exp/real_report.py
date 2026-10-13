"""Real-network datasets (CTU-13, RoEduNet-SIMARGL2021, APPRAISE H2020): RQ1 protocol, unchanged defaults, five seeds.

Per run: the metrics of evaluate.run_metrics. In addition, for CTU-13, the false-positive rate on the flows that the
dataset verifies as 'Normal' (its 'Background' flows are unverified), and for SIMARGL_all (published labels) the
metrics with the 24.4M-flow unlabelled FIN scan kept as normal traffic.
Statistics over datasets (seeds are sub-samples, not datasets): Friedman test over the five detectors and ERNP, and
per-baseline sign counts and one-sided Wilcoxon signed-rank tests of the per-dataset mean differences, on the four
benchmarks of the paper and on all seven datasets.
Output: reports/real_runs.csv, reports/real.txt
"""
import os, sys, json, importlib, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.stats import friedmanchisquare, wilcoxon
from evaluate import run_metrics, seeded_name

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
REAL = ["CTU13", "SIMARGL", "APPRAISE"]; OLD = ["CIDDS_ctx", "ToN_ctx", "CIC17", "CIC18"]
M = ["ernp", "abl_evid_rep", "ernp_B0", "abl_prior_none", "ssf_causal", "owad", "owad_lr3", "cade", "cade_t1", "trident", "trident_t1",
     "memstream", "knn", "iforest", "ae_static"]
SOTA = ["ssf_causal", "owad", "cade", "trident", "memstream"]
COLS = ["F1", "MCC", "FPR", "recall", "zero_day_recall", "zd_recall_at1fpr", "AUROC", "AUPRC", "labels"]

def collect():
    rows = []
    for b in REAL + ["SIMARGL_all"]:
        for s in range(5):
            os.environ["EXP_SEED"] = str(s); import common; importlib.reload(common)
            ds = seeded_name(b, s); df = pd.read_parquet(common.STREAMS + f"{ds}.parquet", columns=["t", "fam"] + (["lab"] if b == "CTU13" else []))
            anc = (df["t"] < common.anchor_end(ds, df["t"].min())).values; s_idx = np.where(~anc)[0]
            for m in M:
                r = run_metrics(b, m, s)
                if r is None: continue
                if b == "CTU13":
                    fl = np.load(common.RESULTS + f"{ds}__{m}__s{s}.npz")["flag"].astype(bool); lab = df["lab"].values[s_idx]
                    r["FPR_normal"] = float(fl[lab == "normal"].mean()); r["n_normal"] = int((lab == "normal").sum())
                rows.append({k: v for k, v in r.items() if not k.startswith("t_")})
        print("collected", b, flush=True)
    return pd.DataFrame(rows)

def tests(dm, label, fh):
    """dm: dataset x method table of mean F1 (ERNP and the five detectors)"""
    fh.write(f"\n{label}: {len(dm)} datasets\n" + dm.round(3).to_string() + "\n")
    ranks = dm.rank(axis=1, ascending=False).mean().round(2).to_dict()
    fh.write(f"mean rank {ranks}; Friedman p = {friedmanchisquare(*[dm[c] for c in dm.columns]).pvalue:.4f}\n")
    for b in SOTA:
        d = dm["ernp"] - dm[b]; p = wilcoxon(dm["ernp"], dm[b], alternative="greater").pvalue
        fh.write(f"  ERNP vs {b:11s}: wins {int((d > 0).sum())}/{len(d)}, mean diff {d.mean():+.3f}, one-sided Wilcoxon p = {p:.4f}\n")

def main():
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 300)
    run = collect(); run.to_csv(os.path.join(OUT, "real_runs.csv"), index=False)
    g = run.groupby(["dataset", "method"]); mean = g[COLS].mean(); mean.insert(0, "n", g.size())
    old = pd.concat([pd.read_csv(os.path.join(OUT, f)) for f in ("rq1_runs.csv", "rq1_ext_runs.csv", "rq4_runs.csv", "owad_lr3_rq1.csv", "t1_runs.csv")]).drop_duplicates(["dataset", "method", "seed"])
    with open(os.path.join(OUT, "real.txt"), "w", encoding="utf-8") as fh:
        fh.write(__doc__ + "\nSanity gate (runs with AUROC < 0.5 or F1 < 0.05):\n")
        bad = run[(run.AUROC < 0.5) | (run.F1 < 0.05)][["dataset", "method", "seed", "F1", "FPR", "recall", "AUROC"]]
        fh.write((bad.round(3).to_string(index=False) if len(bad) else "none") + "\n")
        fh.write("\nMean over seeds (n = runs)\n" + mean.round(3).to_string() + "\n")
        c = run[run.dataset == "CTU13"].groupby("method")[["FPR", "FPR_normal", "n_normal"]].mean()
        fh.write("\nCTU-13: FPR on all benign-labelled flows and on the verified 'Normal' flows\n" + c.round(4).to_string() + "\n")
        fam = [x for x in run.columns if x.startswith("rec::")]
        for b in REAL:
            t = run[run.dataset == b].groupby("method")[fam].mean().dropna(axis=1, how="all").T.round(2)
            fh.write(f"\n{b}: recall per family (own rule)\n" + t[[m for m in M if m in t.columns]].to_string() + "\n")
        piv = run[run.dataset.isin(REAL)].pivot_table(index=["dataset", "seed"], columns="method", values="F1")
        fh.write("\nPer dataset: F1 difference ERNP - method (mean over seeds; seeds won)\n")
        for m in [x for x in M if x != "ernp" and x in piv.columns]:
            d = (piv["ernp"] - piv[m]).dropna()
            fh.write(f"  {m:15s} " + "; ".join(f"{k} {v.mean():+.3f} ({int((v > 0).sum())}/{len(v)})" for k, v in d.groupby(level=0)) + "\n")
        allr = pd.concat([old[old.dataset.isin(OLD)], run[run.dataset.isin(REAL)]])
        dm = allr[allr.method.isin(["ernp"] + SOTA)].groupby(["dataset", "method"]).F1.mean().unstack()[["ernp"] + SOTA]
        fh.write("\nStatistics over datasets, F1 (per-dataset means)\n")
        tests(dm.loc[OLD], "four benchmarks of the paper", fh); tests(dm.loc[REAL], "three real-network datasets", fh); tests(dm.loc[OLD + REAL], "all seven datasets", fh)
        ref = allr[allr.method.isin(["ernp", "knn", "abl_prior_none", "abl_evid_rep"])].groupby(["dataset", "method"]).F1.mean().unstack().loc[OLD + REAL]
        fh.write("\nReferences, mean F1 per dataset\n" + ref.round(3).to_string() + "\nmean " + ref.mean().round(3).to_dict().__str__() + "\n")
    print(open(os.path.join(OUT, "real.txt"), encoding="utf-8").read())

if __name__ == "__main__":
    main()
