"""Metrics, seed aggregation and statistics for all RQs.

Per run (dataset, method, seed): precision, recall, F1, FPR, AUROC, AUPRC, zero-day recall (families never labelled
in the anchor period), per-family recall, online labels used, and - for injection streams - payload recall and the
absorption day (first stream day from which the ramp recall stays below 10%).

Statistics (fixed before running, see ERNP_RQs_Experiments.md):
  * per dataset: mean +- std over seeds, and #seeds where ERNP > baseline
  * primary test: one-sided Wilcoxon signed-rank ERNP > baseline over all (dataset x seed) pairs (n = 20 in RQ1)
    (with 5 seeds a per-dataset two-sided Wilcoxon cannot reach p < 0.05 - min p = 0.0625 - so it is not used)
  * ranking: Friedman test + Nemenyi critical difference over (dataset x seed) blocks
"""
import sys, os, json, glob, argparse, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("EXP_SEED", "0")
from common import RESULTS
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import wilcoxon, friedmanchisquare

Q_NEMENYI = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}   # alpha = 0.05

def seeded_name(base, s):
    """base dataset name (seed 0) -> stream name of seed s (CIDDS_ctx_injinit -> CIDDS_ctx_s1_injinit, CIC18 -> CIC18_s1)"""
    if s == 0 or base.endswith("_full"): return base
    for root in ("CIDDS_ctx", "ToN_ctx"):
        if base.startswith(root): return f"{root}_s{s}" + base[len(root):]
    return f"{base}_s{s}"

def run_metrics(base, method, s):
    os.environ["EXP_SEED"] = str(s)
    import importlib, common; importlib.reload(common)
    ds = seeded_name(base, s)
    f = RESULTS + f"{ds}__{method}__s{s}.npz"
    if not os.path.exists(f): return None
    df, X, src, dst, tr, ho, s_idx, anc = common.load(ds)
    r = np.load(f); flag, score = r["flag"].astype(bool), r["score"]; info = json.load(open(f.replace(".npz", ".json")))
    fam = df["fam"].values[s_idx]; y = fam != "BENIGN"
    tp, fp = (flag & y).sum(), (flag & ~y).sum(); fn = (~flag & y).sum(); tn = (~flag & ~y).sum()
    P = tp / max(1, tp + fp); R = tp / max(1, tp + fn)
    known = set(df["fam"].values[anc & (df["fam"].values != "BENIGN")]); unk = y & ~np.isin(fam, list(known))
    den = np.sqrt(float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn))
    out = dict(dataset=base, stream=ds, method=method, seed=s, precision=P, recall=R, F1=2 * P * R / max(1e-9, P + R),
               MCC=(float(tp) * tn - float(fp) * fn) / den if den > 0 else 0.0,   # 0 for alert-all / alert-none
               bal_acc=0.5 * (R + tn / max(1, tn + fp)),
               FPR=fp / max(1, fp + tn), zero_day_recall=(flag & unk).sum() / max(1, unk.sum()) if unk.any() else np.nan,
               labels=0 if method.startswith(("trident", "cade")) else int(info.get("labels_used", 0)))   # ONLINE labels only
    try: out["AUROC"] = roc_auc_score(y, score); out["AUPRC"] = average_precision_score(y, score)
    except Exception: out["AUROC"] = out["AUPRC"] = np.nan
    # detector quality at a COMMON operating point, independent of each method's own decision rule:
    # threshold = 99th percentile of the method's scores on stream benign flows (FPR = 1%)
    thr = np.quantile(score[~y], 1 - 0.01) if (~y).any() else np.inf
    out["recall_at1fpr"] = float((score[y] > thr).mean()) if y.any() else np.nan
    out["zd_recall_at1fpr"] = float((score[unk] > thr).mean()) if unk.any() else np.nan
    out["seconds"] = info.get("seconds", np.nan); out["peak_mem_mb"] = info.get("peak_mem_mb", np.nan)
    for f_ in sorted(set(fam) - {"BENIGN"}): out["rec::" + f_] = float(flag[fam == f_].mean())
    if (fam == "INJ_attack").any():                                   # RQ2 injection streams
        out["payload_recall"] = float(flag[fam == "INJ_attack"].mean())
        day = pd.to_datetime(df["t"].values[s_idx], unit="s").normalize()
        ramp = pd.Series(flag[fam == "INJ_ramp"]).groupby(day[fam == "INJ_ramp"]).mean()
        below = ramp < 0.10; absorbed = [i for i in range(len(below)) if below.iloc[i:].all()]
        out["absorption_day"] = int(absorbed[0]) if absorbed else -1   # -1: never absorbed
    if "timing" in info: out.update({"t_" + k: v for k, v in info["timing"].items()})
    return out

def collect(datasets, methods, seeds):
    rows = [run_metrics(d, m, s) for d in datasets for m in methods for s in seeds]
    return pd.DataFrame([r for r in rows if r is not None])

def summary(df, metrics=("F1", "MCC", "zero_day_recall", "zd_recall_at1fpr", "FPR", "AUROC", "AUPRC", "labels")):
    g = df.groupby(["dataset", "method"])[list(metrics)]
    return g.mean().round(4).astype(str) + " ± " + g.std().round(4).astype(str)

def stats(df, ours="ernp", metric="F1"):
    piv = df.pivot_table(index=["dataset", "seed"], columns="method", values=metric).dropna()
    res = {}
    for b in [c for c in piv.columns if c != ours]:
        d = piv[ours] - piv[b]
        p = wilcoxon(piv[ours], piv[b], alternative="greater").pvalue if (d != 0).any() else 1.0
        res[b] = dict(mean_diff=float(d.mean()), wins=int((d > 0).sum()), n=len(d), p_one_sided=float(p))
    ranks = piv.rank(axis=1, ascending=False).mean().sort_values()
    k, N = piv.shape[1], piv.shape[0]
    fr = friedmanchisquare(*[piv[c] for c in piv.columns]).pvalue if k >= 3 else np.nan
    cd = Q_NEMENYI.get(k, np.nan) * np.sqrt(k * (k + 1) / (6 * N))
    return pd.DataFrame(res).T, ranks, dict(friedman_p=fr, nemenyi_CD=cd, n_blocks=N, k=k)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True); ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4]); ap.add_argument("--out", default="report")
    ap.add_argument("--metric", default="F1")
    ap.add_argument("--compare", nargs="+", default=None, help="methods entering the tests/ranking (default: all); "
                    "e.g. RQ1 ranks ERNP only against the 5 SOTA under the causal protocol")
    a = ap.parse_args(); df = collect(a.datasets, a.methods, a.seeds)
    os.makedirs(RESULTS + "../reports", exist_ok=True); df.to_csv(RESULTS + f"../reports/{a.out}_runs.csv", index=False)
    print(summary(df).to_string())
    dc = df if a.compare is None else df[df["method"].isin(a.compare)]
    if "ernp" in dc["method"].values and dc["method"].nunique() > 1:
        print("\ntests/ranking over:", sorted(dc["method"].unique()))
        for metric in (a.metric, "MCC", "zero_day_recall", "zd_recall_at1fpr"):
            t, ranks, fr = stats(dc, metric=metric); print(f"\n== {metric}: ERNP vs baselines (one-sided Wilcoxon over dataset x seed)\n", t.round(4).to_string())
            print("mean rank:", ranks.round(2).to_dict(), fr)
