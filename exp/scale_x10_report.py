"""Scale study report: CSE-CIC-IDS2018 at 10x the main sample (4M flows) vs the main 400k streams, same metrics as
evaluate.run_metrics. Each 4M stream is loaded once per seed. Output: reports/x10_runs.csv and a summary table."""
import os, sys, json, importlib, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.metrics import roc_auc_score, average_precision_score

M = ["ernp", "ssf_causal", "owad", "cade", "trident", "memstream"]
rows = []
for s in (0, 1, 2):
    os.environ["EXP_SEED"] = str(s); import common; importlib.reload(common)
    ds = "CIC18_x10" if s == 0 else f"CIC18_x10_s{s}"
    df, X, src, dst, tr, ho, s_idx, anc = common.load(ds); del X
    fam = df["fam"].values[s_idx]; y = fam != "BENIGN"
    known = set(df["fam"].values[anc & (df["fam"].values != "BENIGN")]); unk = y & ~np.isin(fam, list(known))
    for m in M:
        r = np.load(common.RESULTS + f"{ds}__{m}__s{s}.npz"); flag, score = r["flag"].astype(bool), r["score"]
        info = json.load(open(common.RESULTS + f"{ds}__{m}__s{s}.json"))
        tp, fp = (flag & y).sum(), (flag & ~y).sum(); fn, tn = (~flag & y).sum(), (~flag & ~y).sum()
        P, R = tp / max(1, tp + fp), tp / max(1, tp + fn)
        den = np.sqrt(float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn))
        thr = np.quantile(score[~y], 0.99)
        rows.append(dict(dataset="CIC18_x10", method=m, seed=s, F1=2 * P * R / max(1e-9, P + R),
                         MCC=(float(tp) * tn - float(fp) * fn) / den if den else 0.0, FPR=fp / max(1, fp + tn),
                         zero_day_recall=(flag & unk).sum() / max(1, unk.sum()), AUROC=roc_auc_score(y, score),
                         AUPRC=average_precision_score(y, score), zd_recall_at1fpr=float((score[unk] > thr).mean()),
                         seconds=info.get("seconds", np.nan), peak_mem_mb=info.get("peak_mem_mb", np.nan),
                         fit_mean_s=info.get("timing", {}).get("fit_mean_s", np.nan)))
        print(ds, m, {k: round(v, 3) for k, v in rows[-1].items() if isinstance(v, float)}, flush=True)
x = pd.DataFrame(rows); x.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports", "x10_runs.csv"), index=False)
r1 = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports", "rq1_runs.csv"))
base = r1[(r1.dataset == "CIC18") & r1.method.isin(M) & (r1.seed <= 2)]
cols = ["F1", "MCC", "FPR", "zero_day_recall", "AUROC", "AUPRC", "zd_recall_at1fpr"]
print("\n== CIC18 main sample (400k, seeds 0-2)\n" + base.groupby("method")[cols].mean().reindex(M).round(3).to_string())
print("\n== CIC18 x10 (4M, seeds 0-2): mean\n" + x.groupby("method")[cols].mean().reindex(M).round(3).to_string())
print("\n== CIC18 x10: std\n" + x.groupby("method")[["F1", "MCC", "zero_day_recall"]].std().reindex(M).round(3).to_string())
