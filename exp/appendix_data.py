"""Appendix data from the saved runs (no model is run).
  reports/family_recall.csv : per dataset and attack family: known/zero-day, stream flows (seed 0), recall at each
                              method's own decision rule (mean over five seeds)
  reports/group_audit.csv   : per dataset: alert groups decided per day, group size, labels propagated, and impurity of
                              the group labels (from the ernp_audit runs, identical to the ernp runs)
"""
import os, sys, json, glob, importlib, numpy as np, pandas as pd
EXP = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, EXP)
REP = os.path.join(EXP, "reports"); RES = os.path.join(EXP, "results")
DS = ["CIDDS_ctx", "ToN_ctx", "CIC17", "CIC18", "CTU13", "SIMARGL", "APPRAISE"]
M = ["ernp", "ssf_causal", "owad", "cade", "trident", "memstream", "knn"]

def family_table():
    r = pd.concat([pd.read_csv(os.path.join(REP, f)) for f in ("rq1_runs.csv", "rq1_ext_runs.csv", "real_runs.csv")
                   if os.path.exists(os.path.join(REP, f))])
    r = r.drop_duplicates(["dataset", "method", "seed"])
    os.environ["EXP_SEED"] = "0"; import common; importlib.reload(common)
    rows = []
    for ds in DS:
        df, X, src, dst, tr, ho, s_idx, anc = common.load(ds); del X
        fam = df["fam"].values; known = set(fam[anc & (fam != "BENIGN")])
        stream = pd.Series(fam[s_idx]).value_counts()
        for f in known - set(stream.index): stream[f] = 0     # known families that occur only in the anchor
        for f, n in stream.items():
            if f == "BENIGN": continue
            row = dict(dataset=ds, family=f, kind="known" if f in known else "zero-day", stream_flows=int(n))
            for m in M:
                g = r[(r.dataset == ds) & (r.method == m)]
                row[m] = g[f"rec::{f}"].mean() if f"rec::{f}" in g else np.nan
            rows.append(row)
    out = pd.DataFrame(rows).sort_values(["dataset", "kind", "stream_flows"], ascending=[True, True, False])
    out.to_csv(os.path.join(REP, "family_recall.csv"), index=False); print(out.round(3).to_string())

def group_audit():
    rows = []
    for f in sorted(glob.glob(os.path.join(RES, "*__ernp_audit__s*.json"))):
        ds = os.path.basename(f).split("__")[0]
        base = next(d for d in DS if ds == d or ds.startswith(d + "_s") or (d.endswith("_ctx") and ds.startswith(d.replace("_ctx", "") + "_ctx")))
        rec = json.load(open(f))["audit"]
        a = pd.DataFrame(rec, columns=["day", "benign", "size", "wrong", "rep_family", "radius"][:len(rec[0])])
        b, m = a[a.benign == 1], a[a.benign == 0]
        rows.append(dict(dataset=base, stream=ds, days=a.day.nunique(), groups_per_day=len(a) / max(1, a.day.nunique()),
                         mean_group_size=a["size"].mean(), flows_labelled=int(a["size"].sum()),
                         benign_groups=len(b), attack_groups=len(m),
                         attack_in_benign_groups=b.wrong.sum() / max(1, b["size"].sum()),
                         benign_in_attack_groups=m.wrong.sum() / max(1, m["size"].sum()),
                         impure_benign_groups=(b.wrong > 0).mean() if len(b) else np.nan))
    out = pd.DataFrame(rows); out.to_csv(os.path.join(REP, "group_audit.csv"), index=False)
    print(out.groupby("dataset").mean(numeric_only=True).round(4).to_string())

def group_footprint():
    """Prop. 1(iii): a wrongly confirmed group G changes retention only within Delta + rho_delta of G, Delta <= 2 x radius.
    reports/group_footprint.csv: per dataset, the group diameter bound 2r in units of epsilon (benign-labelled groups,
    all groups, and impure benign groups, i.e. the analyst-error case)."""
    rows = []
    for f in sorted(glob.glob(os.path.join(RES, "*__ernp_diam__s*.json"))):
        ds = os.path.basename(f).split("__")[0]
        base = next(d for d in DS if ds == d or ds.startswith(d.split("_")[0]))
        info = json.load(open(f)); eps = float(info["eps"])
        a = pd.DataFrame(info["audit"], columns=["day", "benign", "size", "wrong", "rep_family", "radius"])
        a["diam_eps"] = 2 * a.radius / eps; b = a[a.benign == 1]; imp = b[b.wrong > 0]
        rows.append(dict(dataset=base, stream=ds, eps=eps,
                         benign_median=b.diam_eps.median(), benign_p90=b.diam_eps.quantile(0.9), benign_max=b.diam_eps.max(),
                         impure_median=imp.diam_eps.median() if len(imp) else np.nan,
                         impure_p90=imp.diam_eps.quantile(0.9) if len(imp) else np.nan, n_impure=len(imp),
                         all_median=a.diam_eps.median()))
    out = pd.DataFrame(rows); out.to_csv(os.path.join(REP, "group_footprint.csv"), index=False)
    print(out.groupby("dataset").mean(numeric_only=True).round(3).to_string())

if __name__ == "__main__":
    family_table()
    if glob.glob(os.path.join(RES, "*__ernp_audit__s*.json")): group_audit()
    if glob.glob(os.path.join(RES, "*__ernp_diam__s*.json")): group_footprint()
