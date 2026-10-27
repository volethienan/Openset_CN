"""Review 3 (28/10/2026): what retraction changes end to end, and the prior under ERNP K5A's feedback schedule.
No model is run; every value comes from saved runs.

1. Analyst load with and without retraction (seven datasets, five seeds): ernp_K5A against abl_retr_none_K5A, each
   with its own feedback. Per run: decisions, benign decisions (groups the analyst confirms benign), flows in those
   groups, benign alerts per stream day (as in e23_report.py), FPR and F1.
2. The prior under a fixed feedback schedule (four testbeds, five seeds): ernp_K5A, rpl_retr_none_K5A (retention
   weight one) and rpl_prior_none_K5A (no prior), the replays refitted on ernp_K5A's alerts, groups and decisions.
   A replay must make as many decisions as the run it replays, or the report stops.
3. ToN at the 4 h cadence: ernp_P4_K5A, the no-prior classifier with its own feedback (abl_prior_none_P4_K5A) and
   replaying ernp_P4_K5A (rpl_prior_none_P4_K5A), next to the daily runs (ernp_K5A, abl_prior_none_K5A,
   rpl_prior_none_K5A).
Output: reports/review3.txt, reports/review3_retraction_load.csv, reports/review3_prior.csv
"""
import os, sys, json, importlib, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import seeded_name, run_metrics

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
SEVEN = ["CIDDS_ctx", "ToN_ctx", "CIC17", "CIC18", "CTU13", "SIMARGL", "APPRAISE"]
TESTBEDS = ["CIDDS_ctx", "ToN_ctx", "CIC17", "CIC18"]
LOAD = ["ernp_K5A", "abl_retr_none_K5A"]
REPLAY = {"ernp_K5A": None, "rpl_retr_none_K5A": "ernp_K5A", "rpl_prior_none_K5A": "ernp_K5A"}
CADENCE = {"ernp_K5A": None, "abl_prior_none_K5A": None, "rpl_prior_none_K5A": "ernp_K5A",
           "ernp_P4_K5A": None, "abl_prior_none_P4_K5A": None, "rpl_prior_none_P4_K5A": "ernp_P4_K5A"}
COLS = ["F1", "MCC", "FPR", "AUROC", "zd_recall_at1fpr"]


def runs(bases, methods):
    rows = []
    for b in bases:
        for s in range(5):
            os.environ["EXP_SEED"] = str(s); import common; importlib.reload(common)
            ds = seeded_name(b, s); df, X, src, dst, tr, ho, s_idx, anc = common.load(ds)
            ben = df["fam"].values[s_idx] == "BENIGN"; n_days = len(set(common.stream_days(df, s_idx)))
            for m, ref in methods.items():
                r = run_metrics(b, m, s)
                assert r is not None, ("missing run", ds, m, s)
                info = json.load(open(common.RESULTS + f"{ds}__{m}__s{s}.json"))
                fl = np.load(common.RESULTS + f"{ds}__{m}__s{s}.npz")["flag"].astype(bool)
                if ref is not None:   # a replay decides on the alerts of the run it replays
                    rinfo = json.load(open(common.RESULTS + f"{ds}__{ref}__s{s}.json"))
                    # a K5A run identical to its K5 run is a copy of it (exp/k5a_copy.py); replaying either is the same
                    assert info["cfg"]["replay"] in (ref, rinfo.get("k5a_copied_from")), (ds, m, info["cfg"]["replay"], ref)
                    n_ref = int(rinfo["labels_used"])
                    assert int(info["labels_used"]) == n_ref, (ds, m, "decisions", info["labels_used"], "vs", n_ref)
                au = info.get("audit") or []
                rows.append(dict(dataset=b, seed=s, method=m, **{c: r[c] for c in COLS}, decisions=int(info["labels_used"]),
                                 benign_decisions=sum(int(a[1]) for a in au) if au else np.nan,
                                 benign_confirmed_flows=sum(int(a[2]) for a in au if int(a[1])) if au else np.nan,
                                 benign_alerts_per_day=float((fl & ben).sum()) / n_days))
            print("review3", ds, flush=True)
    return pd.DataFrame(rows)


def paired(d, a, b, cols):
    p = d.pivot_table(index=["dataset", "seed"], columns="method", values=cols)
    out = []
    for c in cols:
        x = p[(c, b)] - p[(c, a)]
        g = x.groupby(level=0)
        out.append(pd.DataFrame({"metric": c, "mean_diff": g.mean(), "seeds_higher": g.apply(lambda z: int((z > 0).sum())),
                                 "seeds": g.size()}))
    return pd.concat(out).reset_index()


if __name__ == "__main__":
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    load = runs(SEVEN, dict.fromkeys(LOAD))
    load.to_csv(os.path.join(OUT, "review3_retraction_load.csv"), index=False)
    if sys.argv[1:] == ["load"]:   # part 1 only (needs no new run)
        print(paired(load, "ernp_K5A", "abl_retr_none_K5A", ["benign_decisions", "benign_alerts_per_day", "FPR", "F1"]).round(4).to_string(index=False))
        sys.exit(0)
    prior = pd.concat([runs(TESTBEDS, REPLAY).assign(block="replay"),
                       runs(["ToN_ctx"], {k: v for k, v in CADENCE.items() if k not in REPLAY or k == "abl_prior_none_K5A"}).assign(block="cadence")])
    # daily ToN rows of the cadence block come from the replay block (same runs)
    ton_daily = prior[(prior.block == "replay") & (prior.dataset == "ToN_ctx") & prior.method.isin(["ernp_K5A", "rpl_prior_none_K5A"])]
    prior = pd.concat([prior, ton_daily.assign(block="cadence")], ignore_index=True)
    load.to_csv(os.path.join(OUT, "review3_retraction_load.csv"), index=False)
    prior.to_csv(os.path.join(OUT, "review3_prior.csv"), index=False)
    lc = ["benign_decisions", "benign_confirmed_flows", "benign_alerts_per_day", "FPR", "F1"]
    with open(os.path.join(OUT, "review3.txt"), "w", encoding="utf-8") as fh:
        fh.write(__doc__ + "\n1. Analyst load, mean over seeds\n")
        fh.write(load.groupby(["dataset", "method"])[lc + ["decisions"]].mean().round(4).to_string() + "\n\n")
        fh.write("   no retraction minus ERNP (mean; seeds on which it is higher)\n")
        fh.write(paired(load, "ernp_K5A", "abl_retr_none_K5A", lc).round(4).to_string(index=False) + "\n\n")
        fh.write("2. Fixed feedback schedule of ernp_K5A, mean over seeds\n")
        r = prior[prior.block == "replay"]
        fh.write(r.groupby(["dataset", "method"])[COLS + ["benign_alerts_per_day", "decisions"]].mean().round(4).to_string() + "\n\n")
        fh.write("3. ToN, daily and 4 h cadence, mean over seeds\n")
        c = prior[prior.block == "cadence"]
        fh.write(c.groupby("method")[COLS + ["benign_alerts_per_day", "decisions"]].mean().round(4).to_string() + "\n")
        fh.write("   per seed MCC\n" + c.pivot_table(index="seed", columns="method", values="MCC").round(3).to_string() + "\n")
    print(open(os.path.join(OUT, "review3.txt"), encoding="utf-8").read())
