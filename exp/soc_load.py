"""E5 (part a): analyst / SOC load implied by each detector's decisions on the RQ1 streams (seeds 0-4). No re-training.

Per run: alerts and false alerts per stream day (sampled streams), false alerts per million benign flows (= FPR x 1e6;
independent of the benign sampling rate), alert precision, and the triage work these alerts create:
  * flow-level triage with a capacity of CAP flows/day: alerts left uninspected per day (backlog growth);
  * ERNP's group triage: decisions per day (<= B), alerts per decision, share of alerts covered by a decided group.
Cadence: daily runs and the 4-hour runs of the cadence study (suffix _P4, same daily budget).
Output: reports/soc_load_runs.csv, reports/soc_load.txt
"""
import os, sys, json, importlib, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import seeded_name

BASES = ["CIDDS_ctx", "ToN_ctx", "CIC17", "CIC18"]
METHODS = ["ernp", "ernp_P4", "abl_evid_rep", "ssf_causal", "ssf_causal_P4", "owad", "owad_P4", "owad_lr3", "memstream",
           "trident_t1", "cade_t1", "trident", "cade", "knn", "iforest", "ae_static"]
CAP = 200                                       # flow-level triage capacity per day (= the flow-label budget of SSF/OWAD)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")

def main():
    rows = []
    for b in BASES:
        for s in range(5):
            os.environ["EXP_SEED"] = str(s); import common; importlib.reload(common)
            ds = seeded_name(b, s); df, X, src, dst, tr, ho, s_idx, anc = common.load(ds)
            y = df["fam"].values[s_idx] != "BENIGN"; day = common.stream_days(df, s_idx); days = list(dict.fromkeys(day))
            for m in METHODS:
                p = common.RESULTS + f"{ds}__{m}__s{s}"
                if not os.path.exists(p + ".npz"): continue
                fl = np.load(p + ".npz")["flag"].astype(bool); info = json.load(open(p + ".json"))
                al_d = np.array([fl[day == d].sum() for d in days]); fa_d = np.array([(fl & ~y)[day == d].sum() for d in days])
                r = dict(dataset=b, seed=s, method=m, days=len(days), alerts_per_day=al_d.mean(), false_alerts_per_day=fa_d.mean(),
                         max_alerts_day=int(al_d.max()), false_per_1M_benign=1e6 * (fl & ~y).sum() / max(1, (~y).sum()),
                         precision=(fl & y).sum() / max(1, fl.sum()),
                         uninspected_per_day_flow_triage=np.maximum(al_d - CAP, 0).mean(),
                         share_alerts_inspected_flow_triage=np.minimum(al_d, CAP).sum() / max(1, al_d.sum()))
                if m.startswith("ernp") or m == "abl_evid_rep":
                    dec = int(info.get("labels_used", 0)); r.update(decisions_per_day=dec / len(days), alerts_per_decision=fl.sum() / max(1, dec))
                rows.append(r)
            print("done", ds, flush=True)
    run = pd.DataFrame(rows); run.to_csv(os.path.join(OUT, "soc_load_runs.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 200)
    t = run.drop(columns="seed").groupby(["dataset", "method"]).mean(numeric_only=True)
    with open(os.path.join(OUT, "soc_load.txt"), "w", encoding="utf-8") as fh:
        fh.write(__doc__ + "\nMean over seeds\n" + t.round(3).to_string() + "\n")
    print(open(os.path.join(OUT, "soc_load.txt"), encoding="utf-8").read())

if __name__ == "__main__":
    main()
