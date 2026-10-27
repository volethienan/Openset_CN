"""RQ2 with a naive analyst (job group `naive`). No model is run.

Protocol-valid slow poisoning of CIDDS-001 (data/inject_valid.py, streams CIDDS_ctx*_injvalid{init,resp}_bf, five seeds).
Ground truth marks every injected ramp flow as an attack (INJ_ramp), including the first ramp records, which are
copies of the compromised host's benign behaviour. In the main protocol the analyst answers with that ground truth.
Here the analyst answers BENIGN for ramp flows with ramp fraction a below a level (common.naive_ramp_mask):
  R0   correct analyst (the existing runs: ernp_K5A, abl_retr_none_K5A, abl_prior_none_K5A, ssf_causal)
  R50  ramp flows with a < 0.5, whose ports, flags and ToS are those of the benign template, are answered BENIGN
  R100 every ramp flow is answered BENIGN; only the payload (a = 1) is recognised as an attack
Evaluation always uses the ground truth. Context rows: ERNP without online labels (ernp_B0_K5A) and ERNP fed its own
verdicts (abl_self_K5A), as in RQ2.
Per run: F1, FPR, AUROC, recall of the ramp and of the payload at the method's own rule, payload recall at the 1%
test-benign diagnostic, absorption (ramp recall < 10% from some day on). For the ERNP-family runs the analyst decisions
are replayed from the saved alerts (same grouping as ernp.run) with the same analyst model; the replay must reproduce
the run's own counts of decisions and of naive answers, or the report stops. It counts what the attacker obtains:
decisions spent on injected representatives, naive BENIGN answers, injected flows entering the benign evidence P
(and, of those, ramp flows with a < 0.5 and the payload) and injected flows entering the attack evidence A.
Paired differences: level R50 / R100 minus R0 of the same method, stream and seed.
Output: reports/naive_analyst.txt, reports/naive_runs.csv, reports/naive_channel.csv
"""
import os, sys, json, importlib, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.cluster import MiniBatchKMeans
from evaluate import seeded_name, run_metrics

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
STREAMS = ["CIDDS_ctx_injvalidinit_bf", "CIDDS_ctx_injvalidresp_bf"]
FAMILY = {"ERNP": "ernp_K5A", "ERNP, no retraction": "abl_retr_none_K5A", "GBM+AL (no prior)": "abl_prior_none_K5A",
          "SSF": "ssf_causal"}
LEVELS = {"R0": 0.0, "R50": 0.5, "R100": 1.0}
CONTEXT = {"ERNP, no online labels": "ernp_B0_K5A", "ERNP-self (counterfactual)": "abl_self_K5A"}
GROUPED = {"ernp_K5A", "abl_retr_none_K5A", "abl_prior_none_K5A"}


def name_of(base, level):
    return base if level == "R0" else f"{base}_{level}"


def label_channel(ds, df, X, s_idx, flag, s, budget, a_level):
    """replays ERNP's grouping on the saved alerts with the analyst model of the run"""
    import common
    fam = df["fam"].values; naive = common.naive_ramp_mask(ds, df, a_level)
    half = common.naive_ramp_mask(ds, df, 0.5); pay = fam == "INJ_attack"
    inj = np.char.startswith(fam.astype(str), "INJ_"); day = common.stream_days(df, s_idx)
    o = dict(decisions=0, naive_answers=0, dec_on_injected_rep=0, injected_into_P=0, ramp_a_lt_half_into_P=0,
             payload_into_P=0, injected_into_A=0)
    for d in dict.fromkeys(day):
        idx = np.where(day == d)[0]; al = s_idx[idx][flag[idx]]
        if len(al) < 2: continue
        k = min(budget, len(al)); km = MiniBatchKMeans(k, random_state=s, n_init=3).fit(X[al])
        for cl in range(k):
            mem = al[km.labels_ == cl]
            if not len(mem): continue
            rep = mem[np.argmin(np.linalg.norm(X[mem] - km.cluster_centers_[cl], axis=1))]; o["decisions"] += 1
            is_ben = fam[rep] == "BENIGN" or naive[rep]; o["naive_answers"] += int(naive[rep])
            o["dec_on_injected_rep"] += int(inj[rep])
            if is_ben:
                o["injected_into_P"] += int(inj[mem].sum()); o["ramp_a_lt_half_into_P"] += int(half[mem].sum())
                o["payload_into_P"] += int(pay[mem].sum())
            else: o["injected_into_A"] += int(inj[mem].sum())
    return o


def collect():
    rows, chan = [], []
    for base in STREAMS:
        for s in range(5):
            os.environ["EXP_SEED"] = str(s); import common; importlib.reload(common)
            ds = seeded_name(base, s); df, X, src, dst, tr, ho, s_idx, anc = common.load(ds)
            fam = df["fam"].values[s_idx]; ben = fam == "BENIGN"; pay = fam == "INJ_attack"; ramp = fam == "INJ_ramp"
            runs = [(lab, lv, name_of(m, lv)) for lab, m in FAMILY.items() for lv in LEVELS] + \
                   [(lab, "R0", m) for lab, m in CONTEXT.items()]
            for lab, lv, m in runs:
                r = run_metrics(base, m, s)
                if r is None: print("  missing", ds, m, flush=True); continue
                z = np.load(common.RESULTS + f"{ds}__{m}__s{s}.npz"); sc, fl = z["score"], z["flag"].astype(bool)
                info = json.load(open(common.RESULTS + f"{ds}__{m}__s{s}.json"))
                if lv != "R0" and m.split("_R")[0] in GROUPED:     # the run used the level it is filed under
                    assert abs(float(info["cfg"]["naive_ramp"]) - LEVELS[lv]) < 1e-12, (ds, m, info["cfg"].get("naive_ramp"))
                if lv == "R0" and m in GROUPED: assert float(info["cfg"].get("naive_ramp", 0.0)) == 0.0, (ds, m)
                rows.append(dict(scenario=base.split("_")[-2].replace("injvalid", ""), method=lab, level=lv, run=m, seed=s,
                                 F1=r["F1"], FPR=r["FPR"], AUROC=r["AUROC"], ramp_recall=float(fl[ramp].mean()),
                                 payload_recall=float(fl[pay].mean()),
                                 payload_at1fpr=float((sc[pay] > np.quantile(sc[ben], 0.99)).mean()),
                                 absorbed=bool(r["absorption_day"] >= 0), labels=r["labels"]))
                if m.split("_R")[0] in GROUPED:
                    c = label_channel(ds, df, X, s_idx, fl, s, int(info["cfg"]["budget"]), LEVELS[lv])
                    assert c["decisions"] == int(info["labels_used"]), (ds, m, c["decisions"], info["labels_used"])
                    if "naive_ramp_answers" in info:
                        assert c["naive_answers"] == int(info["naive_ramp_answers"]), (ds, m, c["naive_answers"], info["naive_ramp_answers"])
                    else: assert c["naive_answers"] == 0, (ds, m)
                    chan.append(dict(scenario=rows[-1]["scenario"], method=lab, level=lv, seed=s,
                                     n_injected=int((pay | ramp).sum()), **c))
            print("naive", ds, flush=True)
    return pd.DataFrame(rows), pd.DataFrame(chan)


def write(run, ch):
    run.to_csv(os.path.join(OUT, "naive_runs.csv"), index=False); ch.to_csv(os.path.join(OUT, "naive_channel.csv"), index=False)
    order = {k: i for i, k in enumerate(list(FAMILY) + list(CONTEXT))}
    g = run.groupby(["scenario", "method", "level"])
    t = g[["payload_recall", "payload_at1fpr", "ramp_recall", "FPR", "F1", "AUROC"]].mean()
    t["payload_min"] = g["payload_recall"].min(); t["absorbed"] = g["absorbed"].sum(); t["n"] = g.size()
    t = t.reset_index().sort_values(["scenario", "method", "level"], key=lambda c: c.map(order) if c.name == "method" else c)
    with open(os.path.join(OUT, "naive_analyst.txt"), "w", encoding="utf-8") as fh:
        fh.write(__doc__ + "\nPer scenario, method and analyst level (mean over seeds; payload_min = worst seed; absorbed = #seeds)\n")
        fh.write(t.round(3).to_string(index=False) + "\n\n")
        if len(ch):
            c = ch.copy(); c["share_dec_on_injected"] = c["dec_on_injected_rep"] / c["decisions"]
            c["share_injected_in_P"] = c["injected_into_P"] / c["n_injected"]
            fh.write("Label channel of the ERNP-family runs (replayed; mean over seeds)\n")
            fh.write(c.drop(columns="seed").groupby(["scenario", "method", "level"]).mean().round(3).to_string() + "\n\n")
        fh.write("Paired differences (level - R0, same method, stream and seed): mean, #seeds where the level is lower\n")
        piv = run.pivot_table(index=["scenario", "method", "seed"], columns="level", values=["payload_recall", "FPR", "F1"])
        for metric in ("payload_recall", "FPR", "F1"):
            for lv in ("R50", "R100"):
                if (metric, lv) not in piv.columns or (metric, "R0") not in piv.columns: continue
                d = (piv[(metric, lv)] - piv[(metric, "R0")]).dropna()
                if not len(d): continue
                agg = d.groupby(level=[0, 1]).agg(["mean", lambda x: int((x < 0).sum()), "size"])
                for (sc_, me), v in agg.iterrows():
                    fh.write(f"  {metric:15s} {lv:4s} {sc_:5s} {me:22s} {v['mean']:+.3f}  lower on {int(v.iloc[1])}/{int(v['size'])}\n")
    print(open(os.path.join(OUT, "naive_analyst.txt"), encoding="utf-8").read())


if __name__ == "__main__":
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 300)
    write(*collect())
