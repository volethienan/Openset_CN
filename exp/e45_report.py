"""Q1 revision: E4 (slow poisoning with protocol-valid flow records) and E5 (analyst label delay). No model is run.

E4. Streams CIDDS_ctx*_injvalid{init,resp}_bf (data/inject_valid.py) against the interpolated-feature streams of RQ2
(CIDDS_ctx*_inj{init,resp}_bf) and the clean stream, same seeds. Per run: F1, FPR, recall of the ramp and of the final
payload at the method's own rule, payload recall at 1% stream FPR (oracle point), absorption (ramp recall < 10% from
some day on), and the benign FPR before / during the campaign. For the ERNP variants the analyst decisions are
replayed from the saved alerts (same grouping as ernp.run, gated on labels_used) to count what the attacker gains
through the label channel: decisions spent on groups whose representative is an injected flow, groups labelled benign
that contain injected flows, and injected flows that enter the benign evidence P.

E5. Labels arrive D days after the alerts (ernp_D1, ernp_D2, ssf_causal_D1, owad_D1) against D = 0, on the three
streams with more than two days. ToN is also re-scored without the gateway burst hour (ton_burst_analysis.py) and
per stream day, because pooled AUROC below 0.5 there comes from that hour and from families that last 1-2 days.
Output: reports/e4_valid_poisoning.txt, e4_runs.csv, e4_label_channel.csv, e5_delay.txt, e5_runs.csv, e5_ton_days.csv
"""
import os, sys, json, importlib, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import roc_auc_score
from evaluate import seeded_name, run_metrics

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
E4_DS = ["CIDDS_ctx_injvalidinit_bf", "CIDDS_ctx_injvalidresp_bf", "CIDDS_ctx_injinit_bf", "CIDDS_ctx_injresp_bf"]
E4_M = ["ernp", "ernp_B25", "abl_evid_rep", "abl_retr_none", "abl_prior_none", "abl_self", "ernp_B0", "ssf_causal", "owad", "memstream", "trident", "cade"]
GROUPED = ["ernp", "ernp_B25", "abl_evid_rep", "abl_retr_none", "abl_prior_none"]   # ERNP variants with analyst decisions
E5_DS = ["CIDDS_ctx", "ToN_ctx", "CIC18"]
E5_M = ["ernp", "ernp_D1", "ernp_D2", "ssf_causal", "ssf_causal_D1", "owad", "owad_D1"]

def label_channel(X, fam, s_idx, day, flag, s, B, rep_only):
    """what the injected traffic obtains from the analyst decisions (replayed groups)"""
    inj = np.char.startswith(fam.astype(str), "INJ_"); o = dict(decisions=0, dec_on_injected_rep=0, benign_groups_with_injected=0,
                                                                 injected_into_P=0, injected_into_A=0)
    for d in dict.fromkeys(day):
        idx = np.where(day == d)[0]; al = s_idx[idx][flag[idx]]
        if len(al) < 2: continue
        k = min(B, len(al)); km = MiniBatchKMeans(k, random_state=s, n_init=3).fit(X[al])
        for cl in range(k):
            mem = al[km.labels_ == cl]
            if not len(mem): continue
            rep = mem[np.argmin(np.linalg.norm(X[mem] - km.cluster_centers_[cl], axis=1))]; o["decisions"] += 1
            grp = np.array([rep]) if rep_only else mem; n_inj = int(inj[grp].sum())
            o["dec_on_injected_rep"] += int(inj[rep])
            if fam[rep] == "BENIGN": o["benign_groups_with_injected"] += int(n_inj > 0); o["injected_into_P"] += n_inj
            else: o["injected_into_A"] += n_inj
    return o

def e4():
    rows, chan = [], []
    for base in E4_DS:
        for s in range(5):
            os.environ["EXP_SEED"] = str(s); import common; importlib.reload(common)
            ds = seeded_name(base, s); df, X, src, dst, tr, ho, s_idx, anc = common.load(ds)
            fam_all = df["fam"].values; fam = fam_all[s_idx]; ben = fam == "BENIGN"; pay = fam == "INJ_attack"; ramp = fam == "INJ_ramp"
            t = df["t"].values[s_idx]; t_inj = t[pay | ramp].min(); day = common.stream_days(df, s_idx)
            for m in E4_M:
                r = run_metrics(base, m, s)
                if r is None: continue
                z = np.load(common.RESULTS + f"{ds}__{m}__s{s}.npz"); sc, fl = z["score"], z["flag"].astype(bool)
                info = json.load(open(common.RESULTS + f"{ds}__{m}__s{s}.json"))
                rows.append(dict(dataset=base, method=m, seed=s, F1=r["F1"], FPR=r["FPR"], AUROC=r["AUROC"], ramp_recall=float(fl[ramp].mean()),
                                 payload_recall=float(fl[pay].mean()), payload_at1fpr=float((sc[pay] > np.quantile(sc[ben], 0.99)).mean()),
                                 absorbed=r["absorption_day"] >= 0, fpr_before=float(fl[ben & (t < t_inj)].mean()),
                                 fpr_during=float(fl[ben & (t >= t_inj)].mean()), labels=r["labels"]))
                if m in GROUPED:
                    c = label_channel(X, fam_all, s_idx, day, fl, s, int(info["cfg"]["budget"]), m == "abl_evid_rep")
                    assert c["decisions"] == int(info["labels_used"]), (ds, m, c["decisions"], info["labels_used"])
                    chan.append(dict(dataset=base, method=m, seed=s, n_injected=int((pay | ramp).sum()), **c))
            print("e4", ds, flush=True)
    run = pd.DataFrame(rows); ch = pd.DataFrame(chan)
    run.to_csv(os.path.join(OUT, "e4_runs.csv"), index=False); ch.to_csv(os.path.join(OUT, "e4_label_channel.csv"), index=False)
    g = run.groupby(["dataset", "method"]); t1 = g[["F1", "FPR", "ramp_recall", "payload_recall", "payload_at1fpr", "fpr_before", "fpr_during", "AUROC", "labels"]].mean()
    t1["absorbed_seeds"] = g["absorbed"].sum(); t1["payload_min"] = g["payload_recall"].min()
    ch["share_dec_on_injected"] = ch["dec_on_injected_rep"] / ch["decisions"]; ch["share_injected_in_P"] = ch["injected_into_P"] / ch["n_injected"]
    with open(os.path.join(OUT, "e4_valid_poisoning.txt"), "w", encoding="utf-8") as fh:
        fh.write(__doc__ + "\nE4 per scenario and method (mean over 5 seeds; payload_min = worst seed)\n" + t1.round(3).to_string() + "\n\n")
        fh.write("Label channel of the ERNP variants (mean over seeds)\n" + ch.drop(columns="seed").groupby(["dataset", "method"]).mean().round(3).to_string() + "\n")

def e5():
    rows, days = [], []
    for base in E5_DS:
        for s in range(5):
            os.environ["EXP_SEED"] = str(s); import common; importlib.reload(common)
            ds = seeded_name(base, s); df, X, src, dst, tr, ho, s_idx, anc = common.load(ds); del X
            fam = df["fam"].values[s_idx]; y = fam != "BENIGN"; day = common.stream_days(df, s_idx)
            tt = pd.to_datetime(df["t"].values[s_idx], unit="s")
            burst = (df["src"].values[s_idx] == "192.168.1.1") & (tt.strftime("%Y-%m-%d") == "2019-04-29") & (tt.hour == 12) if base == "ToN_ctx" else np.zeros(len(y), bool)
            for m in E5_M:
                r = run_metrics(base, m, s)
                if r is None: continue
                z = np.load(common.RESULTS + f"{ds}__{m}__s{s}.npz"); sc, fl = z["score"], z["flag"].astype(bool)
                row = {k: r[k] for k in ("dataset", "method", "seed", "F1", "MCC", "FPR", "recall", "zero_day_recall", "zd_recall_at1fpr", "AUROC", "labels")}
                if burst.any():
                    k = ~burst; tp, fp = (fl & y & k).sum(), (fl & ~y & k).sum(); fn, tn = (~fl & y & k).sum(), (~fl & ~y & k).sum()
                    den = np.sqrt(float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn))
                    row.update(AUROC_noburst=roc_auc_score(y[k], sc[k]), MCC_noburst=(float(tp) * tn - float(fp) * fn) / den if den else 0.0, FPR_noburst=fp / max(1, fp + tn))
                    for i, d in enumerate(dict.fromkeys(day)):
                        j = day == d
                        days.append(dict(method=m, seed=s, day_no=i, day=d, auroc=roc_auc_score(y[j], sc[j]), recall=float(fl[j & y].mean()), fpr=float(fl[j & ~y].mean())))
                rows.append(row)
            print("e5", ds, flush=True)
    run = pd.DataFrame(rows); run.to_csv(os.path.join(OUT, "e5_runs.csv"), index=False)
    dd = pd.DataFrame(days); dd.to_csv(os.path.join(OUT, "e5_ton_days.csv"), index=False)
    cols = ["F1", "MCC", "FPR", "zero_day_recall", "zd_recall_at1fpr", "AUROC", "labels", "AUROC_noburst", "MCC_noburst", "FPR_noburst"]
    piv = run.pivot_table(index=["dataset", "seed"], columns="method", values=["F1", "MCC", "zero_day_recall"])
    with open(os.path.join(OUT, "e5_delay.txt"), "w", encoding="utf-8") as fh:
        fh.write(__doc__ + "\nE5 per dataset and method (mean over 5 seeds; *_noburst: ToN without the gateway burst hour)\n")
        fh.write(run.groupby(["dataset", "method"])[cols].mean().round(3).to_string() + "\n\n")
        fh.write("Paired at the same delay (D = 1 day): ERNP - baseline, mean over seeds and #seeds ERNP > baseline\n")
        for b in ("ssf_causal_D1", "owad_D1"):
            for c in ("F1", "MCC", "zero_day_recall"):
                d = (piv[c]["ernp_D1"] - piv[c][b]).groupby(level=0).agg(["mean", lambda x: int((x > 0).sum()), "size"])
                fh.write(f"  ernp_D1 - {b} {c}: " + "; ".join(f"{k} {v['mean']:+.3f} ({int(v.iloc[1])}/{int(v['size'])})" for k, v in d.iterrows()) + "\n")
        fh.write("\nCost of the delay for each method (D1 - D0), mean over seeds\n")
        for a, b in (("ernp_D1", "ernp"), ("ernp_D2", "ernp"), ("ssf_causal_D1", "ssf_causal"), ("owad_D1", "owad")):
            for c in ("F1", "MCC", "zero_day_recall"):
                d = (piv[c][a] - piv[c][b]).groupby(level=0).mean()
                fh.write(f"  {a} - {b} {c}: " + "; ".join(f"{k} {v:+.3f}" for k, v in d.items()) + "\n")
        fh.write("\nToN per stream day (mean over seeds): AUROC within the day | recall | benign FPR\n")
        fh.write(dd.groupby(["method", "day"])[["auroc", "recall", "fpr"]].mean().round(2).unstack("day").to_string() + "\n")

if __name__ == "__main__":
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 300)
    which = sys.argv[1:] or ["e4", "e5"]
    if "e4" in which: e4()
    if "e5" in which: e5()
    for f in (["e4_valid_poisoning.txt"] if "e4" in which else []) + (["e5_delay.txt"] if "e5" in which else []):
        print(open(os.path.join(OUT, f), encoding="utf-8").read())
