"""E1: zero-day detection BEFORE feedback vs adaptation AFTER feedback (RQ1 streams, seeds 0-4). No model is re-trained.

Every causal method scores a whole period before it learns from it (predict-then-learn, daily), so the flows of a
zero-day family in the period of its FIRST appearance are scored before any label of that family can reach any model
("first-period recall": exact and identical for every method, whatever its query rule).

For ERNP variants the feedback exposure of each family is reconstructed exactly from the saved decisions (same
MiniBatchKMeans call as ernp.run; gate: reconstructed decisions == labels_used). A family is EXPOSED at the end of the
first period in which any of its flows enters the model through an analyst decision, whatever the group label
(also when it is laundered into the benign evidence). Pre-feedback flows = flows scored up to and including that
period; flows of a never-exposed family are all pre-feedback. Post-feedback = flows scored after it.

Time to first alert (TTFA) runs from the family's first flow to its first alerted flow; a family that is never
alerted is censored (reported as not detected, never as 0 h).
Output: reports/prefeedback_family.csv (one row per dataset x seed x method x zero-day family), reports/prefeedback.txt.
usage: python prefeedback_analysis.py [CIDDS_ctx ToN_ctx CIC17 CIC18]     (one process per dataset, then --report)
"""
import os, sys, json, importlib, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.cluster import MiniBatchKMeans
from evaluate import seeded_name

BASES = ["CIDDS_ctx", "ToN_ctx", "CIC17", "CIC18"]
REPLAY = ["ernp", "abl_retr_none", "abl_evid_rep"]                     # ERNP variants with group decisions (B = 200)
OTHER = ["ernp_B0", "ssf_causal", "owad", "owad_lr3", "memstream", "trident", "trident_t1", "cade", "cade_t1",
         "knn", "iforest", "ae_static"]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")

def exposure(s, X, fam_all, s_idx, day_no, n_days, flag, B, rep_only):
    """first period (day number) in which each family enters the model through an analyst decision; decisions count"""
    first, n_dec = {}, 0
    for d in range(n_days):
        idx = np.where(day_no == d)[0]; al = s_idx[idx][flag[idx]]
        if len(al) < 2: continue
        k = min(B, len(al)); km = MiniBatchKMeans(k, random_state=s, n_init=3).fit(X[al])
        for cl in range(k):
            mem = al[km.labels_ == cl]
            if not len(mem): continue
            rep = mem[np.argmin(np.linalg.norm(X[mem] - km.cluster_centers_[cl], axis=1))]; n_dec += 1
            for f in (set(fam_all[[rep]]) if rep_only else set(fam_all[mem])): first.setdefault(f, d)
    return first, n_dec

def analyse(base):
    rows, gate = [], []
    for s in range(5):
        t0 = time.time(); os.environ["EXP_SEED"] = str(s); import common; importlib.reload(common)
        ds = seeded_name(base, s); df, X, src, dst, tr, ho, s_idx, anc = common.load(ds)
        fam_all = df["fam"].values; fam = fam_all[s_idx]; t = df["t"].values[s_idx]; y = fam != "BENIGN"
        known = set(fam_all[anc & (fam_all != "BENIGN")]); zfam = sorted(set(fam[y]) - known)
        day = common.stream_days(df, s_idx); order = list(dict.fromkeys(day))
        day_no = pd.Series(range(len(order)), index=order)[day].values
        for v in REPLAY + OTHER:
            p = common.RESULTS + f"{ds}__{v}__s{s}"
            if not os.path.exists(p + ".npz"): print("missing", p, flush=True); continue
            r_ = np.load(p + ".npz"); flag = r_["flag"].astype(bool); sc = r_["score"]; info = json.load(open(p + ".json"))
            thr1 = np.quantile(sc[~y], 0.99)          # oracle common operating point: 1% FPR on stream benign (diagnostic)
            exp_day = None
            if v in REPLAY:
                cfg = info["cfg"] if isinstance(info["cfg"], dict) else eval(info["cfg"])
                exp_day, n_dec = exposure(s, X, fam_all, s_idx, day_no, len(order), flag, int(cfg["budget"]),
                                          cfg["evidence"] == "representative" or v == "abl_evid_rep")
                ok = n_dec == int(info["labels_used"])
                gate.append(dict(dataset=base, seed=s, method=v, reconstructed=n_dec, labels_used=info["labels_used"], ok=ok))
                if not ok: print("GATE FAIL", ds, v, n_dec, info["labels_used"], flush=True); exp_day = None
            for f in zfam:
                i = np.where(fam == f)[0]; d0 = day_no[i].min(); first = day_no[i] == d0
                ben0 = ~y & (day_no == d0); hit = i[flag[i]]
                r = dict(dataset=base, seed=s, method=v, family=f, n_flows=len(i), first_day=int(d0),
                         n_first=int(first.sum()), rec_first=float(flag[i][first].mean()),
                         fpr_first_day=float(flag[ben0].mean()) if ben0.any() else np.nan,
                         recall=float(flag[i].mean()), detected=bool(len(hit)),
                         ttfa_h=float((t[hit].min() - t[i].min()) / 3600) if len(hit) else np.nan,
                         detected_first_day=bool(flag[i][first].any()), rec_first_at1fpr=float((sc[i][first] > thr1).mean()))
                if exp_day is not None:
                    e = exp_day.get(f, np.inf); pre = day_no[i] <= e
                    r.update(exposure_day=e if np.isfinite(e) else np.nan, n_pre=int(pre.sum()), n_post=int((~pre).sum()),
                             rec_pre=float(flag[i][pre].mean()) if pre.any() else np.nan,
                             rec_post=float(flag[i][~pre].mean()) if (~pre).any() else np.nan)
                rows.append(r)
        print(f"done {ds} ({len(zfam)} zero-day families, {time.time() - t0:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(OUT, f"prefeedback_family_{base}.csv"), index=False)
        pd.DataFrame(gate).to_csv(os.path.join(OUT, f"prefeedback_gate_{base}.csv"), index=False)

def report():
    fr = pd.concat([pd.read_csv(os.path.join(OUT, f"prefeedback_family_{b}.csv")) for b in BASES])
    gt = pd.concat([pd.read_csv(os.path.join(OUT, f"prefeedback_gate_{b}.csv")) for b in BASES])
    fr.to_csv(os.path.join(OUT, "prefeedback_family.csv"), index=False); gt.to_csv(os.path.join(OUT, "prefeedback_gate.csv"), index=False)
    def per_run(g):                         # macro over families of one run, pooled over flows of one run
        o = dict(n_fam=len(g), macro_rec_first=g["rec_first"].mean(),
                 pooled_rec_first=(g["rec_first"] * g["n_first"]).sum() / g["n_first"].sum(),
                 fam_detected_first_day=g["detected_first_day"].mean(), fam_detected=g["detected"].mean(),
                 macro_rec_first_at1fpr=g["rec_first_at1fpr"].mean(),
                 pooled_rec_first_at1fpr=(g["rec_first_at1fpr"] * g["n_first"]).sum() / g["n_first"].sum(),
                 median_ttfa_h=g["ttfa_h"].median(), fpr_first_day=g["fpr_first_day"].mean(), macro_recall=g["recall"].mean())
        if "rec_pre" in g and g["n_pre"].notna().any():
            o.update(macro_rec_pre=g["rec_pre"].mean(), pooled_rec_pre=(g["rec_pre"].fillna(0) * g["n_pre"]).sum() / g["n_pre"].sum(),
                     macro_rec_post=g["rec_post"].mean(), fam_exposed=g["exposure_day"].notna().mean(),
                     share_flows_pre=g["n_pre"].sum() / g["n_flows"].sum())
        return o
    run = pd.DataFrame([dict(dataset=k[0], method=k[1], seed=k[2], **per_run(g)) for k, g in fr.groupby(["dataset", "method", "seed"])])
    agg = run.drop(columns="seed").groupby(["dataset", "method"]).mean()
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 200)
    with open(os.path.join(OUT, "prefeedback.txt"), "w", encoding="utf-8") as fh:
        fh.write(__doc__ + "\nGATE: %d/%d replayed runs reconstruct labels_used exactly\n\n" % (gt["ok"].sum(), len(gt)))
        fh.write("Mean over seeds of per-run statistics (macro = mean over the run's zero-day families)\n")
        fh.write(agg.round(3).to_string() + "\n\n")
        fh.write("Per family (mean over seeds): first-period recall\n")
        fh.write(fr.pivot_table(index=["dataset", "family"], columns="method", values="rec_first").round(3).to_string() + "\n\n")
        fh.write("Per family (mean over seeds), ERNP: first-period / pre-feedback / post-feedback recall, exposure day\n")
        e = fr[fr["method"] == "ernp"].groupby(["dataset", "family"])[["n_flows", "first_day", "n_first", "rec_first", "rec_pre", "rec_post", "exposure_day"]].mean()
        fh.write(e.round(3).to_string() + "\n")
    print(open(os.path.join(OUT, "prefeedback.txt"), encoding="utf-8").read())

if __name__ == "__main__":
    if sys.argv[1:] == ["--report"]: report()
    else:
        for b in (sys.argv[1:] or BASES): analyse(b)
