"""Q1 revision: E3 (retraction under a fixed feedback schedule) and E2 (analyst budget in matched units).

E3 / mechanism control. rpl_* runs replay ERNP's own alerts -> groups -> decisions (ernp.py --replay ernp), so the
evidence, the prior draws (same rng sequence) and the GBM seed are identical to ERNP; only the model differs:
  ernp            full ERNP
  rpl_retr_none   same evidence, retention weight = 1 (no retraction)
  rpl_prior_none  same evidence, no novelty prior (GBM + analyst feedback)
Reported per dataset: paired differences over seeds, benign alerts per stream day, and FPR / zero-day recall by distance
to the benign evidence confirmed before each day (bins in units of eps; evidence = ERNP's, common to all three runs).

E2. B = 25 group decisions per day: full group labels (ernp_B25), at most 200 flows per day entering the evidence
(ernp_B25_cap200, = SSF's 200 labelled flows per day) and representatives only (abl_evid_rep_B25, 25 flows per day),
against SSF / OWAD with 200 and 25 flow labels per day. Units: decisions, labelled flows that enter the model.
Output: reports/e3_retraction.txt, e3_runs.csv, e3_regions.csv, e2_budget.txt, e2_runs.csv
"""
import os, sys, json, importlib, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import MiniBatchKMeans
from evaluate import seeded_name, run_metrics
from retraction_time_analysis import evidence

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
BASES = ["CIDDS_ctx", "ToN_ctx", "CIC17", "CIC18"]
E3 = ["ernp", "rpl_retr_none", "rpl_prior_none", "abl_retr_none", "abl_prior_none"]
E2 = ["ernp", "abl_evid_rep", "ernp_B25", "ernp_B25_cap200", "abl_evid_rep_B25", "ssf_causal", "ssf_causal_B25", "owad", "owad_B25"]
BINS = [(0, 1), (1, 3), (3, np.inf)]; CAP = 15000
COLS = ["F1", "MCC", "FPR", "recall", "zero_day_recall", "zd_recall_at1fpr", "AUROC"]

def flows_in_evidence(common, method, info, X, df, s_idx, flag, s):
    """labelled flows that entered the model: logged by new runs; for older group-label ERNP runs the groups are
    replayed from the saved alerts (same MiniBatchKMeans call as ernp.run, gated on labels_used)"""
    if "flows_in_evidence" in info: return int(info["flows_in_evidence"])
    cfg = info.get("cfg")
    if not isinstance(cfg, dict) or cfg.get("evidence") != "cluster": return int(info.get("labels_used", 0))   # one flow per label
    if info.get("audit"): return int(sum(a[2] for a in info["audit"]))
    day = common.stream_periods(df, s_idx, int(cfg.get("period", 24))); n = n_dec = 0
    for d in dict.fromkeys(day):
        idx = np.where(day == d)[0]; al = s_idx[idx][flag[idx]]; Bk = common.period_budget(d, int(cfg.get("period", 24)), int(cfg["budget"]))
        if len(al) < 2 or Bk <= 0: continue
        k = min(Bk, len(al)); lab = MiniBatchKMeans(k, random_state=s, n_init=3).fit(X[al]).labels_
        c = np.bincount(lab, minlength=k); n += int(c.sum()); n_dec += int((c > 0).sum())
    assert n_dec == int(info["labels_used"]), f"group replay gate failed: {n_dec} vs {info['labels_used']}"
    return n

def metrics(bases, methods, seeds=range(5)):
    rows = []
    for b in bases:
        for m in methods:
            for s in seeds:
                r = run_metrics(b, m, s)
                if r is None: continue
                info = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", f"{r['stream']}__{m}__s{s}.json")))
                os.environ["EXP_SEED"] = str(s); import common; importlib.reload(common)
                df, X, src, dst, tr, ho, s_idx, anc = common.load(r["stream"]); n_days = len(set(common.stream_days(df, s_idx)))
                y = df["fam"].values[s_idx] != "BENIGN"; fl = np.load(common.RESULTS + f"{r['stream']}__{m}__s{s}.npz")["flag"].astype(bool)
                r.update(benign_alerts_per_day=float((fl & ~y).sum()) / n_days, decisions_per_day=r["labels"] / n_days,
                         flows_per_day=flows_in_evidence(common, m, info, X, df, s_idx, fl, s) / n_days)
                rows.append({k: v for k, v in r.items() if not k.startswith("rec::")})
    return pd.DataFrame(rows)

def regions(base, s):
    """FPR / zero-day recall by distance to ERNP's confirmed benign evidence (same evidence for ernp and rpl_*)"""
    os.environ["EXP_SEED"] = str(s); import common; importlib.reload(common)
    ds = seeded_name(base, s); df, X, src, dst, tr, ho, s_idx, anc = common.load(ds)
    fam_all = df["fam"].values; fam = fam_all[s_idx]; y = fam != "BENIGN"
    zd = y & ~np.isin(fam, list(set(fam_all[anc & (fam_all != "BENIGN")])))
    day = common.stream_days(df, s_idx); order = list(dict.fromkeys(day)); rng = np.random.default_rng(s)
    flags = {v: np.load(common.RESULTS + f"{ds}__{v}__s{s}.npz")["flag"].astype(bool) for v in E3[:3]}
    info = json.load(open(common.RESULTS + f"{ds}__ernp__s{s}.json"))
    ev_idx, ev_day, dec = evidence(common, ds, s, X, fam_all, s_idx, day, order, flags["ernp"])
    assert sum(n for _, n, _ in dec) == info["labels_used"], "evidence replay gate failed"
    rows = []
    for i, d in enumerate(order[1:], 1):
        idx = np.where(day == d)[0]; ben = idx[~y[idx]]; zdi = idx[zd[idx]]
        ben = ben[rng.choice(len(ben), min(CAP, len(ben)), replace=False)] if len(ben) else ben
        zdi = zdi[rng.choice(len(zdi), min(CAP, len(zdi)), replace=False)] if len(zdi) else zdi
        prev = ev_idx[np.isin(ev_day, order[:i])]; sel = np.r_[ben, zdi]; isb = np.r_[np.ones(len(ben), bool), np.zeros(len(zdi), bool)]
        dist = (NearestNeighbors(n_neighbors=1).fit(X[prev]).kneighbors(X[s_idx[sel]])[0][:, 0] / info["eps"]
                if len(prev) else np.full(len(sel), np.inf))
        for v, fl in flags.items():
            for lo, hi in BINS:
                m = (dist >= lo) & (dist < hi); f = fl[sel]
                rows.append(dict(dataset=base, seed=s, method=v, day_no=i, bin=f"{lo}-{hi}", n_benign=int((m & isb).sum()),
                                 fp=int((m & isb & f).sum()), n_zd=int((m & ~isb).sum()), tp_zd=int((m & ~isb & f).sum())))
    return rows

def paired(run, base_m, other):
    """mean difference (base - other) over seeds present for both, and #seeds where base > other"""
    a = run[run.method == base_m].set_index(["dataset", "seed"]); b = run[run.method == other].set_index(["dataset", "seed"])
    j = a[COLS].join(b[COLS], rsuffix="_o", how="inner"); out = []
    for ds, g in j.groupby(level=0):
        r = dict(dataset=ds, vs=other, n=len(g))
        for c in COLS: r[f"d_{c}"] = (g[c] - g[c + "_o"]).mean(); r[f"win_{c}"] = int((g[c] > g[c + "_o"]).sum())
        out.append(r)
    return pd.DataFrame(out)

def main():
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 300)
    # ---------------- E3
    run = metrics(BASES, E3); run.to_csv(os.path.join(OUT, "e3_runs.csv"), index=False)
    reg = pd.DataFrame([r for b in BASES for s in range(5) for r in regions(b, s)]); reg.to_csv(os.path.join(OUT, "e3_regions.csv"), index=False)
    agg = reg.groupby(["dataset", "method", "bin"])[["n_benign", "fp", "n_zd", "tp_zd"]].sum()
    agg["FPR"] = agg.fp / agg.n_benign.clip(lower=1); agg["zd_recall"] = agg.tp_zd / agg.n_zd.clip(lower=1)
    with open(os.path.join(OUT, "e3_retraction.txt"), "w", encoding="utf-8") as fh:
        fh.write(__doc__ + "\nE3 per method (mean over seeds)\n")
        fh.write(run.groupby(["dataset", "method"])[COLS + ["benign_alerts_per_day", "decisions_per_day", "flows_per_day"]].mean().round(3).to_string() + "\n\n")
        fh.write("Paired differences ernp - X under the same feedback schedule (win = #seeds ernp > X)\n")
        fh.write(pd.concat([paired(run, "ernp", o) for o in ("rpl_retr_none", "rpl_prior_none")]).round(3).to_string(index=False) + "\n\n")
        fh.write("By distance to confirmed benign evidence (pooled over seeds and days; evidence common to the three runs)\n")
        fh.write(agg.round(4).to_string() + "\n")
    # ---------------- E2
    r2 = metrics(["ToN_ctx", "CIC18"], E2); r2.to_csv(os.path.join(OUT, "e2_runs.csv"), index=False)
    with open(os.path.join(OUT, "e2_budget.txt"), "w", encoding="utf-8") as fh:
        fh.write(__doc__ + "\nE2 (mean over seeds; n = seeds available)\n")
        g = r2.groupby(["dataset", "method"]); t = g[COLS + ["decisions_per_day", "flows_per_day", "benign_alerts_per_day"]].mean()
        t.insert(0, "n", g.size()); fh.write(t.round(3).to_string() + "\n\n")
        fh.write("Same seeds only (seeds where every E2 method exists)\n")
        ok = r2.groupby(["dataset", "seed"]).method.nunique(); keep = ok[ok == ok.max()].index
        r3 = r2.set_index(["dataset", "seed"]).loc[lambda x: x.index.isin(keep)].reset_index()
        g = r3.groupby(["dataset", "method"]); t = g[COLS + ["decisions_per_day", "flows_per_day"]].mean(); t.insert(0, "n", g.size())
        fh.write(t.round(3).to_string() + "\n")
    for f in ("e3_retraction.txt", "e2_budget.txt"): print(open(os.path.join(OUT, f), encoding="utf-8").read())

if __name__ == "__main__":
    main()
