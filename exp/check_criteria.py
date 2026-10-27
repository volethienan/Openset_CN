"""Verdicts against the success criteria fixed before the campaign (read from reports/*_runs.csv of evaluate.py).

Operationalisation (decided before reading the full results):
  RQ1  ranking/tests only over ERNP + 5 SOTA under the causal protocol (ernp_ssfproto/ssf = protocol (b), new_ip = sanity).
       C1a  ERNP has the best mean rank (Friedman blocks = source dataset means) on F1 and zero-day recall @ 1% FPR.
       C1b  historical five-seed consistency is reported descriptively, not as independent-dataset significance.
            Inference revised 2026-10-23 uses dataset means and Holm; it is not retroactive pre-registration.
       C1c  mean F1 of ERNP >= best SOTA mean F1 - 0.01 on every dataset.
  RQ2  C2a  ERNP payload recall >= 0.9 x its recall on the same (real) payload family in the clean stream, every scenario
            (mean over seeds).
       C2b  >= 3/5 SOTA absorbed in at least one scenario; absorbed = the ramp was detected (>= 10% on some day) and then
            recall stayed < 10% (absorption_day >= 1), in the majority of seeds.
  RQ3  C3   the smallest ERNP budget whose mean F1 and mean zero-day recall @ 1% FPR both reach SSF(B=200) is <= 100,
            on ToN and on CIC18.
  RQ5  C5a  |mean F1 - default| < 0.02 for every tested setting (settings breaking it are listed);
       C5b  ERNP scoring throughput >= 100k flows/s (CPU).
"""
import os, sys, re, glob, numpy as np, pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare
from inference import stats, dataset_means
REP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
SOTA5 = ["ssf_causal", "owad", "memstream", "trident", "cade"]
PAYLOAD = {"injinit": "dos", "injresp": "dos", "injinit_bf": "bruteForce", "injresp_bf": "bruteForce"}

def rd(name):
    f = f"{REP}/{name}_runs.csv"; return pd.read_csv(f) if os.path.exists(f) else None

def yes(b): return "PASS" if b else "FAIL"

def mean_ranks(df, metric, methods):
    piv = dataset_means(df[df.method.isin(methods)], metric).dropna()
    fr = friedmanchisquare(*[piv[c] for c in piv.columns]).pvalue if len(piv) >= 3 else np.nan
    return piv.rank(axis=1, ascending=False).mean().sort_values(), fr, len(piv)

def rq1():
    d = rd("rq1")
    if d is None: return print("RQ1: no results")
    M = ["ernp"] + SOTA5; print("=" * 100 + "\nRQ1  zero-day detection vs 5 SOTA (causal protocol)")
    tab = d[d.method.isin(M + ["ernp_ssfproto", "ssf", "new_ip"])].groupby(["dataset", "method"])[
        ["F1", "zd_recall_at1fpr", "zero_day_recall", "FPR", "AUROC"]].agg(["mean", "std"]).round(3)
    print(tab.to_string())
    ok_rank = True
    for met in ("F1", "zd_recall_at1fpr"):
        r, fr, n = mean_ranks(d, met, M); ok_rank &= r.index[0] == "ernp"
        print(f"\nmean rank on {met} (n={n} blocks, Friedman p={fr:.2e}): {r.round(2).to_dict()}")
    print(f"C1a ERNP best mean rank on F1 and zero-day@1%FPR: {yes(ok_rank)}")
    n_sig = 0
    print('C1b below is historical five-seed consistency, not independent-dataset significance.')
    for ds, g in d.groupby("dataset"):
        piv = g.pivot_table(index="seed", columns="method", values="zd_recall_at1fpr")
        res = {}
        for b in SOTA5:
            x = piv[["ernp", b]].dropna(); dif = x["ernp"] - x[b]
            res[b] = (int((dif > 0).sum()), len(dif))
        allsig = all(w == n and n > 0 for w, n in res.values()); n_sig += allsig
        print(f"  {ds:10s} zd@1%FPR seed wins/n: " + ", ".join(f"{b} {w}/{n}" for b, (w, n) in res.items()))
    print(f'C1b descriptive consistency against all baselines in {n_sig} datasets; no population-level p-value from seeds.')
    m = d.groupby(["dataset", "method"])["F1"].mean().unstack(); ok = True
    for ds in m.index:
        best = m.loc[ds, SOTA5].max(); gap = m.loc[ds, "ernp"] - best; ok &= gap >= -0.01
        print(f"  {ds:10s} F1 ERNP {m.loc[ds, 'ernp']:.3f} vs best SOTA {best:.3f} ({m.loc[ds, SOTA5].idxmax()}): {gap:+.3f}")
    print(f"C1c F1 never more than 1 point below the best SOTA: {yes(ok)}")
    fpr = d.groupby(["dataset", "method"])["FPR"].mean().unstack(); ok2 = True
    for ds in m.index:                  # design note: on ToN alert-all already gives F1 ~ 0.70 -> F1 alone is not a claim
        valid = [b for b in SOTA5 if fpr.loc[ds, b] <= 0.5]
        if valid: ok2 &= m.loc[ds, "ernp"] - m.loc[ds, valid].max() >= -0.01
        deg = [b for b in SOTA5 if b not in valid]
        if deg: print(f"  {ds:10s} near alert-all SOTA (mean FPR > 50%): {deg}; best other: "
                      f"{m.loc[ds, valid].max():.3f} ({m.loc[ds, valid].idxmax()}); ERNP FPR {fpr.loc[ds, 'ernp']:.3f}")
    print(f"C1c' same, ignoring near alert-all operating points (reported separately, not a substitute): {yes(ok2)}")
    # Replaces historical pseudo-replicated tests; dataset is the inferential unit.
    for met in ("F1", "zd_recall_at1fpr"):
        t, _, meta = stats(d[d.method.isin(M)], metric=met)
        print(f'\nDataset-level inference {met}: {meta}\n{t.round(5).to_string()}')
    if {"ernp_ssfproto", "ssf"} <= set(d.method):
        print("protocol (b), SSF's native protocol: F1 ernp_ssfproto - ssf per dataset:",
              (m["ernp_ssfproto"] - m["ssf"]).round(3).to_dict())
    if "new_ip" in m.columns: print("sanity new_ip F1 per dataset:", m["new_ip"].round(3).to_dict())

def rq2():
    d, c = rd("rq2"), rd("rq1")
    if d is None or c is None: return print("RQ2: no results")
    print("=" * 100 + "\nRQ2  robustness to attacks on adaptation (CIDDS injection)")
    c = c[c.dataset == "CIDDS_ctx"].set_index(["method", "seed"])
    ok_a, absorbed = True, {m: [] for m in SOTA5}
    for ds, g in d.groupby("dataset"):
        v = ds.replace("CIDDS_ctx_", ""); fam = "rec::" + PAYLOAD[v]
        rows = []
        for m_, gm in g.groupby("method"):
            ref = np.array([c.loc[(m_, s), fam] if (m_, s) in c.index else np.nan for s in gm.seed])
            rows.append(dict(method=m_, payload=gm.payload_recall.mean(), clean_ref=np.nanmean(ref),
                             ratio=np.nanmean(gm.payload_recall.values / np.maximum(ref, 1e-9)),
                             absorbed_seeds=f"{(gm.absorption_day >= 1).sum()}/{len(gm)}",
                             blind_seeds=f"{(gm.absorption_day == 0).sum()}/{len(gm)}", F1=gm.F1.mean(), FPR=gm.FPR.mean()))
            if m_ in absorbed and (gm.absorption_day >= 1).sum() > len(gm) / 2: absorbed[m_].append(v)
        t = pd.DataFrame(rows).set_index("method"); print(f"\n[{v}] payload = {PAYLOAD[v]}\n{t.round(3).to_string()}")
        r = t.loc["ernp", "payload"] / max(t.loc["ernp", "clean_ref"], 1e-9); ok_a &= r >= 0.9
        print(f"  ERNP payload recall / clean reference = {r:.3f}")
    print(f"\nC2a ERNP keeps >= 90% of its clean payload recall in every scenario: {yes(ok_a)}")
    n_abs = sum(bool(v) for v in absorbed.values())
    print(f"SOTA absorbed (majority of seeds) in: {absorbed}")
    print(f"C2b >= 3/5 SOTA absorbed in at least one scenario: {n_abs}/5 -> {yes(n_abs >= 3)}")
    sw = rd("rq2_sweep")
    if sw is not None:
        sw["ramp_rate"] = sw.dataset.str.extract(r"_r(\d+)_q(\d+)$").fillna({0: "7", 1: "30"}).agg("d/".join, axis=1) + "/h"
        print("\nsweep (initiator, bruteForce): payload recall / absorbed seeds\n" +
              sw.groupby(["ramp_rate", "method"]).agg(payload=("payload_recall", "mean"),
                                                      absorbed=("absorption_day", lambda a: f"{(a >= 1).sum()}/{len(a)}"))
              .unstack("method").to_string())

def rq3():
    d, base, b0 = rd("rq3"), rd("rq1"), rd("rq4")
    if d is None or base is None: return print("RQ3: no results")
    print("=" * 100 + "\nRQ3  label efficiency (seeds 0-2)")
    seeds = sorted(d.seed.unique()); ok = True
    extra = [base[base.method.isin(["ernp", "ssf_causal", "owad"]) & base.seed.isin(seeds)].assign(
        method=lambda x: x.method + "_B200")]
    if b0 is not None: extra.append(b0[(b0.method == "ernp_B0") & b0.seed.isin(seeds)])
    d = pd.concat([d] + extra); d = d[d.dataset.isin(["ToN_ctx", "CIC18"])]
    d["family"] = d.method.str.replace(r"_B\d+$", "", regex=True); d["B"] = d.method.str.extract(r"_B(\d+)$")[0].fillna(200).astype(int)
    for ds, g in d.groupby("dataset"):
        t = g.groupby(["family", "B"])[["F1", "zd_recall_at1fpr", "labels"]].mean().unstack("family").round(3)
        print(f"\n[{ds}]\n{t.to_string()}")
        tgt = g[(g.family == "ssf_causal") & (g.B == 200)][["F1", "zd_recall_at1fpr"]].mean()
        e = g[g.family == "ernp"].groupby("B")[["F1", "zd_recall_at1fpr"]].mean()
        hit = e[(e.F1 >= tgt.F1) & (e.zd_recall_at1fpr >= tgt.zd_recall_at1fpr)].index
        bmin = int(hit.min()) if len(hit) else None; ok &= bmin is not None and bmin <= 100
        print(f"  SSF(B=200): F1 {tgt.F1:.3f}, zd@1%FPR {tgt.zd_recall_at1fpr:.3f} -> smallest ERNP budget reaching both: {bmin}")
    print(f"C3 ERNP reaches SSF(B=200) with B <= 100 on ToN and CIC18: {yes(ok)}")

def rq4():
    d = rd("rq4")
    if d is None: return print("RQ4: no results")
    print("=" * 100 + "\nRQ4  ablations (delta = variant - ERNP, mean over seeds; wins = #(dataset x seed) where ERNP is better)")
    for met in ("F1", "zd_recall_at1fpr"):
        piv = d.pivot_table(index=["dataset", "seed"], columns="method", values=met)
        rows = {}
        for v in [c for c in piv.columns if c != "ernp"]:
            x = piv[["ernp", v]].dropna(); dif = x[v] - x["ernp"]
            rows[v] = dict(**{ds: dif.xs(ds).mean() for ds in sorted(d.dataset.unique()) if ds in dif.index.get_level_values(0)},
                           seed_wins_descriptive=f"{(dif < 0).sum()}/{len(dif)}")
        print(f"\n{met}\n{pd.DataFrame(rows).T.round(4).to_string()}")
        t, _, meta = stats(d, metric=met)
        print(f'Dataset-level inference: {meta}\n{t.round(5).to_string()}')
    print("\nP2/P3: see reports/analysis_p23.txt")

def rq5():
    d, r1 = rd("rq5"), rd("rq1")
    if d is None: return print("RQ5: no results")
    print("=" * 100 + "\nRQ5  sensitivity (seeds 0-2) and cost")
    m = d.groupby(["dataset", "method"])["F1"].mean().unstack(); dl = m.sub(m["ernp"], axis=0).drop(columns="ernp")
    print(f"delta F1 vs default:\n{dl.round(4).to_string()}")
    bad = [(ds, v, round(x, 4)) for ds in dl.index for v, x in dl.loc[ds].items() if abs(x) >= 0.02]
    print(f"C5a all settings within 2 F1 points: {yes(not bad)}" + (f"  (outside: {bad})" if bad else ""))
    if r1 is not None and "t_score_flows_per_s" in r1:
        tp = r1[r1.method == "ernp"].groupby("dataset")["t_score_flows_per_s"].mean()
        print(f"ERNP scoring throughput (flows/s): {tp.round(0).to_dict()}")
        print(f"C5b throughput >= 100k flows/s: {yes((tp >= 1e5).all())}")
    wall = []
    for f in glob.glob(os.path.join(os.path.dirname(REP), "logs", "rq1_s*_runner.log")):
        for line in open(f):
            mm = re.match(r"OK\s+(\S+) (\S+) s(\d) \((\d+)s\)", line)
            if mm: wall.append(dict(stream=mm[1], method=mm[2], seed=int(mm[3]), sec=int(mm[4])))
    if wall:
        w = pd.DataFrame(wall); w["dataset"] = w.stream.str.replace(r"_s\d$", "", regex=True)
        print("wall-clock per run (s, RQ1, mean over seeds; jobs shared the machine):\n" +
              w.groupby(["dataset", "method"]).sec.mean().unstack().round(0).to_string())

def cadence():
    """added analysis (no pre-registered criterion): same labels/day, update every 24 h / 4 h / 1 h"""
    d, base = rd("cadence"), rd("rq1")
    if d is None or base is None: return print("CADENCE: no results")
    print("=" * 100 + "\nCADENCE study (seeds 0-2; 200 labels/day for every cadence; label-using methods)")
    seeds = sorted(d.seed.unique())
    d = pd.concat([d, base[base.method.isin(["ernp", "ssf_causal", "owad"]) & base.seed.isin(seeds)]])
    d["family"] = d.method.str.replace(r"_P\d+$", "", regex=True)
    d["period_h"] = d.method.str.extract(r"_P(\d+)$")[0].fillna(24).astype(int)
    for met in ("F1", "zd_recall_at1fpr", "FPR"):
        print(f"\n{met} (mean over seeds)\n" + d.groupby(["dataset", "period_h", "family"])[met].mean()
              .unstack("family").round(3).to_string())
    for h in (24, 4, 1):
        g = d[d.period_h == h]; piv = g.pivot_table(index=["dataset", "seed"], columns="family", values="F1")
        if {"ernp", "ssf_causal", "owad"} <= set(piv.columns):
            best = piv[["ssf_causal", "owad"]].max(1); dif = piv["ernp"] - best
            print(f"period {h:2d} h: ERNP - best(SSF, OWAD) F1 per dataset: {dif.groupby(level=0).mean().round(3).to_dict()}"
                  f"  (ERNP better in {(dif > 0).sum()}/{len(dif)} dataset x seed)")
    print("labels actually used (mean):\n" + d.groupby(["dataset", "period_h", "family"])["labels"].mean()
          .unstack("family").round(0).to_string())

def reviewer_checks():
    """answers to the objections a strict reviewer is expected to raise (added 28/09/2026, after RQ1)"""
    d, e = rd("rq1"), rd("rq1_ext")
    if d is None or e is None: return print("REVIEWER CHECKS: no results")
    print("=" * 100 + "\nREVIEWER CHECKS")
    x = pd.concat([d, e]).drop_duplicates(["dataset", "method", "seed"])
    t = x.groupby(["dataset", "method"])[["F1", "MCC", "zd_recall_at1fpr", "FPR"]].mean().round(3)
    print("\n(a) label fairness: ERNP with representative-only labels (abl_evid_rep, same label semantics as SSF)")
    for met in ("F1", "MCC", "zd_recall_at1fpr"):
        t, _, meta = stats(x[x.method.isin(['abl_evid_rep'] + SOTA5)], ours='abl_evid_rep', metric=met)
        print(f'  {met}: representative-only, dataset-level {meta}\n{t.round(5).to_string()}')
    print("\n(b) all methods (mean over seeds)\n" + t.unstack("dataset").to_string())
    for m in ("ssf_causal_E180", "ssf_causal_W20000"):
        if m in set(x.method):
            g = x[x.method.isin(["ssf_causal", m, "ernp"])]; sd = sorted(g[g.method == m].seed.unique())
            print(f"\n(c) SSF with the authors' settings ({m}, seeds {sd}) vs our SSF config and ERNP, F1 / zd@1%FPR:\n" +
                  g[g.seed.isin(sd)].groupby(["dataset", "method"])[["F1", "zd_recall_at1fpr"]].mean().unstack().round(3).to_string())
    sc = rd("scale")
    if sc is not None:
        print("\n(d) scale check: CICIDS2017 without sampling (2.1M flows, seed 0) vs the 300k-flow streams (mean of 5 seeds)")
        ref = d[d.dataset == "CIC17"].groupby("method")[["F1", "zd_recall_at1fpr"]].mean()
        print(sc.set_index("method")[["F1", "zd_recall_at1fpr"]].join(ref, rsuffix="_300k").round(3).to_string())
    b = os.path.join(os.path.dirname(os.path.dirname(REP)), 'exp', 'reports', 'access', 'benchmark_k5.json')
    if os.path.exists(b):
        import json
        bench = json.load(open(b))
        print(f'\nExplicit K={bench["n_prior"]} score-only benchmark, threads={bench["threads"]}, median after warmup')
        print(pd.DataFrame([{k: row[k] for k in ['dataset', 'batch_size', 'initial_fit_s', 'median_flows_per_s']}
                            for row in bench['rows']]).round(2).to_string(index=False))
    else:
        print('\nK5 cost benchmark not yet available. Historical copied K1 costs are not used as K5 evidence.')

def robust_stats(B=10000):
    """reviewer check W4 (pseudo-replication): seeds are sub-samples of the same dataset, so (dataset x seed) pooling
    overstates n. Report (i) per-dataset mean difference ERNP - SOTA with a 95% bootstrap CI over seeds and
    (ii) dataset-level tests on per-dataset means (n = 4 datasets, honest but low power)."""
    d = rd("rq1")
    if d is None: return print("ROBUST STATS: no results")
    print("=" * 100 + "\nROBUST STATISTICS (dataset-level; seeds are not independent samples)")
    rng = np.random.default_rng(0)
    for met in ("F1", "MCC", "zd_recall_at1fpr"):
        rows = []
        for ds, g in d.groupby("dataset"):
            piv = g.pivot_table(index="seed", columns="method", values=met)
            for b in SOTA5:
                x = (piv["ernp"] - piv[b]).dropna().values
                bs = rng.choice(x, (B, len(x))).mean(1)
                rows.append(dict(dataset=ds, sota=b, diff=x.mean(), ci_lo=np.quantile(bs, .025), ci_hi=np.quantile(bs, .975),
                                 excludes_0=bool(np.quantile(bs, .025) > 0 or np.quantile(bs, .975) < 0)))
        t = pd.DataFrame(rows)
        print(f"\n{met}: ERNP - SOTA per dataset, mean over seeds [95% bootstrap CI]")
        print(t.pivot(index="dataset", columns="sota", values="diff").round(3).astype(str).add(" [") .add(
              t.pivot(index="dataset", columns="sota", values="ci_lo").round(3).astype(str)).add(", ").add(
              t.pivot(index="dataset", columns="sota", values="ci_hi").round(3).astype(str)).add("]").to_string())
        dm = d[d.method.isin(["ernp"] + SOTA5)].groupby(["dataset", "method"])[met].mean().unstack()
        ranks = dm.rank(axis=1, ascending=False).mean().sort_values()
        fr = friedmanchisquare(*[dm[c] for c in dm.columns]).pvalue
        wins = {b: int((dm["ernp"] > dm[b]).sum()) for b in SOTA5}
        print(f"dataset-level (n=4): mean rank {ranks.round(2).to_dict()}; Friedman p={fr:.3f}; datasets where ERNP wins: {wins}")

def rq2lb():
    d, ref = rd("rq2lb"), rd("rq2")
    if d is None: return print("RQ2-LOW-BUDGET: no results")
    print("=" * 100 + "\nRQ2 at 25 labels/day (vs 200 in RQ2): payload recall / absorbed seeds / FPR")
    x = pd.concat([d, ref[ref.method.isin(["ernp", "ssf_causal", "owad"])]])
    x["B"] = np.where(x.method.str.endswith("_B25"), 25, 200); x["family"] = x.method.str.replace("_B25", "", regex=False)
    print(x.groupby(["dataset", "family", "B"]).agg(payload=("payload_recall", "mean"),
          absorbed=("absorption_day", lambda a: f"{(a >= 1).sum()}/{len(a)}"), FPR=("FPR", "mean"), F1=("F1", "mean"))
          .round(3).to_string())

def noise():
    d, r1, r2 = rd("noise"), rd("rq1"), rd("rq2")
    if d is None: return print("NOISY ANALYST: no results")
    print("=" * 100 + "\nNOISY ANALYST (each label flipped w.p. p; seeds 0-2)")
    seeds = sorted(d.seed.unique())
    clean = pd.concat([r for r in (r1, r2) if r is not None]); clean = clean[clean.method.isin(["ernp", "ssf_causal", "owad"]) & clean.seed.isin(seeds)]
    x = pd.concat([d, clean]); x["p"] = x.method.str.extract(r"_N(\d+)$")[0].fillna(0).astype(int)
    x["family"] = x.method.str.replace(r"_N\d+$", "", regex=True)
    cols = [c for c in ("F1", "MCC", "payload_recall") if c in x]
    print(x.groupby(["dataset", "family", "p"])[cols].mean().round(3).to_string())

if __name__ == "__main__":
    for f in (rq1, rq2, rq3, rq4, rq5, cadence, reviewer_checks, robust_stats, rq2lb, noise): f()
