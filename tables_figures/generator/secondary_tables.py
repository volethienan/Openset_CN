"""LaTeX tables of the secondary experiments from the *_runs.csv reports (no number is typed by hand).
Run through tables_figures/build_tables.py, which sets REP (exp/reports) and OUT on the line that starts with `REP =`.
"""
import os, sys, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP = os.environ.get("ERNP_REP", os.path.join(ROOT, "..", "exp", "reports")); OUT = os.path.join(ROOT, "paper", "tables"); os.makedirs(OUT, exist_ok=True)
DS = ["CIDDS_ctx", "ToN_ctx", "CIC17", "CIC18"]                       # testbeds: RQ1-RQ5
REAL = ["CTU13", "SIMARGL", "APPRAISE"]; DS7 = DS + REAL               # real networks: RQ1
DSN = {"CIDDS_ctx": "CIDDS-001", "ToN_ctx": "NF-ToN-IoT-v3", "CIC17": "CIC-IDS2017$^\\dagger$", "CIC18": "CSE-CIC-IDS2018",
       "CTU13": "CTU-13", "SIMARGL": "SIMARGL2021", "APPRAISE": "APPRAISE"}
DSS = {"CIDDS_ctx": "CIDDS", "ToN_ctx": "ToN", "CIC17": "CIC17$^\\dagger$", "CIC18": "CIC18", "CTU13": "CTU-13",
       "SIMARGL": "SIMARGL", "APPRAISE": "APPRAISE"}                    # short names in the seven-dataset tables
def grp7(first, per=1, extra=0):
    """header row grouping the seven dataset columns into testbeds and real networks; first = index of the first dataset
    column, per = columns per dataset, extra = columns after the datasets (e.g. Mean)"""
    a, b, c = first, first + 4 * per - 1, first + 7 * per - 1
    return (" & " * (first - 1) + f"\\multicolumn{{{4 * per}}}{{c}}{{Testbeds}} & \\multicolumn{{{3 * per}}}{{c}}{{Real networks}}" +
            " & " * extra + "\\\\\n" + f"\\cmidrule(lr){{{a}-{b}}}\\cmidrule(lr){{{b + 1}-{c}}}\n")
SOTA = ["ssf_causal", "owad", "cade", "trident", "memstream"]
NAME = {"ernp": "\\textbf{ERNP}", "ssf_causal": "SSF", "owad": "OWAD", "cade": "CADE", "trident": "Trident",
        "memstream": "MemStream", "knn": "$k$NN", "iforest": "IForest", "ae_static": "AE", "abl_prior_none": "GBM+AL",
        "abl_evid_rep": "ERNP (rep.-only labels)", "new_ip": "new-IP", "owad_lr3": "OWAD (lr $10^{-3}$)"}
rd = lambda f: pd.read_csv(os.path.join(REP, f))

def fmt(m, s=None, bold=False):
    t = f"{m:.3f}" + (f"$\\pm${s:.3f}" if s is not None and not np.isnan(s) else "")
    return f"\\textbf{{{t}}}" if bold else t

def write(name, body):
    open(os.path.join(OUT, name), "w", encoding="utf-8").write(body); print("wrote", name)

# ------------------------------------------------------------------ Table: RQ1 main
r1, ext = rd("rq1_runs.csv"), rd("rq1_ext_runs.csv"); r4 = rd("rq4_runs.csv"); real = rd("real_runs.csv")
real = real[real.dataset.isin(REAL)]
d = pd.concat([r1, ext, r4[r4.method.isin(["abl_prior_none", "abl_evid_rep"])], rd("owad_lr3_rq1.csv"), real]).drop_duplicates(["dataset", "method", "seed"])
M = ["ernp", "abl_evid_rep", "ssf_causal", "owad", "owad_lr3", "cade", "trident", "memstream", "abl_prior_none", "knn", "iforest", "ae_static"]
rows = []
for met, lab in (("F1", "F1"), ("MCC", "MCC"), ("zd_recall_at1fpr", "Zero-day recall @1\\%FPR")):
    g = d[d.method.isin(M)].groupby(["dataset", "method"])[met].agg(["mean", "std"])
    rows.append(f"\\midrule\n\\multicolumn{{{len(DS7) + 2}}}{{l}}{{\\emph{{{lab}}}}}\\\\")
    for m in M:
        cells = []
        for ds in DS7:
            best = g.loc[ds]["mean"].max()
            mu, sd = g.loc[(ds, m)]
            cells.append(fmt(mu, sd, bold=np.isclose(mu, best)))
        avg = g.xs(m, level="method")["mean"].mean()
        bestavg = max(g.xs(mm, level="method")["mean"].mean() for mm in M)
        rows.append(f"{NAME[m]} & " + " & ".join(cells) + f" & {fmt(avg, bold=np.isclose(avg, bestavg))}\\\\")
write("rq1_main.tex", "\\begin{tabular}{l" + "c" * (len(DS7) + 1) + "}\n\\toprule\n" + grp7(2, extra=1) + "Method & " +
      " & ".join(DSS[x] for x in DS7) + " & Mean\\\\\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------ Table: per-dataset effect sizes with bootstrap CI + tests over datasets
from scipy.stats import friedmanchisquare, wilcoxon
rng = np.random.default_rng(0); rows = []
for ds in DS7:
    piv = d[d.dataset == ds].pivot_table(index="seed", columns="method", values="F1")
    cells = []
    for b in SOTA:
        x = (piv["ernp"] - piv[b]).dropna().values; bs = rng.choice(x, (10000, len(x))).mean(1)
        lo, hi = np.quantile(bs, [.025, .975]); sig = lo > 0 or hi < 0
        t = f"{x.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]"
        cells.append(f"\\textbf{{{t}}}" if sig and x.mean() > 0 else (f"\\underline{{{t}}}" if sig else t))
    rows.append(("\\midrule\n" if ds == REAL[0] else "") + f"{DSN[ds]} & " + " & ".join(cells) + "\\\\")
STAT = {}                                    # tests over the seven datasets (per-dataset means; seeds are sub-samples)
for met in ("F1", "MCC", "zd_recall_at1fpr", "AUROC"):
    dm_ = d[d.method.isin(["ernp"] + SOTA)].groupby(["dataset", "method"])[met].mean().unstack().loc[DS7, ["ernp"] + SOTA]
    STAT[met] = dict(fr=friedmanchisquare(*[dm_[c] for c in dm_.columns]).pvalue, rank=dm_.rank(axis=1, ascending=False).mean(),
                     wins={b: int((dm_["ernp"] > dm_[b]).sum()) for b in SOTA},
                     p={b: wilcoxon(dm_["ernp"], dm_[b], alternative="greater").pvalue for b in SOTA})
ranks, fr = STAT["F1"]["rank"], STAT["F1"]["fr"]
def holm(ps):
    """Holm step-down adjusted p-values"""
    out, run = {}, 0.0
    for i, (k, v) in enumerate(sorted(ps.items(), key=lambda z: z[1])): run = max(run, min(1.0, (len(ps) - i) * v)); out[k] = run
    return out
hp = holm(STAT["F1"]["p"])
write("rq1_effects.tex", "\\begin{tabular}{l" + "c" * len(SOTA) + "}\n\\toprule\nDataset & " + " & ".join(NAME[b] for b in SOTA) +
      "\\\\\n\\midrule\n" + "\n".join(rows) + "\n\\midrule\nMean rank (ERNP " + f"{ranks['ernp']:.2f})" + " & " +
      " & ".join(f"{ranks[b]:.2f}" for b in SOTA) + "\\\\\nDatasets won by ERNP & " + " & ".join(f"{STAT['F1']['wins'][b]}/7" for b in SOTA) +
      "\\\\\nWilcoxon $p$ (Holm) & " + " & ".join(f"{STAT['F1']['p'][b]:.3f} ({hp[b]:.3f})" for b in SOTA) + "\\\\\n\\bottomrule\n\\end{tabular}\n")
open(os.path.join(OUT, "macros.tex"), "w").write(
    "".join(f"\\newcommand{{\\{n}}}{{{STAT[m]['fr']:.3f}}}\n" for n, m in (("friedmanp", "F1"), ("friedmanpmcc", "MCC"),
                                                                         ("friedmanpzd", "zd_recall_at1fpr"), ("friedmanpauc", "AUROC"))))

# ------------------------------------------------------------------ Table: SOTA at their best configuration + fidelity
cad = rd("cadence_runs.csv"); t1 = rd("t1_runs.csv")
x = pd.concat([r1, ext, cad, t1, rd("owad_lr3_rq1.csv"), real]).drop_duplicates(["dataset", "method", "seed"])
FAM = {"ssf_causal": "SSF", "ssf_causal_E180": "SSF", "ssf_causal_W20000": "SSF", "ssf_causal_P4": "SSF", "ssf_causal_P1": "SSF",
       "owad": "OWAD", "owad_P4": "OWAD", "owad_P1": "OWAD", "owad_lr3": "OWAD", "memstream": "MemStream", "memstream_adapt": "MemStream",
       "trident": "Trident", "trident_t1": "Trident", "cade": "CADE", "cade_t1": "CADE", "ernp": "ERNP", "ernp_P4": "ERNP", "ernp_P1": "ERNP", "knn": "kNN"}
x = x[x.method.isin(FAM)].assign(fam=lambda z: z.method.map(FAM))
m = x.groupby(["dataset", "fam", "method"])["F1"].mean().reset_index()
best = m.loc[m.groupby(["dataset", "fam"])["F1"].idxmax()].pivot(index="dataset", columns="fam", values="F1")
order = ["ERNP", "SSF", "CADE", "Trident", "OWAD", "MemStream", "kNN"]
best = best.loc[DS7]
rows = [("\\midrule\n" if ds == REAL[0] else "") + f"{DSN[ds]} & " + " & ".join(fmt(best.loc[ds, f], bold=np.isclose(best.loc[ds, f], best.loc[ds].max())) for f in order) + "\\\\" for ds in DS7]
rows.append("\\midrule\nMean & " + " & ".join(fmt(best[f].mean(), bold=np.isclose(best[f].mean(), best.mean().max())) for f in order) + "\\\\")
write("sota_best.tex", "\\begin{tabular}{l" + "c" * len(order) + "}\n\\toprule\nDataset & " + " & ".join(order).replace("kNN", "$k$NN") +
      "\\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------ Table: RQ2 (payload recall, FPR) at 200 and 25 labels/day + counterfactual
gate = rd("rq2gate_runs.csv")      # review 2: other evidence-gated designs (GBM+AL, ERNP without retraction)
r2 = pd.concat([rd("rq2_runs.csv"), gate[~gate.dataset.str.contains("camo")]]).drop_duplicates(["dataset", "method", "seed"])
lb = rd("rq2lb_runs.csv")
SC = [("CIDDS_ctx_injinit", "init/DoS"), ("CIDDS_ctx_injresp", "resp/DoS"), ("CIDDS_ctx_injinit_bf", "init/BF"), ("CIDDS_ctx_injresp_bf", "resp/BF")]
MR = [("ernp", 200, "ERNP"), ("ernp_B25", 25, "ERNP"), ("abl_evid_rep", 200, "ERNP, rep.-only labels"),
      ("abl_retr_none", 200, "ERNP, no retraction"), ("abl_prior_none", 200, "GBM+AL (no prior)"),
      ("abl_self", 0, "ERNP-self (counterfactual)"), ("ssf_causal", 200, "SSF"),
      ("ssf_causal_B25", 25, "SSF"), ("owad", 200, "OWAD"), ("owad_B25", 25, "OWAD"), ("cade", 0, "CADE"), ("trident", 0, "Trident"), ("memstream", 0, "MemStream"),
      ("memstream_adapt", 0, "MemStream, adaptive $\\beta$")]
x = pd.concat([r2, lb]); rows = []
for mth, B, lab in MR:
    cells = []
    for ds, _ in SC:
        g = x[(x.dataset == ds) & (x.method == mth)]
        cells.append(f"{g.payload_recall.mean():.3f} / {100 * g.FPR.mean():.1f}")
    rows.append(f"{lab} & {B if B else '--'} & " + " & ".join(cells) + "\\\\")
write("rq2.tex", "\\begin{tabular}{llcccc}\n\\toprule\nMethod & $B$ & " + " & ".join(s for _, s in SC) + "\\\\\n\\midrule\n" +
      "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------ Table: RQ4 ablations (delta F1 vs ERNP)
AB = [("abl_prior_none", "No novelty prior (GBM+AL)"), ("abl_prior_uniform", "Uniform prior (Sipple)"),
      ("abl_prior_marginal", "Product-of-marginals prior"), ("abl_prior_scarf", "Independent-donor corruption (SCARF/VIME)"), ("abl_retr_none", "No retraction"), ("abl_retr_hard", "Hard retraction"),
      ("abl_evid_rep", "Representative-only evidence"), ("abl_feat_flow", "Per-flow features only"), ("ernp_B0", "No online labels"),
      ("abl_self", "Self-evidence (counterfactual)")]
x = pd.concat([r4, ext[ext.method == "abl_self"]]).drop_duplicates(["dataset", "method", "seed"])
piv = x.pivot_table(index=["dataset", "seed"], columns="method", values="F1"); rows = []
for mth, lab in AB:
    dlt = (piv[mth] - piv["ernp"]).groupby(level=0).mean()
    rows.append(f"{lab} & " + " & ".join(f"{dlt[ds]:+.3f}" for ds in DS) + f" & {dlt.mean():+.3f}\\\\")
write("rq4.tex", "\\begin{tabular}{l" + "c" * (len(DS) + 1) + "}\n\\toprule\nVariant ($\\Delta$F1 vs. ERNP) & " +
      " & ".join(DSN[x_] for x_ in DS) + " & Mean\\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------ Table: RQ5 sensitivity
r5 = rd("rq5_runs.csv"); SE = [("sens_rho0.1", "$\\rho=0.1$"), ("sens_rho0.2", "$\\rho=0.2$"), ("sens_rho0.5", "$\\rho=0.5$"),
      ("sens_mass0.25", "$m=0.25$"), ("sens_mass0.5", "$m=0.5$"), ("sens_mass2", "$m=2$"), ("sens_eps0.5", "$\\epsilon\\times0.5$"),
      ("sens_eps2", "$\\epsilon\\times2$")]
piv = r5.pivot_table(index=["dataset", "seed"], columns="method", values="F1"); rows = []
D5 = ["CIDDS_ctx", "ToN_ctx", "CIC18"]
for mth, lab in SE:
    dlt = (piv[mth] - piv["ernp"]).groupby(level=0).mean()
    rows.append(f"{lab} & " + " & ".join(f"{dlt[ds]:+.3f}" for ds in D5) + "\\\\")
write("rq5.tex", "\\begin{tabular}{lccc}\n\\toprule\nSetting ($\\Delta$F1 vs. default) & " + " & ".join(DSN[x_] for x_ in D5) +
      "\\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------ Table: full-scale CIC-IDS2017 + cost
sc = rd("scale_runs.csv"); cost = rd("bench_cost.csv"); rows = []
for mth in ["ernp"] + SOTA:
    g = sc[sc.method == mth]
    t = cost[(cost.method == ("ernp_K5" if mth == "ernp" else mth))]   # clean benchmark of the K = 5 configuration
    rows.append(f"{NAME[mth]} & {fmt(g.F1.mean(), g.F1.std())} & {fmt(g.MCC.mean(), g.MCC.std())} & {fmt(g.zd_recall_at1fpr.mean(), g.zd_recall_at1fpr.std())} & "
                f"{t.init_s.mean():.0f} & {t.peak_mem_mb.mean():.0f}\\\\")
write("scale_cost.tex", "\\begin{tabular}{lccccc}\n\\toprule\n & \\multicolumn{3}{c}{CIC-IDS2017, full (2.1M flows, 3 seeds)} & \\multicolumn{2}{c}{Cost (sampled streams)}\\\\\n"
      "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}\nMethod & F1 & MCC & ZD recall@1\\%FPR & Initial training (s) & Peak RAM (MB)\\\\\n\\midrule\n" +
      "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------ Table: ranking quality independent of the decision rule
M2 = ["ernp"] + SOTA
# review 2: the representative-only variant and the classical references are ranked too; bold is the best of ALL rows
MRK = ["ernp", "abl_evid_rep", "ssf_causal", "owad", "owad_lr3", "cade", "trident", "memstream", "knn", "iforest", "ae_static"]; g = d[d.method.isin(MRK)]; rows = []
for met, lab in (("AUROC", "AUROC"), ("AUPRC", "AUPRC"), ("recall_at1fpr", "Attack recall @1\\%FPR")):
    s = g.groupby(["dataset", "method"])[met].agg(["mean", "std"])
    rows.append(f"\\midrule\n\\multicolumn{{{len(DS7) + 2}}}{{l}}{{\\emph{{{lab}}}}}\\\\")
    avg = {m: s.xs(m, level="method")["mean"].loc[DS7].mean() for m in MRK}
    for m in MRK:
        cells = [fmt(*s.loc[(ds, m)], bold=np.isclose(s.loc[(ds, m)]["mean"], s.loc[ds]["mean"].max())) for ds in DS7]
        rows.append((f"\\cmidrule(lr){{1-{len(DS7) + 2}}}\n" if m == "knn" else "") +
                    f"{NAME[m]} & " + " & ".join(cells) + f" & {fmt(avg[m], bold=np.isclose(avg[m], max(avg.values())))}\\\\")
write("rq1_rank.tex", "\\begin{tabular}{l" + "c" * (len(DS7) + 1) + "}\n\\toprule\n" + grp7(2, extra=1) + "Method & " +
      " & ".join(DSS[x] for x in DS7) + " & Mean\\\\\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------ Table: matched update cadence (seeds 0-2)
x = pd.concat([cad, r1]).drop_duplicates(["dataset", "method", "seed"])
CR = [("ernp", "\\textbf{ERNP}", "24\\,h"), ("ernp_P4", "\\textbf{ERNP}", "4\\,h"), ("ernp_P1", "\\textbf{ERNP}", "1\\,h"),
      ("ssf_causal", "SSF", "24\\,h"), ("ssf_causal_P4", "SSF", "4\\,h"),
      ("owad", "OWAD", "24\\,h"), ("owad_P4", "OWAD", "4\\,h"), ("owad_P1", "OWAD", "1\\,h"),
      ("cade", "CADE", "--"), ("trident", "Trident", "--"), ("memstream", "MemStream", "--")]
s = x[x.method.isin([c[0] for c in CR])].groupby(["dataset", "method"])[["F1", "MCC"]].mean()
rows = []
for i, (mth, lab, per) in enumerate(CR):
    if i in (3, 8): rows.append("\\midrule")
    cells = []
    for ds in DS:
        for met in ("F1", "MCC"):
            v = s.loc[(ds, mth), met]; cells.append(fmt(v, bold=np.isclose(v, s.loc[ds][met].max())))
    rows.append(f"{lab} & {per} & " + " & ".join(cells) + "\\\\")
write("cadence.tex", "\\begin{tabular}{ll" + "cc" * len(DS) + "}\n\\toprule\n & & " +
      " & ".join(f"\\multicolumn{{2}}{{c}}{{{DSN[d]}}}" for d in DS) + "\\\\\n" +
      "".join(f"\\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(len(DS))) +
      "\nMethod & Update & " + " & ".join(["F1 & MCC"] * len(DS)) + "\\\\\n\\midrule\n" + "\n".join(rows) +
      "\n\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------ Table: zero-day recall vs FPR at each method's own decision rule
M3 = M2 + ["knn", "iforest", "ae_static"]          # Pareto over the detectors and the classical references
s = pd.concat([r1, ext, real]).drop_duplicates(["dataset", "method", "seed"])
s = s[s.method.isin(M3)].groupby(["dataset", "method"])[["zero_day_recall", "FPR"]].mean()
body = []
for panel in (DS, REAL):                                  # two stacked panels: testbeds, then real networks (4 column pairs each)
    body.append(" & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{DSN[d]}}}" for d in panel) + " & " * (2 * (4 - len(panel))) + "\\\\\n" +
                "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(panel))) +
                "\nMethod & " + " & ".join(["ZD rec. & FPR\\,\\%"] * len(panel)) + " & " * (2 * (4 - len(panel))) + "\\\\\n\\midrule")
    for m in M3:
        cells = []
        for ds in panel:
            z, f = s.loc[(ds, m)]
            dominated = any((s.loc[(ds, o)].zero_day_recall >= z) and (s.loc[(ds, o)].FPR <= f) and
                            ((s.loc[(ds, o)].zero_day_recall > z) or (s.loc[(ds, o)].FPR < f)) for o in M3 if o != m)
            cells.append(f"{z:.3f} & {100 * f:.2f}" + ("" if dominated else "$^\\star$"))
        if m == "knn": body.append("\\midrule")
        body.append(f"{NAME[m]} & " + " & ".join(cells) + " & " * (2 * (4 - len(panel))) + "\\\\")
    if panel is DS: body.append("\\midrule[\\heavyrulewidth]")
write("zd_fpr.tex", "\\begin{tabular}{l" + "cc" * 4 + "}\n\\toprule\n" + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------ Table: scale study, CSE-CIC-IDS2018 at 10x (4M flows)
x10 = rd("x10_runs.csv")   # campaign wall-clock omitted (shared hardware); controlled cost is in benchmark.tex
b = r1[(r1.dataset == "CIC18") & r1.method.isin(M2) & (r1.seed <= 2)].groupby("method")[["F1", "MCC"]].mean()
g = x10.groupby("method")[["F1", "MCC", "zero_day_recall", "FPR", "AUPRC"]].agg(["mean", "std"])
rows = []
for m in M2:
    f1, mcc = g.loc[m, ("F1", "mean")], g.loc[m, ("MCC", "mean")]
    best = lambda c: np.isclose(g.loc[m, (c, "mean")], g[(c, "mean")].max())
    rows.append(f"{NAME[m]} & {b.loc[m, 'F1']:.3f} & {fmt(f1, g.loc[m, ('F1', 'std')], bold=best('F1'))} & "
                f"{b.loc[m, 'MCC']:.3f} & {fmt(mcc, g.loc[m, ('MCC', 'std')], bold=best('MCC'))} & "
                f"{fmt(g.loc[m, ('zero_day_recall', 'mean')], bold=best('zero_day_recall'))} & {100 * g.loc[m, ('FPR', 'mean')]:.2f} & "
                f"{fmt(g.loc[m, ('AUPRC', 'mean')], bold=best('AUPRC'))}\\\\")
write("scale_x10.tex", "\\begin{tabular}{lccccccc}\n\\toprule\n & \\multicolumn{2}{c}{F1} & \\multicolumn{2}{c}{MCC} & "
      "\\multicolumn{3}{c}{4M flows}\\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-8}\n"
      "Method & 400k & 4M & 400k & 4M & ZD rec. & FPR\\,\\% & AUPRC\\\\\n\\midrule\n" + "\n".join(rows) +
      "\n\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------ Table: ToN sensitivity to the gateway burst hour
tb = rd("ton_burst_analysis.csv")
TB = [("ernp", "\\textbf{ERNP}, 24\\,h"), ("ernp_P4", "\\textbf{ERNP}, 4\\,h"), ("ssf_causal", "SSF, 24\\,h"),
      ("ssf_causal_P4", "SSF, 4\\,h"), ("owad", "OWAD"), ("cade", "CADE"), ("trident", "Trident"), ("memstream", "MemStream"),
      ("memstream_adapt", "MemStream, adapt. $\\beta$"), ("iforest", "IForest")]
g = tb.groupby(["method", "eval"])[["MCC", "FPR", "AUROC"]].mean()
rows = []
for m, lab in TB:
    a, w = g.loc[(m, "all flows")], g.loc[(m, "without burst hour")]
    rows.append(f"{lab} & {a.MCC:.3f} & {w.MCC:.3f} & {100 * a.FPR:.1f} & {100 * w.FPR:.1f} & {a.AUROC:.3f} & {w.AUROC:.3f}\\\\")
write("ton_burst.tex", "\\begin{tabular}{lcccccc}\n\\toprule\n & \\multicolumn{2}{c}{MCC} & \\multicolumn{2}{c}{FPR\\,\\%} & "
      "\\multicolumn{2}{c}{AUROC}\\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
      "Method & all & w/o & all & w/o & all & w/o\\\\\n\\midrule\n" + "\n".join(rows) +
      "\n\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------ Table: where retraction acts, under ONE feedback schedule
# (exp/e23_report.py: rpl_retr_none replays ERNP's alerts -> groups -> decisions, so only the retention weight differs)
rr, e3 = rd("e3_regions.csv"), rd("e3_runs.csv")
rr = rr[rr.method.isin(["ernp", "rpl_retr_none"])].copy(); rr["near"] = np.where(rr.bin == "3-inf", "far", "near")
def _rate(df, num, den, keys):
    s = df.groupby(keys + ["seed"])[[num, den]].sum(); return (s[num] / s[den].replace(0, np.nan)).groupby(keys).mean()
fpr_b = 100 * _rate(rr, "fp", "n_benign", ["dataset", "method", "bin"]); zd_n = _rate(rr, "tp_zd", "n_zd", ["dataset", "method", "near"])
apd = e3.groupby(["dataset", "method"]).benign_alerts_per_day.mean()
pf = e3.pivot_table(index=["dataset", "seed"], columns="method", values="F1"); dF = pf["ernp"] - pf["rpl_retr_none"]
arrow = lambda a, b, f: f"{a:{f}}$\\rightarrow${b:{f}}"
rows = []
for ds in DS:
    c = [arrow(fpr_b[(ds, "rpl_retr_none", b)], fpr_b[(ds, "ernp", b)], ".2f") for b in ("0-1", "1-3", "3-inf")]
    c.append(arrow(apd[(ds, "rpl_retr_none")], apd[(ds, "ernp")], ",.0f"))
    c += [arrow(zd_n[(ds, "rpl_retr_none", n)], zd_n[(ds, "ernp", n)], ".2f") for n in ("near", "far")]
    c.append(f"{dF[ds].mean():+.3f} ({int((dF[ds] > 0).sum())}/{len(dF[ds])})")
    rows.append(f"{DSN[ds]} & " + " & ".join(c) + "\\\\")
write("retraction.tex", "\\begin{tabular}{lccccccc}\n\\toprule\n & \\multicolumn{3}{c}{Benign FPR (\\%) at distance} & Benign alerts & "
      "\\multicolumn{2}{c}{Zero-day recall at distance} & $\\Delta$F1\\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){6-7}\n"
      "Dataset & $\\le\\epsilon$ & $\\epsilon$--$3\\epsilon$ & $>3\\epsilon$ & per day & $\\le3\\epsilon$ & $>3\\epsilon$ & (seeds won)\\\\\n\\midrule\n" +
      "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------ Table: both replays of ERNP's feedback schedule
# (rpl_retr_none: retention weight 1; rpl_prior_none: no prior; same alerts -> groups -> decisions as ERNP)
RP = [("ernp", "\\ERNP"), ("rpl_retr_none", "no retraction"), ("rpl_prior_none", "no prior")]
RC = [("F1", "F1", ".3f", 1), ("MCC", "MCC", ".3f", 1), ("FPR", "FPR\\,\\%", ".1f", 100), ("AUROC", "AUROC", ".3f", 1),
      ("zd_recall_at1fpr", "ZD@1\\%", ".3f", 1), ("benign_alerts_per_day", "Benign alerts/day", ",.0f", 1)]
em = e3.groupby(["dataset", "method"])[[c for c, *_ in RC]].mean()
assert (e3[e3.method.isin([m for m, _ in RP])].groupby(["dataset", "method"]).size() == 5).all(), "replay table needs 5 seeds per cell"
rows = []
for i, ds in enumerate(DS):
    if i: rows.append("\\midrule")
    for j, (m, lab) in enumerate(RP):
        rows.append((DSN[ds] if j == 0 else "") + f" & {lab} & " + " & ".join(f"{k * em.loc[(ds, m), c]:{fm}}" for c, _, fm, k in RC) + "\\\\")
write("replay.tex", "\\begin{tabular}{llcccccc}\n\\toprule\nDataset & Model & " + " & ".join(h for _, h, _, _ in RC) +
      "\\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------ Table: group camouflage (decoys hide the payload in benign-labelled groups)
cm = pd.concat([rd("camo_runs.csv"), gate[gate.dataset.str.contains("camo")]]).drop_duplicates(["dataset", "method", "seed"])
base = r2[r2.dataset == "CIDDS_ctx_injinit_bf"]
CM = [("ernp", "\\textbf{ERNP}, group labels"), ("abl_evid_rep", "ERNP, rep.-only labels"), ("abl_retr_none", "ERNP, no retraction"),
      ("ernp_B0", "ERNP, no online labels"), ("abl_prior_none", "GBM+AL (no prior)"), ("ssf_causal", "SSF")]
rows = []
for m, lab in CM:
    cells = [f"{base[base.method == m].payload_recall.mean():.3f}" if (base.method == m).any() else "--"]
    for r in (1, 3):
        g = cm[(cm.dataset == f"CIDDS_ctx_injcamo{r}_bf") & (cm.method == m)]
        cells += [f"{g['rec::INJ_camo'].mean():.3f}", f"{g.payload_recall.mean():.3f}", f"{100 * g.FPR.mean():.1f}"]
    rows.append(f"{lab} & " + " & ".join(cells) + "\\\\")
write("camo.tex", "\\begin{tabular}{lc" + "ccc" * 2 + "}\n\\toprule\n & no decoys & \\multicolumn{3}{c}{1 decoy per payload flow} & "
      "\\multicolumn{3}{c}{3 decoys per payload flow}\\\\\n\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\n"
      "Method & payload & camo. & payload & FPR\\,\\% & camo. & payload & FPR\\,\\%\\\\\n\\midrule\n" + "\n".join(rows) +
      "\n\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------ Table: recall of every attack family (appendix)
fr = rd("family_recall.csv"); FM = ["ernp", "ssf_causal", "owad", "cade", "trident", "memstream", "knn"]
fp = pd.concat([r1, ext, real]).drop_duplicates(["dataset", "method", "seed"]).groupby(["dataset", "method"]).FPR.mean()
FAM = {}   # known families with no flow in the seed-0 test stream are omitted (nothing to recall)
for ds in [x for x in DS7 if x in set(fr.dataset)]:
    rows = FAM.setdefault(ds, [])
    rows.append(f"\\midrule\n\\multicolumn{{10}}{{l}}{{\\emph{{{DSN[ds]}}}}}\\\\")
    g = fr[fr.dataset == ds].sort_values(["kind", "stream_flows"], ascending=[True, False])
    for _, x in g[g.stream_flows > 0].iterrows():
        vals = [round(x[m], 2) for m in FM]; best = np.nanmax(vals) if x.stream_flows > 0 else np.nan   # bold on the printed values
        cells = ["--" if np.isnan(v) or x.stream_flows == 0 else (f"\\textbf{{{v:.2f}}}" if v == best else f"{v:.2f}") for v in vals]
        name = x.family.replace("&", "\\&")
        rows.append(f"{name} & {'K' if x.kind == 'known' else 'Z'} & {int(x.stream_flows):,} & " + " & ".join(cells) + "\\\\")
    rows.append("\\emph{FPR (\\%)} & & & " + " & ".join(f"{100 * fp.loc[(ds, m)]:.1f}" for m in FM) + "\\\\")
for fn, part in (("families.tex", DS7[:4]), ("families_b.tex", DS7[4:])):
    write(fn, "\\begin{tabular}{lcrccccccc}\n\\toprule\nFamily & & Flows & \\textbf{ERNP} & SSF & OWAD & CADE & Trident & MemStream & $k$NN\\\\\n" +
          "\n".join(r for ds in part for r in FAM.get(ds, [])) + "\n\\bottomrule\n\\end{tabular}\n")
# zero-day families only, all seven datasets in one table (same values as the two tables above)
ZD = []
for ds in [x for x in DS7 if x in set(fr.dataset)]:
    g = fr[(fr.dataset == ds) & (fr.kind != "known") & (fr.stream_flows > 0)].sort_values("stream_flows", ascending=False)
    if g.empty: continue
    ZD.append(f"\\midrule\n\\multicolumn{{9}}{{l}}{{\\emph{{{DSN[ds]}}}}}\\\\")
    for _, x in g.iterrows():
        vals = [round(x[m], 2) for m in FM]; best = np.nanmax(vals)
        cells = ["--" if np.isnan(v) else (f"\\textbf{{{v:.2f}}}" if v == best else f"{v:.2f}") for v in vals]
        ZD.append(f"{x.family.replace('&', chr(92) + '&')} & {int(x.stream_flows):,} & " + " & ".join(cells) + "\\\\")
    ZD.append("\\emph{FPR (\\%)} & & " + " & ".join(f"{100 * fp.loc[(ds, m)]:.1f}" for m in FM) + "\\\\")
write("families_zd.tex", "\\begin{tabular}{lrccccccc}\n\\toprule\nZero-day family & Flows & \\textbf{ERNP} & SSF & OWAD & CADE & Trident & MemStream & $k$NN\\\\\n" +
      "\n".join(ZD) + "\n\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------ Table: purity of the alert-group labels (appendix)
ga = rd("group_audit.csv"); gf = rd("group_footprint.csv").groupby("dataset")[["impure_median", "impure_p90", "benign_median"]].mean()
g = ga.groupby("dataset").agg(groups=("groups_per_day", "mean"), size=("mean_group_size", "mean"), flows=("flows_labelled", "mean"),
                              ab=("attack_in_benign_groups", "mean"), ba=("benign_in_attack_groups", "mean"),
                              imp=("impure_benign_groups", "mean"), n=("stream", "nunique"))
rows = [f"{DSN[ds]} & {g.loc[ds, 'groups']:.0f} & {g.loc[ds, 'size']:.1f} & {g.loc[ds, 'flows']:,.0f} & {100 * g.loc[ds, 'ab']:.2f} & "
        f"{100 * g.loc[ds, 'ba']:.2f} & {100 * g.loc[ds, 'imp']:.1f} & {gf.loc[ds, 'benign_median']:.1f} & "
        f"{gf.loc[ds, 'impure_median']:.1f} / {gf.loc[ds, 'impure_p90']:.1f}\\\\" for ds in DS7 if ds in g.index]
write("groups.tex", "\\begin{tabular}{lcccccccc}\n\\toprule\n & & & & \\multicolumn{3}{c}{Label purity (\\%)} & \\multicolumn{2}{c}{Diameter bound $2r/\\epsilon$}\\\\\n"
      "\\cmidrule(lr){5-7}\\cmidrule(lr){8-9}\n"
      "Dataset & Groups/day & Flows/group & Flows labeled/run & Attack in benign & Benign in attack & Impure benign groups & "
      "Benign groups (median) & Impure benign (median / p90)\\\\\n\\midrule\n" +
      "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------ Table: zero-day detection BEFORE feedback (exp/prefeedback_analysis.py)
pfm = rd("prefeedback_family.csv")
PM = [("ernp", "\\textbf{ERNP}"), ("abl_evid_rep", "ERNP (rep.-only labels)"), ("ernp_B0", "ERNP (no online labels)"), ("ssf_causal", "SSF"),
      ("owad", "OWAD"), ("cade", "CADE"), ("trident", "Trident"), ("memstream", "MemStream"), ("knn", "$k$NN")]
run = pfm.groupby(["dataset", "method", "seed"]).agg(at1=("rec_first_at1fpr", "mean"), own=("rec_first", "mean"), fpr=("fpr_first_day", "mean"),
                                                     nfam=("family", "nunique")).reset_index()
g = run.groupby(["dataset", "method"])[["at1", "own", "fpr", "nfam"]].mean()
e = pfm[pfm.method == "ernp"].groupby(["dataset", "seed"])[["n_pre", "n_flows"]].sum(); share = (e.n_pre / e.n_flows).groupby("dataset").mean()
pf_ds = [x for x in DS7 if x in set(pfm.dataset)]
rows = []
for panel in (DS, [x for x in REAL if x in pf_ds]):          # stacked panels: testbeds, real networks (Mean: over the panel's datasets)
    if not panel: continue
    pad = " & " * (2 * (4 - len(panel)))
    rows.append(" & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{DSN[d_]}}}" for d_ in panel) + pad + " & Mean\\\\\n" +
                "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(panel))) +
                "\nMethod & " + " & ".join(["@1\\% & own (FPR\\,\\%)"] * len(panel)) + pad + " & @1\\% FPR\\\\\n\\midrule")
    for m, lab in PM:
        cells = []
        for ds in panel:
            best = max(g.loc[(ds, mm), "at1"] for mm, _ in PM)
            cells.append(fmt(g.loc[(ds, m), "at1"], bold=np.isclose(g.loc[(ds, m), "at1"], best)) + f" & {g.loc[(ds, m), 'own']:.3f} ({100 * g.loc[(ds, m), 'fpr']:.1f})")
        avg = np.mean([g.loc[(ds, m), "at1"] for ds in panel]); bavg = max(np.mean([g.loc[(ds, mm), "at1"] for ds in panel]) for mm, _ in PM)
        rows.append(("\\midrule\n" if m == "knn" else "") + f"{lab} & " + " & ".join(cells) + pad + f" & {fmt(avg, bold=np.isclose(avg, bavg))}\\\\")
    rows.append("\\midrule\nFamilies; flows before feedback (\\%) & " +
                " & ".join(f"\\multicolumn{{2}}{{c}}{{{g.loc[(ds, 'ernp'), 'nfam']:.0f}; {100 * share[ds]:.0f}}}" for ds in panel) + pad + " & \\\\")
    if panel is DS and len(pf_ds) > 4: rows.append("\\midrule[\\heavyrulewidth]")
write("prefeedback.tex", "\\begin{tabular}{l" + "cc" * 4 + "c}\n\\toprule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------ Table: analyst budget in two units (exp/e23_report.py)
e2 = rd("e2_runs.csv"); D2 = ["CIC18", "ToN_ctx"]
SSFG = os.path.join(os.path.dirname(os.path.dirname(REP)), "exp", "reports", "ssfg_runs.csv")   # exp/ssfg_report.py (review M5)
if os.path.exists(SSFG):
    _g = pd.read_csv(SSFG); e2 = pd.concat([e2, _g[_g.method.str.endswith("_G")]], ignore_index=True)
BU = [("ernp", "\\textbf{ERNP}, group labels"), ("abl_evid_rep", "ERNP, rep.-only labels"), ("ernp_B25", "\\textbf{ERNP}, group labels"),
      ("ernp_B25_cap200", "ERNP, $\\le200$ flows per day"), ("abl_evid_rep_B25", "ERNP, rep.-only labels"), ("ssf_causal", "SSF"),
      ("ssf_causal_B25", "SSF$^\\ddagger$"), ("ssf_causal_G", "SSF, group labels"), ("ssf_causal_B25_G", "SSF$^\\ddagger$, group labels"),
      ("owad", "OWAD")]
BU = [b for b in BU if not b[0].endswith("_G") or b[0] in set(e2.method)]
g = e2.groupby(["dataset", "method"])[["decisions_per_day", "flows_per_day", "F1", "MCC", "zero_day_recall"]].mean()
rows = []
for i, (m, lab) in enumerate(BU):
    if i in (2, 5): rows.append("\\midrule")
    cells = []
    for ds in D2:
        x = g.loc[(ds, m)]; cells.append(f"{x.decisions_per_day:.0f} & {x.flows_per_day:,.0f} & {x.F1:.3f} & {x.MCC:.3f} & {x.zero_day_recall:.3f}")
    rows.append(f"{lab} & " + " & ".join(cells) + "\\\\")
write("budget_units.tex", "\\begin{tabular}{l" + "rrccc" * 2 + "}\n\\toprule\n & " +
      " & ".join(f"\\multicolumn{{5}}{{c}}{{{DSN[d_]}}}" for d_ in D2) + "\\\\\n\\cmidrule(lr){2-6}\\cmidrule(lr){7-11}\n"
      "Method & " + " & ".join(["Decisions/day & Flows/day & F1 & MCC & ZD rec."] * 2) + "\\\\\n\\midrule\n" + "\n".join(rows) +
      "\n\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------ Table: analyst label delay (exp/e45_report.py)
e5 = rd("e5_runs.csv"); D5d = ["CIDDS_ctx", "ToN_ctx", "CIC18"]
DL = [("ernp", "\\textbf{ERNP}", "0"), ("ernp_D1", "\\textbf{ERNP}", "1"), ("ernp_D2", "\\textbf{ERNP}", "2"), ("ssf_causal", "SSF", "0"),
      ("ssf_causal_D1", "SSF", "1"), ("owad", "OWAD", "0"), ("owad_D1", "OWAD", "1")]
g = e5.groupby(["dataset", "method"])[["F1", "MCC", "zero_day_recall"]].mean(); rows = []
for i, (m, lab, D) in enumerate(DL):
    if i in (3, 5): rows.append("\\midrule")
    cells = []
    for ds in D5d:
        same = [mm for mm, _, DD in DL if DD == D]                       # bold: best at the same delay
        cells += [fmt(g.loc[(ds, m), c], bold=len(same) > 1 and np.isclose(g.loc[(ds, m), c], max(g.loc[(ds, mm), c] for mm in same))) for c in ("F1", "MCC", "zero_day_recall")]
    rows.append(f"{lab} & {D} & " + " & ".join(cells) + "\\\\")
write("delay.tex", "\\begin{tabular}{lc" + "ccc" * len(D5d) + "}\n\\toprule\n & Delay & " +
      " & ".join(f"\\multicolumn{{3}}{{c}}{{{DSN[d_]}}}" for d_ in D5d) + "\\\\\n" +
      "".join(f"\\cmidrule(lr){{{3 + 3 * i}-{5 + 3 * i}}}" for i in range(len(D5d))) +
      "\nMethod & (days) & " + " & ".join(["F1 & MCC & ZD rec."] * len(D5d)) + "\\\\\n\\midrule\n" + "\n".join(rows) +
      "\n\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------ Table: alert load for the analyst (exp/soc_load.py)
so = rd("soc_load_runs.csv"); SL = ["ernp", "ssf_causal", "owad", "cade", "trident", "memstream", "knn"]
g = so.groupby(["dataset", "method"]).mean(numeric_only=True); rows = []
for m in SL:
    cells = [f"{g.loc[(ds, m), 'alerts_per_day']:,.0f} & {g.loc[(ds, m), 'false_per_1M_benign'] / 1000:,.1f} & {100 * g.loc[(ds, m), 'share_alerts_inspected_flow_triage']:.1f}" for ds in DS]
    rows.append(("\\midrule\n" if m == "knn" else "") + f"{NAME[m]} & " + " & ".join(cells) + "\\\\")
rows.append("\\midrule\nERNP: group decisions per day; alerts per decision & " +
            " & ".join(f"\\multicolumn{{3}}{{c}}{{{g.loc[(ds, 'ernp'), 'decisions_per_day']:.0f}; {g.loc[(ds, 'ernp'), 'alerts_per_decision']:.0f}}}" for ds in DS) + "\\\\")
write("soc_load.tex", "\\begin{tabular}{l" + "rrr" * len(DS) + "}\n\\toprule\n & " +
      " & ".join(f"\\multicolumn{{3}}{{c}}{{{DSN[d_]}}}" for d_ in DS) + "\\\\\n" +
      "".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(DS))) +
      "\nMethod & " + " & ".join(["Alerts/day & False/$10^6$ ($\\times10^3$) & Insp.\\,\\%"] * len(DS)) + "\\\\\n\\midrule\n" + "\n".join(rows) +
      "\n\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------ Table: slow poisoning with protocol-valid flow records (exp/e45_report.py)
e4p = os.path.join(REP, "e4_runs.csv")
if os.path.exists(e4p):
    e4 = pd.read_csv(e4p); VS = [("CIDDS_ctx_injvalidinit_bf", "CIDDS_ctx_injinit_bf"), ("CIDDS_ctx_injvalidresp_bf", "CIDDS_ctx_injresp_bf")]
    VM = [("ernp", "200", "\\textbf{ERNP}"), ("ernp_B25", "25", "\\textbf{ERNP}"), ("abl_evid_rep", "200", "ERNP, rep.-only labels"),
          ("abl_retr_none", "200", "ERNP, no retraction"), ("abl_prior_none", "200", "GBM+AL (no prior)"), ("abl_self", "--", "ERNP-self (counterfactual)"),
          ("ernp_B0", "--", "ERNP, no online labels"), ("ssf_causal", "200", "SSF"), ("owad", "200", "OWAD"), ("cade", "--", "CADE"),
          ("trident", "--", "Trident"), ("memstream", "--", "MemStream")]
    g = e4.groupby(["dataset", "method"]); mu = g[["ramp_recall", "payload_recall", "payload_at1fpr", "FPR"]].mean(); ab = g["absorbed"].sum(); n = g.size()
    rows = []
    for i, (m, B, lab) in enumerate(VM):
        if (VS[0][0], m) not in mu.index: continue
        if i == 7: rows.append("\\midrule")
        cells = []
        for v, old in VS:
            x = mu.loc[(v, m)]; o = f"{mu.loc[(old, m), 'payload_recall']:.3f}" if (old, m) in mu.index else "--"
            cells.append(f"{x.ramp_recall:.3f} & {x.payload_recall:.3f} & {x.payload_at1fpr:.3f} & {100 * x.FPR:.1f} & {int(ab[(v, m)])}/{int(n[(v, m)])} & {o}")
        rows.append(f"{lab} & {B} & " + " & ".join(cells) + "\\\\")
    write("rq2_valid.tex", "\\begin{tabular}{ll" + "cccccc" * 2 + "}\n\\toprule\n & & \\multicolumn{6}{c}{Compromised initiator} & "
          "\\multicolumn{6}{c}{Compromised responder}\\\\\n\\cmidrule(lr){3-8}\\cmidrule(lr){9-14}\n"
          "Method & $B$ & " + " & ".join(["ramp & payload & payload@1\\% & FPR\\,\\% & absorbed & interp."] * 2) + "\\\\\n\\midrule\n" +
          "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
