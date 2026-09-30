"""Generate every LaTeX table of the manuscript from exp/reports/*_runs.csv (no number is typed by hand).
usage: python paper/make_tables.py   ->  paper/tables/*.tex
"""
import os, sys, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP = os.path.join(ROOT, "exp", "reports"); OUT = os.path.join(ROOT, "paper", "tables"); os.makedirs(OUT, exist_ok=True)
DS = ["CIDDS_ctx", "ToN_ctx", "CIC17", "CIC18"]
DSN = {"CIDDS_ctx": "CIDDS-001", "ToN_ctx": "NF-ToN-IoT-v3", "CIC17": "CIC-IDS2017$^\\dagger$", "CIC18": "CSE-CIC-IDS2018"}
SOTA = ["ssf_causal", "owad", "cade", "trident", "memstream"]
NAME = {"ernp": "\\textbf{ERNP (ours)}", "ssf_causal": "SSF", "owad": "OWAD", "cade": "CADE", "trident": "Trident",
        "memstream": "MemStream", "knn": "$k$NN", "iforest": "IForest", "ae_static": "AE", "abl_prior_none": "GBM+AL",
        "abl_evid_rep": "ERNP (rep.-only labels)", "new_ip": "new-IP"}
rd = lambda f: pd.read_csv(os.path.join(REP, f))

def fmt(m, s=None, bold=False):
    t = f"{m:.3f}" + (f"$\\pm${s:.3f}" if s is not None and not np.isnan(s) else "")
    return f"\\textbf{{{t}}}" if bold else t

def write(name, body):
    open(os.path.join(OUT, name), "w", encoding="utf-8").write(body); print("wrote", name)

# ------------------------------------------------------------------ Table: RQ1 main
r1, ext = rd("rq1_runs.csv"), rd("rq1_ext_runs.csv"); r4 = rd("rq4_runs.csv")
d = pd.concat([r1, ext, r4[r4.method == "abl_prior_none"]]).drop_duplicates(["dataset", "method", "seed"])
M = ["ernp"] + SOTA + ["abl_prior_none", "knn"]
rows = []
for met, lab in (("F1", "F1"), ("MCC", "MCC"), ("zd_recall_at1fpr", "Zero-day recall @1\\%FPR")):
    g = d[d.method.isin(M)].groupby(["dataset", "method"])[met].agg(["mean", "std"])
    rows.append(f"\\midrule\n\\multicolumn{{{len(DS) + 2}}}{{l}}{{\\emph{{{lab}}}}}\\\\")
    for m in M:
        cells = []
        for ds in DS:
            best = g.loc[ds]["mean"].max()
            mu, sd = g.loc[(ds, m)]
            cells.append(fmt(mu, sd, bold=np.isclose(mu, best)))
        avg = g.xs(m, level="method")["mean"].mean()
        bestavg = max(g.xs(mm, level="method")["mean"].mean() for mm in M)
        rows.append(f"{NAME[m]} & " + " & ".join(cells) + f" & {fmt(avg, bold=np.isclose(avg, bestavg))}\\\\")
write("rq1_main.tex", "\\begin{tabular}{l" + "c" * (len(DS) + 1) + "}\n\\toprule\nMethod & " +
      " & ".join(DSN[x] for x in DS) + " & Mean\\\\\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------ Table: per-dataset effect sizes with bootstrap CI
rng = np.random.default_rng(0); rows = []
for ds in DS:
    piv = r1[r1.dataset == ds].pivot_table(index="seed", columns="method", values="F1")
    cells = []
    for b in SOTA:
        x = (piv["ernp"] - piv[b]).dropna().values; bs = rng.choice(x, (10000, len(x))).mean(1)
        lo, hi = np.quantile(bs, [.025, .975]); sig = lo > 0 or hi < 0
        t = f"{x.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]"
        cells.append(f"\\textbf{{{t}}}" if sig and x.mean() > 0 else (f"\\underline{{{t}}}" if sig else t))
    rows.append(f"{DSN[ds]} & " + " & ".join(cells) + "\\\\")
dm = r1[r1.method.isin(["ernp"] + SOTA)].groupby(["dataset", "method"])["F1"].mean().unstack()
ranks = dm.rank(axis=1, ascending=False).mean()
from scipy.stats import friedmanchisquare
fr = friedmanchisquare(*[dm[c] for c in dm.columns]).pvalue
write("rq1_effects.tex", "\\begin{tabular}{l" + "c" * len(SOTA) + "}\n\\toprule\nDataset & " + " & ".join(NAME[b] for b in SOTA) +
      "\\\\\n\\midrule\n" + "\n".join(rows) + "\n\\midrule\nMean rank (ERNP " + f"{ranks['ernp']:.2f})" + " & " +
      " & ".join(f"{ranks[b]:.2f}" for b in SOTA) + "\\\\\n\\bottomrule\n\\end{tabular}\n")
open(os.path.join(OUT, "macros.tex"), "w").write(r"\newcommand{\friedmanp}{" + f"{fr:.3f}" + "}\n")

# ------------------------------------------------------------------ Table: SOTA at their best configuration + fidelity
cad = rd("cadence_runs.csv")
x = pd.concat([r1, ext, cad]).drop_duplicates(["dataset", "method", "seed"])
FAM = {"ssf_causal": "SSF", "ssf_causal_E180": "SSF", "ssf_causal_W20000": "SSF", "ssf_causal_P4": "SSF", "ssf_causal_P1": "SSF",
       "owad": "OWAD", "owad_P4": "OWAD", "owad_P1": "OWAD", "memstream": "MemStream", "memstream_fn": "MemStream",
       "trident": "Trident", "cade": "CADE", "ernp": "ERNP", "ernp_P4": "ERNP", "ernp_P1": "ERNP", "knn": "kNN"}
x = x[x.method.isin(FAM)].assign(fam=lambda z: z.method.map(FAM))
m = x.groupby(["dataset", "fam", "method"])["F1"].mean().reset_index()
best = m.loc[m.groupby(["dataset", "fam"])["F1"].idxmax()].pivot(index="dataset", columns="fam", values="F1")
order = ["ERNP", "SSF", "CADE", "Trident", "OWAD", "MemStream", "kNN"]
rows = [f"{DSN[ds]} & " + " & ".join(fmt(best.loc[ds, f], bold=np.isclose(best.loc[ds, f], best.loc[ds].max())) for f in order) + "\\\\" for ds in DS]
rows.append("\\midrule\nMean & " + " & ".join(fmt(best[f].mean(), bold=np.isclose(best[f].mean(), best.mean().max())) for f in order) + "\\\\")
write("sota_best.tex", "\\begin{tabular}{l" + "c" * len(order) + "}\n\\toprule\nDataset & " + " & ".join(order).replace("kNN", "$k$NN") +
      "\\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------ Table: RQ2 (payload recall, FPR) at 200 and 25 labels/day + counterfactual
r2, lb = rd("rq2_runs.csv"), rd("rq2lb_runs.csv")
SC = [("CIDDS_ctx_injinit", "init/DoS"), ("CIDDS_ctx_injresp", "resp/DoS"), ("CIDDS_ctx_injinit_bf", "init/BF"), ("CIDDS_ctx_injresp_bf", "resp/BF")]
MR = [("ernp", 200, "ERNP"), ("ernp_B25", 25, "ERNP"), ("abl_self", 0, "ERNP-self (counterfactual)"), ("ssf_causal", 200, "SSF"),
      ("ssf_causal_B25", 25, "SSF"), ("owad", 200, "OWAD"), ("owad_B25", 25, "OWAD"), ("cade", 0, "CADE"), ("trident", 0, "Trident"), ("memstream", 0, "MemStream")]
x = pd.concat([r2, lb]); rows = []
for mth, B, lab in MR:
    cells = []
    for ds, _ in SC:
        g = x[(x.dataset == ds) & (x.method == mth)]
        cells.append(f"{g.payload_recall.mean():.3f} / {100 * g.FPR.mean():.1f}")
    rows.append(f"{lab} & {B if B else '--'} & " + " & ".join(cells) + "\\\\")
write("rq2.tex", "\\begin{tabular}{llcccc}\n\\toprule\nMethod & $B$ & " + " & ".join(s for _, s in SC) + "\\\\\n\\midrule\n" +
      "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------ Table: RQ3 label budget
r3 = rd("rq3_runs.csv")
b0 = r4[(r4.method == "ernp_B0") & r4.seed.isin([0, 1, 2])]
b200 = r1[r1.method.isin(["ernp", "ssf_causal", "owad"]) & r1.seed.isin([0, 1, 2])].assign(method=lambda z: z.method + "_B200")
x = pd.concat([r3, b0.assign(method="ernp_B0"), b200]); x = x[x.dataset.isin(["ToN_ctx", "CIC18"])]
x["fam"] = x.method.str.replace(r"_B\d+$", "", regex=True); x["B"] = x.method.str.extract(r"_B(\d+)$")[0].astype(int)
rows = []
for ds in ("CIC18", "ToN_ctx"):
    rows.append(f"\\midrule\n\\multicolumn{{7}}{{l}}{{\\emph{{{DSN[ds]}}}}}\\\\")
    for fam, lab in (("ernp", "ERNP"), ("ssf_causal", "SSF"), ("owad", "OWAD")):
        g = x[(x.dataset == ds) & (x.fam == fam)].groupby("B")["F1"].mean()
        rows.append(f"{lab} & " + " & ".join(fmt(g[B]) if B in g else "--" for B in (0, 25, 50, 100, 200, 400)) + "\\\\")
write("rq3.tex", "\\begin{tabular}{lcccccc}\n\\toprule\nMethod & $B{=}0$ & 25 & 50 & 100 & 200 & 400\\\\\n" + "\n".join(rows) +
      "\n\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------ Table: RQ4 ablations (delta F1 vs ERNP)
AB = [("abl_prior_none", "No novelty prior (GBM+AL)"), ("abl_prior_uniform", "Uniform prior (Sipple)"),
      ("abl_prior_marginal", "Product-of-marginals prior"), ("abl_retr_none", "No retraction"), ("abl_retr_hard", "Hard retraction"),
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
      ("sens_eps2", "$\\epsilon\\times2$"), ("sens_clf_rf", "RF instead of GBM")]
piv = r5.pivot_table(index=["dataset", "seed"], columns="method", values="F1"); rows = []
D5 = ["CIDDS_ctx", "ToN_ctx", "CIC18"]
for mth, lab in SE:
    dlt = (piv[mth] - piv["ernp"]).groupby(level=0).mean()
    rows.append(f"{lab} & " + " & ".join(f"{dlt[ds]:+.3f}" for ds in D5) + "\\\\")
write("rq5.tex", "\\begin{tabular}{lccc}\n\\toprule\nSetting ($\\Delta$F1 vs. default) & " + " & ".join(DSN[x_] for x_ in D5) +
      "\\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------ Table: noisy analyst
nz = rd("noise_runs.csv"); clean = pd.concat([r1, r2]); clean = clean[clean.method.isin(["ernp", "ssf_causal", "owad"]) & clean.seed.isin([0, 1, 2])]
x = pd.concat([nz, clean]); x["p"] = x.method.str.extract(r"_N(\d+)$")[0].fillna(0).astype(int)
x["fam"] = x.method.str.replace(r"_N\d+$", "", regex=True); rows = []
for ds in DS + ["CIDDS_ctx_injinit_bf"]:
    for fam, lab, ps in (("ernp", "ERNP", (0, 5, 10, 20)), ("ssf_causal", "SSF", (0, 10))):
        g = x[(x.dataset == ds) & (x.fam == fam)].groupby("p")["F1"].mean()
        rows.append(f"{DSN.get(ds, 'CIDDS-001 (init/BF)')} & {lab} & " + " & ".join(fmt(g[p]) if p in ps and p in g else "--" for p in (0, 5, 10, 20)) + "\\\\")
write("noise.tex", "\\begin{tabular}{llcccc}\n\\toprule\nDataset & Method & 0\\% & 5\\% & 10\\% & 20\\%\\\\\n\\midrule\n" +
      "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------ Table: full-scale CIC-IDS2017 + cost
sc = rd("scale_runs.csv"); cost = rd("bench_cost.csv"); rows = []
for mth in ["ernp"] + SOTA:
    g = sc[sc.method == mth]
    t = cost[(cost.method == mth)]
    rows.append(f"{NAME[mth]} & {fmt(g.F1.mean(), g.F1.std())} & {fmt(g.MCC.mean(), g.MCC.std())} & {fmt(g.zd_recall_at1fpr.mean(), g.zd_recall_at1fpr.std())} & "
                f"{t.init_s.mean():.0f} & {t.peak_mem_mb.mean():.0f}\\\\")
write("scale_cost.tex", "\\begin{tabular}{lccccc}\n\\toprule\n & \\multicolumn{3}{c}{CIC-IDS2017, full (2.1M flows, 3 seeds)} & \\multicolumn{2}{c}{Cost (sampled streams)}\\\\\n"
      "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}\nMethod & F1 & MCC & ZD recall@1\\%FPR & Initial training (s) & Peak RAM (MB)\\\\\n\\midrule\n" +
      "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
