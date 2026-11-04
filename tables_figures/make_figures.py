"""Data figures of the manuscript, generated from the K5A reports (no value typed by hand).
Inputs : exp/reports/*  (bash exp/report_k5a.sh, exp/access_report.py, exp/ablation_report.py)
Output : tables_figures/figures/fig_*.pdf, or <ERNP_PAPER_DIR>/figures/ (FIG_PNG=<dir> also writes PNG previews)
Every figure is drawn at its printed width (the \\textwidth and \\columnwidth of the two-column template) and saved
without cropping, so it is included at scale 1 and every text element prints at 8 pt, the size used in all figures.
usage (from the repository root): python tables_figures/make_figures.py [figure names]
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
REP = os.environ.get("ERNP_REP", os.path.join(ROOT, "exp", "reports"))
ACC = REP
OUT = os.path.join(os.environ.get("ERNP_PAPER_DIR", HERE), "figures"); os.makedirs(OUT, exist_ok=True)
FULL, COL = 505.12 / 72.27, 242.67 / 72.27              # \textwidth and \columnwidth of the template (in)
FS = 8                                                  # one font size for every text element of every figure

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": FS, "axes.titlesize": FS, "axes.labelsize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS,
    "legend.fontsize": FS, "legend.title_fontsize": FS, "figure.titlesize": FS,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .6, "axes.edgecolor": "#52514e",
    "xtick.color": "#52514e", "ytick.color": "#52514e", "xtick.major.width": .6, "ytick.major.width": .6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.pad": 2, "ytick.major.pad": 2,
    "axes.labelpad": 2, "axes.titlepad": 3,
    "axes.grid": True, "grid.color": "#d9d8d4", "grid.linewidth": .5, "legend.frameon": False,
    "legend.handlelength": 1.6, "legend.handletextpad": .4, "legend.columnspacing": 1.0, "legend.borderaxespad": .2,
    "figure.constrained_layout.use": True, "figure.constrained_layout.h_pad": .02, "figure.constrained_layout.w_pad": .02,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": None, "savefig.pad_inches": 0})

# one colour per method, fixed order (validated: CVD-separated adjacent pairs); markers carry identity as well
M = {"ernp":       ("ERNP",      "#2a78d6", "o"),
     "ssf_causal": ("SSF",       "#eb6834", "s"),
     "cade":       ("CADE",      "#1baf7a", "D"),
     "trident":    ("Trident",   "#eda100", "v"),
     "memstream":  ("MemStream", "#e87ba4", "p"),
     "owad":       ("OWAD",      "#008300", "^"),
     "knn":        ("$k$NN",     "#6250d6", "X")}
GREY = "#8a8984"
DSN = {"CIDDS_ctx": "CIDDS-001", "ToN_ctx": "NF-ToN-IoT-v3", "CIC17": "CIC-IDS2017", "CIC18": "CSE-CIC-IDS2018",
       "CTU13": "CTU-13", "SIMARGL": "SIMARGL2021", "APPRAISE": "APPRAISE H2020"}
DS7 = list(DSN)


def rd(name, base=REP):
    return pd.read_csv(os.path.join(base, name))


def save(fig, name):
    w, h = fig.get_size_inches()
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    if os.environ.get("FIG_PNG"): fig.savefig(os.path.join(os.environ["FIG_PNG"], name + ".png"), dpi=200)
    plt.close(fig); print(f"wrote {name} ({w:.2f} x {h:.2f} in)")


def panel_title(ax, s):
    ax.set_title(s, loc="left")


# ---------------------------------------------------------------- zero-day recall vs. FPR at each method's own rule
def fig_pareto():
    runs = rd("main_runs.csv", ACC).replace({"method": {"ernp_K5A": "ernp"}})
    fig, axs = plt.subplots(2, 4, figsize=(FULL, 3.7), sharey=True)
    for i, ds in enumerate(DS7):
        ax = axs.flat[i]
        g = runs[runs.dataset.eq(ds) & runs.method.isin(M)].groupby("method")[["FPR", "zero_day_recall"]].agg(["mean", "std"])
        fx, zy = 100 * g[("FPR", "mean")].clip(lower=1e-2), g[("zero_day_recall", "mean")]
        front = [m for m in g.index if not any(zy[o] >= zy[m] and fx[o] <= fx[m] and (zy[o] > zy[m] or fx[o] < fx[m]) for o in g.index)]
        fr = sorted(front, key=lambda m: fx[m])
        ax.step([fx[m] for m in fr], [zy[m] for m in fr], where="post", color="#52514e", lw=.6, ls=":", zorder=1)
        for m, (lab, col, mk) in M.items():
            if m not in g.index: continue
            ax.errorbar(fx[m], zy[m], yerr=g.loc[m, ("zero_day_recall", "std")], fmt=mk, ms=5.5 if m == "ernp" else 4.2,
                        color=col, mfc=col if m in front else "white", mec=col, mew=.9, capsize=1.5, elinewidth=.6, zorder=4 if m == "ernp" else 3)
        ax.set_xscale("log"); ax.set_xlim(0.05, 150); ax.set_ylim(-0.04, 1.06)
        ax.set_xticks([0.1, 1, 10, 100]); ax.set_xticklabels(["0.1", "1", "10", "100"]); ax.minorticks_off()
        panel_title(ax, f"({'abcdefg'[i]}) {DSN[ds]}")
        if i >= 3: ax.set_xlabel("FPR (%)")
        if i % 4 == 0: ax.set_ylabel("zero-day recall")
    lg = axs.flat[7]; lg.axis("off")
    h = [Line2D([], [], ls="", marker=mk, color=col, mfc=col, ms=5) for lab, col, mk in M.values()]
    h += [Line2D([], [], ls="", marker="o", color=GREY, mfc="white", ms=5), Line2D([], [], color="#52514e", lw=.6, ls=":")]
    lg.legend(h, [v[0] for v in M.values()] + ["dominated (hollow)", "Pareto front"], loc="center", ncol=2,
              title="mean $\\pm$ s.d., 5 seeds")
    save(fig, "fig_pareto")


# ---------------------------------------------------------------- component ablations (Delta F1 per dataset)
def fig_ablation():
    a = rd("ablation_runs.csv", ACC)
    f1 = a.groupby(["dataset", "method"]).F1.mean().unstack()
    rows = [("abl_prior_none_K5A", "no novelty prior"), ("ernp_B0_K5A", "no online labels"),
            ("abl_prior_scarf_K5A", "SCARF prior"), ("abl_evid_rep_K5A", "rep.-only labels"),
            ("abl_retr_none_K5A", "no retraction")]
    mk = dict(zip(DS7, ["o", "s", "D", "v", "^", "p", "X"]))
    fig, ax = plt.subplots(figsize=(COL, 2.9))
    rng = np.random.default_rng(0); means = []
    for y, (m, lab) in enumerate(rows):
        d = f1[m] - f1["ernp_K5A"]; means.append(d.mean())
        for ds in DS7:
            ax.plot(d[ds], y + rng.uniform(-.12, .12), ls="", marker=mk[ds], ms=3.6, mfc="white", mec=GREY, mew=.7, zorder=2)
        ax.plot(d.mean(), y, ls="", marker="|", ms=13, mew=2, color=M["ernp"][1], zorder=3)
    ax.axvline(0, color="#52514e", lw=.7)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[1] for r in rows]); ax.set_ylim(len(rows) - .4, -.6)
    ax.set_xlabel("$\\Delta$F1 vs. ERNP (5-seed mean)"); ax.grid(axis="y", visible=False)
    sec = ax.secondary_yaxis("right"); sec.set_yticks(range(len(rows))); sec.set_yticklabels([f"{v:+.3f}" for v in means])
    sec.tick_params(length=0, colors="#0b0b0b"); sec.spines["right"].set_visible(False); sec.set_ylabel("mean $\\Delta$F1")
    h = [Line2D([], [], ls="", marker=mk[ds], mfc="white", mec=GREY, ms=4) for ds in DS7] + \
        [Line2D([], [], ls="", marker="|", color=M["ernp"][1], ms=9, mew=2)]
    fig.legend(h, [DSN[ds] for ds in DS7] + ["mean of 7"], loc="outside lower center", ncol=2, columnspacing=.8)
    save(fig, "fig_ablation")


# ---------------------------------------------------------------- label budget and analyst label noise
def fig_budget_noise():
    r1, r3, r4, nz, r2 = rd("rq1_runs.csv"), rd("rq3_runs.csv"), rd("rq4_runs.csv"), rd("noise_runs.csv"), rd("rq2_runs.csv")
    b200 = r1[r1.method.isin(["ernp", "ssf_causal", "owad"]) & r1.seed.isin([0, 1, 2])].assign(method=lambda z: z.method + "_B200")
    b0 = r4[(r4.method == "ernp_B0") & r4.seed.isin([0, 1, 2])]
    x = pd.concat([r3, b0, b200]); x["fam"] = x.method.str.replace(r"_B\d+$", "", regex=True)
    x["B"] = x.method.str.extract(r"_B(\d+)$")[0].astype(int)
    BS = [0, 25, 50, 100, 200, 400]
    fig, ax = plt.subplots(1, 3, figsize=(FULL, 2.45))
    for i, ds in enumerate(("CIC18", "ToN_ctx")):
        for fam in ("ernp", "ssf_causal", "owad"):
            lab, col, m = M[fam]
            g = x[(x.dataset == ds) & (x.fam == fam)].groupby("B")["F1"].agg(["mean", "std"]).sort_index()
            ax[i].errorbar([BS.index(b) for b in g.index], g["mean"], yerr=g["std"], marker=m, ms=3.5, lw=1.2, capsize=1.5,
                           elinewidth=.6, color=col)
        ax[i].set_xticks(range(6)); ax[i].set_xticklabels(BS); ax[i].set_ylim(0, 1.02)
        ax[i].set_xlabel("analyst queries per day $B$"); panel_title(ax[i], f"({'ab'[i]}) {DSN[ds]}")
    ax[0].set_ylabel("F1")
    clean = pd.concat([r1, r2]); clean = clean[clean.method.isin(["ernp", "ssf_causal"]) & clean.seed.isin([0, 1, 2])]
    y = pd.concat([nz, clean]); y["p"] = y.method.str.extract(r"_N(\d+)$")[0].fillna(0).astype(int)
    y["fam"] = y.method.str.replace(r"_N\d+$", "", regex=True)
    for ds, ls in (("CIC18", "-"), ("CIDDS_ctx", "--"), ("ToN_ctx", ":")):
        for fam in ("ernp", "ssf_causal"):
            lab, col, m = M[fam]
            g = y[(y.dataset == ds) & (y.fam == fam)].groupby("p")["F1"].mean().sort_index()
            ax[2].plot(g.index, g.values, ls=ls, marker=m, ms=3.2, lw=1.1, color=col)
    ax[2].set_xticks([0, 5, 10, 20]); ax[2].set_ylim(0, 1.02)
    ax[2].set_xlabel("analyst label-flip rate $p$ (%)"); panel_title(ax[2], "(c) noisy analyst")
    hs = [Line2D([], [], color=M[f][1], marker=M[f][2], ms=3.5, lw=1.2) for f in ("ernp", "ssf_causal", "owad")]
    hs += [Line2D([], [], color="#52514e", ls=ls, lw=1) for ls in ("-", "--", ":")]
    fig.legend(hs, ["ERNP", "SSF", "OWAD", f"(c) {DSN['CIC18']}", f"(c) {DSN['CIDDS_ctx']}", f"(c) {DSN['ToN_ctx']}"],
               loc="outside lower center", ncol=6)
    save(fig, "fig_budget_noise")


# ---------------------------------------------------------------- zero-day recall vs. distance to benign evidence
def fig_distance():
    fig, axs = plt.subplots(2, 2, figsize=(COL, 3.35), sharey=True)
    LB = {"[0.0, 0.5)": "<.5", "[0.5, 1.0)": ".5–1", "[1.0, 2.0)": "1–2", "[2.0, 4.0)": "2–4", "[4.0, inf)": ">4"}
    for i, ds in enumerate(("CIDDS_ctx", "ToN_ctx", "CIC17", "CIC18")):
        ax = axs.flat[i]
        p = rd(f"p3_{ds}.csv")
        g = p.groupby("bin", sort=False)[["n", "recall", "recall_B0", "retained_prior_at_mid"]].mean()
        g = g[g["n"] >= 5]
        xs = np.arange(len(g)); w = .38
        ax.bar(xs - w / 2 - .01, g["recall"], w, color=M["ernp"][1], label="ERNP", zorder=2)
        ax.bar(xs + w / 2 + .01, g["recall_B0"], w, color="#b9b8b2", label="no online labels", zorder=2)
        ax.plot(xs, g["retained_prior_at_mid"], color="#0b0b0b", ls="--", lw=.8, marker=".", ms=4, label="retained prior $r_t$", zorder=3)
        for x, n in zip(xs, g["n"]):                       # zero-day flows per bin (mean over seeds)
            ax.text(x, 1.05, f"{n / 1000:.1f}k" if n >= 1000 else f"{n:.0f}", ha="center", va="bottom", color="#52514e")
        ax.set_xticks(xs); ax.set_xticklabels([LB[b] for b in g.index]); ax.grid(axis="x", visible=False)
        panel_title(ax, f"({'abcd'[i]}) {DSN[ds]}")
        if i >= 2: ax.set_xlabel("$d(z,\\mathcal{E}_t)/\\epsilon$")
        if i % 2 == 0: ax.set_ylabel("zero-day recall")
    axs.flat[0].set_ylim(0, 1.2)
    h, l = axs.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=3, handlelength=1.2, columnspacing=.7)
    save(fig, "fig_distance")


# ---------------------------------------------------------------- poisoning timeline (protocol-valid injection)
def fig_rq2_timeline():
    tl = rd("fig_rq2_timeline.csv"); tl = tl[(tl.dataset == "CIDDS_ctx_injvalidinit_bf") & tl.day.between(2, 12)]
    L = [("ernp", "ERNP, $B=200$", M["ernp"][1], "-", "o"), ("ernp_B25", "ERNP, $B=25$", M["ernp"][1], "--", "o"),
         ("ernp_B0", "ERNP, no online labels", GREY, "-.", "d"), ("abl_self", "ERNP fed its own verdicts", "#86b6ef", ":", "x"),
         ("ssf_causal", "SSF", M["ssf_causal"][1], "-", "s"), ("owad", "OWAD", M["owad"][1], "-", "^"),
         ("memstream", "MemStream", M["memstream"][1], "-", "p")]
    fig, ax = plt.subplots(1, 2, figsize=(FULL, 2.75))
    for m, lab, col, ls, mk in L:
        g = tl[tl.method == m].groupby("day")[["inj_alert", "FPR"]].mean()
        sd = tl[tl.method == m].groupby("day")[["inj_alert", "FPR"]].std().fillna(0)
        if g.empty: raise RuntimeError(f"no timeline rows for {m}")
        ax[0].plot(g.index, g.inj_alert, ls=ls, marker=mk, ms=3, lw=1.1, color=col, label=lab)
        ax[0].fill_between(g.index, (g.inj_alert - sd.inj_alert).clip(lower=0), (g.inj_alert + sd.inj_alert).clip(upper=1), color=col, alpha=.12, lw=0)
        ax[1].plot(g.index, 100 * g.FPR.clip(lower=1e-3), ls=ls, marker=mk, ms=3, lw=1.1, color=col)
        ax[1].fill_between(g.index, 100 * (g.FPR - sd.FPR).clip(lower=1e-3), 100 * (g.FPR + sd.FPR).clip(lower=1e-3), color=col, alpha=.12, lw=0)
    for a in ax:
        a.axvspan(3.5, 10.5, color="#fbe3d6", lw=0, zorder=0); a.axvspan(10.5, 12.5, color="#f4c2b0", lw=0, zorder=0)
        a.set_xlim(1.5, 12.5); a.set_xlabel("stream day")
    ax[0].text(7, 1.04, "ramp", ha="center"); ax[0].text(11.5, 1.04, "payload", ha="center")
    ax[0].set_ylim(-0.03, 1.12); ax[0].set_ylabel("alert rate on injected flows"); panel_title(ax[0], "(a) detection of the ramp and payload")
    ax[1].set_yscale("log"); ax[1].set_ylabel("FPR on benign flows (%)"); panel_title(ax[1], "(b) cost in false alarms")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=4)
    save(fig, "fig_rq2_timeline")


# ---------------------------------------------------------------- RQ2: analyst who accepts the ramp (exp/naive_report.py)
def fig_naive():
    nv = pd.read_csv(os.path.join(ROOT, "exp", "reports", "naive_runs.csv"))
    MS = [("ERNP", "ERNP", M["ernp"][1]), ("ERNP, rep.-only labels", "ERNP, rep.-only labels", "#86b6ef"),
          ("ERNP, no retraction", "ERNP, no retraction", "#1b4f8a"), ("GBM+AL (no prior)", "GBM+AL (no prior)", GREY),
          ("SSF", "SSF", M["ssf_causal"][1])]
    LV = [("R0", "correct analyst"), ("R50", "accepts first half"), ("R100", "accepts whole ramp")]
    sel = nv[nv.method.isin([m for m, _, _ in MS])]
    if not sel.groupby(["scenario", "method", "level"]).size().eq(5).all(): raise RuntimeError("naive figure needs 5 seeds per cell")
    g = sel.groupby(["scenario", "method", "level"]).payload_recall.agg(["mean", "std"])
    ref = nv[nv.method.isin(["ERNP, no online labels", "ERNP-self (counterfactual)"])].groupby(["scenario", "method"]).payload_recall.mean()
    fig, ax = plt.subplots(1, 2, figsize=(FULL, 2.3), sharey=True)
    w = .16; xs = np.arange(len(LV))
    for k, sc in enumerate(("init", "resp")):
        a = ax[k]
        for j, (m, lab, col) in enumerate(MS):
            mu = [g.loc[(sc, m, lv), "mean"] for lv, _ in LV]; sd = [g.loc[(sc, m, lv), "std"] for lv, _ in LV]
            a.bar(xs + (j - 2) * w, mu, w * .92, yerr=sd, color=col, label=lab, error_kw=dict(lw=.6, capsize=1.2), zorder=2)
        a.axhline(ref[(sc, "ERNP, no online labels")], color="#0b0b0b", ls="--", lw=.8, label="ERNP, no online labels", zorder=3)
        a.axhline(ref[(sc, "ERNP-self (counterfactual)")], color="#0b0b0b", ls=":", lw=1, label="ERNP fed its own verdicts", zorder=3)
        a.set_xticks(xs); a.set_xticklabels([l for _, l in LV]); a.set_ylim(0, 1.05); a.grid(axis="x", visible=False)
        panel_title(a, f"({'ab'[k]}) compromised {'initiator' if sc == 'init' else 'responder'}")
    ax[0].set_ylabel("payload recall")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=4)
    save(fig, "fig_naive")


# ---------------------------------------------------------------- RQ1: paired per-dataset differences (replaces the p-value table in the main text)
def fig_paired():
    dm = pd.read_csv(os.path.join(ACC, "dataset_means.csv")); dm = dm[dm.dataset.isin(DS7)].set_index(["dataset", "method"])
    ts = pd.read_csv(os.path.join(ACC, "dataset_tests.csv")); ts = ts[ts.cohort == "historical_filtered_7"].set_index(["metric", "baseline"])
    DET = ["ssf_causal", "cade", "trident", "memstream", "owad"]
    MET = [("F1", "F1"), ("MCC", "MCC"), ("AUROC", "AUROC"), ("zd_recall_at1fpr", "ZD@1%")]
    DMK = {"CIDDS_ctx": "o", "ToN_ctx": "s", "CIC17": "^", "CIC18": "v", "CTU13": "D", "SIMARGL": "P", "APPRAISE": "X"}
    fig, axs = plt.subplots(2, 2, figsize=(FULL, 3.15), sharey=True); axs = list(axs.flat)
    for a, (col, lab) in zip(axs, MET):
        for i, m in enumerate(DET):
            y = len(DET) - 1 - i
            d = np.array([dm.loc[(ds, "ernp_K5A"), col] - dm.loc[(ds, m), col] for ds in DS7])
            for j, ds in enumerate(DS7):
                a.scatter(d[j], y + (j - 3) * .07, marker=DMK[ds], s=11, facecolor="none" if d[j] < 0 else "#52514e",
                          edgecolor="#52514e", lw=.6, zorder=3)
            a.plot([d.mean()] * 2, [y - .32, y + .32], color=M[m][1], lw=2, zorder=4)
            t = ts.loc[(col, m)]
            a.text(1.02, y, f"{t.wins:.0f}/7  {t.p_two_sided:.3f}  {t.p_two_sided_holm:.3f}", transform=a.get_yaxis_transform(), va="center", ha="left")
        a.text(1.02, len(DET) - .5, "wins  $p_2$     Holm", transform=a.get_yaxis_transform(), va="bottom", ha="left", color="#52514e")
        a.axvline(0, color="#0b0b0b", lw=.6, zorder=2); a.grid(axis="y", visible=False)
        panel_title(a, lab); a.set_xlabel("ERNP $-$ detector")
    axs[0].set_yticks(range(len(DET))); axs[0].set_yticklabels([M[m][0] for m in DET][::-1]); axs[0].set_ylim(-.6, len(DET) - .4)
    for a in axs[:2]: a.set_xlabel("")
    h = [Line2D([], [], marker=DMK[ds], ls="", mfc="#52514e", mec="#52514e", ms=4, label=DSN[ds]) for ds in DS7]
    fig.legend(handles=h, loc="outside lower center", ncol=7, handletextpad=.2, columnspacing=.8)
    fig.get_layout_engine().set(w_pad=.04, wspace=.08)
    save(fig, "fig_paired")


# ---------------------------------------------------------------- NF-ToN-IoT-v3: FPR per 4 h bin and benign volume
def fig_ton_timeline():
    tt = rd("fig_ton_timeline.csv"); tt["ts"] = pd.to_datetime(tt.bin_start, unit="s") + pd.Timedelta(hours=2)
    fig, ax = plt.subplots(2, 1, figsize=(FULL, 3.0), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for m, lab, col, ls, mk in (("ernp", "ERNP, daily updates", M["ernp"][1], "-", "o"), ("ernp_P4", "ERNP, 4 h updates", M["ernp"][1], "--", "o"),
                                ("ssf_causal", "SSF, daily", M["ssf_causal"][1], "-", "s"), ("ssf_causal_P4", "SSF, 4 h", M["ssf_causal"][1], "--", "s"),
                                ("owad", "OWAD", M["owad"][1], "-", "^")):
        g = tt[tt.method == m].groupby("ts").FPR.mean(); sd = tt[tt.method == m].groupby("ts").FPR.std().fillna(0)
        ax[0].plot(g.index, 100 * g.values, ls=ls, marker=mk, ms=2.4, lw=1, color=col, label=lab)
        ax[0].fill_between(g.index, 100 * (g - sd).clip(lower=0), 100 * (g + sd).clip(upper=1), color=col, alpha=.12, lw=0)
    v = tt[tt.method == "ernp"].groupby("ts")[["n_benign", "n_burst"]].mean(); w = pd.Timedelta(hours=3.4)
    ax[1].bar(v.index, v.n_benign - v.n_burst, width=w, color="#b9b8b2", label="benign flows")
    ax[1].bar(v.index, v.n_burst, bottom=(v.n_benign - v.n_burst).clip(lower=1), width=w, color="#d03b3b", label="gateway burst (192.168.1.1)")
    for a in ax:
        for d in pd.date_range(tt.ts.min().normalize() + pd.Timedelta(days=1), tt.ts.max(), freq="D"):
            a.axvline(d, color="#52514e", lw=.4, ls=":", zorder=0)
        a.grid(axis="x", visible=False)
    ax[0].set_ylabel("FPR (%)"); ax[0].set_ylim(-3, 103)
    h0, l0 = ax[0].get_legend_handles_labels(); h1, l1 = ax[1].get_legend_handles_labels()
    fig.legend(h0 + h1, l0 + l1, loc="outside upper center", ncol=4)
    ax[1].set_yscale("log"); ax[1].set_ylabel("benign flows")
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b")); ax[1].set_xlabel("stream time (4 h bins; dotted lines: daily updates)")
    save(fig, "fig_ton_timeline")


if __name__ == "__main__":
    import sys
    todo = sys.argv[1:] or ["pareto", "ablation", "budget_noise", "distance", "rq2_timeline", "naive", "paired", "ton_timeline"]
    for name in todo: globals()["fig_" + name]()
