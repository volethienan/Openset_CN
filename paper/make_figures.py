"""Figures of the manuscript, generated from exp/reports (no value typed by hand) -> paper/figures/*.pdf"""
import os, re, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP = os.path.join(ROOT, "exp", "reports"); OUT = os.path.join(ROOT, "paper", "figures"); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25, "legend.frameon": False, "pdf.fonttype": 42})
C = {"ERNP": "#1f4e9c", "SSF": "#c0392b", "OWAD": "#7f8c8d", "B0": "#95a5a6"}
rd = lambda f: pd.read_csv(os.path.join(REP, f))

# ---- Fig. 2: label budget (RQ3) and noisy analyst
r1, r3, r4, nz, r2 = rd("rq1_runs.csv"), rd("rq3_runs.csv"), rd("rq4_runs.csv"), rd("noise_runs.csv"), rd("rq2_runs.csv")
b200 = r1[r1.method.isin(["ernp", "ssf_causal", "owad"]) & r1.seed.isin([0, 1, 2])].assign(method=lambda z: z.method + "_B200")
b0 = r4[(r4.method == "ernp_B0") & r4.seed.isin([0, 1, 2])]
x = pd.concat([r3, b0, b200]); x["fam"] = x.method.str.replace(r"_B\d+$", "", regex=True)
x["B"] = x.method.str.extract(r"_B(\d+)$")[0].astype(int)
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.2))
for i, (ds, name) in enumerate((("CIC18", "CSE-CIC-IDS2018"), ("ToN_ctx", "NF-ToN-IoT-v3"))):
    for fam, lab in (("ernp", "ERNP"), ("ssf_causal", "SSF"), ("owad", "OWAD")):
        g = x[(x.dataset == ds) & (x.fam == fam)].groupby("B")["F1"].agg(["mean", "std"]).sort_index()
        pos = [[0, 25, 50, 100, 200, 400].index(b) for b in g.index]
        ax[i].errorbar(pos, g["mean"], yerr=g["std"], marker="o", ms=3, lw=1.2, capsize=2, color=C[lab], label=lab)
    ax[i].set_xticks(range(6)); ax[i].set_xticklabels([0, 25, 50, 100, 200, 400])
    ax[i].set_xlabel("labels per day $B$"); ax[i].set_title(f"({'ab'[i]}) {name}", fontsize=8); ax[i].set_ylim(0, 1)
ax[0].set_ylabel("F1"); ax[0].legend(loc="lower right")
clean = pd.concat([r1, r2]); clean = clean[clean.method.isin(["ernp", "ssf_causal"]) & clean.seed.isin([0, 1, 2])]
y = pd.concat([nz, clean]); y["p"] = y.method.str.extract(r"_N(\d+)$")[0].fillna(0).astype(int)
y["fam"] = y.method.str.replace(r"_N\d+$", "", regex=True)
for ds, name, ls in (("CIC18", "CSE-CIC-IDS2018", "-"), ("CIDDS_ctx", "CIDDS-001", "--"), ("ToN_ctx", "NF-ToN-IoT-v3", ":")):
    for fam, lab in (("ernp", "ERNP"), ("ssf_causal", "SSF")):
        g = y[(y.dataset == ds) & (y.fam == fam)].groupby("p")["F1"].mean().sort_index()
        ax[2].plot(g.index, g.values, ls=ls, marker="o", ms=3, lw=1.2, color=C[lab])
    ax[2].plot([], [], ls=ls, color="k", lw=1, label=name)
ax[2].set_xlabel("analyst label-flip rate $p$ (%)"); ax[2].set_title("(c) Noisy analyst", fontsize=8); ax[2].set_ylim(0, 1)
ax[2].legend(fontsize=6, loc="center left", bbox_to_anchor=(1.02, 0.5), title="blue: ERNP\nred: SSF", title_fontsize=6)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_budget_noise.pdf")); plt.close(fig)

# ---- Fig. 3: zero-day recall versus distance to confirmed benign evidence (Proposition 3)
fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.0), sharey=True)
for i, (ds, name) in enumerate((("CIDDS_ctx", "CIDDS-001"), ("ToN_ctx", "NF-ToN-IoT-v3"), ("CIC17", "CIC-IDS2017"), ("CIC18", "CSE-CIC-IDS2018"))):
    p = pd.read_csv(os.path.join(REP, f"p3_{ds}.csv"))
    g = p.groupby("bin", sort=False)[["n", "recall", "recall_B0", "retained_prior_at_mid"]].mean()
    g = g[g["n"] >= 5]
    lab = [{"[0.0, 0.5)": "<.5", "[0.5, 1.0)": ".5-1", "[1.0, 2.0)": "1-2", "[2.0, 4.0)": "2-4", "[4.0, inf)": ">4"}[b] for b in g.index]
    xs = np.arange(len(g)); w = .38
    ax[i].bar(xs - w / 2, g["recall"], w, color=C["ERNP"], label="ERNP")
    ax[i].bar(xs + w / 2, g["recall_B0"], w, color=C["B0"], label="no online labels")
    ax[i].plot(xs, g["retained_prior_at_mid"], "k.--", lw=.8, ms=4, label="retained prior $r_t$")
    ax[i].set_xticks(xs); ax[i].set_xticklabels(lab, fontsize=6.5); ax[i].set_title(name, fontsize=8)
    ax[i].set_xlabel(r"$d(z,\mathcal{E}_t)/\epsilon$")
ax[0].set_ylabel("zero-day recall"); ax[0].set_ylim(0, 1.05)
h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=3, fontsize=7)
fig.tight_layout(rect=(0, 0, 1, .88)); fig.savefig(os.path.join(OUT, "fig_p3_distance.pdf")); plt.close(fig)
print("figures written")
