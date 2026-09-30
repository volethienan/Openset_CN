"""Dataset/protocol table (seed-0 streams) -> paper/tables/datasets.tex"""
import os, sys, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, "exp"))
os.environ["EXP_SEED"] = "0"; import common
DSN = {"CIDDS_ctx": r"CIDDS-001", "ToN_ctx": r"NF-ToN-IoT-v3", "CIC17": r"CIC-IDS2017$^\dagger$", "CIC18": r"CSE-CIC-IDS2018"}
ORIG = {"CIDDS_ctx": "31.3M", "ToN_ctx": "27.5M", "CIC17": "2.1M", "CIC18": "63.2M"}
rows = []
for ds in DSN:
    df, X, src, dst, tr, ho, s_idx, anc = common.load(ds); fam = df["fam"].values
    known = sorted(set(fam[anc & (fam != "BENIGN")])); stream = fam[s_idx]
    zd = sorted(set(stream[stream != "BENIGN"]) - set(known))
    days = len(set(common.stream_days(df, s_idx)))
    rows.append(f"{DSN[ds]} & {ORIG[ds]} & {len(df)/1e3:.0f}k & {X.shape[1]} & {len(known)} & {len(zd)} & {days} & {100*(stream!='BENIGN').mean():.1f}" + r"\\")
head = (r"\begin{tabular}{lccccccc}" + "\n" + r"\toprule" + "\n" +
        r"Dataset & Original flows & Flows/seed & Features & Known fam. & Zero-day fam. & Stream days & Attack \%\\" + "\n" + r"\midrule" + "\n")
tail = "\n" + r"\bottomrule" + "\n" + r"\end{tabular}" + "\n"
open(os.path.join(ROOT, "paper", "tables", "datasets.tex"), "w").write(head + "\n".join(rows) + tail)
print("ok")
