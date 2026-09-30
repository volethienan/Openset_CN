"""Development-seed comparison (seeds 0-1; CIC17 and seeds 2-4 are held out). Usage: python exp/dev_compare.py [method ...]"""
import os, sys, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import collect

DEV = ["CIDDS_ctx", "ToN_ctx", "CIC18"]
BASE = ["ssf", "owad", "cade", "trident", "memstream", "ernp"]
methods = BASE + sys.argv[1:]
r = collect(DEV, methods, [0, 1])
for m in ["F1", "MCC", "zd_recall_at1fpr", "FPR"]:
    t = r.pivot_table(index="method", columns="dataset", values=m).reindex(methods)
    t["mean"] = t.mean(1); print(f"\n== {m} (mean of seeds 0-1)\n" + t.round(3).to_string())
