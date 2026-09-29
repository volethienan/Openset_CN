"""Clean benchmark report (RQ5 cost) + regression check against the campaign results (same seed, same code path)."""
import os, json, glob, numpy as np, pandas as pd
EXP = os.path.dirname(os.path.abspath(__file__)); B = EXP + "/bench/results/"; R = EXP + "/results/"
rows = []
for f in sorted(glob.glob(B + "*.json")):
    tag = os.path.basename(f)[:-5]; ds, m, s = tag.split("__"); info = json.load(open(f)); t = info.get("timing", {})
    row = dict(dataset=ds, method=m, seconds=info.get("seconds"), init_s=t.get("init_s", t.get("fit_mean_s")),
               score_flows_per_s=t.get("score_flows_per_s"), update_s_mean=t.get("update_s_mean", t.get("fit_mean_s")),
               peak_mem_mb=info.get("peak_mem_mb"))
    ref = R + tag + ".npz"
    if os.path.exists(ref):
        a, b = np.load(B + tag + ".npz"), np.load(ref)
        row["same_flags"] = bool(np.array_equal(a["flag"], b["flag"])); row["max_score_diff"] = float(np.abs(a["score"] - b["score"]).max())
    rows.append(row)
d = pd.DataFrame(rows); pd.set_option("display.width", 200)
print(d.round(3).to_string(index=False))
chk = d.dropna(subset=["same_flags"]) if "same_flags" in d else d.iloc[:0]
bad = chk[~chk.same_flags.astype(bool)]
print(f"\nregression check: {len(chk) - len(bad)}/{len(chk)} runs reproduce the campaign flags exactly"
      + ("" if bad.empty else f"; DIFFERENT: {bad[['dataset', 'method', 'max_score_diff']].to_dict('records')}"))
d.to_csv(EXP + "/reports/bench_cost.csv", index=False)
print("BENCH OK" if bad.empty else "BENCH MISMATCH")
