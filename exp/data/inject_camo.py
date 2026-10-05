"""Group-camouflage injection into a real stream (CIDDS_ctx): an attack on the alert-group labelling of ERNP.

A compromised initiator A (the same host as the 'init' scenario of inject.py) sends the zero-day payload (copies of
real bruteForce flows, 30 per hour) for CAMO days, together with RATIO x as many DECOY flows. A decoy is a payload flow
with a small jitter (5% of the benign per-feature std) and is benign in truth: traffic the attacker generates so that
it looks like the payload but would be judged benign on inspection. Decoys and payload flows fall into the same alert
groups; when a group's representative is a decoy, the analyst correctly labels it benign and the group label covers
the payload flows in it. This is the worst case of Proposition 1(iii): the analyst makes no error on the flow shown.
After the camouflage phase, the payload is sent alone for ATTACK days.
Labels: payload during camouflage 'INJ_camo', decoys 'BENIGN', final payload 'INJ_attack' (the metric that matters).
"""
import os, sys, numpy as np, pandas as pd
MOCK = os.environ.get("ERNP_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))).replace("\\", "/").rstrip("/") + "/mock/"
START_DAY, CAMO, ATTACK, RATE_PER_H, JITTER = 4.0, 7.0, 2.0, 30, 0.05

def build(base, ratio):
    df = pd.read_parquet(MOCK + f"streams/{base}.parquet"); rng = np.random.default_rng(42)
    f = [c for c in df.columns if c.startswith("f_")]; t0 = df["t"].min()
    ben = df[df["fam"] == "BENIGN"]; pay = df[df["fam"] == "bruteForce"][f].values
    attackers = set(df.loc[df["fam"] != "BENIGN", "src"])
    cand = ben["src"].value_counts(); cand = cand[[s.startswith("192.168.2") and s not in attackers for s in cand.index]]
    A = cand.index[0]; partners = ben[ben["src"] == A]["dst"].values
    sd = ben[f].values.std(0) + 1e-9
    rows = []
    ts = np.arange(t0 + START_DAY * 86400, t0 + (START_DAY + CAMO + ATTACK) * 86400, 3600 / RATE_PER_H)
    for t in ts:
        camo = t < t0 + (START_DAY + CAMO) * 86400
        x = pay[rng.integers(len(pay))]
        rows.append([*x, t, A, partners[rng.integers(len(partners))], "INJ_camo" if camo else "INJ_attack"])
        if camo:
            for k in range(ratio):
                d = pay[rng.integers(len(pay))] + rng.normal(0, JITTER, len(f)) * sd
                rows.append([*d, t + (k + 1) * 3600 / RATE_PER_H / (ratio + 1), A, partners[rng.integers(len(partners))], "BENIGN"])
    inj = pd.DataFrame(rows, columns=f + ["t", "src", "dst", "fam"])
    out = pd.concat([df, inj]).sort_values("t", kind="stable").reset_index(drop=True)
    name = f"{base}_injcamo{ratio}_bf"
    out.to_parquet(MOCK + f"streams/{name}.parquet")
    print(name, "A =", A, "injected", len(inj), inj["fam"].value_counts().to_dict())

if __name__ == "__main__":
    # usage: inject_camo.py BASE [ratio ...]
    for r in (map(int, sys.argv[2:]) if len(sys.argv) > 2 else (1, 3)): build(sys.argv[1], r)
