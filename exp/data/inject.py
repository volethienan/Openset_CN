"""Controlled frog-boiling injection into a real stream (CIDDS_ctx).

A compromised domain A ramps its traffic from its own benign behaviour towards real DoS flows over RAMP days
(x = (1-a) * benign_A + a * dos, a: 0 -> 1), then launches the real attack (copies of real DoS flows) for
ATTACK days. Labels: ramp flows 'INJ_ramp', final attack flows 'INJ_attack' (the metric that matters).

variant 'init': A initiates the flows (A -> its usual partners)
variant 'resp': A is a server; flows are initiated by many honest clients towards A (responder laundering)
"""
import pandas as pd, numpy as np, sys
import os
MOCK = os.environ.get("ERNP_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))).replace("\\", "/").rstrip("/") + "/mock/"
RAMP_START_DAY, ATTACK = 4.0, 2.0

def build(base, variant, payload="dos", RAMP=7.0, RATE_PER_H=30):
    """payload: attack family whose real flows are the end point of the ramp. 'bruteForce' is the only CIDDS
    family that never appears in the anchor period, i.e. a true zero-day for supervised methods too."""
    df = pd.read_parquet(MOCK + f"streams/{base}.parquet"); rng = np.random.default_rng(42)
    f = [c for c in df.columns if c.startswith("f_")]; t0 = df["t"].min()
    ben = df[df["fam"] == "BENIGN"]; dos = df[df["fam"] == payload]
    attackers = set(df.loc[df["fam"] != "BENIGN", "src"])
    if variant == "init":   # a client that never attacks in the original data, with plenty of benign traffic
        cand = ben["src"].value_counts(); cand = cand[[s.startswith("192.168.2") and s not in attackers for s in cand.index]]
        A = cand.index[0]; A_ben = ben[ben["src"] == A]; partners = A_ben["dst"].values
    else:                   # an internal server (192.168.100.x) that is a responder for many clients
        # real server = most distinct client initiators; exclude broadcast addresses
        # (servers appear as 'attack' sources in CIDDS only as victims replying, so no attacker filter here)
        internal = ben[ben["src"].str.startswith("192.168.2")]
        cand = internal.groupby("dst")["src"].nunique().sort_values(ascending=False)
        cand = cand[[s.startswith("192.168.100.") and not s.endswith(".255") for s in cand.index]]
        A = cand.index[0]; A_ben = internal[internal["dst"] == A]; partners = A_ben["src"].values
    rows = []
    ts = np.arange(t0 + RAMP_START_DAY * 86400, t0 + (RAMP_START_DAY + RAMP + ATTACK) * 86400, 3600 / RATE_PER_H)
    for t in ts:
        a = min(1.0, (t - t0 - RAMP_START_DAY * 86400) / (RAMP * 86400))
        b = A_ben[f].values[rng.integers(len(A_ben))]; d = dos[f].values[rng.integers(len(dos))]
        x = (1 - a) * b + a * d
        p = partners[rng.integers(len(partners))]
        src, dst = (A, p) if variant == "init" else (p, A)
        rows.append([*x, t, src, dst, "INJ_attack" if a >= 1.0 else "INJ_ramp"])
    inj = pd.DataFrame(rows, columns=f + ["t", "src", "dst", "fam"])
    out = pd.concat([df, inj]).sort_values("t", kind="stable").reset_index(drop=True)
    name = f"{base}_inj{variant}" + ("" if payload == "dos" else "_bf")
    if (RAMP, RATE_PER_H) != (7.0, 30): name += f"_r{int(RAMP)}_q{int(RATE_PER_H)}"      # RQ2 sweep variants
    out.to_parquet(MOCK + f"streams/{name}.parquet")
    print(name, "A =", A, "payload", payload, "injected", len(inj), inj["fam"].value_counts().to_dict())

if __name__ == "__main__":
    # usage: inject.py BASE [payload=dos|bruteForce] [ramp_days=7] [rate_per_hour=30]
    payload = sys.argv[2] if len(sys.argv) > 2 else "dos"
    ramp = float(sys.argv[3]) if len(sys.argv) > 3 else 7.0; rate = float(sys.argv[4]) if len(sys.argv) > 4 else 30
    for v in ("init", "resp"): build(sys.argv[1], v, payload, ramp, rate)
