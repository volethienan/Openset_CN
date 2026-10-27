"""Naive-analyst model of RQ2 (common.naive_ramp_mask) and the _R suffix of the SSF wrapper.

The mask must reproduce the ramp fraction a of data/inject_valid.py exactly, select only ramp flows, and be empty by
default so that every existing run is unchanged. Stream tests are skipped where the streams are not present."""
import os, re, sys, unittest
from pathlib import Path
import numpy as np
import pandas as pd
EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))
import common
from common import naive_ramp_mask

STREAMS = [f"CIDDS_ctx{'' if s == 0 else f'_s{s}'}_injvalid{v}_bf" for s in range(5) for v in ("init", "resp")]
SOTA_RE = None


def sota_regex():
    """the method-name regex of baselines/sota.py, read from its source (importing it needs torch and the SOTA code)"""
    src = (EXP / "baselines" / "sota.py").read_text(encoding="utf8")
    return re.search(r're\.fullmatch\(r"(.+?)", method\)', src).group(1)


class ScheduleTests(unittest.TestCase):
    def test_constants_equal_generator(self):
        src = (EXP / "data" / "inject_valid.py").read_text(encoding="utf8")
        m = re.search(r"^START, RAMP, ATTACK, RATE_PER_H, N_PART = ([\d.]+), ([\d.]+), ([\d.]+), (\d+), (\d+)", src, re.M)
        self.assertIsNotNone(m)
        self.assertEqual((float(m[1]), float(m[2]), float(m[3]), int(m[4])),
                         (common.INJ_START_D, common.INJ_RAMP_D, common.INJ_ATTACK_D, common.INJ_RATE_H))
        # the generator's ramp fraction and its discrete-field switch, which the 0.5 level follows
        self.assertIn("a = min(1.0, (t - t0 - START * 86400) / (RAMP * 86400))", src)
        self.assertIn("disc = b if a < 0.5 else d", src)
        self.assertIn('"INJ_attack" if a >= 1.0 else "INJ_ramp"', src)

    def synthetic(self, shift=0.0):
        """a stream laid out exactly as inject_valid builds it: clean flows from t0, injections on the 120 s grid"""
        t0 = 1_489_000_000.0; slot = 3600 / common.INJ_RATE_H
        n_ramp = int(common.INJ_RAMP_D * 24 * common.INJ_RATE_H); n_pay = int(common.INJ_ATTACK_D * 24 * common.INJ_RATE_H)
        t_inj = t0 + common.INJ_START_D * 86400 + slot * np.arange(n_ramp + n_pay) + shift
        clean_t = t0 + np.linspace(0, 20 * 86400, 1000)
        df = pd.DataFrame(dict(t=np.r_[clean_t, t_inj],
                               fam=["BENIGN"] * 990 + ["bruteForce"] * 10 + ["INJ_ramp"] * n_ramp + ["INJ_attack"] * n_pay))
        return df.sort_values("t", kind="stable").reset_index(drop=True)

    def test_default_is_empty(self):
        df = self.synthetic()
        self.assertFalse(naive_ramp_mask("CIDDS_ctx_injvalidinit_bf", df, 0.0).any())
        self.assertFalse(naive_ramp_mask("CIDDS_ctx", df, 0.0).any())     # no check, no effect, on any stream

    def test_levels(self):
        df = self.synthetic(); fam = df["fam"].values
        m50 = naive_ramp_mask("CIDDS_ctx_injvalidinit_bf", df, 0.5)
        m100 = naive_ramp_mask("CIDDS_ctx_injvalidinit_bf", df, 1.0)
        self.assertEqual(int(m50.sum()), 2520)           # a = 0, 1/5040, ..., 2519/5040 < 0.5; a = 0.5 is not naive
        self.assertEqual(int(m100.sum()), 5040)          # the whole ramp; the payload is never naive
        self.assertTrue((fam[m100] == "INJ_ramp").all() and (m50 <= m100).all())
        t = df["t"].values[m50]; first_attack_like = df["t"].values[(fam == "INJ_ramp") & ~m50].min()
        self.assertTrue(t.max() < first_attack_like)     # the naive flows are the earliest part of the ramp

    def test_wrong_stream_or_grid_fails(self):
        with self.assertRaises(AssertionError):
            naive_ramp_mask("CIDDS_ctx", self.synthetic(), 0.5)
        with self.assertRaises(AssertionError):           # injections off the generator's grid (t0 inconsistent)
            naive_ramp_mask("CIDDS_ctx_injvalidinit_bf", self.synthetic(shift=37.0), 0.5)

    def test_real_streams(self):
        found = 0
        for ds in STREAMS:
            p = common.STREAMS + ds + ".parquet"
            if not os.path.exists(p): continue
            df = pd.read_parquet(p, columns=["t", "fam"]); found += 1
            self.assertEqual(int(naive_ramp_mask(ds, df, 0.5).sum()), 2520, ds)
            self.assertEqual(int(naive_ramp_mask(ds, df, 1.0).sum()), 5040, ds)
            anc = df["t"].values < common.anchor_end(ds, df["t"].min())
            self.assertFalse((naive_ramp_mask(ds, df, 1.0) & anc).any(), ds)
        if not found: self.skipTest("protocol-valid poisoning streams not present")


class SotaSuffixTests(unittest.TestCase):
    def test_existing_names_parse_as_before(self):
        rx = sota_regex(); old = rx.replace(r"(?:_R(\d+))?", "")
        names = ["ssf_causal", "ssf", "owad", "owad_lr3", "memstream", "memstream_fn", "trident", "cade", "trident_t1",
                 "cade_t1", "ssf_causal_D1", "owad_D1", "ssf_causal_N10", "owad_P1", "owad_B100", "ssf_causal_W20000",
                 "ssf_causal_E180", "ssf_causal_P4", "owad_P4", "ssf_causal_B25", "owad_B25"]
        for n in names:
            a, b = re.fullmatch(rx, n), re.fullmatch(old, n)
            self.assertEqual(a.groups()[:7], b.groups(), n); self.assertIsNone(a[8], n)

    def test_naive_suffix(self):
        m = re.fullmatch(sota_regex(), "ssf_causal_R50")
        self.assertEqual((m[1], m[8]), ("ssf_causal", "50"))
        m = re.fullmatch(sota_regex(), "ssf_causal_R100")
        self.assertEqual((m[1], m[8]), ("ssf_causal", "100"))


class JobTests(unittest.TestCase):
    def test_naive_jobs(self):
        import run_jobs
        jobs = [j for s in range(5) for j in run_jobs.naive_jobs(s)]
        self.assertEqual(len(jobs), 100)
        self.assertEqual(sum(j[0] == "gpu" for j in jobs), 20)
        for kind, ds, name, args in jobs:
            self.assertIn("_injvalid", ds)
            if kind == "cpu":    # every ERNP-family job carries its level and the K5A evidence window
                lvl = int(name.rsplit("_R", 1)[1])
                self.assertEqual(args[args.index("--naive_ramp") + 1], str(lvl / 100))
                self.assertEqual(args[args.index("--keep") + 1], "A")
                self.assertEqual(args[args.index("--name") + 1], name)
        rep = [j for j in jobs if j[2].startswith("abl_evid_rep_K5A_R")]
        self.assertEqual(len(rep), 20)
        for _, _, _, a in rep:
            self.assertEqual(a[a.index("--evidence") + 1], "rep"); self.assertEqual(a[a.index("--n_prior") + 1], "5")


if __name__ == "__main__":
    unittest.main()
