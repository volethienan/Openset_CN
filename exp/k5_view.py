"""Build the K5 view of the experiments: exp_k5/ next to exp/, where every ERNP run is replaced by its K = 5 rerun.

ERNP's main configuration averages K = 5 GBMs, each fitted on its own prior draw (--n_prior 5). The secondary experiments
were rerun with that setting under the name <run>_K5 (run_jobs groups k5, k5abl, k5rq, k5cad). The report scripts and the
paper tables address ERNP runs by their K = 1 names, so instead of renaming inside every script this view hard-links
  results/<stream>__<name>_K5__s<seed>.*  ->  exp_k5/results/<stream>__<name>__s<seed>.*   (ernp_K5 -> ernp)
and every other result file unchanged (baselines; variants with no prior draw, e.g. abl_prior_none, are the same at any K).
Copied cost text/CSV reports remain historical K1 measurements; they are NOT K5 benchmarks. Use benchmark_k5.py
and exp/reports/access/benchmark_k5.json for explicitly measured K5 scoring. For revised inference, use access_report.py.
The original K = 1 ERNP run is kept as <name>_K1 (used where a K = 1 variant must be compared with its own default, e.g. the
random-forest sensitivity run). The report scripts are copied into exp_k5/ and write to exp_k5/reports/.
usage: python exp/k5_view.py           then run the report scripts inside exp_k5/ with EXP_RESULTS=<root>/exp_k5/results/
"""
import os, re, shutil, glob

EXP = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(EXP); V = os.path.join(ROOT, "exp_k5")

def link(src, dst):
    if os.path.exists(dst): os.remove(dst)
    os.link(src, dst)

if __name__ == "__main__":
    os.makedirs(os.path.join(V, "results"), exist_ok=True); os.makedirs(os.path.join(V, "reports"), exist_ok=True)
    for f in glob.glob(os.path.join(EXP, "*.py")) + glob.glob(os.path.join(EXP, "*.sh")): shutil.copy2(f, V)
    for d in ("baselines", "data", "bench"):
        if os.path.isdir(os.path.join(EXP, d)): shutil.copytree(os.path.join(EXP, d), os.path.join(V, d), dirs_exist_ok=True,
                                                                ignore=shutil.ignore_patterns("__pycache__"))
    for f in glob.glob(os.path.join(EXP, "reports", "*")):      # inputs that are not regenerated (bench cost, OWAD checks, ...); never overwrites a K5 report
        if os.path.isfile(f) and not os.path.exists(os.path.join(V, "reports", os.path.basename(f))): shutil.copy2(f, os.path.join(V, "reports"))
    files = os.listdir(os.path.join(EXP, "results")); k5 = {}
    for f in files:
        m = re.match(r"(.*)__(.*)_K5__s(\d)\.(json|npz)$", f)
        if m: k5[(m[1], "ernp" if m[2] == "ernp" else m[2], m[3], m[4])] = f
    n_rep = 0
    for f in files:
        src = os.path.join(EXP, "results", f); m = re.match(r"(.*)__(.*)__s(\d)\.(json|npz)$", f)
        if m and (m[1], m[2], m[3], m[4]) in k5:
            link(src, os.path.join(V, "results", f"{m[1]}__{m[2]}_K1__s{m[3]}.{m[4]}"))
            link(os.path.join(EXP, "results", k5[(m[1], m[2], m[3], m[4])]), os.path.join(V, "results", f)); n_rep += 1
        else:
            link(src, os.path.join(V, "results", f))
    for key, f in k5.items():                                     # K5 runs without a K = 1 counterpart
        dst = os.path.join(V, "results", f"{key[0]}__{key[1]}__s{key[2]}.{key[3]}")
        if not os.path.exists(dst): link(os.path.join(EXP, "results", f), dst)
    # the group-audit and group-diameter logs (ernp_audit, ernp_diam: K = 1 reruns of ernp with the log enabled) are part of
    # every K5 run's record, so on the main streams the K5 run stands for both
    n_aud = 0
    for key, f in k5.items():
        if key[1] != "ernp" or any(t in key[0] for t in ("_inj", "_all", "_full", "_x10")): continue
        for nm in ("ernp_audit", "ernp_diam"):
            link(os.path.join(EXP, "results", f), os.path.join(V, "results", f"{key[0]}__{nm}__s{key[2]}.{key[3]}")); n_aud += 1
    print(f"{len(files)} result files, {n_rep} ERNP files replaced by their K5 rerun, {n_aud} audit/diameter links -> {V}")
