"""Job runner for the RQ experiments: two pools (GPU jobs, CPU jobs), skip finished runs, one log per job.

usage: python run_jobs.py rq1 --seeds 0 1 2 3 4 [--gpu 2 --cpu 2] [--gpu-ids 0,1] [--dry]
--gpu-ids: physical GPUs this runner may use; each GPU job is pinned (CUDA_VISIBLE_DEVICES) to the least-busy one.
"""
import os, sys, subprocess, argparse, time, threading
from concurrent.futures import ThreadPoolExecutor
MAX_GPU_PROCS = int(os.environ.get("MAX_GPU_PROCS", 99))   # machine-wide cap on concurrent GPU jobs (memory commit)
_LAUNCH = threading.Lock()
GPU_IDS = []                                   # set from --gpu-ids; empty = leave CUDA_VISIBLE_DEVICES untouched
THREADS = 0                                    # set from --threads; 0 = libraries use every visible core (unchanged)
_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
_GPU_BUSY = {}
EXP = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(EXP, "results"); LOG = os.path.join(EXP, "logs")
os.makedirs(LOG, exist_ok=True)

def ds_names(seed):
    sfx = "" if seed == 0 else f"_s{seed}"
    return dict(CIDDS=f"CIDDS_ctx{sfx}", ToN=f"ToN_ctx{sfx}", CIC17=f"CIC17{sfx}", CIC18=f"CIC18{sfx}")

def rq1_jobs(seed):
    jobs = []
    for ds in ds_names(seed).values():
        jobs += [("cpu", ds, "ernp", ["ernp.py", ds]),
                 ("cpu", ds, "ernp_ssfproto", ["ernp.py", ds, "--protocol", "ssf", "--name", "ernp_ssfproto"])]
        for m in ["ssf_causal", "ssf", "owad", "memstream", "trident", "cade"]:
            jobs.append(("gpu", ds, m, ["baselines/sota.py", ds, m]))
    return jobs

SOTA5 = ["ssf_causal", "owad", "memstream", "trident", "cade"]
ADAPTIVE = ["ssf_causal", "owad", "memstream"]
SWEEP = [(r, q) for r in (3, 7, 14) for q in (10, 30, 100) if (r, q) != (7, 30)]      # (7,30) = main scenario

def cidds(seed): return "CIDDS_ctx" + ("" if seed == 0 else f"_s{seed}")

def rq2_jobs(seed):
    """4 main injection scenarios x (ERNP, ERNP without labels [P1], 5 SOTA); sweep on initiator-bruteForce (seeds 0-2)."""
    jobs = []
    for v in ("injinit", "injresp", "injinit_bf", "injresp_bf"):
        ds = f"{cidds(seed)}_{v}"
        jobs += [("cpu", ds, "ernp", ["ernp.py", ds]), ("cpu", ds, "ernp_B0", ["ernp.py", ds, "--budget", "0", "--name", "ernp_B0"])]
        jobs += [("gpu", ds, m, ["baselines/sota.py", ds, m]) for m in SOTA5]
    if seed <= 2:
        for r, q in SWEEP:
            ds = f"{cidds(seed)}_injinit_bf_r{r}_q{q}"
            jobs.append(("cpu", ds, "ernp", ["ernp.py", ds]))
            jobs += [("gpu", ds, m, ["baselines/sota.py", ds, m]) for m in ADAPTIVE]
    return jobs

def rq3_jobs(seed):
    """label budget sweep (B=200 comes from RQ1) on ToN and CIC18, seeds 0-2"""
    if seed > 2: return []
    n = ds_names(seed); jobs = []
    for ds in (n["ToN"], n["CIC18"]):
        for B in (25, 50, 100, 400):
            jobs.append(("cpu", ds, f"ernp_B{B}", ["ernp.py", ds, "--budget", str(B), "--name", f"ernp_B{B}"]))
            jobs += [("gpu", ds, f"{m}_B{B}", ["baselines/sota.py", ds, f"{m}_B{B}"]) for m in ("ssf_causal", "owad")]
    return jobs

ABLATIONS = {"abl_prior_none": ["--prior", "none"], "abl_prior_uniform": ["--prior", "uniform"],
             "abl_prior_marginal": ["--prior", "marginal"], "abl_retr_none": ["--retraction", "none"],
             "abl_retr_hard": ["--retraction", "hard"], "abl_evid_rep": ["--evidence", "rep"],
             "abl_feat_flow": ["--features", "flow"], "ernp_B0": ["--budget", "0"]}
SENSITIVITY = {"sens_rho0.1": ["--rho", "0.1"], "sens_rho0.2": ["--rho", "0.2"], "sens_rho0.5": ["--rho", "0.5"],
               "sens_mass0.25": ["--prior_mass", "0.25"], "sens_mass0.5": ["--prior_mass", "0.5"], "sens_mass2": ["--prior_mass", "2.0"],
               "sens_eps0.5": ["--eps_mult", "0.5"], "sens_eps2": ["--eps_mult", "2.0"], "sens_clf_rf": ["--clf", "rf"]}

def rq4_jobs(seed):
    return [("cpu", ds, name, ["ernp.py", ds, *args, "--name", name]) for ds in ds_names(seed).values() for name, args in ABLATIONS.items()]

def rq5_jobs(seed):
    if seed > 2: return []
    n = ds_names(seed)
    return [("cpu", ds, name, ["ernp.py", ds, *args, "--name", name]) for ds in (n["CIDDS"], n["ToN"], n["CIC18"])
            for name, args in SENSITIVITY.items()]

def cad_jobs(seed):
    """cadence study (added after the RQ1 preview showed day-level benign novelty on ToN): update every 4 h / 1 h
    with the SAME total label budget (200/day, split exactly over the periods of a day), label-using methods only"""
    if seed > 2: return []
    jobs = []
    for ds in ds_names(seed).values():
        for h in (1, 4):
            jobs.append(("cpu", ds, f"ernp_P{h}", ["ernp.py", ds, "--period", str(h), "--name", f"ernp_P{h}"]))
            for m in ("ssf_causal", "owad"):
                # SSF retrains 20 epochs on its whole buffer at every update (~5 min on ToN) -> 1 h cadence ~ 8 h/run:
                # SSF at 1 h only on ToN seed 0 (the dataset motivating the study); SSF at 4 h everywhere
                if m == "ssf_causal" and h == 1 and not (ds == "ToN_ctx" and seed == 0): continue
                jobs.append(("gpu", ds, f"{m}_P{h}", ["baselines/sota.py", ds, f"{m}_P{h}"]))
    return jobs

def ext_jobs(seed):
    """reviewer-proofing additions (28/09/2026): classical references, MemStream-fixnorm, the absorbing counterfactual
    of ERNP (its own verdicts as evidence), and SSF run with the authors' own settings"""
    n = ds_names(seed); jobs = []
    for ds in n.values():
        jobs += [("cpu", ds, "memstream_fn", ["baselines/sota.py", ds, "memstream_fn"]),   # CPU: batch-1 streaming
                 ("cpu", ds, "iforest", ["baselines/classic.py", ds, "iforest"]),
                 ("cpu", ds, "knn", ["baselines/classic.py", ds, "knn"]),
                 ("gpu", ds, "ae_static", ["baselines/classic.py", ds, "ae_static"]),
                 ("cpu", ds, "abl_self", ["ernp.py", ds, "--evidence_source", "self", "--name", "abl_self"])]
    for v in ("injinit", "injresp", "injinit_bf", "injresp_bf"):
        ds = f"{cidds(seed)}_{v}"
        jobs += [("cpu", ds, "abl_self", ["ernp.py", ds, "--evidence_source", "self", "--name", "abl_self"]),
                 ("cpu", ds, "memstream_fn", ["baselines/sota.py", ds, "memstream_fn"])]
    for ds in n.values():
        if seed == 0:   # epoch_1 = 180 (authors' UNSW command) instead of 20: ~9x update cost -> one seed
            jobs.append(("gpu", ds, "ssf_causal_E180", ["baselines/sota.py", ds, "ssf_causal_E180"]))
        if seed == 0:   # SSF's native protocol: fixed windows of 20,000 flows, 200 labels per window (seed 0, time)
            jobs.append(("gpu", ds, "ssf_causal_W20000", ["baselines/sota.py", ds, "ssf_causal_W20000"]))
    return jobs

def ext1_jobs(seed):
    """the RQ1 part of ext_jobs only (RQ2 injection streams excluded) - RQ2/RQ3 paused by the user on 28/09/2026"""
    return [j for j in ext_jobs(seed) if "_inj" not in j[1]]

def scale_jobs(seed):
    """scale check: CICIDS2017 without sampling (2.1M flows); seeds 0-2 (seed = anchor split / sampling of model inputs)"""
    if seed > 2: return []
    ds = "CIC17_full"
    return [("cpu", ds, "ernp", ["ernp.py", ds])] + [("gpu", ds, m, ["baselines/sota.py", ds, m]) for m in SOTA5]

def running_elsewhere(args):
    """True if the same job is already running in another process (e.g. left over from a runner that was restarted)"""
    try:
        import psutil
        for p in psutil.process_iter(["cmdline"]):
            cl = p.info["cmdline"] or []
            if len(cl) > len(args) and cl[-len(args):] == args and p.pid != os.getpid(): return True
    except Exception: pass
    return False

def gpu_procs():
    """GPU jobs running on the machine, whichever runner started them"""
    try:
        import psutil
        n = 0
        for p in psutil.process_iter(["cmdline"]):
            cl = " ".join(p.info["cmdline"] or [])
            if ("baselines/sota.py" in cl and "memstream_fn" not in cl) or ("baselines/classic.py" in cl and "ae_static" in cl): n += 1
        return n
    except Exception: return 0

INJ = ("injinit", "injresp", "injinit_bf", "injresp_bf")

def rq2lb_jobs(seed):
    """RQ2 at a small label budget (25/day) for every label-using method: is SSF still not absorbed without 200 labels/day?"""
    jobs = []
    for v in INJ:
        ds = f"{cidds(seed)}_{v}"
        jobs.append(("cpu", ds, "ernp_B25", ["ernp.py", ds, "--budget", "25", "--name", "ernp_B25"]))
        jobs += [("gpu", ds, f"{m}_B25", ["baselines/sota.py", ds, f"{m}_B25"]) for m in ("ssf_causal", "owad")]
    return jobs

def noise_jobs(seed):
    """noisy analyst: each returned label flipped w.p. 10% (ERNP also 5% and 20%), RQ1 datasets + RQ2 scenarios, seeds 0-2"""
    if seed > 2: return []
    jobs = []
    for ds in list(ds_names(seed).values()) + [f"{cidds(seed)}_{v}" for v in INJ]:
        for p in (5, 10, 20):
            jobs.append(("cpu", ds, f"ernp_N{p}", ["ernp.py", ds, "--label_noise", str(p / 100), "--name", f"ernp_N{p}"]))
        jobs += [("gpu", ds, f"{m}_N10", ["baselines/sota.py", ds, f"{m}_N10"]) for m in ("ssf_causal", "owad")]
    return jobs

def run(job, seed):
    pool, ds, name, args = job
    out = os.path.join(RES, f"{ds}__{name}__s{seed}.json")
    if os.path.exists(out): return f"skip {ds} {name}"
    if running_elsewhere(args): return f"skip {ds} {name} (already running in another process)"
    env = dict(os.environ, EXP_SEED=str(seed)); log = os.path.join(LOG, f"{ds}__{name}__s{seed}.log")
    if pool == "cpu": env["CUDA_VISIBLE_DEVICES"] = ""
    if THREADS: env.update({v: str(THREADS) for v in _THREAD_VARS})   # cap BLAS/OpenMP/torch threads per job
    if pool == "gpu":
        with _LAUNCH:                                   # check the cap and start the process atomically
            while gpu_procs() >= MAX_GPU_PROCS: time.sleep(60)
            if os.path.exists(out) or running_elsewhere(args): return f"skip {ds} {name}"
            dev_id = None
            if GPU_IDS:                             # pin to the least-busy allowed GPU
                dev_id = min(GPU_IDS, key=lambda g: _GPU_BUSY.get(g, 0)); _GPU_BUSY[dev_id] = _GPU_BUSY.get(dev_id, 0) + 1
                env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"; env["CUDA_VISIBLE_DEVICES"] = dev_id
            f = open(log, "w"); t0 = time.time()
            proc = subprocess.Popen([sys.executable] + args, cwd=EXP, env=env, stdout=f, stderr=subprocess.STDOUT)
        rc = proc.wait(); f.close()
        if dev_id is not None:
            with _LAUNCH: _GPU_BUSY[dev_id] -= 1
        return f"{'OK ' if rc == 0 else 'ERR'} {ds} {name} s{seed} ({time.time() - t0:.0f}s)" + (f" [gpu {dev_id}]" if dev_id else "")
    else:
        f = open(log, "w"); t0 = time.time()
        proc = subprocess.Popen([sys.executable] + args, cwd=EXP, env=env, stdout=f, stderr=subprocess.STDOUT)
    rc = proc.wait(); f.close()
    return f"{'OK ' if rc == 0 else 'ERR'} {ds} {name} s{seed} ({time.time() - t0:.0f}s)"

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("rq"); ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--gpu", type=int, default=2); ap.add_argument("--cpu", type=int, default=2); ap.add_argument("--dry", action="store_true")
    ap.add_argument("--gpu-ids", default="", help="comma-separated physical GPU ids, e.g. 0,1")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads per job (keep (gpu + cpu) x threads <= cores)")
    a = ap.parse_args(); GPU_IDS[:] = [g.strip() for g in a.gpu_ids.split(",") if g.strip()]; THREADS = a.threads
    GEN = {"rq1": rq1_jobs, "rq2": rq2_jobs, "rq3": rq3_jobs, "rq4": rq4_jobs, "rq5": rq5_jobs, "cad": cad_jobs, "ext": ext_jobs, "ext1": ext1_jobs, "rq2lb": rq2lb_jobs, "noise": noise_jobs,
           "scale": scale_jobs}
    jobs = [(j, s) for rq in a.rq.split("+") for s in a.seeds for j in GEN[rq](s)]
    if a.rq == "cad":   # most informative first: ToN, then CIC18, CIC17, CIDDS; 1 h before 4 h
        prio = {"ToN": 0, "CIC18": 1, "CIC17": 2, "CIDDS": 3}
        jobs.sort(key=lambda js: (prio[js[0][1].split("_")[0]], js[0][2].endswith("_P4"), js[1]))
    if a.dry:
        for j, s in jobs: print(s, j[0], j[1], j[2])
        sys.exit()
    with ThreadPoolExecutor(a.gpu) as G, ThreadPoolExecutor(a.cpu) as C:
        futs = [(G if j[0] == "gpu" else C).submit(run, j, s) for j, s in jobs]
        for f in futs: print(f.result(), flush=True)
    print("ALL JOBS DONE", flush=True)
