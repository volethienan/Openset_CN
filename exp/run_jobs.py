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
             "abl_prior_marginal": ["--prior", "marginal"], "abl_prior_scarf": ["--prior", "scarf"], "abl_retr_none": ["--retraction", "none"],
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
    jobs = []                                  # seeds 0-2 (first campaign); seeds 3-4 added 01/10/2026 for five-seed tables
    for ds in ds_names(seed).values():
        for h in (1, 4):
            jobs.append(("cpu", ds, f"ernp_P{h}", ["ernp.py", ds, "--period", str(h), "--name", f"ernp_P{h}"]))
            for m in ("ssf_causal", "owad"):
                # SSF retrains 20 epochs on its whole buffer at every update (~5 min on ToN) -> 1 h cadence ~ 8 h/run:
                # SSF at 1 h only on ToN seed 0 (the dataset motivating the study); SSF at 4 h everywhere
                if m == "ssf_causal" and h == 1 and not (ds == "ToN_ctx" and seed == 0): continue
                jobs.append(("gpu", ds, f"{m}_P{h}", ["baselines/sota.py", ds, f"{m}_P{h}"]))
    return jobs

def opf_jobs(seed):
    """operating-point check (01/10/2026): ERNP with tau at 0.2% holdout FPR instead of 1%, every dataset - shows what
    closing the F1 gap to SSF on CIDDS-001 costs in zero-day recall (the 1% default is unchanged)"""
    return [("cpu", ds, "ernp_F002", ["ernp.py", ds, "--fpr", "0.002", "--name", "ernp_F002"]) for ds in ds_names(seed).values()]

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

def x10_jobs(seed):
    """scale study (03/10/2026): CSE-CIC-IDS2018 resampled at 10x the main sample (4M flows), same protocol and
    defaults; seeds 0-2 (each seed is an independent sample). Streams: exp/data/build_x10.sh."""
    if seed > 2: return []
    ds = "CIC18_x10" if seed == 0 else f"CIC18_x10_s{seed}"
    return [("cpu", ds, "ernp", ["ernp.py", ds])] + [("gpu", ds, m, ["baselines/sota.py", ds, m]) for m in SOTA5]

def running_elsewhere(args, seed=None):
    """True if the same job (same command AND same EXP_SEED) is already running in another process"""
    try:
        import psutil
        for p in psutil.process_iter(["cmdline"]):
            cl = p.info["cmdline"] or []
            if len(cl) > len(args) and cl[-len(args):] == args and p.pid != os.getpid():
                try: same_seed = seed is None or p.environ().get("EXP_SEED", "0") == str(seed)
                except Exception: same_seed = True          # environment unreadable: stay conservative
                if same_seed: return True
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

def camo_jobs(seed):
    """group camouflage (data/inject_camo.py): payload hidden in alert groups with benign decoys, ratio 1 and 3;
    ERNP (group labels), ERNP representative-only, ERNP without labels, SSF"""
    jobs = []
    for r in (1, 3):
        ds = f"{cidds(seed)}_injcamo{r}_bf"
        jobs += [("cpu", ds, "ernp", ["ernp.py", ds]),
                 ("cpu", ds, "abl_evid_rep", ["ernp.py", ds, "--evidence", "rep", "--name", "abl_evid_rep"]),
                 ("cpu", ds, "ernp_B0", ["ernp.py", ds, "--budget", "0", "--name", "ernp_B0"]),
                 ("gpu", ds, "ssf_causal", ["baselines/sota.py", ds, "ssf_causal"])]
    return jobs

def scarf_jobs(seed):
    """RQ4 add-on: SCARF/VIME-style corruption (independent donor per feature) instead of the single-donor swap"""
    return [("cpu", ds, "abl_prior_scarf", ["ernp.py", ds, "--prior", "scarf", "--name", "abl_prior_scarf"]) for ds in ds_names(seed).values()]

def audit_jobs(seed):
    """RQ1 ERNP rerun with the group-purity audit logged (same defaults; scores must equal the ernp runs)"""
    return [("cpu", ds, "ernp_audit", ["ernp.py", ds, "--name", "ernp_audit"]) for ds in ds_names(seed).values()]

def rq2gate_jobs(seed):
    """review 2: other evidence-gated designs under poisoning and camouflage - GBM+AL (no prior) and ERNP without retraction"""
    dss = [f"{cidds(seed)}_{v}" for v in INJ] + [f"{cidds(seed)}_injcamo{r}_bf" for r in (1, 3)]
    return [("cpu", ds, n, ["ernp.py", ds, *a, "--name", n]) for ds in dss
            for n, a in (("abl_prior_none", ["--prior", "none"]), ("abl_retr_none", ["--retraction", "none"]))]

def e23_jobs(seed):
    """Q1 revision. E3/mechanism control: ERNP's own feedback schedule (alerts -> groups -> decisions) replayed into ERNP
    without retraction and GBM+AL (no prior), so only the model differs. E2: B = 25 groups with at most 200 flows per
    day entering the evidence (= SSF's 200 labelled flows), plus the B = 25 runs of RQ3 extended to seeds 3-4."""
    n = ds_names(seed)
    jobs = [("cpu", ds, nm, ["ernp.py", ds, "--replay", "ernp", *a, "--name", nm]) for ds in n.values()
            for nm, a in (("rpl_retr_none", ["--retraction", "none"]), ("rpl_prior_none", ["--prior", "none"]))]
    for ds in (n["ToN"], n["CIC18"]):
        jobs += [("cpu", ds, nm, ["ernp.py", ds, "--budget", "25", *a, "--name", nm]) for nm, a in
                 (("ernp_B25_cap200", ["--flow_cap", "200"]), ("ernp_B25", []), ("abl_evid_rep_B25", ["--evidence", "rep"]))]
    return jobs

def e5_jobs(seed):
    """Q1 revision E5: analyst labels arrive DELAY days after the alerts (same delay for every label-using method);
    CIC17 has two stream days, so a delay leaves it without any update and it is not included"""
    n = ds_names(seed); jobs = []
    for ds in (n["CIDDS"], n["ToN"], n["CIC18"]):
        jobs += [("cpu", ds, f"ernp_D{D}", ["ernp.py", ds, "--delay", str(D), "--name", f"ernp_D{D}"]) for D in (1, 2)]
        jobs += [("gpu", ds, f"{m}_D1", ["baselines/sota.py", ds, f"{m}_D1"]) for m in ("ssf_causal", "owad")]
    return jobs

def e4_jobs(seed):
    """Q1 revision E4: slow poisoning with protocol-valid flow records (data/inject_valid.py), bruteForce payload;
    ERNP, its evidence-gated variants and ERNP without labels, and the five SOTA, as in RQ2"""
    jobs = []
    for v in ("injvalidinit_bf", "injvalidresp_bf"):
        ds = f"{cidds(seed)}_{v}"
        jobs += [("cpu", ds, n, ["ernp.py", ds, *a, "--name", n]) for n, a in
                 (("ernp", []), ("ernp_B0", ["--budget", "0"]), ("abl_evid_rep", ["--evidence", "rep"]),
                  ("abl_retr_none", ["--retraction", "none"]), ("abl_prior_none", ["--prior", "none"]),
                  ("abl_self", ["--evidence_source", "self"]), ("ernp_B25", ["--budget", "25"]))]   # counterfactual, small budget
        jobs += [("gpu", ds, m, ["baselines/sota.py", ds, m]) for m in SOTA5]
    return jobs

REAL = ("CTU13", "SIMARGL", "APPRAISE")                  # real-network datasets (data/prepare_real.py), held out from development.
# LITNET-2020 was examined and left out: its 'normal' class contains unlabelled scanning and attack traffic
# (exp/reports/litnet_check.txt).

def realsens_jobs(seed):
    """sensitivity of RoEduNet-SIMARGL2021 to the unlabelled FIN scan: the stream with the published labels (SIMARGL_all)"""
    ds = "SIMARGL_all" + ("" if seed == 0 else f"_s{seed}")
    return [("cpu", ds, "ernp", ["ernp.py", ds]), ("cpu", ds, "knn", ["baselines/classic.py", ds, "knn"])] + \
           [("gpu", ds, m, ["baselines/sota.py", ds, m]) for m in SOTA5]

def real_jobs(seed):
    """the RQ1 protocol on the real-network datasets, unchanged defaults: ERNP and its label variants, GBM+AL,
    the five SOTA (plus the OWAD learning-rate variant and CADE/Trident at the anchor threshold) and the classical references"""
    jobs = []
    for ds in (f"{b}" + ("" if seed == 0 else f"_s{seed}") for b in REAL):
        jobs += [("cpu", ds, n, ["ernp.py", ds, *a, "--name", n]) for n, a in
                 (("ernp", []), ("abl_evid_rep", ["--evidence", "rep"]), ("abl_prior_none", ["--prior", "none"]), ("ernp_B0", ["--budget", "0"]))]
        jobs += [("cpu", ds, m, ["baselines/classic.py", ds, m]) for m in ("knn", "iforest")]
        jobs.append(("gpu", ds, "ae_static", ["baselines/classic.py", ds, "ae_static"]))
        jobs += [("gpu", ds, m, ["baselines/sota.py", ds, m]) for m in SOTA5 + ["owad_lr3", "cade_t1", "trident_t1"]]
    return jobs

def stab_jobs(seed):
    """stability study (development seeds first): three changes to ERNP and their combinations, on all seven datasets.
    keepA: the anchor attacks never leave the attack evidence and one attack decision adds at most 100 flows (with the
    FIFO window, one large attack family evicts every earlier one); K5: five prior draws per fit, scores averaged;
    post: decision by the model's own posterior (score > 0) instead of the 1% holdout quantile. Cheap variants first."""
    KA, K5, PO = ["--keep", "A", "--att_cap", "100"], ["--n_prior", "5"], ["--rule", "post"]
    V = (("ernp_keepA", KA), ("ernp_post", PO), ("ernp_keepA_post", KA + PO), ("ernp_keepAP", ["--keep", "AP", "--att_cap", "100"]),
         ("ernp_K5", K5), ("ernp_keepA_K5", KA + K5), ("ernp_K5_post", K5 + PO), ("ernp_keepA_K5_post", KA + K5 + PO))
    dss = [f"{b}" + ("" if seed == 0 else f"_s{seed}") for b in ("SIMARGL", "CTU13")] + [ds_names(seed)[k] for k in ("CIDDS", "CIC17", "ToN", "CIC18")] +           ["APPRAISE" + ("" if seed == 0 else f"_s{seed}")]
    return [("cpu", ds, n, ["ernp.py", ds, *a, "--name", n]) for n, a in V for ds in dss]

def stab2_jobs(seed):
    """stability study, second round, on the K5 base (five prior draws per fit): the mutation operator in the prior
    (25% of prior rows get 1-2 feature values outside what benign has shown), the anchored attack evidence without a
    per-decision cap or with a cap of 1000 flows (the cap of 100 in `stab` cost ToN and CIDDS), and mutation + anchor."""
    K5 = ["--n_prior", "5"]
    V = (("ernp_K5_keepA0", K5 + ["--keep", "A"]), ("ernp_K5_keepA1k", K5 + ["--keep", "A", "--att_cap", "1000"]),
         ("ernp_K5_mut", K5 + ["--mut", "0.25"]), ("ernp_K5_mut_keepA0", K5 + ["--mut", "0.25", "--keep", "A"]))   # anchored evidence first
    dss = [f"{b}" + ("" if seed == 0 else f"_s{seed}") for b in ("SIMARGL", "CTU13")] + [ds_names(seed)[k] for k in ("CIDDS", "CIC17", "ToN", "CIC18")] +           ["APPRAISE" + ("" if seed == 0 else f"_s{seed}")]
    return [("cpu", ds, n, ["ernp.py", ds, *a, "--name", n]) for n, a in V for ds in dss]

def k5_dss(seed):
    sfx = "" if seed == 0 else f"_s{seed}"
    return [f"SIMARGL{sfx}", f"CTU13{sfx}"] + [ds_names(seed)[k] for k in ("CIDDS", "CIC17", "ToN", "CIC18")] + [f"APPRAISE{sfx}"]

def k5_jobs(seed):
    """main configuration from 10/2026 on: ERNP with five prior draws per fit (scores averaged), seven datasets"""
    return [("cpu", ds, "ernp_K5", ["ernp.py", ds, "--n_prior", "5", "--name", "ernp_K5"]) for ds in k5_dss(seed)]

def k5abl_jobs(seed):
    """the main ablations rerun on the K5 base: no online labels, representative-only evidence, no retraction, and the
    SCARF prior (the no-prior ablation has no prior draw, so the existing abl_prior_none run is the K5 counterpart)"""
    V = (("ernp_B0_K5", ["--budget", "0"]), ("abl_evid_rep_K5", ["--evidence", "rep"]),
         ("abl_retr_none_K5", ["--retraction", "none"]), ("abl_prior_scarf_K5", ["--prior", "scarf"]))
    return [("cpu", ds, n, ["ernp.py", ds, "--n_prior", "5", *a, "--name", n]) for n, a in V for ds in k5_dss(seed)]

def k5rq_jobs(seed):
    """every ERNP run of the secondary experiments (RQ2-RQ5, operating point, E2-E5, scale, label noise,
    camouflage, the self-evidence counterfactual, SIMARGL published labels; the cadence study is k5cad) rerun on the K5 base: same arguments plus
    --n_prior 5, name suffixed _K5 (ernp -> ernp_K5), replays of ERNP's feedback schedule replay ernp_K5. Baselines are
    not rerun. The no-prior ablation draws no prior, so its existing runs stand for K5; the random-forest sensitivity run
    stays at K = 1 (K > 1 is implemented for the GBM only). Most important groups first."""
    groups = (rq2_jobs, camo_jobs, rq2gate_jobs, e4_jobs, noise_jobs, rq2lb_jobs, e5_jobs, e23_jobs, rq3_jobs, rq5_jobs,
              rq4_jobs, opf_jobs, ext_jobs, realsens_jobs, scale_jobs, x10_jobs)
    return k5_variant(groups, seed)

def k5_variant(groups, seed):
    jobs, seen = [], set()
    for g in groups:
        for pool, ds, name, args in g(seed):
            if args[0] != "ernp.py" or ("--prior" in args and args[args.index("--prior") + 1] == "none") or "--clf" in args: continue
            nm = "ernp_K5" if name == "ernp" else name + "_K5"
            if (ds, nm) in seen: continue
            a = list(args)
            if "--replay" in a: a[a.index("--replay") + 1] += "_K5"
            if "--name" in a: a[a.index("--name") + 1] = nm
            else: a += ["--name", nm]
            jobs.append(("cpu", ds, nm, a + ["--n_prior", "5"])); seen.add((ds, nm))
    return jobs

def k5cad_jobs(seed):
    """the cadence study on the K5 base, kept apart from k5rq because it costs more than all of k5rq together
    (the 1 h cadence refits 24 times a day): the 4 h cadence first, then the 1 h cadence"""
    return [j for j in k5_variant((cad_jobs,), seed) if j[2] == "ernp_P4_K5"] + [j for j in k5_variant((cad_jobs,), seed) if j[2] == "ernp_P1_K5"]

def k5a_jobs(seed):
    """final configuration (10/2026): K5 + anchored attack evidence (--keep A: the anchor's known attacks never leave the
    attack evidence; with the FIFO window one large confirmed family evicted them, e.g. CIC17 unsampled and SIMARGL).
    Every K5 job of k5, k5abl, k5rq and k5cad with --keep A, name suffix _K5A (ernp_K5 -> ernp_K5A), replays of
    ernp_K5A. Runs whose attack evidence never exceeded the window before the last scored period are identical to
    their K5 counterpart by construction; exp/k5a_copy.py copies those, so only the affected runs are left to run."""
    out = []
    for pool, ds, name, args in k5_jobs(seed) + k5abl_jobs(seed) + k5rq_jobs(seed) + k5cad_jobs(seed):
        nm = name[:-3] + "_K5A"; a = list(args)
        a[a.index("--name") + 1] = nm
        if "--replay" in a: a[a.index("--replay") + 1] = a[a.index("--replay") + 1].replace("_K5", "_K5A")
        out.append((pool, ds, nm, a + ["--keep", "A"]))
    return out

def t1_jobs(seed):
    """review 2: CADE and Trident with the threshold at 1% FPR on held-out anchor flows (same models)"""
    return [("gpu", ds, m, ["baselines/sota.py", ds, m]) for ds in ds_names(seed).values() for m in ("cade_t1", "trident_t1")]

def diam_jobs(seed):
    """review 2: RQ1 ERNP rerun logging each group's radius around its representative (footprint, Prop. 1(iii))"""
    return [("cpu", ds, "ernp_diam", ["ernp.py", ds, "--name", "ernp_diam"]) for ds in ds_names(seed).values()]

def rq2rep_jobs(seed):
    """RQ2 with representative-only labels: does ERNP's robustness need group-label propagation?"""
    return [("cpu", f"{cidds(seed)}_{v}", "abl_evid_rep", ["ernp.py", f"{cidds(seed)}_{v}", "--evidence", "rep", "--name", "abl_evid_rep"])
            for v in INJ]

def run(job, seed):
    pool, ds, name, args = job
    out = os.path.join(RES, f"{ds}__{name}__s{seed}.json")
    if os.path.exists(out): return f"skip {ds} {name}"
    if running_elsewhere(args, seed): return f"skip {ds} {name} (already running in another process)"
    env = dict(os.environ, EXP_SEED=str(seed)); log = os.path.join(LOG, f"{ds}__{name}__s{seed}.log")
    if pool == "cpu": env["CUDA_VISIBLE_DEVICES"] = ""
    if THREADS: env.update({v: str(THREADS) for v in _THREAD_VARS})   # cap BLAS/OpenMP/torch threads per job
    if pool == "gpu":
        with _LAUNCH:                                   # check the cap and start the process atomically
            while gpu_procs() >= MAX_GPU_PROCS: time.sleep(60)
            if os.path.exists(out) or running_elsewhere(args, seed): return f"skip {ds} {name}"
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
    ap.add_argument("--pool", choices=("all", "cpu", "gpu"), default="all", help="run only the CPU or only the GPU jobs")
    a = ap.parse_args(); GPU_IDS[:] = [g.strip() for g in a.gpu_ids.split(",") if g.strip()]; THREADS = a.threads
    GEN = {"rq1": rq1_jobs, "rq2": rq2_jobs, "rq3": rq3_jobs, "rq4": rq4_jobs, "rq5": rq5_jobs, "cad": cad_jobs, "ext": ext_jobs, "ext1": ext1_jobs, "rq2lb": rq2lb_jobs, "noise": noise_jobs,
           "scale": scale_jobs, "opf": opf_jobs, "x10": x10_jobs, "camo": camo_jobs, "rq2rep": rq2rep_jobs, "scarf": scarf_jobs, "audit": audit_jobs, "rq2gate": rq2gate_jobs, "t1": t1_jobs, "e23": e23_jobs, "e5": e5_jobs, "e4": e4_jobs, "real": real_jobs, "stab": stab_jobs, "stab2": stab2_jobs, "k5": k5_jobs, "k5abl": k5abl_jobs, "k5rq": k5rq_jobs, "k5cad": k5cad_jobs, "k5a": k5a_jobs, "realsens": realsens_jobs, "diam": diam_jobs}
    jobs = [(j, s) for rq in a.rq.split("+") for s in a.seeds for j in GEN[rq](s) if a.pool in ("all", j[0])]
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
