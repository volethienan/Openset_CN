"""Run OFFICIAL SOTA implementations under the mock-test protocol (same streams, features, anchor split).

MemStream (WWW'22)  : official class code (exec'd from repo), 0 labels, memory-admission beta = 0.1 (authors' value for
                      the intrusion datasets), alert threshold at 1% holdout FPR. Results named `memstream_adapt` are the
                      earlier runs with beta = the alert threshold (adaptive memory; kept as a sensitivity variant).
Trident (WWW'24)    : the authors' demo retrains every 300 flows on the ground-truth labels of the whole stream so far;
                      under a query budget no such labels exist, so Trident is run update-free after anchor training.
OWAD (NDSS'23)      : official AE + Calibrator + ShiftHunter; daily shift test, explain+adapt with a label
                      budget of LBL labels/day (simulated analyst labelling, as in the official demo).
SSF (INFOCOM'25)    : official AE_classifier / InfoNCE / drift detection / mask optimisation / sample selection;
                      initial labelled training on the anchor period (benign + attacks seen there), then one
                      update interval per day with LBL oracle labels per interval (official protocol:
                      predict each interval after updating on it).
Trident (WWW'24)    : official per-class autoencoders + SPOT/EVT thresholds, trained on anchor-period labels;
                      unknown (no class accepts) or known-attack class => alert. No oracle retraining.
FOSS (ToN'24)       : NOT runnable - the public repo does not contain the FOSS / Testingpro implementation.
"""
import sys, os, json, time, types, copy, math
import numpy as np, pandas as pd, torch
_EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOTA = os.environ.get("ERNP_ROOT", os.path.dirname(_EXP)).replace("\\", "/").rstrip("/") + "/sota/"
sys.path.insert(0, _EXP); sys.path.insert(0, os.path.join(_EXP, "baselines"))
from common import load, calib_upper, FPR_TARGET, save, seed, stream_periods, period_budget
LBL = 200
CAUSAL = False
PERIOD = 24                                          # update cadence in hours (cadence study; 24 = daily)
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPOCH_1 = 20                                         # SSF epoch_1 (suffix _E{n}; the authors' UNSW command uses 180)
WINDOW = 0                                           # SSF native protocol: windows of WINDOW flows (suffix _W{n}; 0 = off)
NOISE = 0.0                                          # oracle label noise (suffix _N{pct}); 0 = exact labels
DELAY = 0                                            # label delay in periods (suffix _D{n}): the labelled batch of period
                                                     # i - DELAY is used by the update at the end of period i (E5)
TIMING = dict(init_s=0.0, score_s=0.0, n_scored=0, update_s=0.0, n_updates=0)   # cost reporting only (RQ5)

def _now():                                          # wall clock after the GPU queue is drained (fair timing)
    if dev.type == "cuda": torch.cuda.synchronize()
    return time.time()

def noisy(lab, key):
    """labels the analyst returns: each flipped w.p. NOISE (deterministic per seed and period); NOISE = 0 -> lab itself"""
    if NOISE <= 0: return lab
    import zlib; r = np.random.default_rng([10_000 + seed(), zlib.crc32(str(key).encode())])   # stable across processes
    return np.where(r.random(len(lab)) < NOISE, 1 - lab, lab)

def days_of(df, idx):
    if WINDOW: return np.arange(len(idx)) // WINDOW   # SSF's sample_interval (stream is time-ordered)
    return stream_periods(df, idx, PERIOD)            # calendar days when PERIOD = 24

# ---------------------------------------------------------------- MemStream (official class)
def run_memstream_fixnorm(ds):
    """Sensitivity variant: identical to official MemStream except the input normalisation (mean/std) is frozen
    after initialisation. The official code recomputes it from memory contents, which degenerates (std -> 0)
    on duplicate-heavy flow data and makes the detector collapse."""
    return run_memstream(ds, fixnorm=True)

def run_memstream(ds, memlen=2048, epochs=5000, lr=1e-2, fixnorm=False, beta=0.1):
    """beta: memory-admission threshold, the authors' value for the intrusion datasets (README: NSL-KDD, UNSW-NB15,
    CICIDS-DoS all use beta = 0.1, memlen 2048). The alert threshold is separate: 1% FPR on the anchor holdout, as for
    every method. (Before 05/10/2026 the wrapper used the alert threshold as beta, 100-900x the authors' value on our
    feature scale; memory then admitted most attacks: CIDDS-001 seed 0 AUROC 0.26 vs 0.998 with beta = 0.1, see
    diagnose_memstream_beta.py. Earlier runs: exp/results_superseded/.)"""
    df, X, src, dst, tr, ho, s_idx, anc = load(ds)
    code = open(SOTA + "MemStream/code/memstream.py").read()
    cls = code[code.index("class MemStream"): code.index("if args.dataset in")]
    import torch.nn as nn, torch.nn.functional as F
    from torch.autograd import Variable
    ns = dict(torch=torch, nn=nn, F=F, Variable=Variable, device=dev, np=np)
    exec(cls, ns); torch.manual_seed(0)
    Xt = torch.tensor(X, dtype=torch.float32, device=dev)
    init = Xt[tr[:memlen]]
    model = ns["MemStream"](X.shape[1], {"beta": 0.0, "memory_len": len(init), "batch_size": 1, "lr": lr}).to(dev)
    ns["model"] = model                       # official code refers to the global name `model`
    t0 = _now(); model.mem_data = init; torch.set_grad_enabled(True); model.train_autoencoder(init, epochs=epochs)
    TIMING["init_s"] += _now() - t0
    torch.set_grad_enabled(False); model.initialize_memory(init)
    def score_only(Z):                        # score without memory update (calibration)
        new = (Z - model.mean) / model.std; new[:, model.std == 0] = 0
        e = model.encoder(new); return torch.cdist(e, model.memory, p=1).min(1).values.cpu().numpy()
    thr = calib_upper(score_only(Xt[ho]))                       # alert threshold (1% holdout FPR)
    model.max_thres = torch.tensor(float(beta)).to(dev)         # memory admission: the authors' beta
    if fixnorm:
        mu0, sd0 = model.mean.clone(), model.std.clone(); orig = model.update_memory
        def upd(output_loss, encoder_output, data):
            r = orig(output_loss, encoder_output, data); model.mean, model.std = mu0, sd0; return r
        model.update_memory = upd
    s = np.zeros(len(s_idx)); t0 = _now()
    for j, g in enumerate(s_idx):                   # streaming: scoring and memory update are one step
        s[j] = float(model(Xt[g: g + 1]))
    TIMING["score_s"] += _now() - t0; TIMING["n_scored"] += len(s_idx)
    return s, s > thr, dict(thr=thr, beta=float(beta), memory_updates=int(model.count), memlen=len(init), labels_used=0)

# ---------------------------------------------------------------- OWAD (official modules)
def run_owad(ds, ae_lr=None):
    """ae_lr: learning rate of OWAD's AE (None = the authors' configs.yml value, 1e-5). Variant owad_lr3 uses 1e-3:
    with the authors' value the AE is under-trained on CSE-CIC-IDS2018 (initial-model AUROC 0.58 vs 0.98 at 1e-3)."""
    import yaml
    _load = yaml.load; yaml.load = lambda f, Loader=yaml.SafeLoader: _load(f, Loader=Loader)   # PyYAML>=6 compat
    for p in ("OWAD", "OWAD/admodels", "OWAD/moudles"): sys.path.insert(0, SOTA + p)
    import myutils as U, AE
    from calibrator import Calibrator
    from shifthunter import ShiftHunter
    from sklearn.preprocessing import MinMaxScaler
    df, X, src, dst, tr, ho, s_idx, anc = load(ds)
    y = (df["fam"].values != "BENIGN").astype(int)
    U.set_random_seed()
    # Input: the shared standardised features, as for every other method. (Before 05/10/2026 the wrapper applied the
    # min-max scaling of OWAD's Kyoto demo on top of them; on features already standardised and clipped at +-10 sd this
    # squeezes most flows into a narrow band and broke OWAD's AE on the CIC datasets: initial-model AUROC on CIC-IDS2017
    # 0.68 with min-max vs 0.96 without, see diagnose_owad_scaling.py. Earlier OWAD runs: exp/results_superseded/.)
    Z = (X if os.environ.get("OWAD_MINMAX", "0") != "1" else MinMaxScaler().fit(X[tr]).transform(X)).astype(np.float32)
    t0 = _now(); model, _ = (AE.train(Z[tr], Z.shape[1]) if ae_lr is None else AE.train(Z[tr], Z.shape[1], lr=ae_lr))
    TIMING["init_s"] += _now() - t0
    _, r_ho = AE.test(model, 0, Z[ho]); thr = calib_upper(r_ho)
    con = tr[:50000]; _, rmse_c = AE.test(model, thr, Z[con])
    day = days_of(df, s_idx); s = np.zeros(len(s_idx)); n_lab = 0; n_adapt = 0
    days = list(pd.unique(day))
    for j, d in enumerate(days):
        m = np.where(day == d)[0]; g = s_idx[m]
        t0 = _now(); _, s[m] = AE.test(model, thr, Z[g])          # predict first (causal)
        TIMING["score_s"] += _now() - t0; TIMING["n_scored"] += len(g)
        sc = s[m]
        if DELAY:                                                  # labels of period j - DELAY arrive now
            if j < DELAY: continue
            d = days[j - DELAY]; m = np.where(day == d)[0]; g = s_idx[m]; _, sc = AE.test(model, thr, Z[g])
        lb = period_budget(d, PERIOD, LBL)                         # = LBL when daily
        if len(g) < 200 or lb <= 0: continue
        t0 = _now(); cb = Calibrator(Z[con], method="Isotonic"); cb.set_calibrator(rmse_c, is_P_mal=True)
        sh = ShiftHunter(rmse_c, sc, cb, ad_type="Tab")
        if sh.permu_test() < U.get_params("ShiftDetector")["test_thres"]:
            sh.explainer(Z[g], noisy(y[g], d), label_num=lb); sh.adapter(model); model = sh.model
            _, rmse_c = AE.test(model, thr, Z[con])   # control scores under the adapted model (next round)
            n_lab += lb; n_adapt += 1
        TIMING["update_s"] += _now() - t0; TIMING["n_updates"] += 1
    return s, s > thr, dict(thr=thr, labels_used=n_lab, adaptations=n_adapt)

# ---------------------------------------------------------------- SSF (official functions)
def run_ssf(ds, epochs=200, epoch_1=None, opt_old_lr=24, opt_new_lr=50, new_sample_weight=60):
    epoch_1 = EPOCH_1 if epoch_1 is None else epoch_1
    # epoch_1 = 20 as in the authors' NSL-KDD example command (runtime); other values from their UNSW command
    sys.path.insert(0, SOTA + "SSF-Strategic-Selection-and-Forgetting")
    import importlib; U = importlib.import_module("utils")
    class _TorchCompat:   # official utils.py crashes (torch.cat of a 0-dim tensor) when exactly ONE unlabeled new sample
        """remains for pseudo-labelling - only reachable with tiny sub-daily batches (cadence study). Promoting 0-dim
        tensors to 1-dim changes nothing else: torch.cat raises on 0-dim inputs, so no working call is affected."""
        def __getattr__(self, k): return getattr(torch, k)
        @staticmethod
        def cat(ts, *a, **kw): return torch.cat([t.reshape(1) if t.dim() == 0 else t for t in ts], *a, **kw)
    U.torch = _TorchCompat()
    import torch.nn as nn, torch.nn.functional as F
    from torch.utils.data import TensorDataset, DataLoader
    df, X, src, dst, tr, ho, s_idx, anc = load(ds)
    y = (df["fam"].values != "BENIGN").astype(int)
    U.setup_seed(5011)
    init = np.sort(np.concatenate([tr, np.where(anc & (y == 1))[0]]))
    xtr = torch.tensor(X[init], device=dev); ytr = torch.tensor(y[init], device=dev).long()
    memory = len(init)
    model = U.AE_classifier(X.shape[1]).to(dev); teacher = U.AE_classifier(X.shape[1]).to(dev)
    crit = U.InfoNCELoss(dev, 0.02); bce = nn.BCELoss(reduction="none")
    opt = torch.optim.SGD(model.parameters(), lr=0.001)
    loader = DataLoader(TensorDataset(xtr, ytr), batch_size=128, shuffle=True)
    model.train(); t_init = _now()
    for ep in range(epochs):
        for xb, yb in loader:
            opt.zero_grad(); _, rv, cl = model(xb)
            loss = crit(rv, yb).mean() + bce(cl.squeeze(-1), yb.float()).mean(); loss.backward(); opt.step()
    teacher.load_state_dict(model.state_dict()); TIMING["init_s"] += _now() - t_init
    day = days_of(df, s_idx); prob = np.zeros(len(s_idx)); n_lab = 0; n_drift = 0
    days = list(pd.unique(day)); assert CAUSAL or not DELAY, "label delay needs the causal protocol"
    for j, d in enumerate(days):
        m = np.where(day == d)[0]; xte = torch.tensor(X[s_idx[m]], device=dev); yte = torch.tensor(noisy(y[s_idx[m]], d), device=dev).long()
        if CAUSAL:
            t0 = _now()
            with torch.no_grad(): prob[m] = model(xte)[2].squeeze(-1).cpu().numpy()   # causal: predict before learning the day
            TIMING["score_s"] += _now() - t0; TIMING["n_scored"] += len(m)
        if DELAY:                                                  # labels of period j - DELAY arrive now
            if j < DELAY: continue
            d = days[j - DELAY]; m = np.where(day == d)[0]
            xte = torch.tensor(X[s_idx[m]], device=dev); yte = torch.tensor(noisy(y[s_idx[m]], d), device=dev).long()
        lb = LBL if PERIOD == 24 else min(period_budget(d, PERIOD, LBL), len(m))   # daily: exactly the original loop
        if PERIOD != 24 and (lb <= 0 or len(m) < 2):               # sub-daily period without labels: no update
            if not CAUSAL:
                with torch.no_grad(): prob[m] = model(xte)[2].squeeze(-1).cpu().numpy()
            continue
        t_upd = _now()
        with torch.no_grad():
            tl = model(xte)[2].squeeze(-1); trl = model(xtr)[2].squeeze(-1)
        drift = U.detect_drift(tl, trl, len(m), 0.05)
        M_c = U.optimize_old_mask(trl.cpu().numpy(), tl.cpu().numpy(), dev, initialization="0.5-1", lr=opt_old_lr)
        M_t = U.optimize_new_mask(trl.cpu().numpy(), tl.cpu().numpy(), M_c, dev, initialization="0-0.5", lr=opt_new_lr)
        if drift:
            n_drift += 1
            xtr, ytr, _, new_mask = U.select_and_update_representative_samples_when_drift(xtr, ytr, xte, yte, M_c, M_t, lb, dev, memory, model)
        else:
            xtr, ytr, _, new_mask = U.select_and_update_representative_samples(xtr, ytr, xte, yte, M_c, M_t, lb, dev)
        n_lab += lb
        loader = DataLoader(TensorDataset(xtr, ytr, new_mask), batch_size=128, shuffle=True); model.train()
        for ep in range(epoch_1):
            for xb, yb, nm in loader:
                opt.zero_grad(); _, rv, cl = model(xb)
                w_con = crit(rv, yb) * ((1 - nm[yb == 0]) + nm[yb == 0] * new_sample_weight)
                w_cls = bce(cl.squeeze(-1), yb.float()) * ((1 - nm) + nm * new_sample_weight)
                loss = w_con.mean() + w_cls.mean()
                if not drift:
                    with torch.no_grad(): tlog = teacher(xb)[2]
                    loss = loss + 0.5 * F.mse_loss(cl, tlog)
                loss.backward(); opt.step()
        teacher.load_state_dict(model.state_dict()); TIMING["update_s"] += _now() - t_upd; TIMING["n_updates"] += 1
        if not CAUSAL:
            with torch.no_grad(): prob[m] = model(xte)[2].squeeze(-1).cpu().numpy()   # official: predict after update
    return prob, prob > 0.5, dict(labels_used=n_lab, drift_intervals=n_drift)

# ---------------------------------------------------------------- Trident (official model + EVT)
def _ae_err(mdl, Xn, bs=65536):
    """per-row reconstruction MSE, computed in GPU batches (same values as one full pass; needed for the 4M-flow streams)"""
    out = []
    with torch.no_grad():
        for i in range(0, len(Xn), bs):
            xb = torch.from_numpy(np.ascontiguousarray(Xn[i:i + bs])).float().to(dev)
            out.append(((mdl(xb) - xb) ** 2).mean(1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, np.float32)

def run_trident(ds, anchor_thr=False):
    sys.path.insert(0, SOTA + "Trident/code")
    import autoencoder as AEm, evt, random
    import torch.nn as nn
    df, X, src, dst, tr, ho, s_idx, anc = load(ds)
    fam = df["fam"].values
    idx = np.sort(np.concatenate([tr, np.where(anc & (fam != "BENIGN"))[0]]))
    data, label = X[idx].astype(np.float64), fam[idx]
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    lf = nn.MSELoss(); classes = [c for c, n in pd.Series(label).value_counts().items() if n > 50]
    mods, thr = [], []; t0 = _now()
    for c in classes:                                   # == official train(): 10 epochs, batch 10, Adam lr .01, SPOT q=.05
        Xc = data[label == c]; mdl = AEm.Autoencoder(X.shape[1], 256, 128, 64, 32).to(dev)   # GPU, same hyper-params
        opt = torch.optim.Adam(mdl.parameters(), lr=0.01, weight_decay=5e-4)
        Xg = torch.from_numpy(Xc).float().to(dev)
        for ep in range(10):
            order = torch.from_numpy(np.random.permutation(len(Xc))).to(dev)
            for b in range(len(Xc) // 10):
                xb = Xg[order[b * 10:(b + 1) * 10]]; loss = lf(mdl(xb), xb)
                opt.zero_grad(); loss.backward(); opt.step()
        mdl.eval()
        e = _ae_err(mdl, Xc)
        try:
            s = evt.SPOT(5e-2); s.fit(e, e); s.initialize(); res = s.run_simp()
            t_c = res["thresholds"][0] if res["thresholds"][0] > 0 else np.sort(s.init_data)[int(0.85 * s.init_data.size)]
        except Exception:   # degenerate (duplicate-heavy) error distribution: official fallback = 85th percentile
            t_c = np.sort(e)[int(0.85 * e.size)]
        thr.append(t_c)
        mods.append(mdl)
    TIMING["init_s"] += _now() - t0; t0 = _now()
    E = np.stack([_ae_err(m, X[s_idx]) for m in mods], 1)
    ok = E < np.array(thr)[None, :]; Em = np.where(ok, E, np.inf); best = Em.argmin(1)
    pred = np.where(ok.any(1), np.array(classes, dtype=object)[best], "UNKNOWN")
    flag = pred != "BENIGN"; TIMING["score_s"] += _now() - t0; TIMING["n_scored"] += len(s_idx)
    bi = classes.index("BENIGN"); score = E[:, bi] / thr[bi]
    if anchor_thr:   # review variant: threshold at 1% FPR on held-out anchor benign flows (as ERNP and kNN)
        t1 = calib_upper(_ae_err(mods[bi], X[ho]) / thr[bi]); flag = score > t1
        return score, flag, dict(classes=classes, labels_used=int(len(idx)), anchor_threshold=float(t1))
    return score, flag, dict(classes=classes, labels_used=int(len(idx)))

# ---------------------------------------------------------------- CADE (USENIX Security 2021; faithful PyTorch port)
def run_cade(ds, anchor_thr=False):
    import cade_port as C
    from sklearn.preprocessing import MinMaxScaler
    df, X, src, dst, tr, ho, s_idx, anc = load(ds); fam = df["fam"].values
    idx = np.sort(np.concatenate([tr, np.where(anc & (fam != "BENIGN"))[0]]))
    sc = MinMaxScaler().fit(X[idx]); Z = np.clip(sc.transform(X), -1, 2).astype(np.float32)   # original data in [0,1]
    t0 = _now(); st = C.train_cade(Z[idx], fam[idx], device=str(dev), seed=seed()); TIMING["init_s"] += _now() - t0
    t0 = _now(); closest, amin, drift, A = C.detect(st, Z[s_idx]); TIMING["score_s"] += _now() - t0; TIMING["n_scored"] += len(s_idx)
    bi = list(st["classes"]).index("BENIGN")
    flag = drift | (closest != "BENIGN")                      # unseen family OR nearest known family is an attack
    score = A[:, bi]                                          # MAD-normalised deviation from the benign family
    if anchor_thr:   # review variant: threshold at 1% FPR on held-out anchor benign flows (as ERNP and kNN)
        t1 = calib_upper(C.detect(st, Z[ho])[3][:, bi]); flag = score > t1
        return score, flag, dict(classes=list(st["classes"]), labels_used=0, anchor_threshold=float(t1))
    return score, flag, dict(classes=list(st["classes"]), labels_used=0, mad_threshold=3.5)

if __name__ == "__main__":
    ds, method = sys.argv[1], sys.argv[2]; t0 = time.time()
    import re                                            # suffixes: _P{hours} update cadence, _B{n} labels per day
    mm = re.fullmatch(r"(.+?)(?:_W(\d+))?(?:_E(\d+))?(?:_P(\d+))?(?:_B(\d+))?(?:_N(\d+))?(?:_D(\d+))?", method); base = mm[1]
    if mm[2]: WINDOW = int(mm[2])      # _W{n}: SSF windows of n flows, _E{n}: SSF epoch_1 (e.g. ssf_causal_W20000,
    if mm[3]: EPOCH_1 = int(mm[3])     # ssf_causal_E180), _P{h}: update cadence (owad_P1), _B{n}: labels/day (owad_B100)
    if mm[4]: PERIOD = int(mm[4])
    if mm[5]: LBL = int(mm[5])
    if mm[6]: NOISE = int(mm[6]) / 100
    if mm[7]: DELAY = int(mm[7])
    if base == "ssf_causal": CAUSAL = True
    fn = {"memstream": run_memstream, "memstream_fn": run_memstream_fixnorm, "owad": run_owad, "owad_lr3": lambda d: run_owad(d, ae_lr=1e-3), "ssf": run_ssf,
          "ssf_causal": run_ssf, "trident": run_trident, "cade": run_cade,
          "trident_t1": lambda d: run_trident(d, anchor_thr=True), "cade_t1": lambda d: run_cade(d, anchor_thr=True)}[base]
    out = fn(ds)
    score, flag, info = out; info["seconds"] = round(time.time() - t0)
    info["timing"] = dict(init_s=TIMING["init_s"], score_flows_per_s=TIMING["n_scored"] / max(TIMING["score_s"], 1e-9),
                          update_s_mean=TIMING["update_s"] / max(TIMING["n_updates"], 1), n_updates=TIMING["n_updates"])
    save(ds, method, score, flag, info)
