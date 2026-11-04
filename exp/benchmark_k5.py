"""Explicit-K scoring benchmark; excludes extraction, triage and analyst work."""
import argparse
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=['CIDDS_ctx', 'ToN_ctx', 'CIC18'])
    ap.add_argument('--n-prior', type=int, default=5)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--batch-size', type=int, default=100_000)
    ap.add_argument('--repeats', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', type=Path, default=Path(__file__).parent / 'reports/benchmark_k5.json')
    a = ap.parse_args()
    if min(a.n_prior, a.threads, a.batch_size, a.repeats) < 1:
        ap.error('K, threads, batch size and repeats must be positive')
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = str(a.threads)
    os.environ['EXP_SEED'] = str(a.seed)
    import hashlib
    import json
    import platform
    import time
    import numpy as np
    import psutil
    import sklearn
    from threadpoolctl import threadpool_limits
    from common import load, seed
    from sklearn.neighbors import NearestNeighbors
    import ernp
    output = dict(scope='initial_fit_and_score_only_complete_flow_features', n_prior=a.n_prior,
                  seed=a.seed, threads=a.threads, repeats=a.repeats, batch_requested=a.batch_size,
                  cpu=platform.processor(), logical_cpus=os.cpu_count(), platform=platform.platform(),
                  python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__,
                  source_sha256=hashlib.sha256(Path(ernp.__file__).read_bytes()).hexdigest(),
                  statistic='median_after_one_untimed_warmup', rows=[])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=a.threads):
        for ds in a.datasets:
            print(f'fit {ds} K={a.n_prior} threads={a.threads}', flush=True)
            c = dict(ernp.DEFAULTS, n_prior=a.n_prior)
            df, X, src, dst, tr, ho, s_idx, anc = load(ds)
            fam = df.fam.to_numpy(); rng = np.random.default_rng(seed())
            P = X[rng.choice(tr, min(c['n_pos_max'], len(tr)), replace=False)]
            ka = np.where(anc & (fam != 'BENIGN'))[0]
            A = X[rng.choice(ka, min(c['n_known_max'], len(ka)), replace=False)] if len(ka) else np.zeros((0, X.shape[1]), X.dtype)
            del df, src, dst, fam
            m = ernp.ERNP(c, rng)
            m.eps = c['eps_mult'] * np.quantile(NearestNeighbors(n_neighbors=2).fit(P).kneighbors(P[:5000])[0][:, 1], .9)
            t0 = time.perf_counter(); m.fit(P, A); fit_s = time.perf_counter() - t0
            Z = X[s_idx[:a.batch_size]]; baseline = m.score(Z)
            elapsed, max_diff = [], 0.
            for _ in range(a.repeats):
                t0 = time.perf_counter(); score = m.score(Z); elapsed.append(time.perf_counter() - t0)
                max_diff = max(max_diff, float(np.max(np.abs(score - baseline))))
            actual_models = len(m.clf) if isinstance(m.clf, list) else 1
            if actual_models != a.n_prior or not np.isfinite(baseline).all():
                raise RuntimeError('benchmark model count or finite-score check failed')
            row = dict(dataset=ds, cfg=c, models_trained=actual_models, batch_size=len(Z), initial_fit_s=fit_s, score_seconds=elapsed,
                       median_score_s=float(np.median(elapsed)), median_flows_per_s=float(len(Z) / np.median(elapsed)),
                       max_repeat_score_diff=max_diff, process_rss_mb=psutil.Process().memory_info().rss / 2**20,
                       available_ram_mb=psutil.virtual_memory().available / 2**20)
            output['rows'].append(row); a.out.write_text(json.dumps(output, indent=2), encoding='utf-8')
            print(f'{ds}: fit {fit_s:.2f}s, median {row["median_flows_per_s"]:,.0f} flows/s, K={a.n_prior}', flush=True)
            del X, Z, P, A, m, score, baseline
    print(f'BENCH COMPLETE {a.out}', flush=True)


if __name__ == '__main__':
    main()
