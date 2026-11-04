"""Main comparison of the paper from saved predictions (per-run metrics, dataset means, tests), keeping the SIMARGL
label policies in separate panels."""
from pathlib import Path
import os
import hashlib
import json
import numpy as np
import pandas as pd
from evaluate import run_metrics, seeded_name
from inference import stats, holm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / os.environ.get('ERNP_ACCESS_OUT', 'exp/reports')
OURS = os.environ.get('ERNP_METHOD', 'ernp_K5A')  # final configuration
DS = ['CIC17', 'CIC18', 'CIDDS_ctx', 'ToN_ctx', 'CTU13', 'SIMARGL', 'APPRAISE', 'SIMARGL_all']
METHODS = [OURS, 'ssf_causal', 'cade', 'trident', 'memstream', 'owad', 'knn']
BASELINES = METHODS[1:-1]
METRICS = ['F1', 'MCC', 'AUROC', 'zd_recall_at1fpr']


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, manifest, missing = [], [], []
    for ds in DS:
        for seed in range(5):
            for method in METHODS:
                row = run_metrics(ds, method, seed)
                if row is None:
                    missing.append([ds, method, seed]); continue
                rows.append(row)
                stem = ROOT / 'exp/results' / f'{seeded_name(ds, seed)}__{method}__s{seed}'
                manifest.append({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (stem.with_suffix('.npz'), stem.with_suffix('.json'))})
        print(f'collected {ds}', flush=True)
    df = pd.DataFrame(rows); df.to_csv(OUT / 'main_runs.csv', index=False)
    values = METRICS + ['FPR', 'zero_day_recall', 'macro_zero_day_recall', 'macro_zd_recall_at1fpr',
                       'macro_zd_first_period_recall', 'macro_zd_first_period_at1fpr', 'pooled_zd_first_period_recall']
    means = df.groupby(['dataset', 'method'])[values].mean()
    means.to_csv(OUT / 'dataset_means.csv')
    cohorts = {'historical_filtered_7': DS[:7], 'published_labels_7': DS[:5] + ['SIMARGL_all', 'APPRAISE'],
               'without_uncertain_SIMARGL_6': DS[:5] + ['APPRAISE']}
    tests, macro = [], []
    for label, datasets in cohorts.items():
        data = df[df.dataset.isin(datasets)]
        avg = data.groupby(['dataset', 'method'])[values].mean().groupby('method').mean()
        avg.insert(0, 'cohort', label); macro.append(avg.reset_index())
        for metric in METRICS:
            t, ranks, meta = stats(data[data.method.isin([OURS] + BASELINES)], ours=OURS, metric=metric)
            t = t.rename_axis('baseline').reset_index(); t['metric'] = metric; t['cohort'] = label
            tests.append(t)
    tests = pd.concat(tests, ignore_index=True)
    for _, indices in tests.groupby('cohort').groups.items():
        tests.loc[indices, 'p_two_sided_holm_all4metrics'] = holm(tests.loc[indices, 'p_two_sided'])
    tests.to_csv(OUT / 'dataset_tests.csv', index=False)
    pd.concat(macro, ignore_index=True).to_csv(OUT / 'cohort_means.csv', index=False)
    (OUT / 'main_result_hashes.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    meta = dict(method=OURS, expected_runs=280, collected_runs=len(df), missing=missing,
                inferential_unit='dataset (seeds and scenarios averaged within a dataset)',
                correction='Holm over the five baselines within each metric',
                code_hashes={n: hashlib.sha256((ROOT / 'exp' / n).read_bytes()).hexdigest()
                             for n in ['evaluate.py', 'inference.py', 'ernp.py', 'access_report.py']})
    (OUT / 'main_manifest.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(f'REPORT COMPLETE {len(df)}/280 runs; missing={len(missing)} -> {OUT}', flush=True)
    if missing:
        raise RuntimeError(f'incomplete main cohort: {missing}')


if __name__ == '__main__':
    main()
