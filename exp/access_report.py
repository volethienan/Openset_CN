"""Regenerate Access evidence from saved predictions, keeping label policies in separate panels."""
from pathlib import Path
import os
import hashlib
import json
import numpy as np
import pandas as pd
from evaluate import run_metrics, seeded_name
from inference import stats, holm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / os.environ.get('ERNP_ACCESS_OUT', 'exp/reports/access')  # K5A: exp/reports/access_k5a
OURS = os.environ.get('ERNP_METHOD', 'ernp_K5')  # final configuration: ernp_K5A
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
    (OUT / 'result_hashes.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    meta = dict(expected_runs=280, collected_runs=len(df), missing=missing,
                scope='post_hoc_audit_of_saved_predictions_not_new_preregistration',
                inferential_unit='source_dataset_seed_and_scenario_averaged',
                correction='Holm_within_each_metric_5_baselines_plus_20_test_two_sided_sensitivity',
                cohorts='all_three_reported_without_selecting_the_best_policy',
                flow_protocol='retrospective_complete_flow_start_order_model_feedback_predict_then_learn',
                labels='group_representative_simulation_not_measured_SOC_workload',
                roc_at_1fpr='retrospective_test_benign_quantile_not_deployable_calibration',
                first_period='causal_first_update_period_per_family_not_all_stream_zero_shot',
                code_hashes={n: hashlib.sha256((ROOT / 'exp' / n).read_bytes()).hexdigest()
                             for n in ['evaluate.py', 'inference.py', 'ernp.py', 'access_report.py']})
    (OUT / 'manifest.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    text = ['# Current evidence, label-policy sensitivity and claim boundaries', '',
            'All values below were regenerated from saved scores/flags. No model was retrained.',
            'SIMARGL exclusion is a suspected label issue; neither label policy is certified ground truth.',
            'Seeds are repeated measurements of a dataset. Both test directions and Holm corrections are exported.',
            'Existing context features are retrospective complete-flow features; feedback order alone is causal.', '',
            f'## ERNP ({OURS}) across the three label-policy cohorts', '']
    for frame in macro:
        text += [frame[frame.method == OURS].round(4).to_string(index=False), '']
    text += ['## Per-dataset evidence', '', means.round(4).to_string(), '',
             '## Limits supported by existing data', '',
             '- ToN: high operating FPR and low low-FPR zero-day recall; no universal benign-drift claim.',
             '- CIC17_full: lower F1 than CADE and substantial seed variation; report separately from sampled CIC17.',
             '- Group camouflage: representative-only can beat full propagation; no universal synergy claim.',
             '- ROC@1% uses evaluation labels; report actual operating FPR beside it.',
             '- First-period and macro-family recall are separate from pooled whole-stream recall.',
             '- CPU benchmark is score-only with explicit K=5; no end-to-end line-rate or SOC labor claim.',
             '- Retraction is local in prior weights; refitting a GBM does not guarantee local score changes.', '',
             '## Inferential results', '', tests.round(6).to_string(index=False), '']
    (OUT / 'EVIDENCE.md').write_text('\n'.join(text), encoding='utf-8')
    print(f'REPORT COMPLETE {len(df)}/280 runs; missing={len(missing)} -> {OUT}', flush=True)
    if missing:
        raise RuntimeError(f'incomplete main cohort: {missing}')


if __name__ == '__main__':
    main()
