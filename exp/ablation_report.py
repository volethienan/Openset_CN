"""Component ablations of the final configuration (K5A) on the seven main datasets, from saved predictions.
Each ablation changes one component of ERNP K5A and keeps everything else (K = 5 prior draws, anchored attack evidence):
  ernp_B0_K5A          no online labels (anchor-only model)
  abl_evid_rep_K5A     only the queried representative is labelled (no group propagation)
  abl_retr_none_K5A    benign evidence never retracts the prior
  abl_prior_scarf_K5A  SCARF-style marginal corruption instead of the single-donor feature-swap prior
  abl_prior_none_K5A   no novelty prior: GBM on benign vs. attack evidence with the same feedback loop (GBM+AL)
-> exp/reports/ablation_runs.csv (one row per dataset, method, seed) and ablation_tests.csv
usage: python ablation_report.py
"""
from pathlib import Path
import os
import pandas as pd
from evaluate import run_metrics
from inference import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / os.environ.get('ERNP_ACCESS_OUT', 'exp/reports')
DS = ['CIDDS_ctx', 'ToN_ctx', 'CIC17', 'CIC18', 'CTU13', 'SIMARGL', 'APPRAISE']
OURS = 'ernp_K5A'
ABL = ['ernp_B0_K5A', 'abl_evid_rep_K5A', 'abl_retr_none_K5A', 'abl_prior_scarf_K5A', 'abl_prior_none_K5A']
METRICS = ['F1', 'MCC', 'AUROC', 'zd_recall_at1fpr']


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, missing = [], []
    for ds in DS:
        for seed in range(5):
            for m in [OURS] + ABL:
                r = run_metrics(ds, m, seed)
                if r is None: missing.append((ds, m, seed)); continue
                rows.append(r)
    if missing: raise RuntimeError(f'missing ablation runs: {missing}')
    df = pd.DataFrame(rows); df.to_csv(OUT / 'ablation_runs.csv', index=False)
    tests = []
    for metric in METRICS:
        t, _, _ = stats(df, ours=OURS, metric=metric)
        t = t.rename_axis('ablation').reset_index(); t['metric'] = metric; tests.append(t)
    pd.concat(tests, ignore_index=True).to_csv(OUT / 'ablation_tests.csv', index=False)
    print(df.groupby(['dataset', 'method']).F1.mean().unstack().round(3).to_string())
    print(f'ABLATION COMPLETE {len(df)} runs -> {OUT}')


if __name__ == '__main__':
    main()
