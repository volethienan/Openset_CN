"""Dataset-level inference. Seeds and scenarios are repeated measurements, not new datasets."""
import re
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

Q_NEMENYI = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
             7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}


def dataset_unit(name):
    # Injection/scale/label-policy variants of one capture do not add independent units.
    for root in ('CIDDS_ctx', 'ToN_ctx', 'CIC17', 'CIC18', 'CTU13', 'SIMARGL', 'APPRAISE'):
        if name == root or name.startswith(root + '_'):
            return root
    return re.sub(r'_s\d+$', '', name)


def holm(pvalues):
    """Holm adjustment; unavailable tests remain NaN and do not enter the family."""
    p = np.asarray(pvalues, dtype=float)
    out = np.full(len(p), np.nan)
    order = np.flatnonzero(np.isfinite(p))
    order = order[np.argsort(p[order], kind='stable')]
    running = 0.
    for i, j in enumerate(order):
        running = max(running, (len(order) - i) * p[j])
        out[j] = min(1., running)
    return out


def signed_rank(delta, alternative):
    x = np.asarray(delta, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan  # a single underlying dataset supports no cross-dataset inference
    if not np.any(x):
        return 1.
    return float(wilcoxon(x, alternative=alternative, method='auto').pvalue)


def dataset_means(df, metric):
    """Average scenarios within seed, then seeds within the source dataset, equally weighted."""
    x = df[['dataset', 'method', 'seed', metric]].copy()
    if x.duplicated(['dataset', 'method', 'seed']).any():
        raise ValueError('duplicate dataset/method/seed rows; resolve provenance before inference')
    x['unit'] = x.dataset.map(dataset_unit)
    return x.groupby(['unit', 'method', 'seed'])[metric].mean().groupby(['unit', 'method']).mean().unstack('method')


def stats(df, ours='ernp', metric='F1'):
    """Pair on the same stream/seed before aggregation; Holm family = baselines in this metric.

    Both directional (historical hypothesis) and two-sided sensitivity are returned.
    Never select a direction based on observed p-values. Seed wins are descriptive only.
    """
    keys = ['dataset', 'seed']
    if df.duplicated(keys + ['method']).any():
        raise ValueError('duplicate dataset/method/seed rows')
    pairs = df.pivot(index=keys, columns='method', values=metric)
    if ours not in pairs:
        raise ValueError(f'missing reference method {ours}')
    rows = {}
    for baseline in pairs.columns.drop(ours):
        matched = pairs[[ours, baseline]].dropna()
        if matched.empty:
            continue
        dif = matched[ours] - matched[baseline]
        # First average scenarios per source dataset/seed, then average seeds.
        tab = dif.rename('delta').reset_index()
        tab['unit'] = tab.dataset.map(dataset_unit)
        d = tab.groupby(['unit', 'seed']).delta.mean().groupby('unit').mean()
        rows[baseline] = dict(mean_diff=float(d.mean()), wins=int((d > 0).sum()), n=len(d),
                              paired_seed_rows=len(dif), seed_wins=int((dif > 0).sum()),
                              p_one_sided=signed_rank(d, 'greater'), p_two_sided=signed_rank(d, 'two-sided'))
    result = pd.DataFrame(rows).T
    for col in ('p_one_sided', 'p_two_sided'):
        result[col + '_holm'] = holm(result[col]) if len(result) else []
    means = dataset_means(df, metric).dropna()
    k, n = len(means.columns), len(means)
    ranks = means.rank(axis=1, ascending=False).mean().sort_values()
    fr = float(friedmanchisquare(*[means[c] for c in means]).pvalue) if k >= 3 and n >= 3 else np.nan
    cd = Q_NEMENYI.get(k, np.nan) * np.sqrt(k * (k + 1) / (6 * n)) if n >= 3 else np.nan
    return result, ranks, dict(friedman_p=fr, nemenyi_CD=cd, n_blocks=n, k=k,
                               unit='source_dataset', correction='Holm across baselines within metric',
                               seed_variability='descriptive; not independent datasets')
