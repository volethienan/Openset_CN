"""Memory-bounded raw-field audit of the historical SIMARGL exclusion; no labels are edited."""
import argparse
from collections import Counter
from pathlib import Path
import hashlib
import json
import numpy as np
import pyarrow.parquet as pq

HOST = '10.9.3.238'
DAY_MS = 1622419200000
START_MS = DAY_MS + (4 * 3600 + 58 * 60 + 17) * 1000
END_MS = DAY_MS + (6 * 3600 + 56 * 60 + 6) * 1000


def masks(ch):
    t = ch['FLOW_START_MILLISECONDS'].to_numpy()
    host = ch['IPV4_SRC_ADDR'].astype(str).to_numpy() == HOST
    normal = ch['LABEL'].astype(str).str.strip().to_numpy() == 'Normal flow'
    return host & (t >= DAY_MS), host & normal & (t >= START_MS) & (t < END_MS)


def audit(path, output):
    f = pq.ParquetFile(path)
    cols = ['FLOW_START_MILLISECONDS', 'FLOW_END_MILLISECONDS', 'IPV4_SRC_ADDR', 'IPV4_DST_ADDR',
            'L4_DST_PORT', 'LABEL', 'TCP_FLAGS', 'IN_PKTS', 'OUT_PKTS', 'PROTOCOL_MAP']
    counts = Counter(); labels = Counter(); signatures = Counter(); outside = Counter(); duration = Counter()
    destinations, ports = set(), set(); lo, hi = None, None
    for i, batch in enumerate(f.iter_batches(batch_size=200_000, columns=cols, use_threads=False)):
        ch = batch.to_pandas(); old, bounded = masks(ch)
        counts.update(rows=len(ch), legacy_excluded=int(old.sum()), bounded_excluded=int(bounded.sum()),
                      legacy_only=int((old & ~bounded).sum()), bounded_only=int((bounded & ~old).sum()))
        end = ch.FLOW_END_MILLISECONDS.to_numpy(); start = ch.FLOW_START_MILLISECONDS.to_numpy()
        duration.update(nonzero_duration=int((end > start).sum()), invalid_end_before_start=int((end < start).sum()))
        if old.any():
            h = ch.loc[old]
            labels.update({str(k): int(v) for k, v in h.LABEL.value_counts().items()})
            signatures.update({str(k): int(v) for k, v in h.groupby(['PROTOCOL_MAP', 'TCP_FLAGS', 'IN_PKTS', 'OUT_PKTS']).size().items()})
            destinations.update(h.IPV4_DST_ADDR.astype(str).unique()); ports.update(map(int, h.L4_DST_PORT.unique()))
            v = h.FLOW_START_MILLISECONDS.to_numpy()
            lo = min(lo if lo is not None else int(v.min()), int(v.min())); hi = max(hi or 0, int(v.max()))
        if (old & ~bounded).any():
            h = ch.loc[old & ~bounded]
            outside.update(map(str, zip(h.LABEL, h.PROTOCOL_MAP, h.TCP_FLAGS, h.IN_PKTS, h.OUT_PKTS)))
        if i % 25 == 0:
            print(f'rows={counts["rows"]:,} excluded={counts["legacy_excluded"]:,} outside={counts["legacy_only"]:,}', flush=True)
    result = dict(raw_path=str(path.resolve()), raw_size=path.stat().st_size, parquet_rows=f.metadata.num_rows,
                  policy=dict(host=HOST, start_ms=START_MS, end_ms_exclusive=END_MS, label='Normal flow',
                              status='suspected_label_anomaly_not_verified_ground_truth'),
                  counts=dict(counts), labels=dict(labels), signatures=dict(signatures), outside_signatures=dict(outside),
                  destinations=len(destinations), destination_ports=len(ports), excluded_min_start_ms=lo,
                  excluded_max_start_ms=hi, availability=dict(duration),
                  masks_identical=counts['legacy_only'] == 0 and counts['bounded_only'] == 0,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True); return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=Path(__file__).parents[1] / 'reports/simargl_raw_audit.json')
    a = ap.parse_args(); audit(a.raw, a.out)
