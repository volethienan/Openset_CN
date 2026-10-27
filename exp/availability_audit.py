"""Quantify complete-flow/start-order overlap in streams whose duration units are explicit in the builder.

This audits the existing retrospective protocol; it does not reconstruct raw export arrival order.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from common import STREAMS, anchor_end


def main():
    rows = []
    for dataset, unit in [('CIDDS_ctx', 1.), ('CTU13', 1.), ('SIMARGL', .001), ('APPRAISE', 1.)]:
        frame = pd.read_parquet(Path(STREAMS) / f'{dataset}.parquet', columns=['t', 'f_0'])
        start = frame.t.to_numpy(); duration = np.expm1(frame.f_0.to_numpy(dtype=float)) * unit
        end = start + duration
        cut = anchor_end(dataset, start.min())
        overlaps = np.maximum.accumulate(end)[:-1] > start[1:]
        rows.append(dict(dataset=dataset, rows=len(frame), duration_seconds_per_original_unit=unit,
                         positive_duration=int((duration > 0).sum()), max_duration_s=float(duration.max()),
                         anchor_flows_completing_after_split=int(((start < cut) & (end >= cut)).sum()),
                         rows_with_prior_record_not_completed_at_start=int(overlaps.sum()),
                         scope='overlap_diagnostic_not_measured_metric_inflation_or_export_order'))
    out = Path(__file__).parent / 'reports/access/feature_availability.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(protocol='retrospective_complete_flow_start_order', packet_causal_verified=False,
                                    rows=rows, other_datasets='duration_units_not_inferred_in_this_audit'), indent=2), encoding='utf-8')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
