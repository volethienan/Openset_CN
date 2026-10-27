# ERNP K5 evaluation and claim contract

Updated 23 October 2026 after the source/result audit. These changes are an audit amendment, not new pre-registration.
Original scores, flags and raw labels remain preserved. The manuscript has not been edited.

## 1. SIMARGL label-policy uncertainty

The dataset publishes the disputed traffic as Normal flow. The historical filtered stream excludes traffic from
10.9.3.238 after the start of 31 May 2021. The reproducible raw-field audit is:

```powershell
python exp/data/audit_simargl.py --raw C:/Users/AnThiwn/Desktop/CN/dataraw/SIMARGL2021/simargl-2021-combined.parquet
```

The bounded policy is source host + original Normal flow label + UTC interval
2021-05-31 04:58:17 inclusive to 06:56:06 exclusive. The audit checks equivalence against the historical rule on
all raw records and exports packet-count, TCP-flag, protocol, destination and port evidence, independently of ERNP scores.
These traffic patterns support a suspected label anomaly; they do not create verified attack ground truth.

All three cohorts are exported together, with no selection of the cohort giving the best result:

- `historical_filtered_7`: historical filtered SIMARGL alongside the other six datasets.
- `published_labels_7`: SIMARGL_all using published labels alongside the same six datasets.
- `without_uncertain_SIMARGL_6`: omit SIMARGL for a robustness analysis.

If the historical and bounded masks differ, do not relabel or overwrite old streams. Rebuild a separately named
stream and rerun every affected method/seed before quoting its aggregate. The raw audit is the gate for this decision.

## 2. Independent units and statistical reporting

`exp/inference.py` is the shared implementation used by `evaluate.py`, `check_criteria.py` and `real_report.py`.
Methods are paired on the same stream and seed; scenario variants are averaged within seed and source dataset,
then seeds are averaged. One dataset with several seeds/scenarios is still one inferential unit.

Reports export mean differences, dataset wins, descriptive seed wins, directional signed-rank p-values from
the historical hypothesis, two-sided sensitivity, and Holm adjustment across the five baseline comparisons
within a metric. The Access report also exports two-sided Holm across all four headline metrics as a wider-family
sensitivity. No direction or correction family is chosen after inspecting significance. Seed SDs/seed bootstrap
intervals describe sampling/model variability, not uncertainty across new deployment environments.

The code revision is dated; it does not retrospectively change what the original campaign pre-registered.
F1/MCC superiority over SSF is not supported by the seven-dataset signed-rank test.

## 3. Correct K5 cost measurement

```powershell
python exp/benchmark_k5.py --n-prior 5 --threads 4 --repeats 5
```

This script fits exactly five GBMs, preserves the fixed seed and learner settings, performs an untimed warmup,
and reports the median of five measured scoring passes on up to 100,000 complete feature vectors.
It records actual batch size, K, threads, versions, source hash, fit time, RSS and all scoring times.
It writes JSON after each dataset. It does not measure feature extraction, clustering, analyst work or end-to-end latency.
RSS is a process snapshot, not a peak measurement. Fit timing here is an initial fit, not a claim about all online updates.
No test labels choose the benchmark configuration. Runs should be measured sequentially without other experiment jobs.

`bench_batch.py` now forwards to this explicit-K entry point. Historical `exp_k5/reports/bench_batch.txt` and
`bench_cost.csv` were copied K1 reports and must not be quoted as K5 measurements.

## 4. Feature availability and causal scope

Current evidence uses retrospective replay of complete flow records ordered by START timestamp. Per-flow duration,
total bytes and final flags can be known only after a flow completes. Context is built on complete records whose
start timestamps fall in the past/current window and includes ties at timestamp resolution.
The model scores an update period before receiving the period's simulated analyst feedback. This verifies the
ordering of model feedback, not packet-causal feature availability or true export-arrival order.

`python exp/availability_audit.py` quantifies positive duration, overlap and anchor-straddling flows for four streams
whose duration units are explicit in the builders. It does not estimate metric inflation or prove export order.

The scope of the present study is therefore **adaptation under retrospective complete-flow replay**. Do not claim
detection at flow start, zero packet latency, feature-level causal deployment or production readiness.
Strict availability-time evaluation would require raw completion/export timestamps, reconstruction of full-data
context at that time, a new anchor split and reruns of every affected method. Merely sorting sampled records by
completion time while keeping old full-data start-time context would not solve the problem and is not done here.

## 5. Metric definitions and bounded conclusions

- The operating threshold for ERNP is fitted on benign anchor holdout and fixed. Its stream FPR can be much higher
  than 1%, as ToN demonstrates. Some baseline rules use their native operating thresholds.
- ROC recall@1% uses the stream benign evaluation-score quantile. It is a retrospective ranking diagnostic,
  not an available deployment calibration procedure.
- Whole-stream zero-day recall counts flows of families absent from the anchor, including post-feedback flows.
  The evidence report also includes family-macro recall and first-update-period recall, before that period's feedback.
- One simulated analyst decision labels a representative and propagates its label to a group. Report decisions,
  propagated flow count, purity and matched-flow-cap controls separately. Do not equate this with human verification
  of every flow or with measured SOC labor savings.
- Report ToN's high FPR/low recall at low FPR, CIC17_full's seed instability and deficit to CADE, and group-camouflage
  cases where representative-only feedback beats full propagation. Neither universal superiority nor universal
  poisoning resistance follows from the experiments.
- Retraction acts locally on prior weights. Refitting a GBM can change scores globally; no locality guarantee for
  final scores is claimed. Each refit samples q(P_t) freshly and applies the current retention weights, rather than
  multiplying all historical kernels recursively.
- The candidate method contribution is the evidence-controlled prior adaptation and its coordination with benign
  and attack evidence. Synthetic negatives, single-donor swapping, feedback, GBM and K5 are not independently
  claimed as inventions. Priority over the closest methods remains a literature-comparison task, not something
  established by winning a benchmark.

## Commands and artifacts

```powershell
python -m unittest discover -s exp/tests -v
python exp/access_report.py
python exp/availability_audit.py
python exp/verify_access.py
```

`exp/reports/access/` contains revised metrics, three cohort panels, signed-rank/Holm results, provenance manifests,
raw-label audit, feature-availability audit and explicit K5 cost. Archived reports outside this directory should
not override these definitions. An artifact being regenerated does not mean a model was retrained.
