# ERNP: Evidence-Retracted Novelty Priors

Code, prepared data streams and per-run results for the manuscript

> **Evidence-Retracted Novelty Priors for Zero-Day Network Intrusion Detection under Benign Drift**

ERNP is a network intrusion detector that adapts only through analyst decisions on alert groups. It estimates a
density ratio between benign evidence and the union of attack evidence and a novelty prior (benign flows with part of
their features swapped with another benign flow). Benign confirmations retract the prior locally. At most `B` groups
are labelled per period, and no flow enters the model without an analyst decision on its group.

The configuration reported in the paper is **ERNP K5A**: five prior draws per fit (`--n_prior 5`) and anchored attack
evidence (`--keep A`). Its runs are stored under the method name `ernp_K5A`. The command-line defaults of
`exp/ernp.py` (`--n_prior 1`, FIFO evidence) are the earlier development configuration.

## Repository layout

| Path | Content |
|---|---|
| `exp/ernp.py` | The proposed method. Ablations are command-line options. |
| `exp/common.py` | Shared protocol: stream loading, anchor/holdout split per seed, standardisation, 1% FPR threshold, result files. |
| `exp/baselines/sota.py` | Wrappers around the official code of SSF, OWAD, MemStream and Trident (in `sota/`). |
| `exp/baselines/cade_port.py` | PyTorch port of CADE, validated against the authors' CSE-CIC-IDS2018 split. |
| `exp/baselines/classic.py` | References: kNN, Isolation Forest, autoencoder. |
| `exp/data/` | Stream builders (`prepare_*.py`), poisoning and camouflage injection, SIMARGL label audit. |
| `exp/run_jobs.py` | Job runner for every experiment group (GPU and CPU pools; finished runs are skipped). |
| `exp/evaluate.py`, `exp/inference.py` | Metrics, aggregation over seeds, paired tests and Holm correction. |
| `exp/tests/` | Unit tests (`python -m unittest discover -s exp/tests`). |
| `exp/results/` | One `<stream>__<method>__s<seed>.npz` (scores, alerts) and `.json` (configuration, analyst decisions, timing) per run. |
| `exp/reports/` | Every result of the paper, aggregated from the runs (see the table below). |
| `tables_figures/` | Scripts that build every table and data figure of the manuscript from `exp/reports/`. |
| `mock/streams/` | Prepared input streams (Parquet). The folder name is historical; the streams are real data. |
| `sota/` | Official code of the baselines, with the authors' READMEs and licences. |

## Installation

Python 3.12.7 was used for all results. The unsampled CICIDS2017 stream is stored with Git LFS.

```bash
git lfs install
git clone https://github.com/volethienan/Openset_CN.git
cd Openset_CN
git lfs pull
pip install -r requirements.txt
```

`requirements.txt` pins the versions that produced the committed results. Install the PyTorch build that matches your
CUDA version from https://pytorch.org if needed. ERNP runs on CPU; the deep baselines use a GPU when one is available.
Paths are derived from the repository location (override with `ERNP_ROOT=/path/to/Openset_CN/`).

## Data

| Stream | Source dataset | Builder |
|---|---|---|
| `CIDDS_ctx` | CIDDS-001 (OpenStack, internal traffic) | `exp/data/prepare_ctx.py CIDDS` |
| `ToN_ctx` | NF-ToN-IoT-v3 | `exp/data/prepare_ctx.py ToN` |
| `CIC17` | CICIDS2017, improved version (Distrinet) | `exp/data/prepare_cic.py CIC17` |
| `CIC18` | CSE-CIC-IDS2018, improved version (Distrinet) | `exp/data/prepare_cic.py CIC18` |
| `CTU13` | CTU-13 | `exp/data/prepare_real.py CTU13` |
| `SIMARGL`, `SIMARGL_all` | SIMARGL2021 (suspected-label exclusion / published labels) | `exp/data/prepare_real.py SIMARGL` |
| `APPRAISE` | APPRAISE | `exp/data/prepare_real.py APPRAISE` |
| `CIC17_full` | CICIDS2017, unsampled (Git LFS) | `exp/data/prepare_cic.py CIC17_full` |
| `CIDDS_ctx_inj*` | CIDDS-001 with injected slow poisoning (RQ2) | `exp/data/inject.py`, `exp/data/inject_valid.py` |

Seed `k` uses the stream `<name>_s<k>` (seed 0 has no suffix). Streams are time ordered and sub-sampled per seed;
per-flow and temporal-context features are computed on the full trace before sampling.

The source datasets are not redistributed; they are available from their providers. To rebuild the streams, place them
under `dataraw/` and run the builders with `EXP_SEED=<k>`. Two groups of streams are not committed. The
group-camouflage streams are rebuilt exactly from the committed ones with
`python exp/data/inject_camo.py <base>` for `<base>` in `CIDDS_ctx`, `CIDDS_ctx_s1`, …, `CIDDS_ctx_s4`.
CSE-CIC-IDS2018 at 4M flows (about 600 MB per seed) is rebuilt from the source dataset with `exp/data/build_x10.sh`.

SIMARGL2021 publishes the traffic of host 10.9.3.238 between 04:58:17 and 06:56:06 UTC on 31 May 2021 as normal, and
this traffic is suspected to be mislabelled. `SIMARGL` excludes it, `SIMARGL_all` keeps the published labels, and both
are reported. The raw-field audit is
`python exp/data/audit_simargl.py --raw dataraw/SIMARGL2021/simargl-2021-combined.parquet`.

## Reproducing the tables and figures

**From the committed reports** (no model is run, about one minute):

```bash
python tables_figures/build_tables.py    # -> tables_figures/tables/*.tex and table_sources.json (SHA-256 of every input)
python tables_figures/make_figures.py    # -> tables_figures/figures/fig_*.pdf
```

`tables_figures/table_sources.json` lists the hash of each report file the tables are built from, so every table can be
traced to its exact input.

**From the saved runs** (no model is retrained):

```bash
python exp/k5a_view.py                    # exp_k5a/: hard links to the ernp_K5A runs under the names the report scripts use
bash exp/report_k5a.sh                    # secondary experiments, copied to exp/reports/
python exp/access_report.py               # main comparison: main_runs, dataset_means, dataset_tests, cohort_means
python exp/ablation_report.py             # ablation_runs, ablation_tests
python exp/footprint_report.py            # footprint_runs
python exp/naive_report.py                # naive_runs
python exp/review3_report.py              # review3_prior, review3_retraction_load
python exp/ssfg_report.py                 # ssfg_runs
python exp/review30_posthoc.py            # review30_l1, review30_l2, review30_l3
python exp/review30_prior_check.py        # review30_prior
python exp/availability_audit.py          # feature_availability.json
python exp/benchmark_k5.py --n-prior 5 --threads 4   # benchmark_k5.json (run on an idle machine)
```

Every script writes to `exp/reports/`. The view `exp_k5a/` is a working folder and is not versioned.

**Rerunning the experiments.** From `exp/`:

```bash
EXP_SEED=1 python ernp.py CIC18_s1 --n_prior 5 --keep A --name ernp_K5A     # one ERNP K5A run
EXP_SEED=1 python baselines/sota.py CIC18_s1 ssf_causal                      # ssf_causal | owad | memstream | trident | cade
EXP_SEED=1 python baselines/classic.py CIC18_s1 knn                          # knn | iforest | ae_static
python run_jobs.py rq1+real --seeds 0 1 2 3 4 --gpu 2 --cpu 2                # baselines on all seven datasets
python run_jobs.py k5a --seeds 0 1 2 3 4 --cpu 4                             # every ERNP K5A run of the paper
python run_jobs.py k5afill+naive+prior_replay+ssfg --seeds 0 1 2 3 4         # remaining K5A variants and analyses
```

`--gpu`/`--cpu` set the number of concurrent jobs and `--threads` the threads per job. Runs that already exist are
skipped. When the attack evidence of a run never exceeds the FIFO window, `--keep A` changes nothing; such runs were
computed once without it and copied to the K5A name (`exp/k5a_copy.py`; field `k5a_copied_from` in the `.json`).

**Determinism check.** Re-running a committed job into a separate folder reproduces its alerts:

```bash
cd exp
EXP_SEED=0 EXP_RESULTS=/tmp/check/ python ernp.py CIDDS_ctx --n_prior 5 --keep A --name ernp_K5A
python -c "import numpy as np; a=np.load('/tmp/check/CIDDS_ctx__ernp_K5A__s0.npz'); b=np.load('results/CIDDS_ctx__ernp_K5A__s0.npz'); print(np.array_equal(a['flag'], b['flag']))"
```

## Where each result comes from

All files are in `exp/reports/` unless a path is given.

| Manuscript item | Source |
|---|---|
| Main comparison (F1, MCC, AUROC, zero-day recall at 1% FPR), label-policy cohorts, ERNP per-dataset detail, paired-comparison figure, Pareto figure | `main_runs.csv`, `dataset_means.csv`, `dataset_tests.csv`, `cohort_means.csv` |
| Ablation table and figure | `ablation_runs.csv`, `ablation_tests.csv` |
| Scoring cost | `benchmark_k5.json` |
| Retraction mechanism (retraction load, replay without the prior, FPR by distance) | `review3_retraction_load.csv`, `review3_prior.csv`, `e3_regions.csv` |
| Analyst budget in two units | `e2_runs.csv`, `ssfg_runs.csv` |
| Slow poisoning with protocol-valid records; analyst who accepts the ramp | `e4_runs.csv`, `fig_rq2_timeline.csv`, `naive_runs.csv` |
| Group camouflage | `camo_runs.csv`, `rq2gate_runs.csv` |
| Budget and label-noise figure, label-noise table | `rq1_runs.csv`, `rq2_runs.csv`, `rq3_runs.csv`, `rq4_runs.csv`, `noise_runs.csv` |
| Update cadence, label delay | `cadence_runs.csv`, `e5_runs.csv` |
| Sensitivity | `rq5_runs.csv` |
| Unsampled CICIDS2017, CSE-CIC-IDS2018 at 4M flows | `scale_runs.csv`, `x10_runs.csv` |
| Distance-to-evidence figure | `p3_*.csv` |
| ToN timeline and gateway burst | `fig_ton_timeline.csv`, `ton_burst_analysis.csv` |
| Zero-day families, alert-group purity and footprint | `family_recall.csv`, `group_audit.csv`, `group_footprint.csv`, `footprint_runs.csv` |
| Past-only thresholds, time to detection, out-of-range zero-days | `review30_*.csv` |
| Feature availability, SIMARGL label audit | `feature_availability.json`, `simargl_raw_audit.json` |
| Dataset table | `tables_figures/evidence/datasets.tex` |

## Method names

| Name | Meaning |
|---|---|
| `ernp_K5A` | ERNP, final configuration |
| `ernp_B0_K5A`, `abl_evid_rep_K5A`, `abl_retr_none_K5A`, `abl_prior_scarf_K5A`, `abl_prior_none_K5A` | Ablations: no online labels, representative-only labels, no retraction, SCARF prior, no prior (GBM with active learning) |
| `ernp_B<n>_K5A`, `ernp_N<p>_K5A`, `ernp_P<h>_K5A`, `ernp_D<d>_K5A` | Budget of `n` groups per day, `p`% label noise, `h`-hour periods, labels delayed by `d` days |
| `ssf_causal`, `owad`, `memstream`, `trident`, `cade` | Baselines, with the same suffixes where applicable |

## Protocol notes

- **Anchor.** The first period(s) of each dataset form the anchor. Anchor benign flows are split 80/20 into reference
  and holdout per seed. Zero-day families are the attack families absent from the anchor.
- **Order.** Each period is scored before its labels are used (predict-then-update). Streams contain complete-flow
  records ordered by flow start time.
- **Threshold.** Every score-based detector alerts at 1% FPR on the anchor holdout; the threshold is not re-tuned.
- **Baselines.** SSF, OWAD, MemStream and Trident run the authors' code from `sota/`. Trident is run update-free,
  because its demo retrains on ground-truth labels of the whole stream. MemStream uses the authors' memory-admission
  `beta = 0.1`. Deviations from the original setups are listed in the paper.

## Licence and third-party code

The code in `exp/` and `tables_figures/` is released for the review and reproduction of the paper. The code in `sota/`
belongs to its authors and keeps their licences. The source datasets are subject to their providers' terms of use.
