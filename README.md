# ERNP: Evidence-Retracted Novelty Prior

Code, data streams and results for the manuscript

> **Adapting Network Intrusion Detectors Only Through Trusted Evidence: Evidence-Retracted Novelty Priors for Zero-Day Detection under Benign Drift** (submitted to *Computer Networks*).

ERNP is a network intrusion detector that adapts **only through analyst decisions on alert groups**. It estimates a
density ratio between benign evidence and the union of attack evidence and a novelty prior (benign flows whose features
are partially swapped with another benign flow). Benign confirmations retract the prior locally; no flow enters the
model without an analyst decision on its group, at most `B` decisions per period.

Every number in the manuscript is computed from the files in `exp/results/` and `exp/reports/` of this repository.

## Repository layout

| Path | Content |
|---|---|
| `exp/ernp.py` | The proposed method. Defaults are the final configuration; every ablation switch is a command-line option. |
| `exp/common.py` | Shared protocol: stream loading, anchor/holdout split per seed, standardisation, 1% FPR threshold, result files. |
| `exp/baselines/sota.py` | Wrappers around the **official code** of SSF, OWAD, MemStream and Trident (in `sota/`). |
| `exp/baselines/cade_port.py` | PyTorch port of CADE (the original runs on TensorFlow 1.12), validated against the authors' IDS2018 split. |
| `exp/baselines/classic.py` | Classical references: kNN, Isolation Forest, autoencoder. |
| `exp/baselines/sanity.py` | Leakage sanity baseline (`new_ip`). |
| `exp/data/` | Stream builders (`prepare*.py`) and the controlled injections for RQ2 (`inject.py`, `inject_camo.py`). |
| `exp/run_jobs.py` | Job runner for all experiment groups (GPU and CPU pools, skips finished runs). |
| `exp/evaluate.py` | Metrics, aggregation over seeds and statistical tests. |
| `exp/report_all.sh` | Regenerates every report in `exp/reports/`. |
| `exp/*analysis*.py`, `exp/diagnose_*.py` | Additional analyses reported in the paper (ToN burst, retraction replay, P2/P3 checks, diagnostics). |
| `exp/results/` | One `<stream>__<method>__s<seed>.npz` (scores, alerts) and `.json` (configuration, labels used, timing) per run. |
| `exp/reports/` | Aggregated results (`*_runs.csv`, `*.txt`). |
| `mock/streams/` | Pre-built input streams (Parquet). The folder name is historical; the streams are real data. |
| `sota/` | Official code of the baselines, with the authors' own READMEs and licences. |

## Installation

Python 3.12 is used for all results. Large stream files are stored with Git LFS.

```bash
git lfs install
git clone https://github.com/volethienan/Openset_CN.git
cd Openset_CN
git lfs pull
pip install -r requirements.txt
```

`requirements.txt` pins the exact versions that produced the committed results. Install the PyTorch build that matches
your CUDA version from https://pytorch.org if needed. ERNP itself runs on CPU only; the deep baselines use a GPU when
one is available. All paths are derived from the repository location (override with `ERNP_ROOT=/path/to/Openset_CN/`).

## Data

| Stream | Source dataset | Role |
|---|---|---|
| `CIDDS_ctx` | CIDDS-001 (OpenStack, internal traffic) | Main benchmark; base of the RQ2 poisoning streams |
| `ToN_ctx` | NF-ToN-IoT-v3 | Strong benign drift, sequential zero-day families |
| `CIC17` | CICIDS2017 (improved version, Distrinet) | Held-out dataset (not used during design) |
| `CIC18` | CSE-CIC-IDS2018 (improved version, Distrinet) | Main benchmark |
| `CIC17_full` | CICIDS2017, unsampled (Git LFS) | Scale check |

Seed `k` uses the stream `<name>_s<k>` (seed 0 has no suffix). Streams are time ordered and sub-sampled (300k–400k flows
per seed), with per-flow features and temporal-context features computed on the full data before sampling.

The original datasets are not redistributed here; they are available from their authors. To rebuild the streams, place
them under `dataraw/` (`CIDDS-001/`, `NF-ToN-IoT-v3/`, `CICIDS2017_improved/`, `CSECICIDS2018_improved/`) and run, for each seed:

```bash
EXP_SEED=0 python exp/data/prepare_ctx.py CIDDS      # -> CIDDS_ctx
EXP_SEED=0 python exp/data/prepare_ctx.py ToN        # -> ToN_ctx
EXP_SEED=0 python exp/data/prepare_cic.py CIC17
EXP_SEED=0 python exp/data/prepare_cic.py CIC18
```

Derived streams that are not committed are rebuilt deterministically from the committed ones:
`python exp/data/inject.py CIDDS_ctx [payload] [ramp_days] [rate]` (slow poisoning, RQ2),
`python exp/data/inject_camo.py CIDDS_ctx` (group camouflage, RQ2), and `exp/data/build_x10.sh`
(CSE-CIC-IDS2018 at 4M flows; needs `dataraw/`).

## Running

All commands are run from `exp/`.

**Single runs**

```bash
EXP_SEED=1 python ernp.py CIC18_s1                                         # ERNP, causal protocol, B = 200
EXP_SEED=1 python ernp.py CIC18_s1 --retraction none --name abl_retr_none  # an ablation (RQ4)
EXP_SEED=1 python ernp.py ToN_ctx_s1 --budget 25 --name ernp_B25          # analyst budget (RQ3)
EXP_SEED=1 python baselines/sota.py CIC18_s1 ssf_causal                    # SSF | owad | memstream | trident | cade
EXP_SEED=1 python baselines/classic.py CIC18_s1 knn                        # knn | iforest | ae_static
```

**Experiment groups**

```bash
python run_jobs.py rq1 --seeds 0 1 2 3 4 --gpu 2 --cpu 2      # groups can be combined: rq1+rq4
```

| Group | Paper |
|---|---|
| `rq1`, `ext`, `ext1`, `opf`, `scale`, `x10`, `cad`, `t1` | RQ1: detection, references, best configurations, scale, update cadence, common operating point |
| `rq2`, `rq2lb`, `rq2rep`, `rq2gate`, `camo` | RQ2: slow poisoning, small budgets, evidence-gated variants, group camouflage |
| `rq3` | RQ3: analyst budget |
| `rq4`, `scarf`, `audit`, `diam` | RQ4: ablations, independent-donor prior, group-label audit and footprint |
| `rq5`, `noise` | RQ5: sensitivity and analyst label noise |

`--gpu`/`--cpu` set the number of concurrent jobs, `--threads` the threads per job, and `MAX_GPU_PROCS` caps
concurrent GPU jobs machine-wide. Finished runs are skipped, so the runner can be restarted at any time.

**Reports**

```bash
bash report_all.sh                       # exp/reports/*_runs.csv and *.txt
python ton_burst_analysis.py             # ToN gateway-burst sensitivity
python retraction_time_analysis.py       # where retraction acts (replays the evidence from saved decisions)
python retraction_time_report.py
```

**Reproducibility check.** Re-running a committed job into a separate folder reproduces its decisions:

```bash
EXP_SEED=0 EXP_RESULTS=/tmp/check/ python ernp.py CIC17
python -c "import numpy as np; a=np.load('/tmp/check/CIC17__ernp__s0.npz'); b=np.load('results/CIC17__ernp__s0.npz'); print(np.array_equal(a['flag'], b['flag']))"
```

## Protocol notes

- **Anchor and stream.** The first day(s) of each dataset form the anchor period (CIDDS-001: 2 days, ToN: 1 day,
  CICIDS2017: before 2017-07-06, CSE-CIC-IDS2018: before 2018-02-17). Anchor benign flows are split 80/20 into reference
  and holdout per seed. Zero-day families are the attack families absent from the anchor.
- **Causal order.** Every period is first scored, then used for the update (predict-then-learn).
- **Threshold.** Every score-based detector alerts at 1% FPR on the anchor holdout; the threshold is never re-tuned.
- **Baselines.** SSF, OWAD, MemStream and Trident run the authors' code from `sota/`. Trident is run update-free,
  because its demo retrains on ground-truth labels of the whole stream. MemStream uses the authors' memory-admission
  `beta = 0.1`; earlier runs with an adaptive `beta` are kept as the variant `memstream_adapt`. Deviations are listed in
  the paper (Table "Baselines").

## Licence and third-party code

The code in `exp/` is released for the review and reproduction of the paper. The code in `sota/` belongs to the
respective authors and keeps their licences. The source datasets are subject to their providers' terms of use.
