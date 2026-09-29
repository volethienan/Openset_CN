#!/usr/bin/env bash
# Clean cost benchmark (RQ5): one job at a time on an otherwise idle machine, seed 0, results in bench/results/.
# Doubles as a regression test: every method that also ran in the campaign must reproduce its campaign output
# (checked by bench_compare.py) - this validates the timing instrumentation added to the wrappers.
cd "$(dirname "$0")"; export EXP_SEED=0 EXP_RESULTS="$(pwd)/bench/results/"; mkdir -p bench/results
for ds in CIDDS_ctx ToN_ctx CIC18; do
  python ernp.py $ds > logs/bench__${ds}__ernp.log 2>&1
  for m in ssf_causal owad memstream trident cade; do python baselines/sota.py $ds $m > logs/bench__${ds}__$m.log 2>&1; done
  for m in iforest knn ae_static; do python baselines/classic.py $ds $m > logs/bench__${ds}__$m.log 2>&1; done
  echo "bench $ds done $(date)"
done
python bench_compare.py
