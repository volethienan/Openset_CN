#!/bin/bash
# operating-point study on development seeds 0-1 (6 jobs in parallel, 2 threads each)
cd "$(dirname "$0")/.."; mkdir -p logs
jobs=()
for f in 0.002 0.005; do tag=${f#0.}; for p in 24 4; do
  name=ernp_F$tag; [ $p = 4 ] && name=ernp_P4_F$tag
  for s in 0 1; do for b in CIDDS_ctx ToN_ctx CIC18; do ds=$b; [ $s = 1 ] && ds=${b}_s1
    jobs+=("$s $ds $name $f $p"); done; done; done; done
printf '%s\n' "${jobs[@]}" | xargs -P 6 -L 1 bash -c 'EXP_SEED=$0 OMP_NUM_THREADS=2 python exp/ernp.py $1 --name $2 --fpr $3 --period $4 > logs/dev_$2_$1.log 2>&1; echo "finished $2 $1"'
