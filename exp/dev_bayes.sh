#!/bin/bash
# Evidence-as-observation study on development seeds 0-1 (6 jobs in parallel, 2 threads each)
#   ernp_prod      : accumulated (product) retraction, daily protocol (same as ERNP)
#   ernp_an24      : analytic 3-class, daily labels, daily refit  (equivalence check vs ERNP)
#   ernp_an24_prod : analytic + product retraction, daily
#   ernp_an1_prod  : analytic + product, labels every hour, refit once a day
#   ernp_an1       : analytic + soft, labels every hour, refit once a day
cd "$(dirname "$0")/.."; mkdir -p logs
jobs=()
for v in "ernp_prod|--retraction prod" "ernp_an24|--analytic 1" "ernp_an24_prod|--analytic 1 --retraction prod" \
         "ernp_an1_prod|--analytic 1 --retraction prod --period 1" "ernp_an1|--analytic 1 --period 1"; do
  name=${v%%|*}; args=${v#*|}
  for s in 0 1; do for b in CIDDS_ctx ToN_ctx CIC18; do ds=$b; [ $s = 1 ] && ds=${b}_s1
    jobs+=("$s $ds $name $args"); done; done; done
printf '%s\n' "${jobs[@]}" | xargs -P 6 -L 1 bash -c 'EXP_SEED=$0 OMP_NUM_THREADS=2 python exp/ernp.py $1 --name $2 "${@:3}" > logs/dev_$2_$1.log 2>&1; echo "finished $2 $1 $(grep -c DONE logs/dev_$2_$1.log)"'
