#!/bin/bash
# OWAD best-configuration variant (AE lr 1e-3), RQ1: 4 datasets x 5 seeds, 2 jobs at a time on the local GPU
cd "$(dirname "$0")"
for s in 0 1 2 3 4; do
  sfx=""; [ $s != 0 ] && sfx="_s$s"
  for b in CIDDS_ctx ToN_ctx CIC17 CIC18; do echo "$s $b$sfx"; done
done | xargs -P 2 -L 1 bash -c 'f=results/$1__owad_lr3__s$0.json; [ -f $f ] && exit 0; EXP_SEED=$0 OMP_NUM_THREADS=4 python baselines/sota.py $1 owad_lr3 > logs/$1__owad_lr3__s$0.log 2>&1; echo "done $1 s$0 $(grep -c DONE logs/$1__owad_lr3__s$0.log)"'
echo OWAD_LR3_DONE
