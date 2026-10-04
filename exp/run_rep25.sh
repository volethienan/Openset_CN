#!/bin/bash
# representative-only evidence at B=25 (closes the RQ3 label-unit question), CIC18 and ToN, seeds 0-2
cd "$(dirname "$0")"
run() { EXP_SEED=$1 OMP_NUM_THREADS=4 python ernp.py $2 --budget 25 --evidence rep --name abl_evid_rep_B25 > ../logs/rep25_$2.log 2>&1; echo "done $2 $(grep -c DONE ../logs/rep25_$2.log)"; }
run 0 CIC18 & run 1 CIC18_s1 & run 2 CIC18_s2 & wait
run 0 ToN_ctx & run 1 ToN_ctx_s1 & run 2 ToN_ctx_s2 & wait
echo REP25_ALL_DONE
