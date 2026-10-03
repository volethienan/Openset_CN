#!/bin/bash
# Scale study streams (03/10/2026): CIDDS-001 at 3M and CSE-CIC-IDS2018 at 4M flows (10x the main sample), seeds 0-2.
# RAW = folder that contains dataraw/ ; streams are written to $RAW/mock/streams and then moved into this repo.
RAW=${RAW:-C:/Users/AnThiwn/Desktop/CN/}
cd "$(dirname "$0")/../.."
for s in 0 1 2; do
  EXP_SEED=$s ERNP_ROOT=$RAW ERNP_TARGET=3000000 ERNP_STREAM_EXTRA=_x10 python exp/data/prepare_ctx.py CIDDS || exit 1
  EXP_SEED=$s ERNP_ROOT=$RAW python exp/data/prepare_cic.py CIC18_x10 || exit 1
done
mv "$RAW"mock/streams/CIDDS_ctx*_x10.parquet "$RAW"mock/streams/CIC18_x10*.parquet mock/streams/
ls -la mock/streams | grep x10
echo BUILD_X10_DONE
