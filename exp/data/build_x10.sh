#!/bin/bash
# Scale study streams (03/10/2026): CSE-CIC-IDS2018 at 4M flows (10x the main sample), seeds 0-2.
# RAW = folder that contains dataraw/ ; streams are written to $RAW/mock/streams and then moved into this repo.
# Idempotent: a stream that already exists (in either folder) is skipped. WAIT_PID: an earlier builder to wait for.
RAW=${RAW:-C:/Users/AnThiwn/Desktop/CN/}
cd "$(dirname "$0")/../.."
if [ -n "$WAIT_PID" ]; then
  while powershell -NoProfile -Command "Get-Process -Id $WAIT_PID -ErrorAction Stop" > /dev/null 2>&1; do sleep 60; done
fi
have() { [ -f "mock/streams/$1.parquet" ] || [ -f "${RAW}mock/streams/$1.parquet" ]; }
for s in 0 1 2; do
  sfx=""; [ $s != 0 ] && sfx="_s$s"
  have "CIC18_x10${sfx}" || EXP_SEED=$s ERNP_ROOT=$RAW python exp/data/prepare_cic.py CIC18_x10 || exit 1
done
mv "${RAW}"mock/streams/CIC18_x10*.parquet mock/streams/ 2> /dev/null
ls -la mock/streams | grep x10
echo BUILD_X10_DONE
