#!/bin/bash
# Steps 1-2 for every (world model, benchmark): state bank, jump table, bridge-cost curve.
#
#   ./scripts/build_artifacts.sh                 # LeWM on all 4 benchmarks, one per GPU
#   WMS="lewm pldm" ./scripts/build_artifacts.sh # also PLDM (needs its weights, see docs/lewm_setup.md)
#   ./scripts/build_artifacts.sh lewm pusht      # one
set -euo pipefail
cd "$(dirname "$0")/.."

one () {  # chained with && because `set -e` is ignored where `one` is called as `one ... && echo`
  python -u scripts/build_bank.py --wm $1 --bench $2 &&
  python -u scripts/build_jump_table.py --wm $1 --bench $2 &&
  python -u scripts/calibrate_edge_cost.py --wm $1 --bench $2
}

if [ $# -eq 2 ]; then one $1 $2; exit; fi

mkdir -p artifacts/logs
GPUS=($(nvidia-smi --query-gpu=index --format=csv,noheader))
i=0
for wm in ${WMS:-lewm}; do for bench in pusht cube tworoom reacher; do
  g=${GPUS[$(( i % ${#GPUS[@]} ))]}
  ( CUDA_VISIBLE_DEVICES=$g one $wm $bench > artifacts/logs/${wm}_${bench}.log 2>&1 && echo "$wm $bench done" \
    || echo "$wm $bench FAILED, see artifacts/logs/${wm}_${bench}.log" ) &
  i=$((i + 1))
  if (( i % ${#GPUS[@]} == 0 )); then wait; fi
done; done
wait
