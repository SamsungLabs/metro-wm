#!/bin/bash
# Metro-LeWM with H-JEPA's gradient-descent planner, on H-JEPA's test tasks, seeds 42/43/44.
# Each run is H-JEPA's evaluation (results in $HJEPA_HOME/ckpts/<bench>/<bench>_lewm/seed<s>/metro_h<h>_H<H>/).
#
#   ./hjepa/reproduce.sh                     # all 4 benchmarks, one run per GPU
#   ./hjepa/reproduce.sh cube pusht          # only these benchmarks
#   GPUS="0 1" ./hjepa/reproduce.sh
#   ARTIFACTS=/path/to/artifacts ./hjepa/reproduce.sh   # graphs elsewhere than artifacts/
#
# Needs HJEPA_PY (the python of H-JEPA's environment), HJEPA_ROOT (the H-JEPA clone) and HJEPA_HOME.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$PWD
ARTIFACTS=$(realpath ${ARTIFACTS:-artifacts})
: "${HJEPA_PY:?set HJEPA_PY to the python of the H-JEPA environment}"
: "${HJEPA_ROOT:?set HJEPA_ROOT to the H-JEPA clone}"
: "${HJEPA_HOME:?set HJEPA_HOME to the H-JEPA data directory}"
GPUS=${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')}
NG=$(echo $GPUS | wc -w)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$(( $(nproc) / NG ))}
mkdir -p results/hjepa/logs

# bench h H: micro plan length h (action blocks of 5 frames) and waypoint budget H (frames)
CELLS="ant 3 9
pusht 3 34
fourroom 1 8
cube 1 12"

for b in "$@"; do grep -q "^$b " <<< "$CELLS" || { echo "unknown benchmark $b"; exit 1; }; done
JOBS=()
while read -r b h H; do
  [ $# -gt 0 ] && [[ ! " $* " =~ " $b " ]] && continue
  for s in 42 43 44; do JOBS+=("$b $h $H $s"); done
done <<< "$CELLS"
echo "${#JOBS[@]} runs"

worker () {  # gpu slot
  local gpu=$1 slot=$2 i
  for (( i = slot; i < ${#JOBS[@]}; i += NG )); do
    read -r b h H s <<< "${JOBS[$i]}"
    local log=$REPO/results/hjepa/logs/${b}_h${h}_H${H}_s$s.log
    # a failed run is reported and skipped
    if (cd $HJEPA_ROOT/h_jepa && CUDA_VISIBLE_DEVICES=$gpu MUJOCO_EGL_DEVICE_ID=$gpu $HJEPA_PY -u \
        $REPO/hjepa/metro_in_hjepa.py --config-name ${b}_flat seed=$s \
        policy=$HJEPA_HOME/ckpts/$b/${b}_lewm/seed$s/${b}_lewm_object.ckpt output.dir=metro_h${h}_H$H \
        plan_config.horizon=$h \
        +metro.artifacts=$ARTIFACTS/hj_lewm_s$s/$b +metro.H=$H) > $log 2>&1; then
      echo "done $b h$h H$H seed $s"
    else
      echo "FAILED $b h$h H$H seed $s, see $log"
    fi
  done
}

slot=0
for g in $GPUS; do worker $g $slot & slot=$((slot + 1)); done
wait
$HJEPA_PY hjepa/summarize.py
