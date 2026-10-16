#!/bin/bash
# Metro frontier cells from configs/frontier.txt, three planner seeds each.
# Every (d, wm, bench, h, H) cell runs once at the maximum budget T = 4d (rounded up to a multiple
# of the macro replan interval 10h); summarize.py reads the success at each frontier budget off it.
#
#   ./reproduce.sh                          # d = 75, Metro-LeWM, highest-budget cell per benchmark
#   D="25 50 75" ./reproduce.sh             # these goal distances
#   FRONTIER=1 ./reproduce.sh               # every (h, H) cell of the frontier, not just the highest
#   ./reproduce.sh pusht cube               # only these benchmarks
#   WMS="lewm pldm" ./reproduce.sh          # also Metro-PLDM (needs its weights, see README)
#   GPUS="0 1" ./reproduce.sh
set -euo pipefail
cd "$(dirname "$0")"
GPUS=${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')}
NG=$(echo $GPUS | wc -w)
WMS=${WMS:-lewm}
D=${D:-75}
FRONTIER=${FRONTIER:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$(( $(nproc) / NG ))}
mkdir -p results/logs

# one line "d wm bench h H" per cell: every distinct (h, H) of the frontier, or only the one at the
# highest budget of each (d, wm, bench)
CELLS=$(awk -v all="$FRONTIER" '!/^#/ && NF {
  g = $1 " " $2 " " $3; c = g " " $4 " " $5
  if (all == 1) { if (!(c in seen)) { seen[c] = 1; out[++n] = c } }
  else { if (!(g in top)) grp[++n] = g; if (!(g in top) || $6 > topT[g]) { top[g] = c; topT[g] = $6 } }
} END { for (i = 1; i <= n; i++) print (all == 1 ? out[i] : top[grp[i]]) }' configs/frontier.txt)

JOBS=()
while read -r d wm bench h H; do
  [[ " $D " =~ " $d " ]] || continue
  [[ " $WMS " =~ " $wm " ]] || continue
  [ $# -gt 0 ] && [[ ! " $* " =~ " $bench " ]] && continue
  T=$(( (4 * d + 10 * h - 1) / (10 * h) * 10 * h ))
  for rs in 7 8 9; do JOBS+=("$d $wm $bench $h $H $T $rs"); done
done <<< "$CELLS"
echo "${#JOBS[@]} runs"

worker () {  # gpu slot
  local gpu=$1 slot=$2 i
  for (( i = slot; i < ${#JOBS[@]}; i += NG )); do
    read -r d wm bench h H T rs <<< "${JOBS[$i]}"
    local log=results/logs/${wm}_${bench}_d${d}_h${h}_H${H}_rs${rs}.log
    # a failed run is reported and skipped; under `set -e` it would end this slot's whole queue
    if CUDA_VISIBLE_DEVICES=$gpu MUJOCO_EGL_DEVICE_ID=$gpu python -u eval.py --wm $wm --bench $bench \
      --d $d --h $h --H $H --T $T --run_seed $rs > $log 2>&1; then
      tail -1 $log
    else
      echo "FAILED $wm $bench d$d h$h H$H rs$rs, see $log"
    fi
  done
}

slot=0
for g in $GPUS; do worker $g $slot & slot=$((slot + 1)); done
wait
python summarize.py --d $D
