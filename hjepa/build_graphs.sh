#!/bin/bash
# Metro graphs for H-JEPA's flat LeWM checkpoints, written to artifacts/hj_lewm_s<seed>/<bench>/.
#   PushT, Cube: port the checkpoint to Metro's LeWM (scripts/convert_hjepa.py), then scripts/build_bank.py.
#   AntMaze, FourRoom: hjepa/build_bank.py with H-JEPA's own model.
#   All: scripts/build_jump_table.py and scripts/calibrate_edge_cost.py, up to H_max frames.
#
#   ./hjepa/build_graphs.sh                  # 4 benchmarks x seeds 42/43/44, one job per GPU
#   ./hjepa/build_graphs.sh pusht 42         # one
#   GPUS="0 1" ./hjepa/build_graphs.sh
#
# Needs METRO_PY / HJEPA_PY (the python of Metro's and of H-JEPA's environment), STABLEWM_HOME and HJEPA_HOME.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${METRO_PY:?set METRO_PY to the python of the Metro environment}"
: "${HJEPA_PY:?set HJEPA_PY to the python of the H-JEPA environment}"
: "${HJEPA_HOME:?set HJEPA_HOME to the H-JEPA data directory}"
declare -A H_MAX=([pusht]=80 [cube]=40 [ant]=150 [fourroom]=150)

one () {  # bench seed; chained with && because `set -e` is ignored where `one` is called as `one ... && echo`
  local b=$1 s=$2 wm=hj_lewm_s$2
  if [ $b = pusht ] || [ $b = cube ]; then
    $HJEPA_PY scripts/convert_hjepa.py export --ckpt $HJEPA_HOME/ckpts/$b/${b}_lewm/seed$s/${b}_lewm_object.ckpt \
      --out artifacts/exports/${b}_lewm_s$s.pt &&
    $METRO_PY scripts/convert_hjepa.py build --export artifacts/exports/${b}_lewm_s$s.pt --bench $b --wm $wm &&
    $METRO_PY -u scripts/build_bank.py --wm $wm --bench $b $([ $b = cube ] && echo --max_episode 9800)
  else
    $HJEPA_PY -u hjepa/build_bank.py --bench $b --seed $s --out artifacts/$wm/$b
  fi &&
  $METRO_PY -u scripts/build_jump_table.py --wm $wm --bench $b &&
  $METRO_PY -u scripts/calibrate_edge_cost.py --wm $wm --bench $b --horizon ${H_MAX[$b]}
}

if [ $# -eq 2 ]; then [ -n "${H_MAX[$1]:-}" ] || { echo "unknown benchmark $1"; exit 1; }; one $1 $2; exit; fi
[ $# -eq 0 ] || { echo "usage: $0 [bench seed]"; exit 1; }

mkdir -p artifacts/logs
GPUS=(${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader)})
JOBS=()
for b in ant pusht fourroom cube; do for s in 42 43 44; do JOBS+=("$b $s"); done; done   # longest first

worker () {  # gpu slot
  local gpu=$1 slot=$2 i
  for (( i = slot; i < ${#JOBS[@]}; i += ${#GPUS[@]} )); do
    read -r b s <<< "${JOBS[$i]}"
    local log=artifacts/logs/hj_lewm_s${s}_$b.log
    CUDA_VISIBLE_DEVICES=$gpu one $b $s > $log 2>&1 && echo "$b seed $s done" || echo "$b seed $s FAILED, see $log"
  done
}

for slot in "${!GPUS[@]}"; do worker ${GPUS[$slot]} $slot & done
wait
