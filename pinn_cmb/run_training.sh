#!/bin/bash
# Resumable pipeline for all remaining phases, two lanes in parallel (2 threads each).
#   Lane A: phase 3/4 training -> phase 3 eval -> phase 4 C_l -> phase 5 train/validate/inverse
#   Lane B: phase 2 (train + eval) -> phase 1 v2 (train + eval)
# Re-run after a container restart: training resumes from the last finished window and
# finished steps (marked by their output files) are skipped. Exits immediately if running.
cd "$(dirname "$0")/.."
if pgrep -f "pinn_cmb.phases" > /dev/null; then echo "already running"; exit 0; fi
export PYTHONPATH=$PWD
PY=${PY:-/home/user/venv-pinn/bin/python}
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2"
R=pinn_cmb/results
mkdir -p $R/phase34 $R/phase2 $R/phase5 $R/phase1_v2

step() {  # step <marker file> <log> <args...>
  local marker=$1 log=$2; shift 2
  [ -f "$marker" ] && return 0
  echo "=== $(date -u +%FT%TZ) $*" >> "$log"
  $PY -m "$@" >> "$log" 2>&1 || { echo "=== FAILED: $*" >> "$log"; return 1; }
}

laneA() {
  step $R/phase34/.trained $R/phase34/nohup.log pinn_cmb.phases.phase34 train && touch $R/phase34/.trained &&
  step $R/phase34/phase3.json $R/phase34/nohup.log pinn_cmb.phases.phase34 phase3 &&
  step $R/phase34/phase4.json $R/phase34/nohup.log pinn_cmb.phases.phase34 phase4 &&
  step $R/phase5/.trained $R/phase5/nohup.log pinn_cmb.phases.phase5 train && touch $R/phase5/.trained &&
  step $R/phase5/validate.json $R/phase5/nohup.log pinn_cmb.phases.phase5 validate &&
  step $R/phase5/inverse.json $R/phase5/nohup.log pinn_cmb.phases.phase5 inverse &&
  touch $R/.laneA_done
}

laneB() {
  step $R/phase2/evaluation.json $R/phase2/nohup.log pinn_cmb.phases.phase2 &&
  step $R/phase1_v2/evaluation.json $R/phase1_v2/nohup.log pinn_cmb.phases.phase1 \
       --config pinn_cmb/configs/phase1_v2.yaml --out $R/phase1_v2 &&
  touch $R/.laneB_done
}

laneA &
laneB &
wait
