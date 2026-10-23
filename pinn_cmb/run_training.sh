#!/bin/bash
# Resumable background training: phases 3-4 and 2 in parallel, 2 threads each.
# Re-run after a container restart; each driver resumes from its last finished window.
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD
PY=${PY:-/home/user/venv-pinn/bin/python}
T2="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2"
mkdir -p pinn_cmb/results/phase34 pinn_cmb/results/phase2
XLA_FLAGS="$T2" nohup $PY -m pinn_cmb.phases.phase34 train >> pinn_cmb/results/phase34/nohup.log 2>&1 &
XLA_FLAGS="$T2" nohup $PY -m pinn_cmb.phases.phase2 >> pinn_cmb/results/phase2/nohup.log 2>&1 &
wait
