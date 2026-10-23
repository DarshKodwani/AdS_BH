# Running the pipeline on an Apple-silicon Mac

Notes for moving the training from the 4-core cloud container to a local Mac (e.g. M5 Max, 48 GB).

## What to expect

- **CPU only.** The pipeline runs in float64 (`jax_enable_x64`). The Apple-GPU JAX plugin (jax-metal)
  has no float64, and the pilots showed float32 is not enough for these stiff equations. Do not install jax-metal.
- The gain comes from more and faster cores. In the container each lane has 2 threads. On the Mac, give each
  lane 4-6 threads and run extra jobs (seeds, variants) side by side.

## Setup

```bash
git clone https://github.com/darshkodwani/ads_bh.git && cd ads_bh
git checkout claude/pinn-cmb-setup
python3.13 -m venv ~/venv-pinn && source ~/venv-pinn/bin/activate
pip install "jax==0.12.0.1" "jaxlib==0.12.0.1" "optax==0.2.8" "numpy==2.5.4" "scipy==1.18.1" \
            "classy==3.4.1.0" "PyYAML==6.0.3" "matplotlib==3.11.2" "pytest==9.1.1"
PYTHONPATH=$PWD python -m pytest -q pinn_cmb/tests -m "not slow"   # expect 36 passed
```

`classy` builds CLASS from source and needs a C compiler (`xcode-select --install`). If the wheel build fails,
install CLASS from https://github.com/lesgourgues/class_public and run `make classy` there.

## Taking over the runs

1. **Stop the cloud session's runs first** (ask it to), then `git pull` so you have its latest checkpoints.
   Only one machine may train a given phase at a time, or the two will push conflicting `params.npz`.
2. Set the thread count. `run_training.sh` hard-codes 2 threads per lane in `XLA_FLAGS`. Change
   `intra_op_parallelism_threads=2` to 5 or 6, and drop `--xla_cpu_multi_thread_eigen=false`.
3. Run the lanes. They resume from `results/*/params.npz` and the intra-window `partial_w*.npz`, and skip
   finished steps:
   ```bash
   PY=~/venv-pinn/bin/python pinn_cmb/run_training.sh AB
   ```
   On a laptop, keep it awake (`caffeinate -dimsu pinn_cmb/run_training.sh AB`) and plugged in.
4. Commit and push `pinn_cmb/results` now and then. Logs (`train.log`, `nohup.log`) and partial checkpoints
   are gitignored.

## State at hand-over (2026-10-23)

- Lane A (`results/phase34`): v2 chain with the slow momentum row (`network.slow_row`), retrained from
  window 3. The v1 chain and its diagnostics are in `results/phase34_v1`. After training: phase3 → phase4 →
  phase5 train/validate/inverse.
- Lane B: Phase 2 is done (`results/phase2/evaluation.txt`). Phase 1 v2 (`results/phase1_v2`) is on its last window.
- Paper: `pinn_cmb/paper` (`pdflatex main && bibtex main && pdflatex main && pdflatex main`; needs a TeX
  distribution such as MacTeX).
