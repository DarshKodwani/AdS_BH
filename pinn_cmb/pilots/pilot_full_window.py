"""Pilot 3: does the chain survive the TCA -> full-hierarchy switch and recombination?

Single k (a narrow bundle around it), chain: TCA windows to tau_sw, then full-hierarchy
windows (photons + polarisation + nu to lmax) through recombination. Thomson rows are
preconditioned by 1 + tau kappa' (model.Window.precond). Compared with the reference
solver on the same segments (TCA until tau_sw, then full).

    python -m pinn_cmb.pilots.pilot_full_window --k 0.05
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from pinn_cmb import background as bgm
from pinn_cmb import equations as eq
from pinn_cmb import reference as ref
from pinn_cmb import train as tr
from pinn_cmb.model import Window
from pinn_cmb.phases.phase1 import EVAL_VARS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=float, default=0.05)
    ap.add_argument("--lmax", type=int, default=8)
    ap.add_argument("--edges-full", default="30,60,100,150,200,240,270,300,340,400")
    ap.add_argument("--adam-steps", type=int, default=8000)
    ap.add_argument("--lbfgs-steps", type=int, default=3000)
    ap.add_argument("--out", type=Path, default=Path("pinn_cmb/docs/pilot3"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    bg = bgm.build_background()
    tca = eq.Layout(tca=True, lmax_nu=args.lmax)
    full = eq.Layout(tca=False, lmax_g=args.lmax, lmax_pol=args.lmax, lmax_nu=args.lmax)
    lk = (float(np.log(args.k * 0.98)), float(np.log(args.k * 1.02)))
    fe = [float(x) for x in args.edges_full.split(",")]
    windows = [Window(tca, 1e-3, 1.0, lk), Window(tca, 1.0, fe[0], lk)]
    windows += [Window(full, a, b, lk) for a, b in zip(fe[:-1], fe[1:])]
    cfg = tr.TrainConfig(adam_steps=args.adam_steps, lbfgs_steps=args.lbfgs_steps, n_k=8, n_t=256,
                         n_k_lbfgs=8, n_t_lbfgs=512, log_every=4000)
    log = lambda s: print(s, flush=True)
    chain, hists = tr.train_chain(windows, lambda c: bg, cfg, 1e-3, log=log)

    taus = np.geomspace(1e-3 * 1.0001, fe[-1], 1500)
    sol = ref.solve_segments(args.k, bg, taus, [(tca, fe[0]), (full, fe[-1])], tau_i=1e-3)
    obs = chain.observables(taus, jnp.float64(args.k), jnp.zeros(0), bg)
    errs = {v: float(np.max(np.abs(obs[v] - sol.values[v])) / np.max(np.abs(sol.values[v]))) for v in EVAL_VARS + ("theta_b", "shear_g")}
    print("RESULT " + " ".join(f"{k}={v:.1e}" for k, v in errs.items()))
    (args.out / "results.json").write_text(json.dumps({"k": args.k, "edges_full": fe, "errors": errs,
                                                        "seconds": sum(h["seconds"] for h in hists)}, indent=1))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vs = ("phi", "delta_c", "delta_g", "theta_b", "shear_g")
    fig, ax = plt.subplots(2, len(vs), figsize=(4 * len(vs), 6), sharex=True)
    for j, v in enumerate(vs):
        ax[0, j].plot(taus, sol.values[v], "k-", lw=1.5, label="reference")
        ax[0, j].plot(taus, obs[v], "C1--", label="PINN")
        ax[1, j].plot(taus, np.abs(obs[v] - sol.values[v]) / np.max(np.abs(sol.values[v])), "C3")
        ax[0, j].set_title(v)
        ax[1, j].set_yscale("log")
        ax[1, j].set_xscale("log")
        for e in fe:
            ax[1, j].axvline(e, color="0.85", lw=0.5)
    ax[0, 0].legend()
    fig.tight_layout()
    fig.savefig(args.out / "pilot.png", dpi=110)


if __name__ == "__main__":
    main()
