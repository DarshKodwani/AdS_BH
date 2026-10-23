"""Time-marching PINN pilot: one small network per time window, chained by hard ICs.

Window 1 runs from tau_i to k tau = 1 (super-horizon). Later windows are one
oscillation wide (Delta k tau = 2 pi). Each window is a pilot_single_k.Problem whose
initial state is the previous window's network evaluated at its end, so continuity is
exact and every window only has to learn ~one oscillation. Same loss (ODE residual +
00 constraint) and optimiser as pilot 1/2.

    python -m pinn_cmb.pilots.pilot_marching --k 0.2 --tau-max 200 --ansatz osc
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from pinn_cmb import background as bgm
from pinn_cmb import equations as eq
from pinn_cmb import initial_conditions as ics
from pinn_cmb import reference as ref
from pinn_cmb.pilots import pilot_single_k as p1


def window_edges(k: float, tau_i: float, tau_max: float, dkt: float = 2 * np.pi) -> list[float]:
    edges = [tau_i]
    t = 1.0 / k
    while t < tau_max:
        if t > edges[-1]:
            edges.append(t)
        t += dkt / k
    edges.append(tau_max)
    return edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=float, default=0.2)
    ap.add_argument("--tau-max", type=float, default=200.0)
    ap.add_argument("--ansatz", default="osc", choices=["none", "osc"])
    ap.add_argument("--adam-steps", type=int, default=5000)
    ap.add_argument("--lbfgs-steps", type=int, default=4000)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--out", type=Path, default=Path("pinn_cmb/docs/pilot2/marching"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    bg = bgm.build_background()
    tau_i = ics.default_tau_i(args.k, bg)
    edges = window_edges(args.k, tau_i, args.tau_max)
    print(f"k={args.k} windows (Mpc): " + ", ".join(f"{e:.3g}" for e in edges))

    windows, u_start, total = [], None, 0.0
    for j, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        prob = p1.Problem(bg, args.k, b, args.ansatz, tau_start=None if j == 0 else a, u_start=u_start)
        params, hist, secs = p1.train(prob, "constraint", adam_steps=args.adam_steps, lbfgs_steps=args.lbfgs_steps,
                                      width=args.width, log=lambda s: None, tag=f"w{j}")
        total += secs
        u_start = prob.u(params, jnp.float64(1.0))
        windows.append((prob, params))
        print(f"window {j}: tau {a:.3g}-{b:.3g}  final loss {hist[-1][2]:.2e}  ({secs:.0f}s)", flush=True)

    tau_eval = np.geomspace(tau_i * 1.0001, args.tau_max, 800)
    sol = ref.solve(args.k, bg, tau_eval, mode="tca_only", lmax_nu=p1.LAYOUT.lmax_nu)
    obs = {}
    for j, (prob, params) in enumerate(windows):
        sel = (tau_eval >= edges[j]) & ((tau_eval < edges[j + 1]) | (j == len(windows) - 1))
        t = jnp.asarray(tau_eval[sel])
        x = jnp.log(t / prob.tau_i) / prob.L
        u = jax.vmap(lambda s: prob.u(params, s))(x)
        o = jax.vmap(lambda tt, y: eq.observables(tt, y, args.k, bg, p1.LAYOUT))(t, u)
        for v, arr in o.items():
            obs.setdefault(v, []).append(np.asarray(arr))
    obs = {v: np.concatenate(a) for v, a in obs.items()}
    errs = {v: float(np.max(np.abs(obs[v] - sol.values[v])) / np.max(np.abs(sol.values[v]))) for v in p1.ERROR_VARS}
    errs["max_constraint_00"] = float(np.max(np.abs(obs["constraint_00"])))
    print("RESULT " + " ".join(f"{k}={v:.1e}" for k, v in errs.items()) + f"  ({total:.0f}s total)")
    (args.out / "results.json").write_text(json.dumps(
        {"k": args.k, "tau_max": args.tau_max, "ansatz": args.ansatz, "edges": edges, "errors": errs,
         "train_seconds": total}, indent=1))
    p1.plot(sol, {"marching": obs}, args.out / "pilot.png")


if __name__ == "__main__":
    main()
