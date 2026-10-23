"""Validate the line-of-sight pipeline on reference-solver sources against CLASS C_l^TT.

    python -m pinn_cmb.phases.los_reference --scheme class|fixed [--tau-rsa ...]

scheme "class": CLASS-like k-dependent switches (TCA trigger, RSA at k tau > 45 after
free streaming), i.e. a test of the LOS code itself.
scheme "fixed": k-independent switch times (TCA off at tau_tca, RSA at tau_rsa for all k),
the schedule a k-bundle PINN chain can use.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from pinn_cmb import background as bgm
from pinn_cmb import equations as eq
from pinn_cmb import los
from pinn_cmb import reference as ref

HERE = Path(__file__).resolve().parent.parent
LMAX = dict(lmax_g=12, lmax_pol=10, lmax_nu=17)  # CLASS defaults


def class_cl(lmax=500):
    from classy import Class

    c = Class()
    c.set({**bgm.class_base_settings(), "output": "tCl", "l_max_scalars": lmax, "A_s": los.PRIMORDIAL["A_s"],
           "n_s": los.PRIMORDIAL["n_s"], "k_pivot": los.PRIMORDIAL["k_pivot"], "ur_fluid_approximation": 3})
    c.compute()
    cl = c.raw_cl(lmax)
    c.struct_cleanup()
    c.empty()
    return cl["ell"], cl["tt"]


def segments_for(k, bg, scheme, tau_tca, tau_rsa):
    tca = eq.Layout(tca=True, lmax_nu=LMAX["lmax_nu"])
    full = eq.Layout(tca=False, **LMAX)
    rsa = eq.Layout(tca=False, rsa=True)
    if scheme == "class":
        t_sw = ref.tca_switch_time(k, bg, ref.TCATrigger(), 1e-3, bg.tau0)
        t_rsa = max(45.0 / k, bg.tau_free_streaming)
    else:
        t_sw, t_rsa = tau_tca, tau_rsa
    segs = [(tca, t_sw), (full, min(t_rsa, bg.tau0 * (1 - 1e-6)))]
    if t_rsa < bg.tau0:
        segs.append((rsa, bg.tau0 * (1 - 1e-6)))
    return segs


def reference_sources(bg, k_src, tau_grid, scheme, tau_tca, tau_rsa):
    S = {n: np.zeros((len(k_src), len(tau_grid))) for n in ("S0", "S1", "S2")}
    extra = lambda t, y, layout, k: los.sources(t, y, k, bg, layout)
    for i, k in enumerate(k_src):
        sol = ref.solve_segments(k, bg, tau_grid, segments_for(k, bg, scheme, tau_tca, tau_rsa), tau_i=min(1e-3, 1e-3 / k),
                                 C=0.5, extra=lambda t, y, layout, k=k: los.sources(t, y, jnp.float64(k), bg, layout))
        for n in S:
            S[n][i] = sol.values[n]
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", default="class", choices=["class", "fixed"])
    ap.add_argument("--tau-tca", type=float, default=30.0)
    ap.add_argument("--tau-rsa", type=float, default=850.0)
    ap.add_argument("--lmax", type=int, default=500)
    ap.add_argument("--k-max", type=float, default=0.15)
    ap.add_argument("--nk", type=int, default=160)
    ap.add_argument("--lmax-hier", type=int, nargs=3, default=None, metavar=("G", "POL", "NU"))
    ap.add_argument("--out", type=Path, default=HERE / "results" / "los_reference")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.lmax_hier:
        LMAX.update(lmax_g=args.lmax_hier[0], lmax_pol=args.lmax_hier[1], lmax_nu=args.lmax_hier[2])
    bg = bgm.build_background()
    tau_grid = los.default_tau_grid(bg)
    k_src = np.concatenate([np.geomspace(1e-5, 2e-3, 30, endpoint=False), np.linspace(2e-3, args.k_max, args.nk)])
    t0 = time.time()
    S = reference_sources(bg, k_src, tau_grid, args.scheme, args.tau_tca, args.tau_rsa)
    t_src = time.time() - t0
    ells = los.default_ells(args.lmax)
    k_fine = los.default_k_fine(args.k_max, bg.tau0)
    delta = los.transfer_tt(k_src, tau_grid, S, ells, bg.tau0, k_fine)
    cl = los.cl_tt(k_fine, delta)
    ell_c, cl_c = class_cl(max(2500, args.lmax))  # CLASS ties its k range to lmax; converge it
    ratio = cl / cl_c[ells] - 1.0
    tag = args.scheme if args.scheme == "class" else f"fixed_tca{args.tau_tca:g}_rsa{args.tau_rsa:g}"
    if args.lmax_hier:
        tag += "_lmax" + "-".join(map(str, args.lmax_hier))
    print(f"[{tag}] sources {t_src:.0f}s; max|C_l/C_l^CLASS - 1| = {np.max(np.abs(ratio)):.2e} "
          f"(l<=30: {np.max(np.abs(ratio[ells <= 30])):.2e}, l>30: {np.max(np.abs(ratio[ells > 30])):.2e})")
    np.savez(args.out / f"{tag}.npz", ells=ells, cl=cl, cl_class=cl_c[ells], k_src=k_src, tau=tau_grid, **S)
    (args.out / f"{tag}.json").write_text(json.dumps(
        {"max_abs_rel": float(np.max(np.abs(ratio))), "ells": ells.tolist(), "rel": ratio.tolist(),
         "source_seconds": t_src}, indent=1))


if __name__ == "__main__":
    main()
