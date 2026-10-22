"""Adiabatic growing-mode initial conditions, conformal Newtonian gauge.

Leading order (M&B astro-ph/9506072 v1, eq. 98), for k tau << 1 in radiation domination:

    psi = 20 C / (15 + 4 R_nu),       phi = (1 + 2 R_nu / 5) psi,
    delta_g = delta_nu = -2 psi,      delta_c = delta_b = (3/4) delta_g,
    theta_{g,nu,c,b} = (1/2) k^2 tau psi,
    sigma_nu = (1/15) (k tau)^2 psi,

with R_nu = rho_nu / (rho_g + rho_nu).

These leading-order values violate the Einstein 00 constraint at O(tau / tau_eq), and the
constraint-violating solution of the evolution system grows like tau relative to the
physical one (a 1e-5 violation at tau_i became 5% by recombination). So the state we
actually integrate follows CLASS (perturbations.c, adiabatic IC): the synchronous-gauge
expansion to first order in (k tau)^2 and in omega tau, with omega = a rho_m / sqrt(rho_r),
followed by the gauge transformation to Newtonian gauge. That transformation takes
alpha = (h' + 6 eta') / (2 k^2) from the 00 and 0i Einstein equations, so the initial
state satisfies the constraint by construction. M&B's C equals half of CLASS's curvature_ini.
"""

from __future__ import annotations

import jax.numpy as jnp

from . import background as bgm
from .equations import Layout


def r_nu(bg: bgm.Background):
    return bg.rho0[3] / (bg.rho0[2] + bg.rho0[3])


def adiabatic_leading_order(k, tau, bg: bgm.Background, C=1.0) -> dict:
    """M&B eq. (98): the leading-order mode (sigma_g and higher multipoles vanish)."""
    rn = r_nu(bg)
    psi = 20.0 * C / (15.0 + 4.0 * rn)
    delta_g = -2.0 * psi
    theta = 0.5 * k**2 * tau * psi
    return {
        "psi": psi,
        "phi": (1.0 + 0.4 * rn) * psi,
        "delta_g": delta_g,
        "delta_nu": delta_g,
        "delta_c": 0.75 * delta_g,
        "delta_b": 0.75 * delta_g,
        "theta_g": theta,
        "theta_b": theta,
        "theta_c": theta,
        "theta_nu": theta,
        "shear_nu": (k * tau) ** 2 * psi / 15.0,
    }


def adiabatic(k, tau, bg: bgm.Background, C=1.0) -> dict:
    """Constraint-consistent adiabatic mode (CLASS scheme), as named Newtonian-gauge variables."""
    b = bgm.at(bg, tau)
    rho_r = b.rho_g + b.rho_nu
    rho_m = b.rho_c + b.rho_b
    fnu = b.rho_nu / rho_r
    fb = b.rho_b / rho_m
    ot = b.a * rho_m / jnp.sqrt(rho_r) * tau
    kt2 = (k * tau) ** 2
    kt3 = k * tau * kt2
    curv = 2.0 * C

    # Synchronous gauge, flat space (CLASS's s2_squared = 1).
    delta_g = -kt2 / 3.0 * (1.0 - ot / 5.0) * curv
    theta_g = -k * kt3 / 36.0 * (1.0 - 3.0 * (1.0 + 5.0 * fb - fnu) / 20.0 / (1.0 - fnu) * ot) * curv
    theta_nu = (
        -k * kt3 / 36.0 / (4.0 * fnu + 15.0)
        * (4.0 * fnu + 23.0 - 3.0 * (8.0 * fnu**2 + 50.0 * fnu + 275.0) / 20.0 / (2.0 * fnu + 15.0) * ot)
        * curv
    )
    shear_nu = 2.0 * kt2 / (45.0 + 12.0 * fnu) * (1.0 + (4.0 * fnu - 5.0) / 4.0 / (2.0 * fnu + 15.0) * ot) * curv
    F_nu3 = 2.0 * kt3 / 7.0 / (12.0 * fnu + 45.0) * curv
    eta = curv * (
        1.0 - kt2 / 12.0 / (15.0 + 4.0 * fnu)
        * (5.0 + 4.0 * fnu - (16.0 * fnu**2 + 280.0 * fnu + 325.0) / 10.0 / (2.0 * fnu + 15.0) * ot)
    )
    sync = {
        "delta_g": delta_g, "theta_g": theta_g,
        "delta_b": 0.75 * delta_g, "theta_b": theta_g,
        "delta_c": 0.75 * delta_g, "theta_c": jnp.zeros_like(delta_g),
        "delta_nu": delta_g, "theta_nu": theta_nu,
    }

    # Newtonian gauge: alpha from Einstein 00 + 0i written in synchronous variables.
    H = b.conf_H
    delta_rho = b.rho_g * delta_g + b.rho_nu * delta_g + b.rho_b * sync["delta_b"] + b.rho_c * sync["delta_c"]
    rho_p_theta = 4.0 / 3.0 * (b.rho_g * theta_g + b.rho_nu * theta_nu) + b.rho_b * theta_g
    alpha = (eta + 1.5 * b.a**2 / k**2 * (delta_rho + 3.0 * H / k**2 * rho_p_theta)) / H
    out = {"phi": eta - H * alpha, "shear_nu": shear_nu, "F_nu3": F_nu3}
    for name, w_plus_1 in (("g", 4.0 / 3.0), ("nu", 4.0 / 3.0), ("b", 1.0), ("c", 1.0)):
        out[f"delta_{name}"] = sync[f"delta_{name}"] - 3.0 * w_plus_1 * H * alpha
        out[f"theta_{name}"] = sync[f"theta_{name}"] + k**2 * alpha
    return out


def initial_state(k, tau, bg: bgm.Background, layout: Layout, C=1.0):
    """State vector for the given layout. theta_b = theta_g; photon F_l>=2 and G_l start at zero."""
    ic = adiabatic(k, tau, bg, C)
    vals = {
        **{n: ic[n] for n in ("phi", "delta_c", "theta_c", "delta_b", "theta_b", "delta_g", "theta_g",
                              "delta_nu", "theta_nu", "F_nu3")},
        "theta_gb": ic["theta_g"],
        "F_nu2": 2.0 * ic["shear_nu"],
    }
    y = jnp.zeros(layout.size)
    for name, i in layout.index.items():
        if name in vals:
            y = y.at[i].set(vals[name])
    return y


def default_tau_i(k: float, bg: bgm.Background, k_tau: float = 1e-3, max_tau: float = 1e-2) -> float:
    """tau_i with k tau_i <= k_tau and tau_i <= max_tau Mpc (omega tau ~ 1e-4), above the tables."""
    return max(min(k_tau / k, max_tau), 2.0 * bg.tau_min)
