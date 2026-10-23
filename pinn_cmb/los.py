"""Line-of-sight integration (Seljak & Zaldarriaga 1996) for the unlensed C_l^TT.

Source functions in conformal Newtonian gauge exactly as CLASS (perturbations.c,
"slightly more complicated form, but more efficient numerically"):

    S_0 = g (delta_g/4 + psi) + g (phi - psi) + 2 e^-kappa phi' + (g theta_b' + g' theta_b) / k^2
    S_1 = e^-kappa k (psi - phi)
    S_2 = g P,   P = (F_2 + G_0 + G_2) / 8

with CLASS's flat-space radial functions j_l(x), j_l'(x) and (3 j_l''(x) + j_l(x)) / 2,
x = k (tau0 - tau):

    Delta_l(k) = int dtau [S_0 j_l + S_1 j_l' + S_2 (3 j_l'' + j_l) / 2],
    C_l = 4 pi int dk/k P_R(k) Delta_l(k)^2,   P_R = A_s (k/k_pivot)^(n_s - 1).

Sources must be normalised to unit initial curvature (CLASS curvature_ini = 1), i.e.
M&B C = 1/2 in initial_conditions.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import spherical_jn

from . import background as bgm
from . import equations as eq

PRIMORDIAL = {"A_s": 2.1e-9, "n_s": 0.9649, "k_pivot": 0.05}


def sources(tau, y, k, bg: bgm.Background, layout: eq.Layout) -> dict:
    """S_0, S_1, S_2 at one (tau, k) from a state vector of any layout."""
    o = eq.observables(tau, y, k, bg, layout)
    dy = eq.rhs(tau, y, k, bg, layout)
    theta_b_dot = dy[layout.index["theta_gb" if layout.tca else "theta_b"]]
    g, emk = bgm.visibility(bg, tau)
    g_dot = jax.grad(lambda t: bgm.visibility(bg, t)[0])(tau)
    s0 = (g * (0.25 * o["delta_g"] + o["psi"]) + g * (o["phi"] - o["psi"]) + 2.0 * emk * o["phi_dot"]
          + (g * theta_b_dot + g_dot * o["theta_b"]) / k**2)
    s1 = emk * k * (o["psi"] - o["phi"])
    s2 = g * o["Pi"] / 8.0
    return {"S0": s0, "S1": s1, "S2": s2}


def default_tau_grid(bg: bgm.Background) -> np.ndarray:
    """Dense through recombination, then ~3 Mpc steps to tau0 (ISW)."""
    t_rec = bg.tau_rec
    return np.unique(np.concatenate([
        np.linspace(0.5 * t_rec, 2.5 * t_rec, 1200),
        np.linspace(2.5 * t_rec, bg.tau0 * (1 - 1e-6), 4500),
    ]))


def default_ells(lmax: int) -> np.ndarray:
    return np.unique(np.concatenate([np.arange(2, 31), np.arange(32, 100, 4), np.arange(100, lmax + 1, 12),
                                     [lmax]])).astype(int)


def transfer_tt(k_src: np.ndarray, tau: np.ndarray, S: dict, ells: np.ndarray, tau0: float,
                k_fine: np.ndarray) -> np.ndarray:
    """Delta_l(k) on k_fine for each l. Sources S[name] are (len(k_src), len(tau)), splined in k."""
    Sf = {n: CubicSpline(k_src, v, axis=0)(k_fine) for n, v in S.items()}
    w = np.gradient(tau)  # trapezoid-like weights on a non-uniform grid
    chi = tau0 - tau
    x = k_fine[:, None] * chi[None, :]
    x_max = float(x.max()) + 1.0
    out = np.empty((len(ells), len(k_fine)))
    for i, l in enumerate(ells):
        xg = np.linspace(0.0, x_max, int(x_max / 0.02) + 2)
        j = spherical_jn(l, xg)
        jp = spherical_jn(l, xg, derivative=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            jpp = np.where(xg > 0, -2.0 / xg * jp + (l * (l + 1) / xg**2 - 1.0) * j, 0.0)
        if l == 2:
            jpp[0] = 2.0 / 15.0  # j_2''(0)
        r0 = np.interp(x, xg, j)
        r1 = np.interp(x, xg, jp)
        r2 = 0.5 * (3.0 * np.interp(x, xg, jpp) + r0)
        out[i] = ((Sf["S0"] * r0 + Sf["S1"] * r1 + Sf["S2"] * r2) * w).sum(axis=1)
    return out


def cl_tt(k_fine: np.ndarray, delta: np.ndarray, prim=PRIMORDIAL) -> np.ndarray:
    pr = prim["A_s"] * (k_fine / prim["k_pivot"]) ** (prim["n_s"] - 1.0)
    integrand = pr[None, :] * delta**2 / k_fine[None, :]
    return 4.0 * np.pi * np.trapezoid(integrand, k_fine, axis=1)


def default_k_fine(k_max: float, tau0: float) -> np.ndarray:
    """Fine k grid resolving the (tau0 - tau_rec) oscillation of Delta_l(k)."""
    dk = 2.0 * np.pi / tau0 / 8.0
    return np.concatenate([np.geomspace(1e-5, 2e-4, 40, endpoint=False), np.arange(2e-4, k_max, dk)])
