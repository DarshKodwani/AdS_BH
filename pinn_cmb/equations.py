"""Linear Einstein–Boltzmann equations, conformal Newtonian gauge (Ma & Bertschinger 1995).

This module is the single source of truth: the PINN residuals and the reference ODE
solver both call these functions.

Conventions (M&B §4, §5): ds^2 = a^2[-(1+2 psi) dtau^2 + (1-2 phi) dx^2]; dots are
d/dtau; theta = i k^j v_j; sigma is the anisotropic stress. Multipoles follow M&B:
F_0 = delta, F_1 = 4 theta / (3k), F_2 = 2 sigma; G_l is the polarisation hierarchy.
Densities are in CLASS units (see background.py), so 4 pi G a^2 rho -> 1.5 a^2 rho.

Equation numbers refer to the arXiv v1 of M&B (astro-ph/9506072), checked against the text:
  Einstein 00, 0i, ij-traceless: eq. (23a), (23b), (23d)
  CDM: eq. (30) with w = 0;  baryons: §5.6
  photons (Newtonian gauge): eq. (64), truncation eq. (65)
  massless neutrinos (Newtonian gauge): eq. (50), truncation eq. (51)
  tight coupling: §5.6, zeroth order in 1/kappa'
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import jax.numpy as jnp

from . import background as bgm


# ---------------------------------------------------------------------------
# State layouts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Layout:
    """Names and positions of the state variables. Hashable, so usable as a jit static arg."""

    tca: bool
    lmax_g: int = 0
    lmax_pol: int = 0
    lmax_nu: int = 4

    def __post_init__(self):
        if self.lmax_nu < 3:
            raise ValueError("lmax_nu must be >= 3")
        if not self.tca and (self.lmax_g < 3 or self.lmax_pol < 3):
            raise ValueError("full photon hierarchy needs lmax_g, lmax_pol >= 3")

    @cached_property
    def names(self) -> tuple[str, ...]:
        nu = ("delta_nu", "theta_nu") + tuple(f"F_nu{l}" for l in range(2, self.lmax_nu + 1))
        if self.tca:
            return ("phi", "delta_c", "theta_c", "delta_b", "delta_g", "theta_gb") + nu
        photons = tuple(f"F_g{l}" for l in range(2, self.lmax_g + 1)) + tuple(
            f"G_g{l}" for l in range(0, self.lmax_pol + 1)
        )
        return ("phi", "delta_c", "theta_c", "delta_b", "theta_b", "delta_g", "theta_g") + photons + nu

    @cached_property
    def index(self) -> dict[str, int]:
        return {n: i for i, n in enumerate(self.names)}

    @property
    def size(self) -> int:
        return len(self.names)

    def block(self, prefix: str) -> slice:
        idx = [i for n, i in self.index.items() if n.startswith(prefix)]
        return slice(idx[0], idx[-1] + 1)


# ---------------------------------------------------------------------------
# Per-equation building blocks
# ---------------------------------------------------------------------------


def psi_from_ij(phi, k, b: bgm.BgAt, sigma_g, sigma_nu):
    """Einstein ij-traceless (M&B 23d): k^2 (phi - psi) = 12 pi G a^2 sum (rho+P) sigma."""
    return phi - 6.0 * b.a**2 / k**2 * (b.rho_g * sigma_g + b.rho_nu * sigma_nu)


def phi_dot_from_0i(k, b: bgm.BgAt, psi, rho_plus_p_theta):
    """Einstein 0i (M&B 23b): k^2 (phi' + H psi) = 4 pi G a^2 sum (rho+P) theta."""
    return -b.conf_H * psi + 1.5 * b.a**2 / k**2 * rho_plus_p_theta


def constraint_00(k, b: bgm.BgAt, phi, phi_dot, psi, delta_rho):
    """Einstein 00 (M&B 23a), returned as a dimensionless residual.

    k^2 phi + 3 H (phi' + H psi) = -4 pi G a^2 delta rho. Normalised by the sum of the
    magnitudes of the three terms, so 0 means satisfied and 1 means violated at O(1).
    """
    lhs1 = k**2 * phi
    lhs2 = 3.0 * b.conf_H * (phi_dot + b.conf_H * psi)
    rhs = -1.5 * b.a**2 * delta_rho
    return (lhs1 + lhs2 - rhs) / (jnp.abs(lhs1) + jnp.abs(lhs2) + jnp.abs(rhs))


def cdm_rhs(k, b: bgm.BgAt, delta, theta, phi_dot, psi):
    """M&B eq. (30): delta' = -theta + 3 phi', theta' = -H theta + k^2 psi."""
    return -theta + 3.0 * phi_dot, -b.conf_H * theta + k**2 * psi


def baryon_rhs(k, b: bgm.BgAt, delta, theta, theta_g, phi_dot, psi):
    """M&B §5.6: theta_b' = -H theta_b + c_s^2 k^2 delta_b + k^2 psi + (kappa'/R)(theta_g - theta_b).

    Here R = 3 rho_b / (4 rho_g), the inverse of M&B's R.
    """
    d_delta = -theta + 3.0 * phi_dot
    d_theta = (
        -b.conf_H * theta + b.cb2 * k**2 * delta + k**2 * psi + b.kappa_dot / b.R * (theta_g - theta)
    )
    return d_delta, d_theta


def streaming(k, tau, F_ext, lmin):
    """Free-streaming part of a Boltzmann hierarchy, dF_l/dtau for l = lmin..L.

    F_ext holds [F_0, ..., F_L]; for lmin = 0 the l F_{l-1} term vanishes. Interior multipoles use
    dF_l = k/(2l+1) [l F_{l-1} - (l+1) F_{l+1}]; the last one uses the M&B truncation
    dF_L = k F_{L-1} - (L+1)/tau F_L (eq. 51 / 64).
    """
    L = F_ext.shape[0] - 1
    l = jnp.arange(lmin, L)
    prev = F_ext[jnp.maximum(l - 1, 0)] * (l > 0)
    interior = k / (2 * l + 1) * (l * prev - (l + 1) * F_ext[l + 1])
    last = k * F_ext[L - 1] - (L + 1) / tau * F_ext[L]
    return jnp.concatenate([interior, last[None]])


def neutrino_rhs(k, tau, delta, theta, F_hi, phi_dot, psi):
    """Massless neutrinos, M&B eq. (50) with truncation (51). F_hi = [F_2, ..., F_lmax]."""
    sigma = 0.5 * F_hi[0]
    d_delta = -4.0 / 3.0 * theta + 4.0 * phi_dot
    d_theta = k**2 * (0.25 * delta - sigma) + k**2 * psi
    F_ext = jnp.concatenate([jnp.stack([delta, 4.0 * theta / (3.0 * k)]), F_hi])
    return d_delta, d_theta, streaming(k, tau, F_ext, 2)


def photon_rhs(k, tau, b: bgm.BgAt, delta, theta, F_hi, G, theta_b, phi_dot, psi):
    """Photons with Thomson scattering and polarisation, M&B eq. (64) with truncation (65).

    F_hi = [F_2, ..., F_lmax_g], G = [G_0, ..., G_lmax_pol].
    """
    kd = b.kappa_dot
    sigma = 0.5 * F_hi[0]
    pi_pol = F_hi[0] + G[0] + G[2]  # Pi = F_2 + G_0 + G_2
    d_delta = -4.0 / 3.0 * theta + 4.0 * phi_dot
    d_theta = k**2 * (0.25 * delta - sigma) + k**2 * psi + kd * (theta_b - theta)
    F_ext = jnp.concatenate([jnp.stack([delta, 4.0 * theta / (3.0 * k)]), F_hi])
    dF = streaming(k, tau, F_ext, 2) - kd * F_hi
    dF = dF.at[0].add(0.1 * kd * pi_pol)
    dG = streaming(k, tau, G, 0) - kd * G
    dG = dG.at[0].add(0.5 * kd * pi_pol).at[2].add(0.1 * kd * pi_pol)
    return d_delta, d_theta, dF, dG


def tca_theta_dot(k, b: bgm.BgAt, delta_b, delta_g, theta_gb, psi):
    """Zeroth-order tight coupling, theta_b = theta_g (M&B eq. 74-75 with the slip dropped).

    Sum the photon and R x baryon Euler equations; the Thomson terms cancel.
    """
    R = b.R
    return (
        -b.conf_H * R * theta_gb + 0.25 * k**2 * delta_g + R * b.cb2 * k**2 * delta_b
    ) / (1.0 + R) + k**2 * psi


# ---------------------------------------------------------------------------
# Full right-hand sides
# ---------------------------------------------------------------------------


def unpack(y, layout: Layout) -> dict:
    """Named view of a state vector; hierarchies are returned as arrays."""
    ix = layout.index
    out = {n: y[i] for n, i in ix.items() if not n.startswith(("F_", "G_"))}
    out["F_nu"] = y[layout.block("F_nu")]
    if not layout.tca:
        out["F_g"] = y[layout.block("F_g")]
        out["G_g"] = y[layout.block("G_g")]
    return out


def metric(tau, y, k, bg: bgm.Background, layout: Layout):
    """Background at tau plus (psi, phi', delta rho) implied by the state."""
    b = bgm.at(bg, tau)
    s = unpack(y, layout)
    sigma_nu = 0.5 * s["F_nu"][0]
    if layout.tca:
        sigma_g = 0.0
        theta_b = theta_g = s["theta_gb"]
    else:
        sigma_g = 0.5 * s["F_g"][0]
        theta_b, theta_g = s["theta_b"], s["theta_g"]
    psi = psi_from_ij(s["phi"], k, b, sigma_g, sigma_nu)
    rho_theta = (
        b.rho_c * s["theta_c"]
        + b.rho_b * theta_b
        + 4.0 / 3.0 * (b.rho_g * theta_g + b.rho_nu * s["theta_nu"])
    )
    phi_dot = phi_dot_from_0i(k, b, psi, rho_theta)
    delta_rho = (
        b.rho_c * s["delta_c"] + b.rho_b * s["delta_b"] + b.rho_g * s["delta_g"] + b.rho_nu * s["delta_nu"]
    )
    return b, s, psi, phi_dot, delta_rho


def rhs(tau, y, k, bg: bgm.Background, layout: Layout):
    """dy/dtau for either layout (TCA or full hierarchy)."""
    b, s, psi, phi_dot, _ = metric(tau, y, k, bg, layout)
    dc = cdm_rhs(k, b, s["delta_c"], s["theta_c"], phi_dot, psi)
    dnu = neutrino_rhs(k, tau, s["delta_nu"], s["theta_nu"], s["F_nu"], phi_dot, psi)
    if layout.tca:
        theta = s["theta_gb"]
        d_delta_b = -theta + 3.0 * phi_dot
        d_delta_g = -4.0 / 3.0 * theta + 4.0 * phi_dot
        d_theta = tca_theta_dot(k, b, s["delta_b"], s["delta_g"], theta, psi)
        head = [phi_dot, dc[0], dc[1], d_delta_b, d_delta_g, d_theta]
        tails = []
    else:
        db = baryon_rhs(k, b, s["delta_b"], s["theta_b"], s["theta_g"], phi_dot, psi)
        dg = photon_rhs(
            k, tau, b, s["delta_g"], s["theta_g"], s["F_g"], s["G_g"], s["theta_b"], phi_dot, psi
        )
        head = [phi_dot, dc[0], dc[1], db[0], db[1], dg[0], dg[1]]
        tails = [dg[2], dg[3]]
    return jnp.concatenate([jnp.stack(head), *tails, jnp.stack([dnu[0], dnu[1]]), dnu[2]])


def observables(tau, y, k, bg: bgm.Background, layout: Layout) -> dict:
    """Physical variables (CLASS naming where possible) plus the 00-constraint residual."""
    b, s, psi, phi_dot, delta_rho = metric(tau, y, k, bg, layout)
    if layout.tca:
        theta_b = theta_g = s["theta_gb"]
        shear_g = jnp.zeros_like(theta_g)
    else:
        theta_b, theta_g, shear_g = s["theta_b"], s["theta_g"], 0.5 * s["F_g"][0]
    return {
        "phi": s["phi"],
        "psi": psi,
        "phi_dot": phi_dot,
        "delta_c": s["delta_c"],
        "theta_c": s["theta_c"],
        "delta_b": s["delta_b"],
        "theta_b": theta_b,
        "delta_g": s["delta_g"],
        "theta_g": theta_g,
        "shear_g": shear_g,
        "delta_nu": s["delta_nu"],
        "theta_nu": s["theta_nu"],
        "shear_nu": 0.5 * s["F_nu"][0],
        "constraint_00": constraint_00(k, b, s["phi"], phi_dot, psi, delta_rho),
    }


def tca_to_full(tau, y_tca, k, bg: bgm.Background, tca: Layout, full: Layout):
    """Map a TCA state onto the full hierarchy at the switch time.

    theta_b = theta_g = theta_gb. The photon shear and polarisation take their first-order
    tight-coupling values: sigma_g = (16/45) theta_g / kappa', and quasi-static Thomson
    balance gives G_0 = (5/4) F_2, G_2 = (1/4) F_2. Higher multipoles start at zero.
    """
    b = bgm.at(bg, tau)
    s = unpack(y_tca, tca)
    F2 = 32.0 / 45.0 * s["theta_gb"] / b.kappa_dot
    y = jnp.zeros(full.size)
    ix = full.index
    for n in ("phi", "delta_c", "theta_c", "delta_b", "delta_g", "delta_nu", "theta_nu"):
        y = y.at[ix[n]].set(s[n])
    y = y.at[ix["theta_b"]].set(s["theta_gb"]).at[ix["theta_g"]].set(s["theta_gb"])
    y = y.at[ix["F_g2"]].set(F2).at[ix["G_g0"]].set(1.25 * F2).at[ix["G_g2"]].set(0.25 * F2)
    return y.at[full.block("F_nu")].set(s["F_nu"])
