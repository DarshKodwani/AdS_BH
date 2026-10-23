"""Background and thermodynamics from CLASS, as JAX-differentiable cubic splines.

Units follow CLASS: conformal time tau in Mpc, k in 1/Mpc, and densities rho_i in
Mpc^-2 normalised so that H^2 = sum_i rho_i (8 pi G / 3 absorbed). Hence
4 pi G a^2 rho_phys = 1.5 a^2 rho_i, which is the factor used in equations.py.

Densities are evaluated analytically from today's values (rho_i ∝ a^-3 or a^-4),
and the conformal Hubble rate from the Friedmann equation, so that the Einstein
equations see an exactly self-consistent background. Only a(tau) and the
thermodynamic quantities (kappa', c_b^2) are interpolated from CLASS tables.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline

jax.config.update("jax_enable_x64", True)

# Planck 2018 (TT,TE,EE+lowE+lensing) with massless neutrinos only.
FIDUCIAL = {
    "h": 0.6736,
    "omega_b": 0.02237,
    "omega_cdm": 0.1200,
    "N_ur": 3.046,
    "T_cmb": 2.7255,
}


class Spline(NamedTuple):
    """Cubic spline in scipy's PPoly convention: knots x (n,), coefficients c (4, n-1)."""

    x: jax.Array
    c: jax.Array


def make_spline(x: np.ndarray, y: np.ndarray) -> Spline:
    cs = CubicSpline(x, y, bc_type="not-a-knot")
    return Spline(jnp.asarray(cs.x), jnp.asarray(cs.c))


def spline_eval(s: Spline, xq: jax.Array) -> jax.Array:
    """Evaluate the spline; differentiable in xq (the end pieces extrapolate)."""
    i = jnp.clip(jnp.searchsorted(s.x, xq, side="right") - 1, 0, s.x.shape[0] - 2)
    dx = xq - s.x[i]
    c = s.c[:, i]
    return ((c[0] * dx + c[1]) * dx + c[2]) * dx + c[3]


class Background(NamedTuple):
    lna: Spline  # ln a vs ln tau
    ln_kappa_dot: Spline  # ln kappa' vs ln tau
    ln_cb2: Spline  # ln c_b^2 vs ln tau
    rs: Spline  # comoving sound horizon r_s vs ln tau
    ln_kd2inv: Spline  # ln(k_D^-2) vs ln tau (Silk damping scale)
    rho0: jax.Array  # today's [cdm, b, g, ur, lambda] densities in Mpc^-2
    tau_min: float  # lower edge of the tables
    tau0: float  # conformal age
    tau_rec: float  # CLASS's tau at the visibility maximum
    vis: Spline | None = None  # visibility g = kappa' e^-kappa [1/Mpc] vs ln tau (CLASS table)
    exp_m_kappa: Spline | None = None  # e^-kappa vs ln tau (CLASS table)
    tau_free_streaming: float = 0.0  # CLASS: first tau with 1/(kappa' tau) > 5 (RSA trigger)


class BgAt(NamedTuple):
    a: jax.Array
    conf_H: jax.Array  # conformal Hubble rate a'/a
    rho_c: jax.Array
    rho_b: jax.Array
    rho_g: jax.Array
    rho_nu: jax.Array
    kappa_dot: jax.Array  # Thomson opacity a n_e sigma_T
    cb2: jax.Array  # baryon sound speed squared
    R: jax.Array  # 3 rho_b / (4 rho_g)


C_KM_S = 299792.458
H100_MPC = 100.0 / C_KM_S  # H0 / h in 1/Mpc


class AnalyticBackground(NamedTuple):
    """Radiation + matter background, analytic in tau and differentiable in (omega_b, omega_c).

    For tau before recombination only (Lambda is ~1e-9 of rho there). Friedmann
    (a')^2 = rho_r0 + rho_m0 a gives a(tau) = sqrt(rho_r0) tau + rho_m0 tau^2 / 4 exactly.
    c_b^2 ∝ 1/a (T_b = T_gamma). kappa' and the Silk scale are the fiducial CLASS tables in
    ln a, scaled by omega_b / omega_b,fid (n_e ∝ omega_b); they only enter the TCA switch
    time and the ansatz envelope, never the TCA equations themselves.
    """

    rho0: jax.Array  # [cdm, b, g, ur, lambda=0]
    cb2_a: jax.Array  # c_b^2 * a (constant while T_b = T_gamma)
    ln_kappa_a: Spline  # fiducial ln kappa' vs ln a
    ln_kd2inv_a: Spline  # fiducial ln k_D^-2 vs ln a
    omega_b_ratio: jax.Array


def analytic_background(bg_fid: Background, omega_b, omega_c, h: float = FIDUCIAL["h"]) -> AnalyticBackground:
    """Analytic background for (omega_b, omega_c), sharing radiation and thermal tables with bg_fid."""
    lnt = jnp.linspace(jnp.log(2 * bg_fid.tau_min), jnp.log(0.8 * bg_fid.tau_rec), 2000)
    b = jax.vmap(lambda t: at(bg_fid, t))(jnp.exp(lnt))
    lna = np.log(np.asarray(b.a))
    s_kappa = make_spline(lna, np.log(np.asarray(b.kappa_dot)))
    s_kd = make_spline(lna, np.asarray(spline_eval(bg_fid.ln_kd2inv, lnt)))
    ob_fid = float(bg_fid.rho0[1]) / H100_MPC**2
    return AnalyticBackground(
        rho0=jnp.stack([omega_c * H100_MPC**2, omega_b * H100_MPC**2, bg_fid.rho0[2], bg_fid.rho0[3], 0.0 * omega_b]),
        cb2_a=jnp.asarray(np.median(np.asarray(b.cb2 * b.a)[:200])),
        ln_kappa_a=s_kappa,
        ln_kd2inv_a=s_kd,
        omega_b_ratio=omega_b / ob_fid,
    )


def with_params(abg: AnalyticBackground, omega_b, omega_c) -> AnalyticBackground:
    """Same tables, new (omega_b, omega_c): cheap and differentiable (used inside the loss)."""
    ob_fid = abg.rho0[1] / (H100_MPC**2 * abg.omega_b_ratio)
    return abg._replace(rho0=abg.rho0.at[0].set(omega_c * H100_MPC**2).at[1].set(omega_b * H100_MPC**2),
                        omega_b_ratio=omega_b / ob_fid)


def _at_analytic(bg: AnalyticBackground, tau) -> BgAt:
    rho_r0 = bg.rho0[2] + bg.rho0[3]
    rho_m0 = bg.rho0[0] + bg.rho0[1]
    a = jnp.sqrt(rho_r0) * tau + 0.25 * rho_m0 * tau**2
    a_dot = jnp.sqrt(rho_r0) + 0.5 * rho_m0 * tau
    lna = jnp.log(a)
    return BgAt(
        a=a,
        conf_H=a_dot / a,
        rho_c=bg.rho0[0] / a**3,
        rho_b=bg.rho0[1] / a**3,
        rho_g=bg.rho0[2] / a**4,
        rho_nu=bg.rho0[3] / a**4,
        kappa_dot=bg.omega_b_ratio * jnp.exp(spline_eval(bg.ln_kappa_a, lna)),
        cb2=bg.cb2_a / a,
        R=0.75 * bg.rho0[1] / bg.rho0[2] * a,
    )


def at(bg: Background, tau: jax.Array) -> BgAt:
    if isinstance(bg, AnalyticBackground):
        return _at_analytic(bg, tau)
    x = jnp.log(tau)
    a = jnp.exp(spline_eval(bg.lna, x))
    rho_c = bg.rho0[0] / a**3
    rho_b = bg.rho0[1] / a**3
    rho_g = bg.rho0[2] / a**4
    rho_nu = bg.rho0[3] / a**4
    rho_tot = rho_c + rho_b + rho_g + rho_nu + bg.rho0[4]
    return BgAt(
        a=a,
        conf_H=a * jnp.sqrt(rho_tot),
        rho_c=rho_c,
        rho_b=rho_b,
        rho_g=rho_g,
        rho_nu=rho_nu,
        kappa_dot=jnp.exp(spline_eval(bg.ln_kappa_dot, x)),
        cb2=jnp.exp(spline_eval(bg.ln_cb2, x)),
        R=0.75 * rho_b / rho_g,
    )


def sound_horizon(bg: Background, tau: jax.Array) -> jax.Array:
    if isinstance(bg, AnalyticBackground):
        # r_s = int da / sqrt(3 (1 + c a)(A + B a)), c = R/a, A = rho_r0, B = rho_m0 (closed form)
        A, B = bg.rho0[2] + bg.rho0[3], bg.rho0[0] + bg.rho0[1]
        c = 0.75 * bg.rho0[1] / bg.rho0[2]
        a = _at_analytic(bg, tau).a
        F = lambda aa: jnp.log(jnp.sqrt(c * (A + B * aa)) + jnp.sqrt(B * (1.0 + c * aa)))
        return 2.0 / jnp.sqrt(3.0 * c * B) * (F(a) - F(0.0))
    return spline_eval(bg.rs, jnp.log(tau))


def damping_wavenumber(bg: Background, tau: jax.Array) -> jax.Array:
    if isinstance(bg, AnalyticBackground):
        lna = jnp.log(_at_analytic(bg, tau).a)
        return jnp.exp(-0.5 * spline_eval(bg.ln_kd2inv_a, lna)) * jnp.sqrt(bg.omega_b_ratio)
    return jnp.exp(-0.5 * spline_eval(bg.ln_kd2inv, jnp.log(tau)))


def visibility(bg: Background, tau: jax.Array) -> tuple[jax.Array, jax.Array]:
    """(g, e^-kappa) at tau; g' follows by autodiff."""
    x = jnp.log(tau)
    return spline_eval(bg.vis, x), spline_eval(bg.exp_m_kappa, x)


def class_base_settings(params: dict | None = None) -> dict:
    """CLASS settings shared by the background build and the validation harness."""
    return {**FIDUCIAL, **(params or {}), "reio_parametrization": "reio_none"}


def _sorted_unique(tau: np.ndarray, *cols: np.ndarray):
    order = np.argsort(tau)
    tau = tau[order]
    keep = np.concatenate([[True], np.diff(tau) > 0]) & (tau > 0)
    return (tau[keep],) + tuple(c[order][keep] for c in cols)


def build_background(params: dict | None = None, tau_min: float = 1e-5) -> Background:
    """Run CLASS (background + thermodynamics) and tabulate everything the perturbations need."""
    from classy import Class

    cosmo = Class()
    cosmo.set(class_base_settings(params))
    cosmo.compute()
    try:
        bgt = cosmo.get_background()
        th = cosmo.get_thermodynamics()
        derived = cosmo.get_current_derived_parameters(["tau_rec", "conformal_age"])
    finally:
        cosmo.struct_cleanup()
        cosmo.empty()

    tau_b, z_b, rho_c, rho_b, rho_g, rho_nu, rho_l = _sorted_unique(
        bgt["conf. time [Mpc]"], bgt["z"], bgt["(.)rho_cdm"], bgt["(.)rho_b"],
        bgt["(.)rho_g"], bgt["(.)rho_ur"], bgt["(.)rho_lambda"],
    )
    today = np.argmin(np.abs(z_b))
    rho0 = np.array([rho_c[today], rho_b[today], rho_g[today], rho_nu[today], rho_l[today]])
    if tau_b[0] > 0.5 * tau_min:
        raise ValueError(f"CLASS background starts at tau={tau_b[0]:.3g} Mpc > tau_min/2")
    sel = tau_b >= 0.5 * tau_min
    lna = make_spline(np.log(tau_b[sel]), -np.log1p(z_b[sel]))

    # Thermodynamics: CLASS tables start near z ~ 1e6-1e7. Earlier, hydrogen and helium are
    # fully ionised, so n_e ∝ a^-3 (kappa' ∝ a^-2) and T_b = T_gamma (c_b^2 ∝ a^-1).
    tau_t, kd_t, cb2_t, g_t, emk_t = _sorted_unique(
        th["conf. time [Mpc]"], th["kappa' [Mpc^-1]"], th["c_b^2"], th["g [Mpc^-1]"], th["exp(-kappa)"])
    fs = np.nonzero(1.0 / (kd_t * tau_t) > 5.0)[0]
    tau_fs = float(tau_t[fs[0]]) if fs.size else float(tau_t[-1])
    lntau_ext = np.linspace(np.log(0.5 * tau_min), np.log(tau_t[0]), 400)[:-1]
    a_ext = np.exp(np.asarray(spline_eval(lna, jnp.asarray(lntau_ext))))
    a_t0 = float(jnp.exp(spline_eval(lna, jnp.log(tau_t[0]))))
    ln_tau_th = np.concatenate([lntau_ext, np.log(tau_t)])
    ln_kd = np.concatenate([np.log(kd_t[0]) - 2 * np.log(a_ext / a_t0), np.log(kd_t)])
    ln_cb2 = np.concatenate([np.log(cb2_t[0]) - np.log(a_ext / a_t0), np.log(cb2_t)])
    ln_kappa_dot = make_spline(ln_tau_th, ln_kd)
    ln_cb2_s = make_spline(ln_tau_th, ln_cb2)

    # Sound horizon and Silk damping scale on a fine log grid. Both integrands are power
    # laws deep in radiation domination, which fixes the starting values.
    tau0 = float(derived["conformal_age"])
    lt = np.linspace(np.log(0.5 * tau_min), np.log(tau0), 20000)
    tau = np.exp(lt)
    a = np.exp(np.asarray(spline_eval(lna, jnp.asarray(lt))))
    R = 0.75 * rho0[1] / rho0[2] * a
    kappa_dot = np.exp(np.asarray(spline_eval(ln_kappa_dot, jnp.asarray(lt))))
    cs = 1.0 / np.sqrt(3.0 * (1.0 + R))
    rs = cs[0] * tau[0] + cumulative_trapezoid(cs, tau, initial=0.0)
    damp = (R**2 + 16.0 / 15.0 * (1.0 + R)) / (6.0 * kappa_dot * (1.0 + R) ** 2)
    kd2inv = damp[0] * tau[0] / 3.0 + cumulative_trapezoid(damp, tau, initial=0.0)

    return Background(
        lna=lna,
        ln_kappa_dot=ln_kappa_dot,
        ln_cb2=ln_cb2_s,
        rs=make_spline(lt, rs),
        ln_kd2inv=make_spline(lt, np.log(kd2inv)),
        rho0=jnp.asarray(rho0),
        tau_min=float(tau_min),
        tau0=tau0,
        tau_rec=float(derived["tau_rec"]),
        vis=make_spline(np.log(tau_t), g_t),
        exp_m_kappa=make_spline(np.log(tau_t), emk_t),
        tau_free_streaming=tau_fs,
    )
