"""One test per equation block of equations.py, plus the initial conditions.

Each test checks a property that a wrong coefficient or sign would break: exact
analytic solutions (free streaming -> spherical Bessel functions), conservation laws
(Thomson momentum exchange), limits (tight coupling), the radiation-era growing mode
(M&B eq. 98) and propagation of the Einstein 00 constraint.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.special import spherical_jn

from pinn_cmb import background as bgm
from pinn_cmb import equations as eq
from pinn_cmb import initial_conditions as ics

FULL = eq.Layout(tca=False, lmax_g=12, lmax_pol=12, lmax_nu=12)
TCA = eq.Layout(tca=True, lmax_nu=12)


def _bessel_table(L, x):
    return np.array([spherical_jn(l, x) for l in range(L + 1)])


# --- Boltzmann hierarchies -------------------------------------------------


def test_truncation_is_exact_for_bessel_functions():
    """M&B eq. (51) extrapolates F_{L+1} with the j_l recurrence, so it is exact for F_l = j_l(k tau)."""
    k, tau, L = 0.7, 13.0, 9
    F = jnp.asarray(_bessel_table(L, k * tau))
    dF = eq.streaming(k, tau, F, 2)
    expected = k * np.array([spherical_jn(l, k * tau, derivative=True) for l in range(2, L + 1)])
    np.testing.assert_allclose(dF, expected, rtol=1e-12, atol=1e-14)


def _free_stream(rhs_fn, k, L, tau0, tau1):
    y0 = _bessel_table(L, k * tau0)
    sol = solve_ivp(lambda t, y: np.asarray(rhs_fn(t, jnp.asarray(y))), (tau0, tau1), y0,
                    method="DOP853", rtol=1e-11, atol=1e-13, dense_output=True)
    return sol


def test_neutrino_free_streaming_gives_bessel_functions():
    """With phi' = psi = 0, M&B eq. (50) has F_l(tau) = j_l(k tau) (M&B §5.3)."""
    k, L = 1.0, 60

    def rhs(t, y):
        delta, theta = y[0], 0.75 * k * y[1]
        d_delta, d_theta, dF = eq.neutrino_rhs(k, t, delta, theta, y[2:], 0.0, 0.0)
        return jnp.concatenate([jnp.stack([d_delta, d_theta / (0.75 * k)]), dF])

    sol = _free_stream(rhs, k, L, 1e-3, 30.0)
    tau = np.linspace(1.0, 30.0, 60)
    np.testing.assert_allclose(sol.sol(tau)[:4], _bessel_table(3, k * tau), atol=1e-7)


def _bg_at(kappa_dot, R=1.0):
    one = jnp.float64(1.0)
    return bgm.BgAt(a=one, conf_H=one, rho_c=one, rho_b=one, rho_g=one, rho_nu=one,
                    kappa_dot=jnp.float64(kappa_dot), cb2=jnp.float64(0.0), R=jnp.float64(R))


def test_photon_free_streaming_without_scattering():
    """kappa' = 0: temperature multipoles stream as j_l, polarisation stays zero (M&B eq. 64)."""
    k, L = 1.0, 60
    b = _bg_at(0.0)

    def rhs(t, y):
        F, G = y[: L + 1], y[L + 1:]
        dd, dth, dF, dG = eq.photon_rhs(k, t, b, F[0], 0.75 * k * F[1], F[2:], G, 0.0, 0.0, 0.0)
        return jnp.concatenate([jnp.stack([dd, dth / (0.75 * k)]), dF, dG])

    y0 = np.concatenate([_bessel_table(L, k * 1e-3), np.zeros(L + 1)])
    sol = solve_ivp(lambda t, y: np.asarray(rhs(t, jnp.asarray(y))), (1e-3, 30.0), y0,
                    method="DOP853", rtol=1e-11, atol=1e-13, dense_output=True)
    tau = np.linspace(1.0, 30.0, 60)
    np.testing.assert_allclose(sol.sol(tau)[:4], _bessel_table(3, k * tau), atol=1e-7)
    assert np.max(np.abs(sol.y[L + 1:])) == 0.0


def test_thomson_scattering_conserves_momentum():
    """(4/3) rho_g theta_g' + rho_b theta_b' must not depend on kappa' (M&B §5.6)."""
    k, tau = 0.05, 100.0
    rng = np.random.default_rng(0)
    F, G = jnp.asarray(rng.normal(size=8)), jnp.asarray(rng.normal(size=9))
    theta_g, theta_b = 0.3, -0.2

    def momentum_rate(kd):
        R = 0.6
        b = _bg_at(kd, R)._replace(rho_b=jnp.float64(4.0 / 3.0 * R))
        th_g = eq.photon_rhs(k, tau, b, 0.1, theta_g, F, G, theta_b, 0.0, 0.0)[1]
        th_b = eq.baryon_rhs(k, b, 0.1, theta_b, theta_g, 0.0, 0.0)[1]
        return 4.0 / 3.0 * b.rho_g * th_g + b.rho_b * th_b

    assert abs(float(jax.grad(momentum_rate)(5.0))) < 1e-14


def test_tight_coupling_balance_of_polarisation():
    """The tca_to_full photon moments are a fixed point of the Thomson terms for G_0 and G_2."""
    F2 = 0.01
    G = jnp.zeros(9).at[0].set(1.25 * F2).at[2].set(0.25 * F2)
    F = jnp.zeros(8).at[0].set(F2)
    b = _bg_at(1e6)
    _, _, _, dG = eq.photon_rhs(1e-8, 1.0, b, 0.0, 0.0, F, G, 0.0, 0.0, 0.0)
    assert abs(float(dG[0])) < 1e-9 * 1e6 * F2 and abs(float(dG[2])) < 1e-9 * 1e6 * F2


def test_tca_momentum_equation_is_weighted_euler(bg):
    """Zeroth-order TCA: theta' = (theta_g' + R theta_b')/(1+R) from the full equations at theta_b = theta_g."""
    k, tau = 0.05, 50.0
    b = bgm.at(bg, tau)
    delta_b, delta_g, theta, psi = -0.4, -0.7, 0.02, 0.6
    th_g = eq.photon_rhs(k, tau, b, delta_g, theta, jnp.zeros(8), jnp.zeros(9), theta, 0.0, psi)[1]
    th_b = eq.baryon_rhs(k, b, delta_b, theta, theta, 0.0, psi)[1]
    expected = (th_g + b.R * th_b) / (1.0 + b.R)
    np.testing.assert_allclose(eq.tca_theta_dot(k, b, delta_b, delta_g, theta, psi), expected, rtol=1e-13)


# --- Initial conditions and Einstein equations ------------------------------


@pytest.mark.parametrize("k", [1e-3, 0.05])
def test_ic_reduces_to_ma_bertschinger_eq98(bg, k):
    tau = ics.default_tau_i(k, bg)
    full, lo = ics.adiabatic(k, tau, bg), ics.adiabatic_leading_order(k, tau, bg)
    for name in ("phi", "delta_g", "delta_nu", "delta_c", "delta_b", "theta_g", "theta_c", "theta_nu", "shear_nu"):
        np.testing.assert_allclose(full[name], lo[name], rtol=2e-3, err_msg=name)
    b = bgm.at(bg, tau)
    psi = eq.psi_from_ij(full["phi"], k, b, 0.0, full["shear_nu"])
    np.testing.assert_allclose(psi, lo["psi"], rtol=2e-3)  # ij-traceless: phi - psi = (2/5) R_nu psi


@pytest.mark.parametrize("layout", [TCA, FULL], ids=["tca", "full"])
@pytest.mark.parametrize("k", [1e-3, 0.01, 0.05])
def test_ic_satisfies_00_constraint(bg, layout, k):
    tau = ics.default_tau_i(k, bg)
    y = ics.initial_state(k, tau, bg, layout)
    assert abs(float(eq.observables(tau, y, k, bg, layout)["constraint_00"])) < 1e-8


@pytest.mark.parametrize("k", [1e-3, 0.05])
def test_ic_is_radiation_era_growing_mode(bg, k):
    """At tau_i each equation reproduces the growing mode: theta ∝ tau, sigma_nu ∝ tau^2, phi, delta const."""
    tau = ics.default_tau_i(k, bg)
    y = ics.initial_state(k, tau, bg, FULL)
    dy = eq.rhs(tau, y, k, bg, FULL)
    ix = FULL.index
    for name in ("theta_c", "theta_b", "theta_g", "theta_nu"):  # Euler equations
        np.testing.assert_allclose(dy[ix[name]] * tau / y[ix[name]], 1.0, atol=2e-3, err_msg=name)
    np.testing.assert_allclose(dy[ix["F_nu2"]] * tau / y[ix["F_nu2"]], 2.0, atol=4e-3)  # shear equation
    for name in ("phi", "delta_c", "delta_b", "delta_g", "delta_nu"):  # 0i + continuity equations
        assert abs(float(dy[ix[name]] * tau / y[ix[name]])) < 2e-3, name


@pytest.mark.parametrize("layout", [TCA, FULL], ids=["tca", "full"])
@pytest.mark.parametrize("tau", [0.1, 10.0, 200.0])
def test_evolution_preserves_00_constraint(bg, layout, tau):
    """d/dtau of the 00 residual vanishes on the constraint surface (Bianchi identity)."""
    k = 0.02

    def C(t, y):
        b, s, psi, phi_dot, drho = eq.metric(t, y, k, bg, layout)
        return k**2 * s["phi"] + 3 * b.conf_H * (phi_dot + b.conf_H * psi) + 1.5 * b.a**2 * drho

    rng = np.random.default_rng(1)
    y = jnp.asarray(rng.normal(size=layout.size)) * 0.1
    y = y.at[0].add(-C(tau, y) / jax.grad(C, 1)(tau, y)[0])  # project onto C = 0 (C is linear in phi)
    Cdot = jax.grad(C, 0)(tau, y) + jnp.dot(jax.grad(C, 1)(tau, y), eq.rhs(tau, y, k, bg, layout))
    b = bgm.at(bg, tau)
    scale = b.conf_H * 1.5 * b.a**2 * (b.rho_g + b.rho_nu + b.rho_c + b.rho_b) * jnp.max(jnp.abs(y))
    assert abs(float(Cdot / scale)) < 1e-9
