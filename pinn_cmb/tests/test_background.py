"""Background tables: interpolation accuracy against CLASS and internal consistency."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pinn_cmb import background as bgm


@pytest.fixture(scope="module")
def class_tables():
    from classy import Class

    c = Class()
    c.set(bgm.class_base_settings())
    c.compute()
    out = c.get_background(), c.get_thermodynamics(), c.get_current_derived_parameters(["rs_rec", "tau_rec"])
    c.struct_cleanup()
    c.empty()
    return out


def _at(bg, tau):
    return jax.vmap(lambda t: bgm.at(bg, t))(jnp.asarray(tau))


def test_conformal_hubble_matches_class(bg, class_tables):
    b_cls, _, _ = class_tables
    tau = b_cls["conf. time [Mpc]"]
    sel = (tau > 1e-4) & (tau < 0.99 * bg.tau0)
    a = 1.0 / (1.0 + b_cls["z"][sel])
    ours = _at(bg, tau[sel]).conf_H
    np.testing.assert_allclose(ours, a * b_cls["H [1/Mpc]"][sel], rtol=1e-6)


def test_hubble_is_derivative_of_lna(bg):
    """The Friedmann conf_H must equal d ln a / d tau of the a(tau) spline."""
    tau = jnp.geomspace(1e-3, 1e4, 50)
    dlna = jax.vmap(jax.grad(lambda t: jnp.log(bgm.at(bg, t).a)))(tau)
    np.testing.assert_allclose(dlna, _at(bg, tau).conf_H, rtol=5e-6)


def test_opacity_and_sound_speed_match_class(bg, class_tables):
    _, th, _ = class_tables
    tau = th["conf. time [Mpc]"]
    sel = (tau > 0) & (tau < 0.99 * bg.tau0)
    # evaluate between knots, where interpolation error is largest
    t_mid = np.sqrt(tau[sel][1:] * tau[sel][:-1])
    kd_cls = np.exp(np.interp(np.log(t_mid), np.log(tau[sel][::-1]), np.log(th["kappa' [Mpc^-1]"][sel][::-1])))
    b = _at(bg, t_mid)
    np.testing.assert_allclose(b.kappa_dot, kd_cls, rtol=1e-3)
    cb2_cls = np.interp(np.log(t_mid), np.log(tau[sel][::-1]), th["c_b^2"][sel][::-1])
    np.testing.assert_allclose(b.cb2, cb2_cls, rtol=1e-3)


def test_early_time_extrapolation_is_fully_ionised(bg):
    """Before the CLASS thermo table: kappa' ∝ a^-2 and c_b^2 ∝ a^-1."""
    tau = jnp.array([2e-5, 2e-4])
    b = _at(bg, tau)
    ratio_a = b.a[1] / b.a[0]
    np.testing.assert_allclose(b.kappa_dot[1] / b.kappa_dot[0], ratio_a**-2, rtol=1e-6)
    np.testing.assert_allclose(b.cb2[1] / b.cb2[0], ratio_a**-1, rtol=1e-6)


def test_sound_horizon_at_recombination(bg, class_tables):
    _, _, derived = class_tables
    rs = float(bgm.sound_horizon(bg, jnp.float64(derived["tau_rec"])))
    np.testing.assert_allclose(rs, derived["rs_rec"], rtol=2e-3)


def test_radiation_era_limits(bg):
    b = bgm.at(bg, jnp.float64(1e-3))
    assert abs(float(b.conf_H) * 1e-3 - 1.0) < 1e-5  # conf_H = 1/tau
    np.testing.assert_allclose(float(bgm.sound_horizon(bg, 1e-3)), 1e-3 / np.sqrt(3.0), rtol=1e-4)
