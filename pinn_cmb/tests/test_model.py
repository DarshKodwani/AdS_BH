"""Model building blocks: hard ICs, chain continuity, scaling and preconditioning."""

import jax
import jax.numpy as jnp
import numpy as np

from pinn_cmb import background as bgm
from pinn_cmb import equations as eq
from pinn_cmb import initial_conditions as ics
from pinn_cmb.model import Chain, Window

TCA = eq.Layout(tca=True, lmax_nu=4)
FULL = eq.Layout(tca=False, lmax_g=6, lmax_pol=6, lmax_nu=4)
LK = (float(np.log(1e-3)), float(np.log(0.2)))


def _chain(bg):
    ws = [Window(TCA, 1e-3, 1.0, LK), Window(TCA, 1.0, 30.0, LK), Window(FULL, 30.0, 300.0, LK)]
    ps = [w.init(jax.random.PRNGKey(i)) for i, w in enumerate(ws)]
    return Chain(ws, ps, 1e-3)


def test_first_window_reproduces_initial_condition_exactly(bg):
    ch = _chain(bg)
    k = jnp.float64(0.03)
    _, u = ch.state(1e-3, k, jnp.zeros(0), bg)
    np.testing.assert_array_equal(u, ics.initial_state(k, 1e-3, bg, TCA))


def test_chain_is_continuous_at_window_edges(bg):
    ch = _chain(bg)
    k, c = jnp.float64(0.07), jnp.zeros(0)
    w0, w1 = ch.windows[0], ch.windows[1]
    end0 = w0.u(ch.params[0], jnp.float64(1.0), k, c, ch.start_state(0, k, c, bg), bg)
    start1 = w1.u(ch.params[1], jnp.float64(0.0), k, c, ch.start_state(1, k, c, bg), bg)
    np.testing.assert_allclose(end0, start1, rtol=1e-14, atol=1e-16)


def test_layout_switch_maps_tca_state_to_full_hierarchy(bg):
    ch = _chain(bg)
    k, c = jnp.float64(0.07), jnp.zeros(0)
    u_full = ch.start_state(2, k, c, bg)
    u_tca = ch.windows[1].u(ch.params[1], jnp.float64(1.0), k, c, ch.start_state(1, k, c, bg), bg)
    assert u_full.shape == (FULL.size,)
    R = float(bgm.at(bg, 30.0).R)
    mean = (u_full[FULL.index["theta_g"]] + R * u_full[FULL.index["theta_b"]]) / (1 + R)
    np.testing.assert_allclose(mean, u_tca[TCA.index["theta_gb"]], rtol=1e-12)
    slip = u_full[FULL.index["theta_b"]] - u_full[FULL.index["theta_g"]]
    assert 0 < abs(float(slip)) < 1e-2 * abs(float(u_tca[TCA.index["theta_gb"]]))


def test_scales_positive_and_superhorizon_powers(bg):
    w = Window(FULL, 30.0, 300.0, LK)
    k, tau = 1e-4, 30.0  # k tau = 3e-3
    s = w.scale(tau, k, bg)
    assert bool(jnp.all(s > 0))
    ix = FULL.index
    np.testing.assert_allclose(s[ix["F_nu4"]] / s[ix["F_nu2"]], (k * tau) ** 2, rtol=1e-4)


def test_preconditioning_only_on_thomson_rows(bg):
    w = Window(FULL, 30.0, 300.0, LK)
    d = w.precond(40.0, 0.05, bg)
    ix = FULL.index
    assert float(d[ix["phi"]]) == 1.0 and float(d[ix["delta_g"]]) == 1.0
    assert float(d[ix["theta_g"]]) > 100.0 and float(d[ix["F_g2"]]) > 100.0


def _late_chain(bg):
    from pinn_cmb.model import ModeWindow

    rsa = eq.Layout(tca=False, rsa=True)
    t_r = lambda k: jnp.maximum(45.0 / k, 365.0)
    ws = [Window(TCA, 1e-3, 30.0, LK), Window(FULL, 30.0, 600.0, LK), Window(FULL, 600.0, 14000.0, LK),
          ModeWindow(rsa, 1.0, 14000.0, LK, tau_start=t_r, frac_a=0.0, frac_b=0.5, max_start=13000.0),
          ModeWindow(rsa, 1.0, 14000.0, LK, tau_start=t_r, frac_a=0.5, frac_b=1.0, max_start=13000.0)]
    return Chain(ws, [w.init(jax.random.PRNGKey(i)) for i, w in enumerate(ws)], 1e-3)


def test_mode_window_starts_from_full_chain_at_rsa_switch(bg):
    ch = _late_chain(bg)
    k, c = jnp.float64(0.05), jnp.zeros(0)  # switch at tau = 900, inside window 2
    t_r = 900.0
    u_full = ch.windows[2].u(ch.params[2], ch.windows[2].x_of(t_r), k, c, ch.start_state(2, k, c, bg), bg)
    u_rsa = ch.start_state(3, k, c, bg)
    np.testing.assert_allclose(u_rsa, u_full[np.array([FULL.index[n] for n in ch.windows[3].layout.names])], rtol=1e-12)
    np.testing.assert_allclose(float(ch.windows[3].tau(0.0, k)), t_r, rtol=1e-12)
    # second RSA window continues the first
    end3 = ch.windows[3].u(ch.params[3], jnp.float64(1.0), k, c, u_rsa, bg)
    np.testing.assert_allclose(ch.start_state(4, k, c, bg), end3, rtol=1e-12)


def test_chain_ranges_route_low_k_through_full_hierarchy(bg):
    ch = _late_chain(bg)
    r_low, r_high = ch.ranges(jnp.float64(1e-3)), ch.ranges(jnp.float64(0.05))
    assert r_low[3][0] > r_low[3][1]  # low k never enters RSA (45/k > tau0)
    assert r_low[2] == (600.0, 14000.0)
    assert r_high[2] == (600.0, 900.0) and abs(r_high[4][1] - 14000.0) < 1e-6
