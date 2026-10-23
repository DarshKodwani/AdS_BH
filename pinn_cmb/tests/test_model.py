"""Model building blocks: hard ICs, chain continuity, scaling and preconditioning."""

import jax
import jax.numpy as jnp
import numpy as np

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
    assert u_full[FULL.index["theta_b"]] == u_tca[TCA.index["theta_gb"]] == u_full[FULL.index["theta_g"]]


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
