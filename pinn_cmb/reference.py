"""Reference solver: scipy Radau on exactly the equations in equations.py.

The RHS and its exact Jacobian (jax.jacfwd) are JIT-compiled once per layout and
called from scipy. Modes:
  "tca"  - zeroth-order tight coupling until the CLASS-style trigger, then the full
           hierarchy (initialised by equations.tca_to_full);
  "full" - full hierarchy from tau_i, including the stiff Thomson terms (Radau copes);
  "tca_only" - tight coupling throughout (what the Phase 1 PINN solves).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from . import background as bgm
from . import equations as eq
from . import initial_conditions as ics


@dataclass(frozen=True)
class TCATrigger:
    """Leave tight coupling when tau_c/tau_H or tau_c/tau_k exceeds these (tau_c = 1/kappa').

    CLASS defaults are 0.015 and 0.01.
    """

    tau_c_over_tau_h: float = 0.015
    tau_c_over_tau_k: float = 0.01


@dataclass
class ReferenceSolution:
    k: float
    tau: np.ndarray
    values: dict[str, np.ndarray]
    tau_switch: float | None
    stats: dict = field(default_factory=dict)


@lru_cache(maxsize=None)
def _compiled(layout: eq.Layout):
    f = jax.jit(lambda t, y, k, bg: eq.rhs(t, y, k, bg, layout))
    jac = jax.jit(jax.jacfwd(lambda t, y, k, bg: eq.rhs(t, y, k, bg, layout), argnums=1))
    obs = jax.jit(jax.vmap(lambda t, y, k, bg: eq.observables(t, y, k, bg, layout), (0, 0, None, None)))
    return f, jac, obs


_to_full = jax.jit(eq.tca_to_full, static_argnums=(4, 5))


def tca_switch_time(k: float, bg: bgm.Background, trig: TCATrigger, tau_lo: float, tau_hi: float) -> float:
    """First tau at which either tight-coupling trigger fires (kappa' falls monotonically)."""

    def g(log_tau):
        b = bgm.at(bg, jnp.exp(log_tau))
        return float(
            jnp.maximum(
                b.conf_H / b.kappa_dot / trig.tau_c_over_tau_h,
                k / b.kappa_dot / trig.tau_c_over_tau_k,
            )
            - 1.0
        )

    lo, hi = np.log(tau_lo), np.log(tau_hi)
    if g(lo) >= 0:
        return tau_lo
    if g(hi) < 0:
        return tau_hi
    return float(np.exp(brentq(g, lo, hi, xtol=1e-10)))


def _integrate(layout, k, bg, y0, t0, t1, t_eval, rtol, atol):
    f, jac, _ = _compiled(layout)
    sol = solve_ivp(
        lambda t, y: np.asarray(f(t, y, k, bg)),
        (t0, t1),
        np.asarray(y0),
        method="Radau",
        jac=lambda t, y: np.asarray(jac(t, y, k, bg)),
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"Radau failed for k={k}: {sol.message}")
    return sol


def solve(
    k: float,
    bg: bgm.Background,
    tau_eval: np.ndarray,
    *,
    mode: str = "tca",
    lmax_g: int = 17,
    lmax_pol: int = 17,
    lmax_nu: int = 17,
    trigger: TCATrigger = TCATrigger(),
    tau_i: float | None = None,
    C: float = 1.0,
    rtol: float = 1e-10,
    atol: float = 1e-13,
) -> ReferenceSolution:
    """Integrate one k-mode and return the observables of equations.observables on tau_eval."""
    tau_eval = np.asarray(tau_eval, dtype=float)
    tau_i = ics.default_tau_i(k, bg) if tau_i is None else tau_i
    tau_end = float(tau_eval[-1])
    if tau_eval[0] < tau_i:
        raise ValueError(f"tau_eval starts before tau_i={tau_i}")
    tca = eq.Layout(tca=True, lmax_nu=lmax_nu)
    full = eq.Layout(tca=False, lmax_g=lmax_g, lmax_pol=lmax_pol, lmax_nu=lmax_nu)
    kj = jnp.float64(k)

    if mode == "full":
        sol = _integrate(full, kj, bg, ics.initial_state(kj, tau_i, bg, full, C), tau_i, tau_end, tau_eval, rtol, atol)
        segments, tau_switch = [(full, sol)], None
    elif mode == "tca_only":
        sol = _integrate(tca, kj, bg, ics.initial_state(kj, tau_i, bg, tca, C), tau_i, tau_end, tau_eval, rtol, atol)
        segments, tau_switch = [(tca, sol)], None
    elif mode == "tca":
        tau_switch = tca_switch_time(k, bg, trigger, tau_i, tau_end)
        before = tau_eval[tau_eval < tau_switch]
        after = tau_eval[tau_eval >= tau_switch]
        s1 = _integrate(tca, kj, bg, ics.initial_state(kj, tau_i, bg, tca, C), tau_i, tau_switch,
                        np.append(before, tau_switch), rtol, atol)
        segments = [(tca, s1)]
        if after.size:
            y_sw = _to_full(jnp.float64(tau_switch), jnp.asarray(s1.y[:, -1]), kj, bg, tca, full)
            segments.append((full, _integrate(full, kj, bg, y_sw, tau_switch, tau_end, after, rtol, atol)))
        segments[0] = (tca, _drop_last(s1))
    else:
        raise ValueError(f"unknown mode {mode!r}")

    values: dict[str, list] = {}
    nfev = 0
    for layout, sol in segments:
        nfev += sol.nfev
        _, _, obs = _compiled(layout)
        out = obs(jnp.asarray(sol.t), jnp.asarray(sol.y.T), kj, bg)
        for name, v in out.items():
            values.setdefault(name, []).append(np.asarray(v))
    values = {n: np.concatenate(v) for n, v in values.items()}
    tau = np.concatenate([sol.t for _, sol in segments])
    return ReferenceSolution(k=k, tau=tau, values=values, tau_switch=tau_switch, stats={"nfev": nfev})


def solve_segments(
    k: float,
    bg: bgm.Background,
    tau_eval: np.ndarray,
    segments: list[tuple[eq.Layout, float]],
    *,
    tau_i: float | None = None,
    C: float = 1.0,
    rtol: float = 1e-8,
    atol: float = 1e-11,
    extra: callable = None,
) -> ReferenceSolution:
    """Integrate through a sequence of (layout, tau_end) segments, switching with eq.switch_layout.

    `extra(tau, y, layout) -> dict` adds per-sample outputs (e.g. source functions); it is
    vmapped over the samples of each segment.
    """
    tau_eval = np.asarray(tau_eval, dtype=float)
    tau_i = ics.default_tau_i(k, bg) if tau_i is None else tau_i
    kj = jnp.float64(k)
    y = ics.initial_state(kj, tau_i, bg, segments[0][0], C)
    t0, nfev, values, taus = tau_i, 0, {}, []
    for j, (layout, t1) in enumerate(segments):
        last = j == len(segments) - 1
        sel = (tau_eval >= t0) & (tau_eval < t1)
        sol = _integrate(layout, kj, bg, y, t0, t1, np.append(tau_eval[sel], t1), rtol, atol)
        nfev += sol.nfev
        keep_end = last and np.any(tau_eval == t1)  # t1 itself requested: keep the appended sample
        n_out = sol.t.size if keep_end else sol.t.size - 1
        t_out, y_out = jnp.asarray(sol.t[:n_out]), jnp.asarray(sol.y[:, :n_out].T)
        if t_out.shape[0]:
            _, _, obs = _compiled(layout)
            out = dict(obs(t_out, y_out, kj, bg))
            if extra is not None:
                out.update(jax.vmap(lambda t, yy: extra(t, yy, layout))(t_out, y_out))
            for name, v in out.items():
                values.setdefault(name, []).append(np.asarray(v))
            taus.append(np.asarray(t_out))
        if not last:
            y = _switch(jnp.float64(t1), jnp.asarray(sol.y[:, -1]), kj, bg, layout, segments[j + 1][0])
        t0 = t1
    values = {n: np.concatenate(v) for n, v in values.items()}
    return ReferenceSolution(k=k, tau=np.concatenate(taus), values=values, tau_switch=None, stats={"nfev": nfev})


_switch = jax.jit(eq.switch_layout, static_argnums=(4, 5))


def _drop_last(sol):
    """Remove the extra sample at tau_switch that was only needed to hand over the state."""
    sol.t, sol.y = sol.t[:-1], sol.y[:, :-1]
    return sol
