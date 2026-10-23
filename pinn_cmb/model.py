"""PINN building blocks: networks, output scaling, oscillation ansatz, time-marching chains.

A *window* is one network covering tau in [tau_a, tau_b] for a family of modes and
cosmologies (a bundle). Its inputs are the window-local log time
x = ln(tau/tau_a)/ln(tau_b/tau_a) in [0, 1], the normalised ln k, and optional
normalised conditioning parameters. Its output is the state vector of one
equations.Layout, written as

    u(x) = u_a + (1 - exp(-x/eps)) * S(tau, k) * B(x, ...),

so the window's initial state u_a is exact (hard IC). S is the physical size of each
variable (pilot 1) and B is either the raw network output or the oscillation ansatz
    B = N0 + D (N1 cos k r_s + N2 sin k r_s) + N3 cos k tau + N4 sin k tau,
with D = exp(-k^2/k_D^2) (spec Rev 2). A *chain* of windows is evaluated by
time-marching: window j starts from window j-1 at its end, mapped between layouts
where needed (TCA -> full hierarchy at the switch).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import jax
import jax.numpy as jnp
import numpy as np

from . import background as bgm
from . import equations as eq
from . import initial_conditions as ics

# Variable kinds for scaling and preconditioning.
_DENSITY_M, _DENSITY_R, _METRIC, _THETA, _F_NU, _F_G, _G_G = range(7)


def init_mlp(key, sizes):
    params = []
    for n_in, n_out in zip(sizes[:-1], sizes[1:]):
        key, sub = jax.random.split(key)
        params.append((jax.random.normal(sub, (n_in, n_out)) * jnp.sqrt(1.0 / n_in), jnp.zeros(n_out)))
    return params


def mlp(params, x):
    h = x
    for W, b in params[:-1]:
        h = jnp.tanh(h @ W + b)
    W, b = params[-1]
    return h @ W + b


def _kinds(layout: eq.Layout):
    kind, ell = [], []
    for n in layout.names:
        if n in ("delta_c", "delta_b"):
            kind.append(_DENSITY_M), ell.append(0)
        elif n in ("delta_g", "delta_nu"):
            kind.append(_DENSITY_R), ell.append(0)
        elif n == "phi":
            kind.append(_METRIC), ell.append(0)
        elif n.startswith("theta"):
            kind.append(_THETA), ell.append(1)
        elif n.startswith("F_nu"):
            kind.append(_F_NU), ell.append(int(n[4:]))
        elif n.startswith("F_g"):
            kind.append(_F_G), ell.append(int(n[3:]))
        elif n.startswith("G_g"):
            kind.append(_G_G), ell.append(int(n[3:]))
        else:
            raise ValueError(n)
    return np.array(kind), np.array(ell, dtype=float)


def psi0(bg) -> jax.Array:
    """|psi| of the adiabatic mode with C = 1 (M&B eq. 98)."""
    return 20.0 / (15.0 + 4.0 * ics.r_nu(bg))


@dataclass(frozen=True)
class Window:
    layout: eq.Layout
    tau_a: float
    tau_b: float
    log_k_range: tuple[float, float]
    n_cond: int = 0
    ansatz: bool = True
    width: int = 64
    depth: int = 4
    eps: float = 0.1

    @cached_property
    def L(self) -> float:
        return float(np.log(self.tau_b / self.tau_a))

    @cached_property
    def _kind_ell(self):
        return _kinds(self.layout)

    @property
    def n_heads(self) -> int:
        return 5 if self.ansatz else 1

    def __hash__(self):
        return hash((self.layout, self.tau_a, self.tau_b, self.log_k_range, self.n_cond, self.ansatz,
                     self.width, self.depth, self.eps))

    def init(self, key):
        return init_mlp(key, [2 + self.n_cond] + [self.width] * self.depth + [self.n_heads * self.layout.size])

    def tau(self, x, k=None):
        return self.tau_a * jnp.exp(self.L * x)

    def x_of(self, tau, k=None):
        return jnp.log(tau / self.tau_a) / self.L

    def L_of(self, k):
        return self.L

    def active(self, k):
        return jnp.ones_like(k)

    def features(self, x, k, cond):
        lo, hi = self.log_k_range
        xi = 2.0 * (jnp.log(k) - lo) / (hi - lo) - 1.0
        return jnp.concatenate([jnp.stack([2.0 * x - 1.0, xi]), jnp.atleast_1d(cond)[: self.n_cond]])

    def scale(self, tau, k, bg) -> jax.Array:
        """Physical size of each variable (time-dependent; see pilot 1 and 2 results)."""
        kind, ell = self._kind_ell
        b = bgm.at(bg, tau)
        kt = k * tau
        g = kt / jnp.sqrt(1.0 + kt**2)  # super-horizon growth factor
        a_eq = bg.rho0[2] / (bg.rho0[0] + bg.rho0[1]) * (1.0 + bg.rho0[3] / bg.rho0[2])
        growth = 1.0 + kt**2 / 6.0 * b.a / (b.a + a_eq)  # matter: delta ~ (k tau)^2 phi / 6 in MD
        q = (k + b.conf_H) / (k + b.conf_H + b.kappa_dot)  # tight-coupling suppression of photon l >= 2
        s = jnp.select(
            [kind == _DENSITY_M, kind == _THETA, kind == _F_NU, kind == _F_G, kind == _G_G],
            [growth, k * g * jnp.sqrt(growth), g**ell, g**ell * q ** (ell - 1),
             g ** jnp.maximum(ell, 2.0) * q ** jnp.maximum(ell, 1.0)],
            default=jnp.ones_like(ell),
        )
        return psi0(bg) * s

    def precond(self, tau, k, bg) -> jax.Array:
        """Row scaling 1 + tau * |stiff Thomson coefficient| so stiff equations become O(1)."""
        if self.layout.tca:
            return jnp.ones(self.layout.size)
        b = bgm.at(bg, tau)
        c = np.zeros(self.layout.size)
        for n, i in self.layout.index.items():
            if n in ("theta_g",) or n.startswith(("F_g", "G_g")):
                c[i] = 1.0
        c = jnp.asarray(c).at[self.layout.index["theta_b"]].set(1.0 / b.R)
        return 1.0 + tau * b.kappa_dot * c

    def bracket(self, params, x, k, cond, bg):
        out = mlp(params, self.features(x, k, cond))
        if not self.ansatz:
            return out
        n = out.reshape(self.n_heads, self.layout.size)
        tau = self.tau(x, k)
        ph_s = k * bgm.sound_horizon(bg, tau)
        damp = jnp.exp(-((k / bgm.damping_wavenumber(bg, tau)) ** 2))
        ph_f = k * tau
        return (n[0] + damp * (n[1] * jnp.cos(ph_s) + n[2] * jnp.sin(ph_s))
                + n[3] * jnp.cos(ph_f) + n[4] * jnp.sin(ph_f))

    def u(self, params, x, k, cond, u_a, bg):
        tau = self.tau(x, k)
        return u_a + (1.0 - jnp.exp(-x / self.eps)) * self.scale(tau, k, bg) * self.bracket(params, x, k, cond, bg)

    def residuals(self, params, x, k, cond, u_a, bg):
        """(preconditioned per-variable ODE residual in log time, normalised 00 residual)."""
        u, du_dx = jax.jvp(lambda s: self.u(params, s, k, cond, u_a, bg), (x,), (jnp.ones_like(x),))
        tau = self.tau(x, k)
        f = eq.rhs(tau, u, k, bg, self.layout)
        r = (du_dx / self.L_of(k) - tau * f) / (self.scale(tau, k, bg) * self.precond(tau, k, bg))
        c = eq.observables(tau, u, k, bg, self.layout)["constraint_00"]
        return r, c


@dataclass(frozen=True, eq=False)
class ModeWindow(Window):
    """Window whose start depends on the mode: tau in [tau_start(k), tau_b], split by fractions.

    Used after the RSA switch, which happens at the k-dependent tau_R(k) = max(45/k, tau_fs)
    (CLASS's trigger). The window covers the fraction [frac_a, frac_b] of
    ln(tau_b / tau_start(k)); a mode is inactive if tau_start(k) >= max_start.
    tau_a is unused (kept for the dataclass); tau_start must be a jnp function of k.
    """

    tau_start: object = None
    frac_a: float = 0.0
    frac_b: float = 1.0
    max_start: float = np.inf

    def __hash__(self):
        return hash((super().__hash__(), self.tau_start, self.frac_a, self.frac_b, self.max_start))

    def _ends(self, k):
        ts = jnp.minimum(self.tau_start(k), 0.999 * self.tau_b)
        lt = jnp.log(self.tau_b / ts)
        return jnp.log(ts) + self.frac_a * lt, (self.frac_b - self.frac_a) * lt

    def tau(self, x, k=None):
        la, L = self._ends(k)
        return jnp.exp(la + L * x)

    def x_of(self, tau, k=None):
        la, L = self._ends(k)
        return (jnp.log(tau) - la) / L

    def L_of(self, k):
        return self._ends(k)[1]

    def active(self, k):
        return (self.tau_start(k) < self.max_start).astype(float)


@dataclass
class Chain:
    """Trained windows in time order. bg_fn(cond) gives the background for a condition vector."""

    windows: list[Window] = field(default_factory=list)
    params: list = field(default_factory=list)
    tau_i: float = 1e-3

    def _fixed_states(self, upto, k, cond, bg):
        """Start states of the fixed (non-mode) windows 0..upto-1 by time-marching."""
        u = ics.initial_state(k, self.tau_i, bg, self.windows[0].layout)
        states = []
        for i in range(upto):
            w = self.windows[i]
            states.append(u)
            if i + 1 < len(self.windows) and not isinstance(self.windows[i + 1], ModeWindow):
                u = w.u(self.params[i], jnp.float64(1.0), k, cond, u, bg)
                u = eq.switch_layout(w.tau_b, u, k, bg, w.layout, self.windows[i + 1].layout)
        return states, u

    def _mode_start(self, j, k, cond, bg, fixed_states):
        """Start of the first ModeWindow: the fixed chain at tau_start(k), mapped to its layout."""
        w = self.windows[j]
        fixed = [i for i in range(j) if not isinstance(self.windows[i], ModeWindow)]
        ts = jnp.minimum(w.tau_start(k), 0.999 * w.tau_b)
        src = self.windows[fixed[-1]].layout
        cands, conds = [], []
        for i in fixed:
            wi = self.windows[i]
            if wi.layout != src:
                continue
            x = jnp.clip(wi.x_of(ts), 0.0, 1.0)
            cands.append(wi.u(self.params[i], x, k, cond, fixed_states[i], bg))
            conds.append((ts >= wi.tau_a) & (ts <= wi.tau_b))
        u = jnp.select(conds, cands, default=cands[-1])
        return eq.switch_layout(ts, u, k, bg, src, w.layout)

    def all_starts(self, k, cond, bg) -> list:
        """Start states of every window in one time-marching pass."""
        n_fixed = sum(not isinstance(w, ModeWindow) for w in self.windows)
        starts, _ = self._fixed_states(n_fixed, k, cond, bg)
        for j in range(n_fixed, len(self.windows)):
            w = self.windows[j]
            if w.frac_a == 0:
                starts.append(self._mode_start(j, k, cond, bg, starts))
            else:
                prev = self.windows[j - 1]
                starts.append(prev.u(self.params[j - 1], jnp.float64(1.0), k, cond, starts[j - 1], bg))
        return starts

    def start_state(self, j, k, cond, bg):
        """Exact start state of window j: the adiabatic IC, or the previous window's end state.

        A ModeWindow with frac_a = 0 starts from the fixed chain evaluated at tau_start(k)
        (selected among the windows of the same layout family), mapped to its layout.
        """
        w = self.windows[j]
        if not isinstance(w, ModeWindow):
            return self._fixed_states(j, k, cond, bg)[1]
        if w.frac_a > 0:
            prev = self.windows[j - 1]
            return prev.u(self.params[j - 1], jnp.float64(1.0), k, cond, self.start_state(j - 1, k, cond, bg), bg)
        fixed = [i for i in range(j) if not isinstance(self.windows[i], ModeWindow)]
        states, _ = self._fixed_states(len(fixed), k, cond, bg)
        return self._mode_start(j, k, cond, bg, states)

    def state(self, tau: float, k, cond, bg):
        """(window, state vector) at a concrete scalar tau for a concrete k."""
        for j, (a, b) in enumerate(self.ranges(k)):
            if a <= tau <= b * (1 + 1e-12):
                w = self.windows[j]
                return w, w.u(self.params[j], w.x_of(tau, k), k, cond, self.start_state(j, k, cond, bg), bg)
        raise ValueError(f"tau={tau} outside the chain")

    def ranges(self, k) -> list[tuple[float, float]]:
        """Concrete tau range of each window for a concrete k (empty ranges have a >= b)."""
        mode = [w for w in self.windows if isinstance(w, ModeWindow)]
        t_switch = float(jnp.minimum(mode[0].tau_start(k), mode[0].tau_b)) if mode else np.inf
        out = []
        for w in self.windows:
            if isinstance(w, ModeWindow):
                if float(w.active(k)) == 0.0:
                    out.append((np.inf, -np.inf))
                else:
                    out.append((float(w.tau(0.0, k)), float(w.tau(1.0, k))))
            elif mode and w.layout == self.windows[[isinstance(x, ModeWindow) for x in self.windows].index(True) - 1].layout:
                out.append((w.tau_a, min(w.tau_b, t_switch)))
            else:
                out.append((w.tau_a, w.tau_b))
        return out

    def states(self, taus: np.ndarray, k, cond, bg) -> list[tuple[eq.Layout, np.ndarray, jax.Array, jax.Array]]:
        """(layout, selection mask, clipped taus, states) per window on an increasing tau grid.

        Every window is evaluated on the full grid clipped to its range, so array shapes do not
        depend on k and the JIT-compiled functions are reused; callers keep the masked samples.
        """
        res = []
        rng = self.ranges(k)
        last_b = max(b for a, b in rng)
        starts = _jit_starts(tuple(self.windows), self.tau_i)(self.params, k, cond, bg)
        for j, w in enumerate(self.windows):
            a, b = rng[j]
            sel = (taus >= a) & ((taus < b) | ((b == last_b) & (taus <= b)))
            if not sel.any():
                continue
            t = jnp.asarray(np.clip(taus, a, b))
            res.append((w.layout, sel, t, _jit_eval(w)(self.params[j], starts[j], k, cond, t, bg)))
        return res

    def observables(self, taus: np.ndarray, k, cond, bg) -> dict:
        """Observables along an increasing tau grid."""
        out: dict[str, list] = {}
        for layout, sel, t, uu in self.states(taus, k, cond, bg):
            o = _jit_obs(layout)(t, uu, k, bg)
            for n, v in o.items():
                out.setdefault(n, []).append(np.asarray(v)[sel])
        return {n: np.concatenate(v) for n, v in out.items()}


_CACHE: dict = {}


def _jit_starts(windows: tuple, tau_i: float):
    key = ("starts", windows, tau_i)
    if key not in _CACHE:
        _CACHE[key] = jax.jit(lambda params, k, cond, bg: Chain(list(windows), params, tau_i).all_starts(k, cond, bg))
    return _CACHE[key]


def _jit_eval(w: Window):
    key = ("eval", w)
    if key not in _CACHE:
        _CACHE[key] = jax.jit(lambda p, u_a, k, cond, t, bg: jax.vmap(lambda s: w.u(p, w.x_of(s, k), k, cond, u_a, bg))(t))
    return _CACHE[key]


def _jit_obs(layout: eq.Layout):
    key = ("obs", layout)
    if key not in _CACHE:
        _CACHE[key] = jax.jit(jax.vmap(lambda t, y, k, bg: eq.observables(t, y, k, bg, layout), (0, 0, None, None)))
    return _CACHE[key]
