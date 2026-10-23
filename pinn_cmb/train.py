"""Training: bundle collocation, per-window Adam -> L-BFGS, and time-marching chains.

Loss for window j (data-free): mean over collocation points of
    |r|^2 / n_vars + w_c * C00^2,
with r the preconditioned log-time ODE residual of model.Window.residuals and C00
the normalised Einstein 00 residual, which pilot 1 showed is mandatory.

Collocation is a tensor product: n_k modes (stratified in ln k, and in the
conditioning parameters if any) times n_t stratified times in the window. The start
state of each mode comes from the frozen earlier windows of the chain.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .model import Chain, Window


@dataclass
class TrainConfig:
    adam_steps: int = 20000
    lr: float = 1e-3
    lbfgs_steps: int = 5000
    n_k: int = 64
    n_t: int = 64
    n_k_lbfgs: int = 96
    n_t_lbfgs: int = 96
    w_constraint: float = 1.0
    seed: int = 0
    log_every: int = 2000


def stratified(key, n):
    return (jnp.arange(n) + jax.random.uniform(key, (n,))) / n


def sample_modes(key, n_k, log_k_range, n_cond):
    """Stratified ln k and Latin-hypercube conditioning parameters in [-1, 1]."""
    lo, hi = log_k_range
    k1, k2 = jax.random.split(key)
    logk = lo + (hi - lo) * stratified(k1, n_k)
    if n_cond == 0:
        return jnp.exp(logk), jnp.zeros((n_k, 0))
    keys = jax.random.split(k2, n_cond)
    cond = jnp.stack([2.0 * jax.random.permutation(kk, stratified(kk, n_k)) - 1.0 for kk in keys], axis=1)
    return jnp.exp(logk), cond


def make_loss(w: Window, chain: Chain, j: int, bg_fn: Callable, w_c: float):
    """loss(params, k[n_k], cond[n_k, c], x[n_k, n_t], u_a[n_k, n], wt[n_k]) for window j.

    wt weights the modes (0 for modes whose active time range ends before this window).
    """

    def per_mode(params, k, cond, xs, u_a):
        bg = bg_fn(cond)
        r, c = jax.vmap(lambda x: w.residuals(params, x, k, cond, u_a, bg))(xs)
        return jnp.mean(r**2) + w_c * jnp.mean(c**2), jnp.mean(r**2), jnp.mean(c**2)

    def loss(params, k, cond, xs, u_a, wt):
        tot, rr, cc = jax.vmap(per_mode, (None, 0, 0, 0, 0))(params, k, cond, xs, u_a)
        norm = jnp.maximum(jnp.sum(wt), 1.0)
        return jnp.sum(wt * tot) / norm, (jnp.sum(wt * rr) / norm, jnp.sum(wt * cc) / norm)

    return loss


def collocation_times(w: Window, key, k, cond, n_t, tau_cap):
    """Stratified x in [0, x_end] per mode; x_end < 1 where the mode's active range ends early."""
    if tau_cap is None:
        x_end = jnp.ones_like(k)
    else:
        x_end = jnp.clip(w.x_of(jax.vmap(tau_cap)(k, cond)), 0.0, 1.0)
    xs = jax.vmap(lambda kk: stratified(kk, n_t))(jax.random.split(key, k.shape[0])) * x_end[:, None]
    return xs, (x_end > 0.02).astype(float)


def start_states(chain: Chain, j: int, k, cond, bg_fn):
    return jax.vmap(lambda kk, cc: chain.start_state(j, kk, cc, bg_fn(cc)))(k, cond)


def train_window(w: Window, chain: Chain, bg_fn: Callable, cfg: TrainConfig, log=print, tag="",
                 tau_cap: Callable | None = None) -> tuple[list, dict]:
    """Train window number len(chain.windows) given the frozen chain; returns (params, history).

    tau_cap(k, cond) -> tau, if given, ends each mode's training range there.
    """
    j = len(chain.windows)
    chain_with = Chain(chain.windows + [w], chain.params + [None], chain.tau_i)
    loss = make_loss(w, chain_with, j, bg_fn, cfg.w_constraint)
    key = jax.random.PRNGKey(cfg.seed + 1000 * j)
    params = w.init(key)

    sched = optax.cosine_decay_schedule(cfg.lr, cfg.adam_steps, alpha=1e-2)
    opt = optax.adam(sched)
    state = opt.init(params)
    u0_fn = jax.jit(lambda k, cond: start_states(chain_with, j, k, cond, bg_fn))

    @jax.jit
    def adam_step(params, state, key):
        k1, k2 = jax.random.split(key)
        k, cond = sample_modes(k1, cfg.n_k, w.log_k_range, w.n_cond)
        xs, wt = collocation_times(w, k2, k, cond, cfg.n_t, tau_cap)
        u_a = start_states(chain_with, j, k, cond, bg_fn)
        (l, aux), g = jax.value_and_grad(loss, has_aux=True)(params, k, cond, xs, u_a, wt)
        upd, state = opt.update(g, state)
        return optax.apply_updates(params, upd), state, l, aux

    hist = {"adam": [], "lbfgs": []}
    t0 = time.time()
    for i in range(cfg.adam_steps):
        key, sub = jax.random.split(key)
        params, state, l, (rr, cc) = adam_step(params, state, sub)
        if i % cfg.log_every == 0 or i == cfg.adam_steps - 1:
            hist["adam"].append((i, float(l), float(rr), float(cc)))
            log(f"{tag} adam {i:6d} loss {float(l):.3e} (ode {float(rr):.2e}, c00 {float(cc):.2e}) {time.time()-t0:.0f}s")

    # L-BFGS on a fixed, denser collocation set.
    kk, cond = sample_modes(jax.random.PRNGKey(cfg.seed + 7 + 1000 * j), cfg.n_k_lbfgs, w.log_k_range, w.n_cond)
    xs, wt = collocation_times(w, jax.random.PRNGKey(cfg.seed + 9), kk, cond, cfg.n_t_lbfgs, tau_cap)
    u_a = u0_fn(kk, cond)
    f = lambda p: loss(p, kk, cond, xs, u_a, wt)[0]
    lb = optax.lbfgs()
    lstate = lb.init(params)
    vg = optax.value_and_grad_from_state(f)

    @jax.jit
    def lbfgs_step(params, lstate):
        l, g = vg(params, state=lstate)
        upd, lstate = lb.update(g, lstate, params, value=l, grad=g, value_fn=f)
        return optax.apply_updates(params, upd), lstate, l

    for i in range(cfg.lbfgs_steps):
        params, lstate, l = lbfgs_step(params, lstate)
        if i % cfg.log_every == 0 or i == cfg.lbfgs_steps - 1:
            hist["lbfgs"].append((i, float(l)))
            log(f"{tag} lbfgs {i:6d} loss {float(l):.3e} {time.time()-t0:.0f}s")
    hist["seconds"] = time.time() - t0
    return params, hist


def train_chain(windows: list[Window], bg_fn: Callable, cfg: TrainConfig, tau_i: float,
                log=print, chain: Chain | None = None, on_window=None, tau_cap=None,
                cfgs: list[TrainConfig] | None = None) -> tuple[Chain, list]:
    """Time-march: train each window in turn, freezing the earlier ones. Resumes from `chain`.

    cfgs, if given, overrides cfg per window.
    """
    chain = chain or Chain([], [], tau_i)
    hists = []
    for j in range(len(chain.windows), len(windows)):
        w = windows[j]
        params, hist = train_window(w, chain, bg_fn, cfgs[j] if cfgs else cfg, log=log,
                                    tag=f"[w{j} {w.tau_a:.3g}-{w.tau_b:.3g}]", tau_cap=tau_cap)
        chain = Chain(chain.windows + [w], chain.params + [params], tau_i)
        hists.append(hist)
        if on_window:
            on_window(chain, hists)
    return chain, hists


def save_params(path, chain: Chain):
    leaves = {f"w{j}_l{i}_{n}": np.asarray(a) for j, p in enumerate(chain.params)
              for i, (W, b) in enumerate(p) for n, a in (("W", W), ("b", b))}
    np.savez(path, **leaves)


def load_params(path, windows: list[Window]) -> list:
    d = np.load(path)
    out = []
    for j, w in enumerate(windows):
        if f"w{j}_l0_W" not in d:
            break
        out.append([(jnp.asarray(d[f"w{j}_l{i}_W"]), jnp.asarray(d[f"w{j}_l{i}_b"])) for i in range(w.depth + 1)])
    return out
