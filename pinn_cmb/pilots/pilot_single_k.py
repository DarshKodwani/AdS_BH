"""Pilot: can a PINN solve the tight-coupled system at one k to ~1e-3?

Single mode k = 0.01 Mpc^-1, zeroth-order TCA layout with lmax_nu = 4, tau in
[tau_i, 200] Mpc. Training is data-free: only the residuals of equations.rhs (plus,
optionally, the Einstein 00 constraint) enter the loss. The reference solver is used
only to measure the error afterwards.

Network: x = log(tau/tau_i)/log(tau_max/tau_i) in [0, 1] -> MLP -> N(x) in R^11.
Hard initial conditions: u(x) = u_IC + (1 - exp(-x/eps)) * S * N(x), with physically
chosen scales S (delta, phi ~ psi_0; theta ~ k psi_0; F_nu,l ~ psi_0).
Residual, per variable, in log time and in units of S:
    r = (du/dx / L - tau f(tau, u)) / S,    L = log(tau_max / tau_i).

    python -m pinn_cmb.pilots.pilot_single_k --variant base|constraint|causal|all
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from pinn_cmb import background as bgm
from pinn_cmb import equations as eq
from pinn_cmb import initial_conditions as ics
from pinn_cmb import reference as ref

K = 0.01
TAU_MAX = 200.0
LAYOUT = eq.Layout(tca=True, lmax_nu=4)
VARIANTS = {
    "base": dict(w_constraint=0.0, causal_eps=0.0),
    "constraint": dict(w_constraint=1.0, causal_eps=0.0),
    "causal": dict(w_constraint=1.0, causal_eps=1.0),
}


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


class Problem:
    def __init__(self, bg, k=K, tau_max=TAU_MAX, eps=0.1):
        self.bg, self.k = bg, k
        self.tau_i = ics.default_tau_i(k, bg)
        self.L = float(np.log(tau_max / self.tau_i))
        self.u0 = ics.initial_state(k, self.tau_i, bg, LAYOUT)
        self.psi0 = 20.0 / (15.0 + 4.0 * float(ics.r_nu(bg)))
        # Power of the super-horizon growth of each variable: theta ∝ k^2 tau, F_l ∝ (k tau)^l.
        power = {n: 0 for n in LAYOUT.names}
        power.update({n: 1 for n in LAYOUT.names if n.startswith("theta")})
        power.update({f"F_nu{l}": l for l in range(2, LAYOUT.lmax_nu + 1)})
        self.power = jnp.array([power[n] for n in LAYOUT.names], dtype=float)
        self.k_factor = jnp.array([k if n.startswith("theta") else 1.0 for n in LAYOUT.names])
        self.eps = eps

    def tau(self, x):
        return self.tau_i * jnp.exp(self.L * x)

    def scale(self, tau):
        """Physical magnitude of each variable: psi0 (k)(g)^p with g = k tau / sqrt(1 + (k tau)^2)."""
        g = self.k * tau / jnp.sqrt(1.0 + (self.k * tau) ** 2)
        return self.psi0 * self.k_factor * g**self.power

    def u(self, params, x):
        """Solution at a scalar x, with the initial condition built in exactly."""
        return self.u0 + (1.0 - jnp.exp(-x / self.eps)) * self.scale(self.tau(x)) * mlp(params, x[None])

    def residuals(self, params, x):
        """(per-variable ODE residual, 00-constraint residual) at a scalar x."""
        u, du_dx = jax.jvp(lambda s: self.u(params, s), (x,), (jnp.ones_like(x),))
        tau = self.tau(x)
        f = eq.rhs(tau, u, self.k, self.bg, LAYOUT)
        r = (du_dx / self.L - tau * f) / self.scale(tau)
        c = eq.observables(tau, u, self.k, self.bg, LAYOUT)["constraint_00"]
        return r, c


def make_loss(prob: Problem, w_constraint: float, causal_eps: float, n_bins: int):
    def loss(params, x):
        r, c = jax.vmap(lambda s: prob.residuals(params, s))(x)
        per_point = jnp.mean(r**2, axis=1) + w_constraint * c**2
        if causal_eps <= 0:
            return jnp.mean(per_point), jnp.ones(n_bins)
        # Wang, Sankaran & Perdikaris 2022: x is sorted and stratified into equal bins.
        bin_loss = jnp.mean(per_point.reshape(n_bins, -1), axis=1)
        w = jax.lax.stop_gradient(jnp.exp(-causal_eps * jnp.concatenate([jnp.zeros(1), jnp.cumsum(bin_loss)[:-1]])))
        return jnp.mean(w * bin_loss), w

    return loss


def stratified(key, n_bins, per_bin):
    """Sorted points, per_bin uniform draws in each of n_bins equal bins of [0, 1]."""
    u = jax.random.uniform(key, (n_bins, per_bin))
    return ((jnp.arange(n_bins)[:, None] + u) / n_bins).ravel()


def train(prob: Problem, variant: str, *, seed=0, width=64, depth=4, adam_steps=20000,
          lbfgs_steps=3000, n_bins=32, per_bin=64, log=print):
    cfg = VARIANTS[variant]
    key = jax.random.PRNGKey(seed)
    params = init_mlp(key, [1] + [width] * depth + [LAYOUT.size])
    loss_fn = make_loss(prob, cfg["w_constraint"], cfg["causal_eps"], n_bins)

    sched = optax.cosine_decay_schedule(1e-3, adam_steps, alpha=1e-2)
    opt = optax.adam(sched)
    state = opt.init(params)

    @jax.jit
    def adam_step(params, state, key):
        x = stratified(key, n_bins, per_bin)
        (l, w), g = jax.value_and_grad(loss_fn, has_aux=True)(params, x)
        upd, state = opt.update(g, state)
        return optax.apply_updates(params, upd), state, l, w

    history = []
    t0 = time.time()
    for i in range(adam_steps):
        key, sub = jax.random.split(key)
        params, state, l, w = adam_step(params, state, sub)
        if i % 2000 == 0 or i == adam_steps - 1:
            history.append(("adam", i, float(l)))
            log(f"[{variant}] adam {i:6d} loss {float(l):.3e} min causal w {float(jnp.min(w)):.2f} ({time.time()-t0:.0f}s)")

    # L-BFGS polish on a fixed, denser collocation set; plain residual loss (no causal weights).
    x_fix = stratified(jax.random.PRNGKey(seed + 1), n_bins, 2 * per_bin)
    plain = make_loss(prob, cfg["w_constraint"], 0.0, n_bins)
    f = lambda p: plain(p, x_fix)[0]
    lb = optax.lbfgs()
    lstate = lb.init(params)
    vg = optax.value_and_grad_from_state(f)

    @jax.jit
    def lbfgs_step(params, lstate):
        l, g = vg(params, state=lstate)
        upd, lstate = lb.update(g, lstate, params, value=l, grad=g, value_fn=f)
        return optax.apply_updates(params, upd), lstate, l

    for i in range(lbfgs_steps):
        params, lstate, l = lbfgs_step(params, lstate)
        if i % 500 == 0 or i == lbfgs_steps - 1:
            history.append(("lbfgs", i, float(l)))
            log(f"[{variant}] lbfgs {i:6d} loss {float(l):.3e} ({time.time()-t0:.0f}s)")
    return params, history, time.time() - t0


def evaluate(prob: Problem, params, sol: ref.ReferenceSolution) -> dict:
    x = jnp.log(jnp.asarray(sol.tau) / prob.tau_i) / prob.L
    u = jax.vmap(lambda s: prob.u(params, s))(x)
    obs = jax.vmap(lambda t, y: eq.observables(t, y, prob.k, prob.bg, LAYOUT))(jnp.asarray(sol.tau), u)
    errs = {}
    for v in ("phi", "psi", "delta_c", "theta_c", "delta_b", "delta_g", "theta_g", "delta_nu", "theta_nu", "shear_nu"):
        p, r = np.asarray(obs[v]), sol.values[v]
        errs[v] = float(np.max(np.abs(p - r)) / np.max(np.abs(r)))
    errs["max_constraint_00"] = float(np.max(np.abs(np.asarray(obs["constraint_00"]))))
    return errs, {v: np.asarray(obs[v]) for v in obs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="all", choices=["all", *VARIANTS])
    ap.add_argument("--adam-steps", type=int, default=20000)
    ap.add_argument("--lbfgs-steps", type=int, default=3000)
    ap.add_argument("--out", type=Path, default=Path("pinn_cmb/docs/pilot1"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    bg = bgm.build_background()
    prob = Problem(bg)
    tau_eval = np.geomspace(prob.tau_i * 1.0001, TAU_MAX, 400)
    sol = ref.solve(K, bg, tau_eval, mode="tca_only", lmax_nu=LAYOUT.lmax_nu)
    print(f"tau_i={prob.tau_i:.3g} Mpc, L={prob.L:.2f}; reference max|C00|={np.max(np.abs(sol.values['constraint_00'])):.1e}")

    results = {}
    curves = {}
    for variant in (VARIANTS if args.variant == "all" else [args.variant]):
        params, hist, secs = train(prob, variant, adam_steps=args.adam_steps, lbfgs_steps=args.lbfgs_steps)
        errs, obs = evaluate(prob, params, sol)
        results[variant] = {"errors": errs, "train_seconds": secs, "final_loss": hist[-1][2], "history": hist}
        curves[variant] = obs
        print(f"[{variant}] " + " ".join(f"{k}={v:.1e}" for k, v in errs.items()) + f"  ({secs:.0f}s)")

    (args.out / "results.json").write_text(json.dumps(results, indent=1))
    plot(sol, curves, args.out / "pilot1.png")


def plot(sol, curves, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vs = ("phi", "delta_c", "delta_g", "theta_g")
    fig, ax = plt.subplots(2, len(vs), figsize=(4 * len(vs), 6), sharex=True)
    for j, v in enumerate(vs):
        r = sol.values[v]
        ax[0, j].plot(sol.tau, r, "k-", lw=2, label="reference")
        for i, (name, obs) in enumerate(curves.items()):
            ax[0, j].plot(sol.tau, obs[v], f"C{i}--", label=name)
            ax[1, j].plot(sol.tau, np.abs(obs[v] - r) / np.max(np.abs(r)), f"C{i}", label=name)
        ax[0, j].set_title(v)
        ax[1, j].set_yscale("log")
        ax[1, j].set_xscale("log")
        ax[1, j].set_xlabel(r"$\tau$ [Mpc]")
    ax[0, 0].legend()
    ax[1, 0].set_ylabel("|PINN - ref| / max|ref|")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


if __name__ == "__main__":
    main()
