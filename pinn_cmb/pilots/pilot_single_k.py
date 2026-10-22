"""Single-k PINN pilots for the tight-coupled system (zeroth-order TCA, lmax_nu = 4).

Training is data-free: only the residuals of equations.rhs and the Einstein 00
constraint enter the loss. The reference solver is used only to measure the error.

Network: x = log(tau/tau_i)/log(tau_max/tau_i) in [0, 1] -> MLP.
Hard initial conditions: u(x) = u_IC + (1 - exp(-x/eps)) * S(tau) * N(x), where S(tau)
is each variable's physical size (theta ∝ k^2 tau and F_nu,l ∝ (k tau)^l outside the
horizon, O(1) inside it).

With --ansatz osc, the bracket N(x) becomes
    N0 + D (N1 cos k r_s + N2 sin k r_s) + N3 cos k tau + N4 sin k tau,
D = exp(-k^2 / k_D^2), i.e. smooth envelopes times the acoustic phase of the
photon-baryon fluid and the free-streaming phase of the neutrinos.
Residual, per variable, in log time and in units of S:
    r = (du/dx / L - tau f(tau, u)) / S(tau),    L = log(tau_max / tau_i).

Pilot 1:  python -m pinn_cmb.pilots.pilot_single_k --variant all
Pilot 2:  python -m pinn_cmb.pilots.pilot_single_k --k 0.05 --tau-max switch --ansatz osc ...
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
    def __init__(self, bg, k, tau_max, ansatz="none", eps=0.1):
        self.bg, self.k, self.tau_max, self.ansatz = bg, k, tau_max, ansatz
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
        self.n_heads = 5 if ansatz == "osc" else 1

    @property
    def n_out(self):
        return self.n_heads * LAYOUT.size

    def tau(self, x):
        return self.tau_i * jnp.exp(self.L * x)

    def scale(self, tau):
        """Physical magnitude of each variable: psi0 (k)(g)^p with g = k tau / sqrt(1 + (k tau)^2)."""
        g = self.k * tau / jnp.sqrt(1.0 + (self.k * tau) ** 2)
        return self.psi0 * self.k_factor * g**self.power

    def bracket(self, params, x):
        out = mlp(params, x[None])
        if self.ansatz != "osc":
            return out
        n = out.reshape(self.n_heads, LAYOUT.size)
        tau = self.tau(x)
        ph_s = self.k * bgm.sound_horizon(self.bg, tau)
        damp = jnp.exp(-((self.k / bgm.damping_wavenumber(self.bg, tau)) ** 2))
        ph_f = self.k * tau
        return (n[0] + damp * (n[1] * jnp.cos(ph_s) + n[2] * jnp.sin(ph_s))
                + n[3] * jnp.cos(ph_f) + n[4] * jnp.sin(ph_f))

    def u(self, params, x):
        """Solution at a scalar x, with the initial condition built in exactly."""
        return self.u0 + (1.0 - jnp.exp(-x / self.eps)) * self.scale(self.tau(x)) * self.bracket(params, x)

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
          lbfgs_steps=3000, n_bins=32, per_bin=64, log=print, tag=None):
    cfg = VARIANTS[variant]
    tag = tag or variant
    key = jax.random.PRNGKey(seed)
    params = init_mlp(key, [1] + [width] * depth + [prob.n_out])
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
            log(f"[{tag}] adam {i:6d} loss {float(l):.3e} min causal w {float(jnp.min(w)):.2f} ({time.time()-t0:.0f}s)")

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
        if i % 1000 == 0 or i == lbfgs_steps - 1:
            history.append(("lbfgs", i, float(l)))
            log(f"[{tag}] lbfgs {i:6d} loss {float(l):.3e} ({time.time()-t0:.0f}s)")
    return params, history, time.time() - t0


ERROR_VARS = ("phi", "psi", "delta_c", "theta_c", "delta_b", "delta_g", "theta_g", "delta_nu", "theta_nu", "shear_nu")


def evaluate(prob: Problem, params, sol: ref.ReferenceSolution) -> dict:
    x = jnp.log(jnp.asarray(sol.tau) / prob.tau_i) / prob.L
    u = jax.vmap(lambda s: prob.u(params, s))(x)
    obs = jax.vmap(lambda t, y: eq.observables(t, y, prob.k, prob.bg, LAYOUT))(jnp.asarray(sol.tau), u)
    errs = {}
    for v in ERROR_VARS:
        p, r = np.asarray(obs[v]), sol.values[v]
        errs[v] = float(np.max(np.abs(p - r)) / np.max(np.abs(r)))
    errs["max_constraint_00"] = float(np.max(np.abs(np.asarray(obs["constraint_00"]))))
    return errs, {v: np.asarray(obs[v]) for v in obs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=float, default=0.01)
    ap.add_argument("--tau-max", default="200", help="Mpc, or 'switch' for the CLASS-default TCA switch time")
    ap.add_argument("--ansatz", default="none", choices=["none", "osc"])
    ap.add_argument("--variant", default="all", choices=["all", *VARIANTS])
    ap.add_argument("--causal-eps", type=float, default=None, help="override the variant's causal epsilon")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--adam-steps", type=int, default=20000)
    ap.add_argument("--lbfgs-steps", type=int, default=3000)
    ap.add_argument("--out", type=Path, default=Path("pinn_cmb/docs/pilot1"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    bg = bgm.build_background()
    tau_i = ics.default_tau_i(args.k, bg)
    if args.tau_max == "switch":
        tau_max = ref.tca_switch_time(args.k, bg, ref.TCATrigger(), tau_i, 2 * bg.tau_rec)
    else:
        tau_max = float(args.tau_max)
    prob = Problem(bg, args.k, tau_max, args.ansatz)
    tau_eval = np.geomspace(prob.tau_i * 1.0001, tau_max, 600)
    sol = ref.solve(args.k, bg, tau_eval, mode="tca_only", lmax_nu=LAYOUT.lmax_nu)
    n_osc = float(args.k * bgm.sound_horizon(bg, tau_max)) / (2 * np.pi)
    print(f"k={args.k} tau_i={prob.tau_i:.3g} tau_max={tau_max:.1f} Mpc, L={prob.L:.2f}, "
          f"acoustic oscillations={n_osc:.2f}, ansatz={args.ansatz}")

    if args.causal_eps is not None:
        for v in VARIANTS.values():
            v["causal_eps"] = args.causal_eps
    results, curves = {}, {}
    for variant in (VARIANTS if args.variant == "all" else [args.variant]):
        tag = f"k={args.k},{args.ansatz},{variant}"
        params, hist, secs = train(prob, variant, seed=args.seed, adam_steps=args.adam_steps,
                                   lbfgs_steps=args.lbfgs_steps, tag=tag)
        errs, obs = evaluate(prob, params, sol)
        results[variant] = {"k": args.k, "tau_max": tau_max, "ansatz": args.ansatz, "seed": args.seed,
                            "causal_eps": VARIANTS[variant]["causal_eps"],
                            "errors": errs, "train_seconds": secs, "final_loss": hist[-1][2], "history": hist}
        curves[variant] = obs
        print(f"[{tag}] RESULT " + " ".join(f"{k}={v:.1e}" for k, v in errs.items()) + f"  ({secs:.0f}s)")

    (args.out / "results.json").write_text(json.dumps(results, indent=1))
    plot(sol, curves, args.out / "pilot.png")


def plot(sol, curves, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vs = ("phi", "delta_c", "delta_g", "theta_g", "delta_nu")
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
