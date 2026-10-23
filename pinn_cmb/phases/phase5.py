"""Phase 5: beyond-LCDM bundle and an inverse problem.

Model: a late-time effective Newton constant for perturbations,
    mu(a) = G_eff / G = 1 + mu0 Omega_Lambda(a) / Omega_Lambda,0,
multiplying G in the perturbed Einstein equations (0i, ij-traceless and, for
diagnostics, 00); the background is unchanged. This is a phenomenological modification
not available in standard CLASS. For mu0 != 0 the 00 equation is not propagated, so these
windows train on the evolution equations only.

The Phase 3/4 chain is reused unchanged up to tau_mu (where |mu - 1| < 1e-4 |mu0|); every
later window is retrained with mu0 as an extra network input (mu0 in [-MU_MAX, MU_MAX]).

    python -m pinn_cmb.phases.phase5 train
    python -m pinn_cmb.phases.phase5 validate      # PINN vs reference solver on held-out (k, mu0)
    python -m pinn_cmb.phases.phase5 inverse       # recover mu0 from mock z=0 spectra
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from pinn_cmb import background as bgm
from pinn_cmb import equations as eq
from pinn_cmb import los
from pinn_cmb import reference as ref
from pinn_cmb import train as tr
from pinn_cmb.model import Chain, ModeWindow
from pinn_cmb.phases import phase34 as p34

HERE = Path(__file__).resolve().parent.parent
MU_MAX = 0.5
TAU_MU = 2400.0  # Mpc, z ~ 30: |mu - 1| < 2e-5 |mu0| before this


def background_fn(bg):
    return lambda cond: bg._replace(mu0=MU_MAX * cond[0])


def build(cfg, bg, out: Path, dir34: Path = HERE / "results" / "phase34"):
    """(windows, frozen prefix chain) for Phase 5; resumes trained Phase 5 windows from out."""
    windows34, chain34 = p34.load_chain(cfg, bg, dir34)
    if len(chain34.windows) < len(windows34):
        raise RuntimeError("Phase 3/4 chain is not fully trained yet")
    n_keep = next(i for i, w in enumerate(windows34) if isinstance(w, ModeWindow) or w.tau_a >= TAU_MU)
    windows = windows34[:n_keep] + [dataclasses.replace(w, n_cond=1) for w in windows34[n_keep:]]
    chain = Chain(windows[:n_keep], chain34.params[:n_keep], cfg["tau_i"])
    ckpt = out / "params.npz"
    if ckpt.exists():
        d = np.load(ckpt)
        extra = []
        for j in range(n_keep, len(windows)):
            if f"w{j}_l0_W" not in d:
                break
            extra.append([(jnp.asarray(d[f"w{j}_l{i}_W"]), jnp.asarray(d[f"w{j}_l{i}_b"]))
                          for i in range(windows[j].depth + 1)])
        chain = Chain(windows[: n_keep + len(extra)], chain.params + extra, cfg["tau_i"])
    return windows, chain, n_keep


def cmd_train(cfg, bg, out, smoke, dir34):
    windows, chain, n_keep = build(cfg, bg, out, dir34)
    base = tr.TrainConfig(**{**cfg["train"], "w_constraint": 0.0})
    if smoke:
        base.adam_steps, base.lbfgs_steps, base.log_every = 100, 10, 50
    log_f = open(out / "train.log", "a")

    def log(s):
        print(s, flush=True)
        log_f.write(s + "\n")
        log_f.flush()

    def on_window(ch, hists):
        tr.save_params(out / "params.npz", ch)
        hf = out / "history.json"
        hf.write_text(json.dumps((json.loads(hf.read_text()) if hf.exists() else []) + [hists[-1]], indent=1))

    log(f"Phase 5: keeping {n_keep} Phase 3/4 windows, training {len(windows) - n_keep} (from {len(chain.windows)})")
    t_r = p34.rsa_time(bg)
    tr.train_chain(windows, background_fn(bg), base, cfg["tau_i"], log=log, chain=chain, on_window=on_window,
                   tau_cap=lambda k, c: t_r(k))


# --- observables at z = 0 ----------------------------------------------------------------------------


def z0_observables(y, k, bg, layout):
    """Comoving matter density Delta_m and Weyl potential (phi + psi)/2 today."""
    tau = bg.tau0 * (1 - 1e-6)
    b, s, psi, _, _ = eq.metric(tau, y, k, bg, layout)
    rho_m = b.rho_c + b.rho_b
    theta = s["theta_c"] if layout.tca else (b.rho_c * s["theta_c"] + b.rho_b * s["theta_b"]) / rho_m
    delta = (b.rho_c * s["delta_c"] + b.rho_b * s["delta_b"]) / rho_m
    return delta + 3.0 * b.conf_H * theta / k**2, 0.5 * (s["phi"] + psi)


def pinn_z0(chain: Chain, bg):
    """Jitted (mu0, k[n]) -> (Delta_m[n], Weyl[n]) for modes that end in an RSA window (k > 45/tau0)."""
    last = chain.windows[-1]

    def one(mu0, k):
        cond = jnp.array([mu0 / MU_MAX])
        b = background_fn(bg)(cond)
        starts = chain.all_starts(k, cond, b)
        y = last.u(chain.params[-1], jnp.float64(1.0), k, cond, starts[-1], b)
        return z0_observables(y, k, b, last.layout)

    return jax.jit(jax.vmap(one, (None, 0)))


def reference_z0(k, mu0, bg, cfg):
    L = cfg["lmax"]
    tca = eq.Layout(tca=True, lmax_nu=L["nu"])
    full = eq.Layout(tca=False, lmax_g=L["g"], lmax_pol=L["pol"], lmax_nu=L["nu"])
    rsa = eq.Layout(tca=False, rsa=True)
    t_end = bg.tau0 * (1 - 1e-6)
    t_r = max(45.0 / k, bg.tau_free_streaming)
    segs = [(tca, cfg["tca_edges"][-1]), (full, min(t_r, t_end))] + ([(rsa, t_end)] if t_r < t_end else [])
    b = bg._replace(mu0=mu0)
    s = ref.solve_segments(k, b, np.array([t_end]), segs, tau_i=cfg["tau_i"])
    layout = segs[-1][0]
    y = jnp.stack([jnp.asarray(s.values[n][-1]) for n in ("phi", "delta_c", "theta_c", "delta_b", "theta_b")]) \
        if layout.rsa else None
    if layout.rsa:
        return tuple(float(v) for v in z0_observables(y, jnp.float64(k), b, layout))
    raise ValueError("reference_z0 expects an RSA mode")


def primordial(k):
    p = los.PRIMORDIAL
    return p["A_s"] * (k / p["k_pivot"]) ** (p["n_s"] - 1.0)


def cmd_validate(cfg, bg, out, dir34):
    _, chain, _ = build(cfg, bg, out, dir34)
    f = pinn_z0(chain, bg)
    rng = np.random.default_rng(11)
    n = 16
    ks = np.exp(rng.uniform(np.log(5e-3), np.log(cfg["k_range"][1]), n))
    mus = rng.uniform(-MU_MAX, MU_MAX, n)
    rows = []
    for k, mu0 in zip(ks, mus):
        d_p, w_p = (float(v[0]) for v in f(jnp.float64(mu0), jnp.array([k])))
        d_r, w_r = reference_z0(float(k), float(mu0), bg, cfg)
        rows.append({"k": float(k), "mu0": float(mu0), "err_Delta_m": abs(d_p / d_r - 1), "err_Weyl": abs(w_p / w_r - 1)})
        print(f"k={k:.4f} mu0={mu0:+.3f}  Delta_m rel err {rows[-1]['err_Delta_m']:.1e}  Weyl rel err {rows[-1]['err_Weyl']:.1e}")
    worst = {q: max(r[q] for r in rows) for q in ("err_Delta_m", "err_Weyl")}
    print("worst", worst)
    (out / "validate.json").write_text(json.dumps({"rows": rows, "worst": worst}, indent=1))


def cmd_inverse(cfg, bg, out, dir34, mu_true=0.25, sigma_rel=0.01, n_grid=201, n_ref=11):
    """Mock P_m and P_Weyl at z=0 with 1% Gaussian errors; recover mu0 with the PINN bundle and the reference."""
    _, chain, _ = build(cfg, bg, out, dir34)
    f = pinn_z0(chain, bg)
    k_obs = np.geomspace(5e-3, 0.1, 20)
    rng = np.random.default_rng(2026)
    t0 = time.time()
    truth = [reference_z0(float(k), mu_true, bg, cfg) for k in k_obs]
    t_ref_one = (time.time() - t0)
    pm_true = primordial(k_obs) * np.array([t[0] for t in truth]) ** 2
    pw_true = primordial(k_obs) * np.array([t[1] for t in truth]) ** 2
    data = np.concatenate([pm_true * (1 + sigma_rel * rng.standard_normal(k_obs.size)),
                           pw_true * (1 + sigma_rel * rng.standard_normal(k_obs.size))])
    err = sigma_rel * np.concatenate([pm_true, pw_true])

    kj = jnp.asarray(k_obs)
    pr = jnp.asarray(primordial(k_obs))

    def model(mu0):
        d, w = f(mu0, kj)
        return jnp.concatenate([pr * d**2, pr * w**2])

    chi2 = jax.jit(lambda mu0: jnp.sum(((model(mu0) - data) / err) ** 2))
    model(jnp.float64(0.0)).block_until_ready()
    t1 = time.time()
    grid = np.linspace(-MU_MAX, MU_MAX, n_grid)
    c_pinn = np.array([float(chi2(jnp.float64(m))) for m in grid])
    t_pinn_grid = time.time() - t1

    # gradient-based fit through the PINN (Newton iterations with autodiff derivatives)
    g1, g2 = jax.grad(chi2), jax.grad(jax.grad(chi2))
    m = jnp.float64(0.0)
    for _ in range(8):
        m = m - g1(m) / g2(m)
    sigma_fisher = float(jnp.sqrt(2.0 / g2(m)))

    # reference likelihood on a coarse grid
    grid_ref = np.linspace(-MU_MAX, MU_MAX, n_ref)
    t2 = time.time()
    c_ref = []
    for mu0 in grid_ref:
        vals = [reference_z0(float(k), float(mu0), bg, cfg) for k in k_obs]
        mod = np.concatenate([primordial(k_obs) * np.array([v[0] for v in vals]) ** 2,
                              primordial(k_obs) * np.array([v[1] for v in vals]) ** 2])
        c_ref.append(float(np.sum(((mod - data) / err) ** 2)))
    t_ref_grid = time.time() - t2

    post = np.exp(-0.5 * (c_pinn - c_pinn.min()))
    post /= np.trapezoid(post, grid)
    mean = float(np.trapezoid(grid * post, grid))
    std = float(np.sqrt(np.trapezoid((grid - mean) ** 2 * post, grid)))
    res = {"mu_true": mu_true, "sigma_rel": sigma_rel, "k_obs": k_obs.tolist(),
           "pinn": {"mean": mean, "std": std, "newton_best_fit": float(m), "fisher_sigma": sigma_fisher,
                    "seconds_per_likelihood": t_pinn_grid / n_grid},
           "reference": {"grid": grid_ref.tolist(), "chi2": c_ref, "seconds_per_likelihood": t_ref_grid / len(grid_ref)},
           "pinn_grid": grid.tolist(), "pinn_chi2": c_pinn.tolist()}
    print(json.dumps({k: v for k, v in res.items() if k in ("mu_true", "pinn", "reference")}, indent=1))
    (out / "inverse.json").write_text(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "validate", "inverse"])
    ap.add_argument("--config", default=HERE / "configs" / "phase34.yaml")
    ap.add_argument("--out", type=Path, default=HERE / "results" / "phase5")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-ref", type=int, default=11, help="reference likelihood grid size (inverse)")
    ap.add_argument("--chain34", type=Path, default=HERE / "results" / "phase34")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    args.out.mkdir(parents=True, exist_ok=True)
    bg = bgm.build_background()
    {"train": lambda: cmd_train(cfg, bg, args.out, args.smoke, args.chain34),
     "validate": lambda: cmd_validate(cfg, bg, args.out, args.chain34),
     "inverse": lambda: cmd_inverse(cfg, bg, args.out, args.chain34, n_ref=args.n_ref)}[args.cmd]()


if __name__ == "__main__":
    main()
