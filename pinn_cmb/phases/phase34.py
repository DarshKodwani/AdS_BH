"""Phases 3 and 4: PINN chain through recombination to today, then C_l^TT.

    python -m pinn_cmb.phases.phase34 train      # time-march all windows (checkpointed, resumable)
    python -m pinn_cmb.phases.phase34 phase3     # perturbations vs CLASS to 2 tau_rec (spec Phase 3)
    python -m pinn_cmb.phases.phase34 phase4     # PINN source functions -> C_l^TT vs CLASS (spec Phase 4)

Phase 3 compares with CLASS run with the same hierarchy truncation, its default TCA,
and RSA/UFA as configured here (RSA at k tau > 45, no UFA).
"""

from __future__ import annotations

import argparse
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
from pinn_cmb import train as tr
from pinn_cmb.model import Chain, ModeWindow, Window
from pinn_cmb.phases import los_reference as lr
from pinn_cmb.phases.phase1 import EVAL_VARS

HERE = Path(__file__).resolve().parent.parent


def rsa_time(bg):
    t_fs = float(bg.tau_free_streaming)
    return lambda k: jnp.maximum(45.0 / k, t_fs)


def build_windows(cfg, bg) -> list[Window]:
    L = cfg["lmax"]
    tca = eq.Layout(tca=True, lmax_nu=L["nu"])
    full = eq.Layout(tca=False, lmax_g=L["g"], lmax_pol=L["pol"], lmax_nu=L["nu"])
    rsa = eq.Layout(tca=False, rsa=True)
    lk = (float(np.log(cfg["k_range"][0])), float(np.log(cfg["k_range"][1])))
    net = cfg["network"]
    kw = dict(ansatz=net["ansatz"], width=net["width"], depth=net["depth"])
    tau_end = bg.tau0 * (1 - 1e-6)
    te, fe = cfg["tca_edges"], [float(x) for x in cfg["full_edges"]] + [tau_end]
    ws = [Window(tca, float(a), float(b), lk, **kw) for a, b in zip(te[:-1], te[1:])]
    ws += [Window(full, a, b, lk, **kw) for a, b in zip(fe[:-1], fe[1:])]
    n = cfg["rsa_windows"]
    ws += [ModeWindow(rsa, 1.0, tau_end, lk, tau_start=rsa_time(bg), frac_a=i / n, frac_b=(i + 1) / n,
                      max_start=0.98 * tau_end, **kw) for i in range(n)]
    return ws


def load_chain(cfg, bg, out: Path):
    windows = build_windows(cfg, bg)
    ckpt = out / "params.npz"
    if ckpt.exists():
        p = tr.load_params(ckpt, windows)
        return windows, Chain(windows[: len(p)], p, cfg["tau_i"])
    return windows, Chain([], [], cfg["tau_i"])


def cmd_train(cfg, bg, out: Path, smoke: bool):
    windows, chain = load_chain(cfg, bg, out)
    base = tr.TrainConfig(**cfg["train"])
    if smoke:
        base.adam_steps, base.lbfgs_steps, base.log_every = 100, 10, 50
    cfgs = [tr.TrainConfig(**{**base.__dict__, **(cfg["train_rsa"] if isinstance(w, ModeWindow) else {})})
            for w in windows]
    log_f = open(out / "train.log", "a")

    def log(s):
        print(s, flush=True)
        log_f.write(s + "\n")
        log_f.flush()

    def on_window(ch, hists):
        tr.save_params(out / "params.npz", ch)
        hf = out / "history.json"
        hf.write_text(json.dumps((json.loads(hf.read_text()) if hf.exists() else []) + [hists[-1]], indent=1))

    if chain.windows:
        log(f"resumed {len(chain.windows)} trained windows")
    t_r = rsa_time(bg)
    tr.train_chain(windows, lambda c: bg, base, cfg["tau_i"], log=log, chain=chain, on_window=on_window,
                   tau_cap=lambda k, c: t_r(k), cfgs=cfgs)


def class_perturbations(k_values, lmax):
    from classy import Class

    c = Class()
    c.set({**bgm.class_base_settings(), "output": "mPk", "gauge": "newtonian", "P_k_max_1/Mpc": 1.0,
           "k_output_values": ", ".join(f"{k:.10g}" for k in k_values), "l_max_g": lmax["g"],
           "l_max_pol_g": lmax["pol"], "l_max_ur": lmax["nu"], "ur_fluid_approximation": 3,
           "tol_perturbations_integration": 1e-8})
    c.compute()
    pts = [{k: np.array(v) for k, v in s.items()} for s in c.get_perturbations()["scalar"]]
    c.struct_cleanup()
    c.empty()
    return pts


VARMAP = {"phi": "phi", "psi": "psi", "delta_c": "delta_cdm", "theta_c": "theta_cdm", "delta_b": "delta_b",
          "theta_b": "theta_b", "delta_g": "delta_g", "theta_g": "theta_g", "delta_nu": "delta_ur",
          "theta_nu": "theta_ur"}


def cmd_phase3(cfg, bg, out: Path):
    _, chain = load_chain(cfg, bg, out)
    rng = np.random.default_rng(cfg["evaluation"]["seed"])
    lo, hi = np.log(1e-3), np.log(cfg["k_range"][1])
    ks = np.sort(np.exp(rng.uniform(lo, hi, cfg["evaluation"]["n_test_k"])))
    t_max = cfg["evaluation"]["tau_max_phase3_over_rec"] * bg.tau_rec
    rows = []
    for k, cl in zip(ks, class_perturbations(ks, cfg["lmax"])):
        tau = cl["tau [Mpc]"]
        sel = (tau <= t_max) & (tau >= cfg["tau_i"] * 1.01)
        tau = tau[sel]
        obs = chain.observables(tau, jnp.float64(k), jnp.zeros(0), bg)
        scale = cl["phi"][sel][0] / obs["phi"][0]
        err = {v: float(np.max(np.abs(scale * obs[v] - cl[c][sel])) / np.max(np.abs(cl[c][sel])))
               for v, c in VARMAP.items()}
        rows.append({"k": float(k), "errors": err, "max_constraint_00": float(np.max(np.abs(obs["constraint_00"])))})
    tol = cfg["evaluation"]["phase3_tolerance"]
    worst = {v: max(r["errors"][v] for r in rows) for v in VARMAP}
    lines = ["k        " + " ".join(f"{v:>9s}" for v in VARMAP)]
    lines += [f"{r['k']:<8.4g} " + " ".join(f"{r['errors'][v]:9.1e}" for v in VARMAP) for r in rows]
    lines.append("max      " + " ".join(f"{worst[v]:9.1e}" for v in VARMAP))
    passed = all(x < tol for x in worst.values())
    lines.append(f"\nPhase 3 acceptance (PINN vs CLASS, tau <= {t_max:.0f} Mpc, all variables < {tol:g}): "
                 f"{'PASS' if passed else 'FAIL'}")
    print("\n".join(lines))
    (out / "phase3.txt").write_text("\n".join(lines) + "\n")
    (out / "phase3.json").write_text(json.dumps({"passed": passed, "worst": worst, "rows": rows}, indent=1))


def pinn_sources(chain: Chain, bg, k_src, tau_grid, C=0.5):
    """S_0, S_1, S_2 on (k_src, tau_grid) from the chain. The chain uses C = 1; rescale to C."""
    S = {n: np.zeros((len(k_src), len(tau_grid))) for n in ("S0", "S1", "S2")}
    src = {}
    for i, k in enumerate(k_src):
        kj = jnp.float64(k)
        for layout, sel, t, states in chain.states(tau_grid, kj, jnp.zeros(0), bg):
            if layout not in src:
                src[layout] = jax.jit(jax.vmap(lambda t, y, kk, L=layout: los.sources(t, y, kk, bg, L), (0, 0, None)))
            vals = src[layout](t, states, kj)
            for n in S:
                S[n][i, sel] = C * np.asarray(vals[n])[sel]
    return S


def cmd_phase4(cfg, bg, out: Path):
    _, chain = load_chain(cfg, bg, out)
    tau_grid = los.default_tau_grid(bg)
    kmin, kmax = cfg["k_range"]
    k_src = np.concatenate([np.geomspace(max(kmin, 1e-5), 2e-3, 30, endpoint=False), np.linspace(2e-3, kmax, 160)])
    t0 = time.time()
    S = pinn_sources(chain, bg, k_src, tau_grid)
    t_src = time.time() - t0
    lmax = cfg["evaluation"]["cl_lmax"]
    ells = los.default_ells(lmax)
    k_fine = los.default_k_fine(kmax, bg.tau0)
    t1 = time.time()
    cl = los.cl_tt(k_fine, los.transfer_tt(k_src, tau_grid, S, ells, bg.tau0, k_fine))
    t_los = time.time() - t1
    _, cl_c = lr.class_cl(2500)
    ratio = cl / cl_c[ells] - 1.0
    ref_file = HERE / "results" / "los_reference" / "class.json"
    ref_rel = json.loads(ref_file.read_text())["rel"] if ref_file.exists() else None
    tol = cfg["evaluation"]["cl_tolerance"]
    passed = bool(np.max(np.abs(ratio)) < tol)
    print(f"PINN sources {t_src:.0f}s, LOS {t_los:.0f}s; max|C_l^PINN/C_l^CLASS - 1| = {np.max(np.abs(ratio)):.2e} "
          f"(l<=30 {np.max(np.abs(ratio[ells <= 30])):.2e}, l>30 {np.max(np.abs(ratio[ells > 30])):.2e}); "
          f"Phase 4 acceptance (< {tol:g}): {'PASS' if passed else 'FAIL'}")
    np.savez(out / "cl_pinn.npz", ells=ells, cl=cl, cl_class=cl_c[ells], k_src=k_src, tau=tau_grid, **S)
    (out / "phase4.json").write_text(json.dumps({"passed": passed, "max_abs_rel": float(np.max(np.abs(ratio))),
                                                 "ells": ells.tolist(), "rel": ratio.tolist(), "ref_rel": ref_rel,
                                                 "source_seconds": t_src, "los_seconds": t_los}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "phase3", "phase4"])
    ap.add_argument("--config", default=HERE / "configs" / "phase34.yaml")
    ap.add_argument("--out", type=Path, default=HERE / "results" / "phase34")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    args.out.mkdir(parents=True, exist_ok=True)
    bg = bgm.build_background()
    {"train": lambda: cmd_train(cfg, bg, args.out, args.smoke),
     "phase3": lambda: cmd_phase3(cfg, bg, args.out),
     "phase4": lambda: cmd_phase4(cfg, bg, args.out)}[args.cmd]()


if __name__ == "__main__":
    main()
