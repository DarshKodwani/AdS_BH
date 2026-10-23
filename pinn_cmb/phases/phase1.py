"""Phase 1: tight-coupled k-bundle PINN, fiducial cosmology.

    python -m pinn_cmb.phases.phase1 [--config ...] [--out ...] [--eval-only]

Trains the time-marching chain (checkpointed after every window, resumable), then
evaluates it on held-out k against the tca_only reference (same equations, Radau),
each mode up to min(tau_max, tau_switch(k)).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import yaml

from pinn_cmb import background as bgm
from pinn_cmb import equations as eq
from pinn_cmb import reference as ref
from pinn_cmb import train as tr
from pinn_cmb.model import Chain, Window

HERE = Path(__file__).resolve().parent.parent
EVAL_VARS = ("phi", "psi", "delta_c", "theta_c", "delta_b", "delta_g", "theta_g", "delta_nu", "theta_nu", "shear_nu")


def build_windows(cfg) -> list[Window]:
    layout = eq.Layout(tca=True, lmax_nu=cfg["lmax_nu"])
    lk = (float(np.log(cfg["k_range"][0])), float(np.log(cfg["k_range"][1])))
    e = cfg["window_edges"]
    net = cfg["network"]
    return [Window(layout, float(a), float(b), lk, ansatz=net["ansatz"], width=net["width"], depth=net["depth"])
            for a, b in zip(e[:-1], e[1:])]


def test_ks(cfg) -> np.ndarray:
    rng = np.random.default_rng(cfg["evaluation"]["seed"])
    lo, hi = np.log(cfg["k_range"])
    return np.sort(np.exp(rng.uniform(lo, hi, cfg["evaluation"]["n_test_k"])))


def evaluate(chain: Chain, bg, cfg, layout) -> dict:
    rows = []
    for k in test_ks(cfg):
        t_end = min(cfg["evaluation"]["tau_max"],
                    ref.tca_switch_time(k, bg, ref.TCATrigger(), cfg["tau_i"], cfg["evaluation"]["tau_max"]))
        taus = np.geomspace(cfg["tau_i"] * 1.0001, t_end, 400)
        sol = ref.solve(k, bg, taus, mode="tca_only", lmax_nu=layout.lmax_nu, tau_i=cfg["tau_i"])
        obs = chain.observables(taus, jnp.float64(k), jnp.zeros(0), bg)
        err = {v: float(np.max(np.abs(obs[v] - sol.values[v])) / np.max(np.abs(sol.values[v]))) for v in EVAL_VARS}
        err["max_constraint_00"] = float(np.max(np.abs(obs["constraint_00"])))
        rows.append({"k": float(k), "tau_end": float(t_end), "errors": err,
                     "curves": {v: (obs[v], sol.values[v]) for v in ("phi", "delta_c", "delta_g", "theta_g")},
                     "tau": taus})
    return rows


def report(rows, cfg, out: Path):
    tol = cfg["acceptance"]["tolerance"]
    worst = {v: max(r["errors"][v] for r in rows) for v in EVAL_VARS}
    rms = {v: float(np.sqrt(np.mean([r["errors"][v] ** 2 for r in rows]))) for v in EVAL_VARS}
    lines = ["k        " + " ".join(f"{v:>9s}" for v in EVAL_VARS)]
    for r in rows:
        lines.append(f"{r['k']:<8.4g} " + " ".join(f"{r['errors'][v]:9.1e}" for v in EVAL_VARS))
    lines.append("max      " + " ".join(f"{worst[v]:9.1e}" for v in EVAL_VARS))
    lines.append("rms      " + " ".join(f"{rms[v]:9.1e}" for v in EVAL_VARS))
    passed = all(worst[v] < tol for v in EVAL_VARS)
    lines.append(f"\nPhase 1 acceptance (max over {len(rows)} held-out k, all variables < {tol:g}): "
                 f"{'PASS' if passed else 'FAIL'}")
    text = "\n".join(lines)
    print(text)
    (out / "evaluation.txt").write_text(text + "\n")
    (out / "evaluation.json").write_text(json.dumps(
        {"passed": passed, "worst": worst, "rms": rms,
         "per_k": [{"k": r["k"], "tau_end": r["tau_end"], "errors": r["errors"]} for r in rows]}, indent=1))
    return passed


def plot(rows, out: Path, n=4):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pick = [rows[i] for i in np.linspace(0, len(rows) - 1, n).astype(int)]
    vs = ("phi", "delta_c", "delta_g", "theta_g")
    fig, ax = plt.subplots(2, len(vs), figsize=(4 * len(vs), 6.5), sharex=True)
    for i, r in enumerate(pick):
        for j, v in enumerate(vs):
            p, s = r["curves"][v]
            ax[0, j].plot(r["tau"], s, "k-", lw=1.5)
            ax[0, j].plot(r["tau"], p, f"C{i}--", label=f"k={r['k']:.3g}")
            ax[1, j].plot(r["tau"], np.abs(p - s) / np.max(np.abs(s)), f"C{i}")
            ax[0, j].set_title(v)
            ax[1, j].set_xscale("log")
            ax[1, j].set_yscale("log")
            ax[1, j].set_xlabel(r"$\tau$ [Mpc]")
    ax[0, 0].legend(fontsize=8)
    ax[1, 0].set_ylabel("|PINN - ref| / max|ref|")
    fig.tight_layout()
    fig.savefig(out / "phase1.png", dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=HERE / "configs" / "phase1.yaml")
    ap.add_argument("--out", type=Path, default=HERE / "results" / "phase1")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny step counts, for testing the pipeline")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    args.out.mkdir(parents=True, exist_ok=True)
    tcfg = tr.TrainConfig(**cfg["train"])
    if args.smoke:
        tcfg.adam_steps, tcfg.lbfgs_steps, tcfg.log_every = 200, 20, 100
        cfg["evaluation"]["n_test_k"] = 3

    bg = bgm.build_background()
    windows = build_windows(cfg)
    ckpt = args.out / "params.npz"
    chain = Chain([], [], cfg["tau_i"])
    if ckpt.exists():
        p = tr.load_params(ckpt, windows)
        chain = Chain(windows[: len(p)], p, cfg["tau_i"])
        print(f"resumed {len(p)} trained windows from {ckpt}")

    log_f = open(args.out / "train.log", "a")

    def log(s):
        print(s, flush=True)
        log_f.write(s + "\n")
        log_f.flush()

    def on_window(ch, hists):
        tr.save_params(ckpt, ch)
        hist_file = args.out / "history.json"
        prev = json.loads(hist_file.read_text()) if hist_file.exists() else []
        hist_file.write_text(json.dumps(prev + [hists[-1]], indent=1))

    if not args.eval_only:
        chain, _ = tr.train_chain(windows, lambda c: bg, tcfg, cfg["tau_i"], log=log, chain=chain, on_window=on_window)
    rows = evaluate(chain, bg, cfg, windows[0].layout)
    report(rows, cfg, args.out)
    plot(rows, args.out)


if __name__ == "__main__":
    main()
