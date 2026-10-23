"""Phase 2: tight-coupled PINN bundle over (k, omega_b, omega_c).

    python -m pinn_cmb.phases.phase2 [--config ...] [--out ...] [--eval-only] [--smoke]

The background is background.AnalyticBackground (differentiable in the parameters), so
the PINN and the tca_only reference solve identical equations. Each mode is trained up
to train_margin x its fiducial TCA switch time and evaluated up to its own switch time
(capped at tau_max) on a held-out Latin hypercube.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from pinn_cmb import background as bgm
from pinn_cmb import equations as eq
from pinn_cmb import reference as ref
from pinn_cmb import train as tr
from pinn_cmb.model import Chain, Window
from pinn_cmb.phases.phase1 import EVAL_VARS

HERE = Path(__file__).resolve().parent.parent


class Params:
    """Maps the conditioning vector in [-1, 1]^2 to (omega_b, omega_c) and backgrounds."""

    def __init__(self, cfg, bg_fid):
        self.ob, self.oc = cfg["omega_b"], cfg["omega_c"]
        self.abg = bgm.analytic_background(bg_fid, 0.5 * sum(self.ob), 0.5 * sum(self.oc))

    def omegas(self, cond):
        ob = self.ob[0] + 0.5 * (cond[0] + 1.0) * (self.ob[1] - self.ob[0])
        oc = self.oc[0] + 0.5 * (cond[1] + 1.0) * (self.oc[1] - self.oc[0])
        return ob, oc

    def cond(self, ob, oc):
        return jnp.array([2 * (ob - self.ob[0]) / (self.ob[1] - self.ob[0]) - 1,
                          2 * (oc - self.oc[0]) / (self.oc[1] - self.oc[0]) - 1])

    def bg(self, cond):
        return bgm.with_params(self.abg, *self.omegas(cond))


def build_windows(cfg) -> list[Window]:
    layout = eq.Layout(tca=True, lmax_nu=cfg["lmax_nu"])
    lk = (float(np.log(cfg["k_range"][0])), float(np.log(cfg["k_range"][1])))
    e, net = cfg["window_edges"], cfg["network"]
    return [Window(layout, float(a), float(b), lk, n_cond=2, ansatz=net["ansatz"], width=net["width"],
                   depth=net["depth"]) for a, b in zip(e[:-1], e[1:])]


def switch_time(k, bg, cfg):
    return min(cfg["tau_max"], ref.tca_switch_time(k, bg, ref.TCATrigger(), cfg["tau_i"], cfg["tau_max"]))


def make_tau_cap(P: Params, cfg):
    ks = np.geomspace(cfg["k_range"][0] * 0.9, cfg["k_range"][1] * 1.1, 60)
    ts = np.array([switch_time(k, P.abg, cfg) for k in ks]) * cfg["train_margin"]
    lk, lt = jnp.log(jnp.asarray(ks)), jnp.log(jnp.asarray(np.minimum(ts, cfg["tau_max"])))
    return lambda k, cond: jnp.exp(jnp.interp(jnp.log(k), lk, lt))


def test_set(cfg):
    rng = np.random.default_rng(cfg["evaluation"]["seed"])
    n = cfg["evaluation"]["n_test"]
    u = np.stack([(rng.permutation(n) + rng.uniform(size=n)) / n for _ in range(3)], axis=1)
    lo, hi = np.log(cfg["k_range"])
    return [(float(np.exp(lo + (hi - lo) * a)), cfg["omega_b"][0] + b * np.diff(cfg["omega_b"])[0],
             cfg["omega_c"][0] + c * np.diff(cfg["omega_c"])[0]) for a, b, c in u]


def evaluate(chain: Chain, P: Params, cfg, layout) -> list[dict]:
    rows = []
    for k, ob, oc in test_set(cfg):
        bg = bgm.with_params(P.abg, ob, oc)
        t_end = switch_time(k, bg, cfg)
        taus = np.geomspace(cfg["tau_i"] * 1.0001, t_end, 400)
        sol = ref.solve(k, bg, taus, mode="tca_only", lmax_nu=layout.lmax_nu, tau_i=cfg["tau_i"])
        obs = chain.observables(taus, jnp.float64(k), P.cond(ob, oc), bg)
        err = {v: float(np.max(np.abs(obs[v] - sol.values[v])) / np.max(np.abs(sol.values[v]))) for v in EVAL_VARS}
        err["max_constraint_00"] = float(np.max(np.abs(obs["constraint_00"])))
        rows.append({"k": k, "omega_b": float(ob), "omega_c": float(oc), "tau_end": float(t_end), "errors": err})
    return rows


def report(rows, cfg, out: Path):
    tol = cfg["acceptance"]["tolerance"]
    worst = {v: max(r["errors"][v] for r in rows) for v in EVAL_VARS}
    rms = {v: float(np.sqrt(np.mean([r["errors"][v] ** 2 for r in rows]))) for v in EVAL_VARS}
    lines = ["k        omega_b omega_c  " + " ".join(f"{v:>9s}" for v in EVAL_VARS)]
    for r in sorted(rows, key=lambda r: r["k"]):
        lines.append(f"{r['k']:<8.4g} {r['omega_b']:.4f}  {r['omega_c']:.4f}  "
                     + " ".join(f"{r['errors'][v]:9.1e}" for v in EVAL_VARS))
    lines.append("max                      " + " ".join(f"{worst[v]:9.1e}" for v in EVAL_VARS))
    lines.append("rms                      " + " ".join(f"{rms[v]:9.1e}" for v in EVAL_VARS))
    passed = all(worst[v] < tol for v in EVAL_VARS)
    lines.append(f"\nPhase 2 acceptance (max over {len(rows)} held-out LHS points, all variables < {tol:g}): "
                 f"{'PASS' if passed else 'FAIL'}")
    text = "\n".join(lines)
    print(text)
    (out / "evaluation.txt").write_text(text + "\n")
    (out / "evaluation.json").write_text(json.dumps({"passed": passed, "worst": worst, "rms": rms, "rows": rows},
                                                    indent=1))
    return passed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=HERE / "configs" / "phase2.yaml")
    ap.add_argument("--out", type=Path, default=HERE / "results" / "phase2")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    args.out.mkdir(parents=True, exist_ok=True)
    tcfg = tr.TrainConfig(**cfg["train"])
    if args.smoke:
        tcfg.adam_steps, tcfg.lbfgs_steps, tcfg.log_every = 200, 20, 100
        cfg["evaluation"]["n_test"] = 3

    P = Params(cfg, bgm.build_background())
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
        hf = args.out / "history.json"
        hf.write_text(json.dumps((json.loads(hf.read_text()) if hf.exists() else []) + [hists[-1]], indent=1))

    if not args.eval_only:
        chain, _ = tr.train_chain(windows, P.bg, tcfg, cfg["tau_i"], log=log, chain=chain, on_window=on_window, partial_dir=args.out,
                                  tau_cap=make_tau_cap(P, cfg))
    report(evaluate(chain, P, cfg, windows[0].layout), cfg, args.out)


if __name__ == "__main__":
    main()
