"""CLASS validation harness: Newtonian-gauge perturbations from classy vs the reference solver.

Run as a script for the Phase 0 acceptance check:

    python -m pinn_cmb.validate_class [--config pinn_cmb/configs/phase0.yaml] [--plots DIR]

Error metric (spec "Open spec issues" #1): for each variable X and mode k,
    err_X = max_tau |X_ref - X_CLASS| / max_tau |X_CLASS|
over the comparison window, which is well defined through zero crossings. The
normalisation of the reference (its C) is fixed by matching phi at CLASS's first sample.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from . import background as bgm
from . import reference as ref

CONFIG = Path(__file__).parent / "configs" / "phase0.yaml"

# CLASS names -> reference names (identical by construction, listed for clarity).
VARIABLES = {
    "phi": "phi", "psi": "psi", "delta_cdm": "delta_c", "theta_cdm": "theta_c",
    "delta_b": "delta_b", "theta_b": "theta_b", "delta_g": "delta_g", "theta_g": "theta_g",
    "shear_g": "shear_g", "delta_ur": "delta_nu", "theta_ur": "theta_nu", "shear_ur": "shear_nu",
}


def load_config(path: Path | str = CONFIG) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_class(k_values, cosmology: dict, class_precision: dict) -> list[dict]:
    """Newtonian-gauge perturbation tables from CLASS, one dict per k (k in 1/Mpc)."""
    from classy import Class

    cosmo = Class()
    cosmo.set({
        **bgm.class_base_settings(cosmology),
        **class_precision,
        "output": "mPk",
        "gauge": "newtonian",
        "k_output_values": ", ".join(f"{k:.10g}" for k in k_values),
        "P_k_max_1/Mpc": max(1.0, 2 * max(k_values)),
    })
    cosmo.compute()
    try:
        return [{key: np.array(v) for key, v in s.items()} for s in cosmo.get_perturbations()["scalar"]]
    finally:
        cosmo.struct_cleanup()
        cosmo.empty()


def compare(k: float, bg: bgm.Background, cls: dict, tau_max: float, solver: dict) -> dict:
    """Reference solution on CLASS's tau samples (tau <= tau_max) and per-variable errors."""
    tau_c = cls["tau [Mpc]"]
    sel = tau_c <= tau_max
    tau = tau_c[sel]
    sol = ref.solve(k, bg, tau, **solver)
    scale = cls["phi"][sel][0] / sol.values["phi"][0]
    errors, curves = {}, {}
    for cname, rname in VARIABLES.items():
        x_cls = cls[cname][sel]
        x_ref = scale * sol.values[rname]
        errors[rname] = float(np.max(np.abs(x_ref - x_cls)) / np.max(np.abs(x_cls)))
        curves[rname] = (x_ref, x_cls)
    return {
        "k": k, "tau": tau, "errors": errors, "curves": curves, "tau_switch": sol.tau_switch,
        "constraint_00": float(np.max(np.abs(sol.values["constraint_00"]))),
    }


def solver_kwargs(cfg: dict) -> dict:
    s = dict(cfg["solver"])
    s["trigger"] = ref.TCATrigger(**s.pop("tca_trigger"))
    return s


def run(cfg: dict) -> list[dict]:
    bg = bgm.build_background(cfg["cosmology"])
    tables = run_class(cfg["k_values"], cfg["cosmology"], cfg["class_precision"])
    tau_max = cfg["tau_max_over_tau_rec"] * bg.tau_rec
    return [compare(k, bg, t, tau_max, solver_kwargs(cfg)) for k, t in zip(cfg["k_values"], tables)]


def plot(results: list[dict], outdir: Path, variables=("phi", "delta_c", "delta_g", "theta_b")):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    for r in results:
        fig, axes = plt.subplots(2, len(variables), figsize=(4 * len(variables), 6), sharex=True)
        for j, v in enumerate(variables):
            x_ref, x_cls = r["curves"][v]
            axes[0, j].plot(r["tau"], x_cls, "k-", lw=2, label="CLASS")
            axes[0, j].plot(r["tau"], x_ref, "C1--", label="reference")
            axes[0, j].set_title(f"{v}, k={r['k']}")
            axes[1, j].plot(r["tau"], (x_ref - x_cls) / np.max(np.abs(x_cls)), "C3")
            axes[1, j].set_xscale("log")
            axes[1, j].set_xlabel(r"$\tau$ [Mpc]")
            if r["tau_switch"]:
                for ax in axes[:, j]:
                    ax.axvline(r["tau_switch"], color="0.6", ls=":")
        axes[0, 0].legend()
        axes[1, 0].set_ylabel("diff / max|CLASS|")
        fig.tight_layout()
        fig.savefig(outdir / f"phase0_k{r['k']}.png", dpi=110)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default=CONFIG)
    p.add_argument("--plots", type=Path, default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    results = run(cfg)
    names = list(VARIABLES.values())
    print("k        " + " ".join(f"{n:>9s}" for n in names) + "   max|C00|")
    for r in results:
        print(f"{r['k']:<8g} " + " ".join(f"{r['errors'][n]:9.2e}" for n in names) + f"   {r['constraint_00']:.1e}")
    tol = cfg["acceptance"]["tolerance"]
    passed = all(r["errors"][v] < tol for r in results for v in cfg["acceptance"]["variables"])
    print(f"\nPhase 0 acceptance ({', '.join(cfg['acceptance']['variables'])} < {tol:g}): "
          f"{'PASS' if passed else 'FAIL'}")
    if args.plots:
        plot(results, args.plots)


if __name__ == "__main__":
    main()
