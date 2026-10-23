"""Controlled test: TCA windows w0-w1 of the phase34 chain, reduced budget, k_power p vs reference."""
import sys, json, time, numpy as np, yaml, jax, jax.numpy as jnp
from pathlib import Path
from pinn_cmb import background as bgm, equations as eq, reference as ref, train as tr
from pinn_cmb.model import Chain
from pinn_cmb.phases import phase34 as p34
p = float(sys.argv[1])
cfg = yaml.safe_load(open("pinn_cmb/configs/phase34.yaml"))
bg = bgm.build_background()
W = p34.build_windows(cfg, bg)[:2]
tc = tr.TrainConfig(**{**cfg["train"], "adam_steps": 3000, "lbfgs_steps": 1000, "k_power": p, "log_every": 1000})
chain, _ = tr.train_chain(W, lambda c: bg, tc, cfg["tau_i"], log=lambda s: print(s, flush=True))
lay = eq.Layout(tca=True, lmax_nu=cfg["lmax"]["nu"])
taus = np.geomspace(1.1e-3, 9.9, 300)
res = {}
for k in [1e-4, 3e-3, 0.03, 0.06, 0.09, 0.12]:
    sol = ref.solve_segments(k, bg, taus, [(lay, 10.0)], tau_i=cfg["tau_i"], C=1.0)
    obs = chain.observables(taus, jnp.float64(k), jnp.zeros(0), bg)
    res[k] = {v: float(np.max(np.abs(np.asarray(obs[v]) - sol.values[v])) / np.max(np.abs(sol.values[v])))
              for v in ["phi", "delta_g", "theta_g", "delta_b", "delta_c", "theta_nu"]}
    print(f"p={p} k={k:<7g} " + " ".join(f"{v}={e:.1e}" for v, e in res[k].items()), flush=True)
