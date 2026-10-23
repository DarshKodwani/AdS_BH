"""Controlled test on window w3 (30-60 Mpc, first full-hierarchy window) on top of the trained w0-w2."""
import sys, numpy as np, yaml, jax.numpy as jnp
from pathlib import Path
from pinn_cmb import background as bgm, equations as eq, reference as ref, train as tr
from pinn_cmb.model import Chain
from pinn_cmb.phases import phase34 as p34
p = float(sys.argv[1]); steps = int(sys.argv[2]); slow = sys.argv[3] == "slow"
cfg = yaml.safe_load(open("pinn_cmb/configs/phase34.yaml"))
bg = bgm.build_background()
W, full_chain = p34.load_chain(cfg, bg, Path("pinn_cmb/results/phase34"))
import dataclasses; W = W[:3] + [dataclasses.replace(W[3], slow_row=slow)] + W[4:]
chain = Chain(full_chain.windows[:3], full_chain.params[:3], cfg["tau_i"])
tc = tr.TrainConfig(**{**cfg["train"], "adam_steps": steps, "lbfgs_steps": 1000, "k_power": p, "log_every": 1000})
chain, _ = tr.train_chain(W[:4], lambda c: bg, tc, cfg["tau_i"], chain=chain, log=lambda s: print(s, flush=True))
L = cfg["lmax"]; tca = eq.Layout(tca=True, lmax_nu=L["nu"]); full = eq.Layout(tca=False, lmax_g=L["g"], lmax_pol=L["pol"], lmax_nu=L["nu"])
taus = np.geomspace(30.5, 59.5, 200)
for k in [3e-3, 0.03, 0.06, 0.09, 0.12]:
    sol = ref.solve_segments(k, bg, taus, [(tca, 30.0), (full, 60.0)], tau_i=cfg["tau_i"], C=1.0)
    obs = chain.observables(taus, jnp.float64(k), jnp.zeros(0), bg)
    e = {v: float(np.max(np.abs(np.asarray(obs[v]) - sol.values[v])) / np.max(np.abs(sol.values[v])))
         for v in ["phi", "delta_g", "theta_g", "delta_b", "theta_b"]}
    print(f"W3 slow={slow} p={p} k={k:<6g} " + " ".join(f"{v}={x:.1e}" for v, x in e.items()), flush=True)
