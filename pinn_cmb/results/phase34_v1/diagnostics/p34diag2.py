import numpy as np, yaml, jax.numpy as jnp
from pathlib import Path
from pinn_cmb import background as bgm, equations as eq, reference as ref
from pinn_cmb.phases import phase34 as p
cfg = yaml.safe_load(open("pinn_cmb/configs/phase34.yaml"))
bg = bgm.build_background()
W, chain = p.load_chain(cfg, bg, Path("pinn_cmb/results/phase34"))
L = cfg["lmax"]; tca = eq.Layout(tca=True, lmax_nu=L["nu"])
full = eq.Layout(tca=False, lmax_g=L["g"], lmax_pol=L["pol"], lmax_nu=L["nu"])
taus = np.geomspace(1.1e-3, 100, 400)
edges = [1e-3, 0.1, 1, 3, 10, 20, 30, 45, 60, 100]
for k in [0.03, 0.09, 0.12]:
    sol = ref.solve_segments(k, bg, taus, [(tca, 30.0), (full, 101.0)], tau_i=cfg["tau_i"], C=1.0)
    obs = chain.observables(taus, jnp.float64(k), jnp.zeros(0), bg)
    print(f"k={k}  phi ratio at start {sol.values['phi'][0]/float(obs['phi'][0]):.6f}")
    for v in ["phi", "delta_g", "theta_g", "delta_b"]:
        r = sol.values[v]; e = np.abs(np.asarray(obs[v]) - r) / np.max(np.abs(r))
        print(f"   {v:8s}", " ".join(f"{a:g}-{b:g}:{np.max(e[(taus>=a)&(taus<b)]):.1e}" for a, b in zip(edges[:-1], edges[1:])))
