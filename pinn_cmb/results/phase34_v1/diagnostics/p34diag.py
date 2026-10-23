import numpy as np, yaml, jax.numpy as jnp
from pathlib import Path
from pinn_cmb import background as bgm, equations as eq, reference as ref
from pinn_cmb.phases import phase34 as p
cfg = yaml.safe_load(open("pinn_cmb/configs/phase34.yaml"))
bg = bgm.build_background()
W, chain = p.load_chain(cfg, bg, Path("pinn_cmb/results/phase34"))
L = cfg["lmax"]; full = eq.Layout(tca=False, lmax_g=L["g"], lmax_pol=L["pol"], lmax_nu=L["nu"])
tca = eq.Layout(tca=True, lmax_nu=L["nu"])
edges = [30, 60, 100, 150, 200, 240, 270, 300, 340]
taus = np.geomspace(31, 339, 300)
print("tau_rec", float(bg.tau_rec), "n windows", len(chain.windows))
print("k        var     " + " ".join(f"{a}-{b}" for a, b in zip(edges[:-1], edges[1:])))
for k in [0.002, 0.03, 0.09, 0.12]:
    sol = ref.solve_segments(k, bg, taus, [(tca, 30.0), (full, 340.0)], tau_i=cfg["tau_i"], C=0.5)
    obs = chain.observables(taus, jnp.float64(k), jnp.zeros(0), bg)
    s0 = sol.values["phi"][0] / float(obs["phi"][0]); obs = {kk: s0 * np.asarray(vv) for kk, vv in obs.items()}
    print("norm ratio", s0)
    for v in ["phi", "delta_g", "theta_b", "delta_b"]:
        ref_v = sol.values[v]; e = np.abs(np.asarray(obs[v]) - ref_v) / np.max(np.abs(ref_v))
        seg = [np.max(e[(taus >= a) & (taus < b)]) for a, b in zip(edges[:-1], edges[1:])]
        print(f"{k:<8} {v:8s} " + " ".join(f"{s:.1e}" for s in seg))
