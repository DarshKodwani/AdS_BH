"""Fig. phase2: held-out errors of the (k, omega_b, omega_c) bundle, and their relation to the 00 residual."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
rows = json.load(open(HERE.parent / "results" / "phase2" / "evaluation.json"))["rows"]
k = np.array([r["k"] for r in rows])
oc = np.array([r["omega_c"] for r in rows])
err = np.array([max(v for n, v in r["errors"].items() if n != "max_constraint_00") for r in rows])
c00 = np.array([r["errors"]["max_constraint_00"] for r in rows])

fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8))
sc = ax[0].scatter(k, err, c=oc, cmap="viridis", s=18)
ax[0].axhline(1e-3, color="k", ls="--", lw=0.8)
ax[0].set(xscale="log", yscale="log", xlabel=r"$k$ [Mpc$^{-1}$]", ylabel="max error over variables")
fig.colorbar(sc, ax=ax[0], label=r"$\omega_c$")
ax[1].scatter(c00, err, c=oc, cmap="viridis", s=18)
ax[1].set(xscale="log", yscale="log", xlabel=r"$\max_\tau |C_{00}|$", ylabel="max error over variables")
r = np.corrcoef(np.log(c00), np.log(err))[0, 1]
ax[1].text(0.05, 0.9, f"corr(log, log) = {r:.2f}", transform=ax[1].transAxes, fontsize=8)
fig.tight_layout()
fig.savefig(HERE / "figures" / "phase2.png", dpi=200)
print("corr", r)
