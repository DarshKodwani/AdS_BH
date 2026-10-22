# PINN solver for linear CMB perturbations — project spec

> Original brief, kept verbatim. Proposed revisions from the literature review and feasibility discussion are listed in `literature_review.md` §6 and in the "Open spec issues" section at the end of this file.

Brief for Claude Code. Goal: a physics-informed neural network (PINN) that solves the linearised Einstein–Boltzmann equations for the photon–baryon–CDM–neutrino system, validated against a stiff ODE reference and CLASS. Build it in staged phases, each with a hard acceptance test, and don't move on until the test passes.

---

## 0. Ground rules

- **Language and stack:** Python with JAX (jax, optax, jaxopt or optax L-BFGS) in **float64** (`jax.config.update("jax_enable_x64", True)`). Use `classy` (the CLASS Python wrapper) for the background, thermodynamics and validation. Use `scipy.integrate.solve_ivp(method="Radau")` for the reference solutions.
- **Conventions:** Ma & Bertschinger 1995 (astro-ph/9506072), conformal Newtonian gauge, conformal time τ in Mpc, k in Mpc⁻¹. Metric: ds² = a²[−(1+2ψ)dτ² + (1−2φ)dx²].
- **Scope:** The PINN solves the **perturbations only**. Take the background a(τ), ℋ = ȧ/a, densities, the free-electron fraction x_e(τ), the opacity κ̇ = a n_e σ_T, the baryon sound speed c_s² and the sound horizon r_s(τ) from CLASS, then tabulate and interpolate them with a JAX-differentiable cubic spline.
- **Verification:** Check every equation and initial-condition coefficient against Ma & Bertschinger before use. Do not trust the coefficients in this spec blindly. Write a unit test per equation.
- **Fiducial cosmology:** Planck 2018 ΛCDM, flat, massless neutrinos (N_eff = 3.046), no reionisation in the early phases.

## 1. Equations (Newtonian gauge, M&B §4 — verify)

Notation: ℋ = ȧ/a, R = 3ρ_b / (4ρ_γ), κ̇ = a n_e σ_T, dots are d/dτ.

**Einstein equations**
- 0i: k²(φ̇ + ℋψ) = 4πG a² Σ(ρ+P)θ
- ij-traceless: k²(φ − ψ) = 12πG a² Σ(ρ+P)σ  (only ν (and γ after decoupling) shear contributes)
- 00 (constraint, use as a **diagnostic loss**, not an evolution eq): k²φ + 3ℋ(φ̇ + ℋψ) = −4πG a² δρ

**CDM**
- δ̇_c = −θ_c + 3φ̇
- θ̇_c = −ℋθ_c + k²ψ

**Baryons**
- δ̇_b = −θ_b + 3φ̇
- θ̇_b = −ℋθ_b + c_s²k²δ_b + k²ψ + (κ̇/R)(θ_γ − θ_b)

**Photons** (temperature hierarchy F_γℓ, polarisation G_γℓ)
- δ̇_γ = −(4/3)θ_γ + 4φ̇
- θ̇_γ = k²(δ_γ/4 − σ_γ) + k²ψ + κ̇(θ_b − θ_γ)
- higher multipoles: M&B eq. (63)

**Massless neutrinos**
- δ̇_ν = −(4/3)θ_ν + 4φ̇
- θ̇_ν = k²(δ_ν/4 − σ_ν) + k²ψ
- 2σ̇_ν = (8/15)θ_ν − (3/5)k F_ν3, plus hierarchy F_νℓ (M&B eq. 49), truncated at ℓ_max with the M&B truncation scheme (eq. 51)

**Tight-coupling approximation (TCA)**, used before recombination in Phase 1: θ_b = θ_γ ≡ θ_γb, σ_γ = 0, and higher photon multipoles = 0.
- θ̇_γb = [ −ℋR θ_γb + k²δ_γ/4 + R c_s²k²δ_b ] / (1+R) + k²ψ  (verify against M&B eq. 74–75; zeroth order is fine for Phase 1)

**Adiabatic initial conditions** (super-horizon, radiation era, M&B eq. 96): set at kτ_i ≈ 10⁻³ and parameterise by a constant C. Pull the exact coefficients (including the R_ν = ρ_ν/(ρ_γ+ρ_ν) dependence of ψ, φ and σ_ν) from the paper. Cross-check: run CLASS with the same k, read its initial values, and confirm the ratios match.

## 2. Network design

- **Inputs:** τ̃ = log(τ/τ_i) normalised to [0, 1], log k normalised, and later (Ω_b h², Ω_c h²) normalised. This makes it a "bundle" PINN: one network over the whole family.
- **Hard initial conditions:** u(τ) = u_IC + (1 − e^{−(τ̃)/ε})·N(τ̃, …). The ICs are then exact and need no IC loss term.
- **Oscillation ansatz** (essential for k ≳ 0.05 Mpc⁻¹). For the oscillating photon–baryon variables:
  X(k,τ) = A(k,τ)·cos(k r_s(τ)) + B(k,τ)·sin(k r_s(τ)) + C(k,τ),
  where A, B and C are smooth network outputs and r_s comes from the background tables. Potentials φ and ψ and the CDM variables are smooth, so they get no ansatz.
- **Architecture:** an MLP of 4–6 layers × 64–128 units with tanh or SiLU activations, plus random Fourier features on τ̃ (σ ≈ 1–5) if the ansatz alone isn't enough.
- **Loss:** a weighted sum of squared ODE residuals (computed with autodiff in τ), each nondimensionalised by a typical scale (e.g. divide the θ equations by k·max|θ|). Add the 00-constraint residual with a small weight.
- **Training:**
  - Use causal weighting (Wang, Sankaran & Perdikaris 2022, "Respecting causality is all you need"), so earlier times converge before later ones are trusted.
  - Use a curriculum: first single k, then a k-range, then the parameter bundle; extend τ_max progressively.
  - Optimise with Adam (lr 1e-3 with cosine decay), then L-BFGS for the final polish.
  - Resample collocation points every N steps (log-uniform in τ, uniform in log k).
- **Metrics:** relative error vs the reference for every variable, max and RMS over τ, with plots.

## 3. Phased plan and acceptance tests

**Phase 0: Infrastructure**
- Build the background and thermo tables from classy, with differentiable interpolation.
- Write a reference solver that implements *exactly the same equations* as the PINN (scipy Radau, rtol 1e-10), with a TCA switch.
- Run the CLASS validation harness: `output: mPk`, `gauge: newtonian`, `k_output_values` set, then read `get_perturbations()`.
- ✅ Accept when the reference solver matches CLASS to <1% on φ, δ_c and δ_γ at k = 0.01 and 0.05 through recombination (residual differences come from the TCA and truncation; document them).

**Phase 1: Tight-coupled fluid, single cosmology**
- Variables: φ, ψ, δ_c, θ_c, δ_b, δ_γ, θ_γb, plus the ν hierarchy truncated at ℓ_max = 4.
- Domain: τ_i up to τ = 200 Mpc (before recombination), k ∈ [10⁻³, 0.05].
- Steps: single k, then a k-bundle.
- ✅ Accept when RMS relative error is <10⁻³ vs the reference solver on all variables, at 20 test k's not seen in training.

**Phase 2: Higher k and parameter bundle**
- Extend to k ≤ 0.2 Mpc⁻¹ using the oscillation ansatz.
- Add Ω_b h² ∈ [0.020, 0.024] and Ω_c h² ∈ [0.10, 0.14] as inputs.
- ✅ Accept when error is <10⁻³ vs the reference across a held-out Latin-hypercube test set.

**Phase 3: Through recombination**
- Drop the TCA once κ̇/ℋ falls below a threshold (CLASS-style switch).
- Add the photon hierarchy (ℓ_max ≈ 10–20) plus polarisation, with the full Thomson coupling.
- Handle the regime boundary either with a second network for τ > τ_switch, whose ICs are taken from the first network's output (hard-enforced), or with a single network and a stiffness-aware loss weighting. Try the two-network approach first.
- ✅ Accept when error is <10⁻² vs CLASS for τ up to 2·τ_rec.

**Phase 4: Source functions → C_ℓ**
- Have the PINN output the temperature source function S_T(k,τ) (M&B / Seljak–Zaldarriaga line-of-sight form) and do the line-of-sight integral with precomputed spherical Bessels.
- Compute C_ℓ^TT for ℓ ≤ 500, initially without reionisation and lensing.
- ✅ Accept when error is <1% vs CLASS C_ℓ^TT at ℓ ≤ 500 (stretch goal: 0.1%).

**Phase 5 (stretch): The scientific payoff**
- Modify the physics where CLASS has no ready implementation: a dark-sector interaction term, or a free function (e.g. w(a) or a time-dependent coupling) parameterised by the network itself.
- Run inverse problems, i.e. fit the free function to mock C_ℓ through the differentiable pipeline.

## 4. Repo layout

```
pinn_cmb/
  background.py      # classy → differentiable tables (ℋ, κ̇, R, c_s², r_s, ρ_i)
  equations.py       # RHS functions shared by PINN residuals AND reference solver
  initial_conditions.py
  reference.py       # scipy Radau solver + TCA switch
  validate_class.py  # CLASS perturbation extraction + comparison plots
  model.py           # networks, ansatz, hard-IC wrapper
  train.py           # losses, causal weighting, curriculum, Adam→L-BFGS
  configs/           # yaml per phase
  tests/             # one test per equation/IC + phase acceptance tests
  notebooks/         # diagnostics plots
```

`equations.py` must be the **single source of truth**: both the PINN residual and the reference solver call the same RHS function. That way any disagreement is a training problem, not a typo.

## 5. Known pitfalls

- **Stiffness:** Don't put the raw Thomson term κ̇(θ_b − θ_γ) into the loss while κ̇/ℋ ≫ 1. That's what the TCA is for.
- **Spectral bias:** Without the cos/sin(k r_s) ansatz, high-k modes will look converged but have the wrong phase. Always plot phase error.
- **Scale disparity:** Variables span orders of magnitude. Nondimensionalise residuals per equation, or one equation will dominate the loss.
- **Gauge mixing:** CLASS's default output gauge is synchronous. Set `gauge: newtonian` or transform.
- **Units:** CLASS's θ and k conventions (h vs no h, Mpc vs h⁻¹Mpc) must be checked explicitly in Phase 0.
- **The 00 constraint:** If its residual grows while the evolution residuals look fine, the solution is drifting off the constraint surface. Treat it as a red flag.

## 6. References

- Ma & Bertschinger 1995, astro-ph/9506072: equations, ICs, truncation
- Blas, Lesgourgues & Tram 2011 (CLASS II), arXiv:1104.2933: TCA and RSA schemes
- Albers et al. 2019 (CosmicNet I), arXiv:1907.05764: physics-driven NN for source functions
- arXiv:2508.08443: bundle PINNs for linear matter perturbations
- Wang, Sankaran & Perdikaris 2022, arXiv:2203.07404: causal training for PINNs
- Seljak & Zaldarriaga 1996, astro-ph/9603033: line-of-sight integration

---

## Open spec issues (from feasibility review, not yet applied)

1. **Error metric.** Pointwise relative error blows up at zero crossings of θ, δ_γ and the multipoles. Use error normalised by max|X| over τ at fixed k, or by the oscillation envelope. This affects the Phase 0–2 acceptance tests.
2. **Phase 4 domain.** ISW needs φ̇ and ψ̇ up to τ₀ ≈ 14,000 Mpc, not 2τ_rec. Add an RSA-style late-time reduction so the late network carries only the metric and matter variables.
3. **k range for ℓ ≤ 500.** ℓ ≈ kτ₀, so k up to ~0.1 Mpc⁻¹ with margin is needed. This is covered by Phase 2.
4. **ψ is algebraic.** It follows from φ and the shears (ij equation). Enforce it exactly instead of learning it.
5. **Phase 0 tolerance.** CLASS uses higher-order TCA and different truncation, so its precision settings may need tuning to get <1% near recombination.
6. **Float64 on GPU** is slow on most GPUs. Benchmark CPU vs GPU early.
7. From the literature review: add the Silk damping envelope to the ansatz, natural-gradient optimisation, a residual-based error bound, and a DISCO-EB/ABCMB benchmark (see `literature_review.md` §6).
