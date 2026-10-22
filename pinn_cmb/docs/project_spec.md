# PINN solver for linear CMB perturbations — project spec

> Revision 2 (2026-10-22). This is the original brief with the revisions from the feasibility review, the literature review (`literature_review.md` §6) and the Phase 0 findings (`phase0_results.md`) applied. Changes are marked **[rev]** and listed in the revision log at the end.

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

**[rev] Initial conditions must satisfy the 00 constraint.** M&B's leading-order ICs (eq. 98 on arXiv v1) violate the 00 constraint at O(τ_i/τ_eq). Our evolution system (φ̇ from 0i, ψ from ij) has a constraint-violating solution that grows like τ relative to the physical one. A 1e-5 violation at τ_i became 2–5% errors by recombination in Phase 0. Use CLASS's scheme instead (`initial_conditions.adiabatic`): the synchronous-gauge expansion to first order in (kτ)² and ωτ (ω = aρ_m/√ρ_r), then the gauge transformation with α taken from the 00 and 0i equations. This applies to the PINN's hard ICs too.

**[rev] ψ is algebraic.** ψ follows from φ and the shears through the ij-traceless equation. Compute it exactly; it is never a network output or an evolved variable.

## 2. Network design

- **Inputs:** τ̃ = log(τ/τ_i) normalised to [0, 1], log k normalised, and later (Ω_b h², Ω_c h²) normalised. This makes it a "bundle" PINN: one network over the whole family.
- **Hard initial conditions:** u(τ) = u_IC + (1 − e^{−(τ̃)/ε})·N(τ̃, …). The ICs are then exact and need no IC loss term.
- **Oscillation ansatz** (essential for k ≳ 0.05 Mpc⁻¹). For the oscillating photon–baryon variables:
  X(k,τ) = A(k,τ)·cos(k r_s(τ)) + B(k,τ)·sin(k r_s(τ)) + C(k,τ),
  where A, B and C are smooth network outputs and r_s comes from the background tables. Potentials φ and ψ and the CDM variables are smooth, so they get no ansatz.
  **[rev]** Multiply the oscillating part by the Silk damping envelope: X = e^{−k²/k_D²(τ)}[A cos(k r_s) + B sin(k r_s)] + C. k_D(τ) is tabulated (`background.damping_wavenumber`). Prior art: CosmicNet I (1907.05764) used the same basis as network *inputs*; ours puts it in the output structure.
- **Architecture:** an MLP of 4–6 layers × 64–128 units with tanh or SiLU activations, plus random Fourier features on τ̃ (σ ≈ 1–5) if the ansatz alone isn't enough.
- **Loss:** a weighted sum of squared ODE residuals (computed with autodiff in τ), each nondimensionalised by a typical scale (e.g. divide the θ equations by k·max|θ|). Add the 00-constraint residual with a small weight.
  **[rev, pilot 1]** The 00-constraint residual is a **mandatory** loss term at O(1) weight. Without it the error was 30× worse at a lower ODE loss. Scale outputs and residuals by each variable's *time-dependent* physical size: θ ∝ k²τ and F_νℓ ∝ (kτ)^ℓ outside the horizon, rolling over to O(1) inside it. Fixed scales make the loss ill-conditioned by ~1/(kτ)⁴. Write the residual in log time: r = (du/dx·1/L − τ f)/S(τ).
- **Training:**
  - Use causal weighting (Wang, Sankaran & Perdikaris 2022, "Respecting causality is all you need"), so earlier times converge before later ones are trusted.
  - Use a curriculum: first single k, then a k-range, then the parameter bundle; extend τ_max progressively.
  - Optimise with Adam (lr 1e-3 with cosine decay), then L-BFGS for the final polish. **[rev]** Also try (energy) natural gradient (Müller & Zeinhofer 2023) for the final-precision stage, and compare.
  - Resample collocation points every N steps (log-uniform in τ, uniform in log k).
- **Metrics [rev]:** the error for every variable X at fixed k is max_τ|X_PINN − X_ref| / max_τ|X_ref| (plus an RMS version). This replaces pointwise relative error, which blows up at the zero crossings of θ, δ_γ and the multipoles. Report it with plots, and always plot phase error at high k.
- **Error certificate [rev]:** derive a residual-based a-posteriori error bound for the linear system (cf. Liu, Huang & Protopapas 2207.01114; Chantada et al. 2411.13848). Report it next to the true error. This is the paper's main methodological contribution beyond 2508.08443.

## 3. Phased plan and acceptance tests

**Phase 0: Infrastructure**
- Build the background and thermo tables from classy, with differentiable interpolation.
- Write a reference solver that implements *exactly the same equations* as the PINN (scipy Radau, rtol 1e-10), with a TCA switch.
- Run the CLASS validation harness: `output: mPk`, `gauge: newtonian`, `k_output_values` set, then read `get_perturbations()`.
- ✅ Accept when the reference solver matches CLASS to <1% on φ, δ_c and δ_γ at k = 0.01 and 0.05 through recombination (residual differences come from the TCA and truncation; document them). **[rev] Done: passes at ≤1.1e-3, see `phase0_results.md`.**

**Phase 1: Tight-coupled fluid, single cosmology**
- Variables: φ, δ_c, θ_c, δ_b, δ_γ, θ_γb, plus the ν hierarchy truncated at ℓ_max = 4. **[rev]** ψ is algebraic, not an output.
- Domain: τ_i up to τ = 200 Mpc (before recombination), k ∈ [10⁻³, 0.05].
  **[rev]** Zeroth-order TCA is not physically valid up to τ = 200 at high k. At k = 0.05 the CLASS trigger fires at τ ≈ 112 Mpc, and the error against the full system grows to ~1e-2 by τ = 200. Two options: cap the domain at τ_max(k) = min(200, τ_switch(k)), or keep τ = 200 but compare only against the `tca_only` reference (same equations) and state that this tests the solver, not the physics. Recommended: the first, since it is what Phase 3 needs anyway.
- Steps: single k, then a k-bundle.
- ✅ Accept when the max-normalised error **[rev]** is <10⁻³ vs the reference solver (mode `tca_only`, ℓ_max,ν = 4) on all variables, at 20 test k's not seen in training.

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
- **[rev]** The source function needs τ up to τ₀ ≈ 14,200 Mpc, not 2τ_rec, for the early and late ISW terms (φ̇ + ψ̇). Add a radiation-streaming-style late-time reduction (CLASS II §4), so the late-time network carries only φ, the matter variables and an approximate radiation sector. Since ℓ ≈ kτ₀, ℓ ≤ 500 needs k up to ~0.1 Mpc⁻¹ with margin, which Phase 2 covers.
- ✅ Accept when error is <1% vs CLASS C_ℓ^TT at ℓ ≤ 500 (stretch goal: 0.1%).

**Phase 5 (stretch): The scientific payoff**
- Modify the physics where CLASS has no ready implementation: a dark-sector interaction term, or a free function (e.g. w(a) or a time-dependent coupling) parameterised by the network itself.
- Run inverse problems, i.e. fit the free function to mock C_ℓ through the differentiable pipeline.
- **[rev]** Benchmark against a differentiable Boltzmann code (DISCO-EB, same JAX ecosystem; or ABCMB), not just CLASS. Compare accuracy, wall-clock time per evaluation and per gradient, and amortised cost over N evaluations, including training. Those codes can also embed a neural free function, so the PINN's advantage must be demonstrated, not asserted.

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

## Revision log

**Rev 2 (2026-10-22)**, from the feasibility review, the literature review and Phase 0:
1. Error metric: max-normalised instead of pointwise relative error (§2 Metrics, Phase 1).
2. Phase 4 domain extended to τ₀, with a late-time radiation-streaming reduction; k ≲ 0.1 needed for ℓ ≤ 500.
3. ψ is algebraic (§1).
4. Phase 0 tolerance: met with CLASS's late-time approximations switched off and ℓ_max matched; no further tuning needed.
5. Float64 on GPU is slow on most GPUs; benchmark CPU vs GPU at the start of Phase 1. The current dev container is CPU-only (4 cores).
6. Silk damping envelope in the ansatz; natural-gradient optimiser; residual-based error certificate; DISCO-EB/ABCMB benchmark.
7. **New from Phase 0:** constraint-consistent ICs (CLASS scheme), and the Phase 1 TCA domain capped at τ_switch(k).
8. **Equation numbering:** this spec cites M&B published numbering. On arXiv v1 the Newtonian-gauge photon hierarchy is eq. 64 (truncation eq. 65), massless neutrinos eq. 50, and the ICs eq. 98. The code cites arXiv v1.
