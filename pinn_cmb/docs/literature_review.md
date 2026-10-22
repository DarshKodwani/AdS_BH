# Literature review: PINNs for linear CMB perturbations

*Scan date: 2026-10-22. Sources: arXiv via alphaXiv. Links are to arXiv abstracts.*

## Bottom line

- **No published work solves the coupled photon–baryon–CDM–neutrino Einstein–Boltzmann system with a PINN.** The closest is a single scalar ODE for sub-horizon matter growth. Phases 1–4 of the spec are therefore novel as a method.
- **The bar is set by differentiable solvers.** DISCO-EB, Bolt.jl, SymBoltz.jl and ABCMB are all differentiable, reach 0.1–0.7% agreement with CLASS, and three of them already compute CMB C_ℓ. A PINN paper must say what it does that these cannot.
- **The oscillation ansatz is not new.** CosmicNet I (2019) fed cos(k r_s), sin(k r_s) and the damping envelope exp(−k²/k_D²) into a network that emulates source functions. That work should be cited, and the damping envelope should be added to our ansatz.

## 1. Direct prior art: PINNs in cosmology

| Paper | What it does | Relevance |
|---|---|---|
| Gomez Bachar, Chantada, Landau, Scóccola & Protopapas 2025, [2508.08443](https://arxiv.org/abs/2508.08443) | A bundle PINN for the **single** sub-horizon growth ODE δ_m'' + h δ_m' + g δ_m = 0, with Ω_m (and a G_eff parameter) as bundle inputs. Two 2×32 MLPs, hard ICs u = u₀ + (1−e^{−(N+1)})·NN (our hard-IC trick), and a variable change to X = ln δ. Its key novelty is an **a-posteriori error bound computed from the residual alone** (<0.4% on fσ₈). | Closest precedent. Their own words call it "an initial effort to address the coupled Boltzmann equations", which is exactly our gap. No k-dependence, radiation, photons or oscillations. |
| Chantada et al. 2023, PRD 107 063523; 2024, PRD 109 123514 ([2311.15955](https://arxiv.org/abs/2311.15955)) | Bundle PINNs for **background** dynamics (several DE/f(R) models), with a speedup reported in MCMC. | Background only. Shows the group's pipeline (NeuroDiffEq). |
| Chantada et al. 2024, [2411.13848](https://arxiv.org/abs/2411.13848) | Exact and approximate residual-based error bounds for PINNs on nonlinear first-order ODEs. | The error-bound machinery could be extended to our *linear* system, possibly more easily, because linear-ODE bounds exist (Liu, Huang & Protopapas 2022/23). |
| Cosmo-PINN 2026, [2605.30139](https://arxiv.org/abs/2605.30139); 2508.12032; 2606.17614; 2509.24327 | PINNs for reconstructing H(z), w(z) and fσ₈ from late-time data (SN, BAO, RSD). | Background or growth reconstruction. None touch CMB perturbations. |
| Neural Boltzmann Equations, Spinner & Shergold 2026, [2608.23022](https://arxiv.org/abs/2608.23022) | Neural distribution functions with Monte Carlo collision integrals and **natural-gradient** time-stepping, applied to N_eff and thermal relics. Homogeneous (FLRW) only. | Different problem: momentum-space f(y) with no spatial perturbations. Methodologically useful: natural gradient beat Adam by orders of magnitude in precision (10⁻⁶ vs 10⁻² at equal iterations), and their implicit fixed-point trick handles stiff, near-equilibrium regimes. |
| PINNs for freeze-in DM, [2502.17597](https://arxiv.org/abs/2502.17597) | A PINN for background Boltzmann (relic-abundance) equations, including inverse problems. | Background number-density equations only. |
| Physics-informed KANs for cosmological Vlasov–Poisson, [2512.11795](https://arxiv.org/abs/2512.11795) | A PINN-style solver for CDM phase-space evolution. | Nonlinear LSS. Different regime. |

## 2. Neural networks inside Boltzmann codes (data-trained)

| Paper | What it does | Relevance |
|---|---|---|
| **CosmicNet I**, Albers et al. 2019, [1907.05764](https://arxiv.org/abs/1907.05764) | Emulates the CMB **source functions** S_T0, S_T1, S_T2(k, τ) inside CLASS, trained on CLASS outputs. Physics-driven inputs: e^{−κ}, g, g′, **cos(k r_s), sin(k r_s), exp(−k²/k_D²)**, and j_ℓ(kτ) for free streaming. Reports ~0.3% C_ℓ^TT, a negligible MCMC bias and a 6–28× speedup of the perturbation module. | **Must cite.** It already used our oscillation ansatz as input features. The differences in our approach: no training data, the equations are the loss, and the ansatz sits in the output structure rather than the inputs. Its target (source functions) is the same as our Phase 4. |
| CosmicNet II, [2207.05707](https://arxiv.org/abs/2207.05707) | Extends CosmicNet I to extended cosmologies. | Shows the retraining cost when the physics changes, which a data-free PINN avoids. |
| CosmoPower ([2106.03846](https://arxiv.org/abs/2106.03846)), CosmoPower-JAX ([2305.06347](https://arxiv.org/abs/2305.06347)), Capse.jl ([2307.14339](https://arxiv.org/abs/2307.14339)), CONNECT, CLiENT ([2512.17509](https://arxiv.org/abs/2512.17509)) | Emulate C_ℓ, P(k) or the likelihood directly from CLASS/CAMB training sets. | The speed benchmark. Every one of them needs a Boltzmann code to generate its training data, so none can be built for physics that has no solver. |

## 3. Differentiable Boltzmann solvers (the main competitors)

| Code | Language | Scope and accuracy | Notes for us |
|---|---|---|---|
| **DISCO-EB** (DISCO-DJ I), Hahn, List & Porqueres 2023, [2311.03291](https://arxiv.org/abs/2311.03291) | JAX + Diffrax (Kvaerno5) | Synchronous gauge. CDM, b, γ, massless and massive ν, w0wa fluid. Per-mille agreement with CLASS and CAMB on P(k). Has TCA, UFA and RSA implemented but **switched off**, since none gave an O(1) speedup on GPU. No CMB C_ℓ (yet). | Forward- and reverse-mode gradients. Also in JAX, so it is the most natural reference implementation to compare against. |
| **ABCMB**, Zhou, Giovanetti & Liu 2026, [2602.15104](https://arxiv.org/abs/2602.15104) | JAX | Lensed TT/TE/EE to ℓ=4000 plus P(k). **0.1–0.7% vs CLASS.** HyRex recombination, LINX BBN. 6–14 s per cosmology on an H100/H200 and ~110 s on 1 CPU core; forward-mode gradients 13–47 s on GPU. **No TCA**: it integrates the stiff system directly and found TCA patching *slower*. No RSA either. | Designed so new species can be added without touching source files. This undercuts the "PINNs make new physics easy" argument, so we must be specific about what is easier. |
| **SymBoltz.jl**, Sletmoen 2025, [2509.24740](https://arxiv.org/abs/2509.24740) | Julia, symbolic-numeric | Approximation-free (no TCA/UFA/RSA). Sparse analytic Jacobians, Rodas5P/KenCarp4. 0.1% vs CLASS. The modeller writes the equations symbolically. | Forward-mode AD only. Again, new models are added by writing equations. |
| **Bolt.jl**, Li et al. 2023 | Julia | Approximation-free, forward-mode differentiable. | |
| nanoCMB, [2602.23466](https://arxiv.org/abs/2602.23466) | Python | Minimal, readable TT/EE/TE calculator for flat ΛCDM. | A useful readable second reference for our Phase 0/4 harness, alongside CLASS. |

**Takeaways:**
1. All of these codes can already compute gradients with respect to parameters and handle new physics written as equations. "A PINN gives you differentiability" is **not** a selling point.
2. Several codes found TCA unnecessary when an implicit solver is used. For a PINN, though, the stiffness problem is about **loss conditioning**, not step size, so the TCA remains essential for us (spec §5). That contrast is itself worth a paragraph in the paper.
3. Runtime per cosmology is seconds on a GPU. A trained bundle PINN evaluates in milliseconds, but training costs hours. The honest comparison is amortised cost over N evaluations.

## 4. PINN methodology relevant to this problem

| Paper | Use |
|---|---|
| Wang, Sankaran & Perdikaris 2022, [2203.07404](https://arxiv.org/abs/2203.07404) | Causal weighting (already in the spec). |
| Flamant, Protopapas & Sondak 2020, [2006.14372](https://arxiv.org/abs/2006.14372) | Original "solution bundle" paper. Cite for the bundle concept. |
| Müller & Zeinhofer 2023 (ICML), "Energy natural gradient" | Natural-gradient PINN training reaching near-machine precision. **Worth trying in place of Adam→L-BFGS** to reach the 10⁻³ targets, since NBE (above) saw the same benefit. |
| Liu, Huang & Protopapas 2022/23, [2207.01114](https://arxiv.org/abs/2207.01114) | Residual-based error bounds for **linear** ODE systems, which applies directly to our perturbation system. |
| Fourier-feature PINNs ([2510.19399](https://arxiv.org/abs/2510.19399), [2610.03378](https://arxiv.org/abs/2610.03378)); "Neither precision nor architecture alone" ([2608.25327](https://arxiv.org/abs/2608.25327)) | Spectral bias and FP64 vs FP32 failure modes. The latter supports the float64 choice but warns it is not sufficient alone. |
| Stiff-PINN benchmarks ([2605.04502](https://arxiv.org/abs/2605.04502), [2606.21236](https://arxiv.org/abs/2606.21236)) | Gradient scaling and loss balancing for stiff, regime-switching systems, which is our Phase 3 problem. |

## 5. Where the gap is: candidate paper claims

Ranked by strength, given the above:

1. **A data-free, error-certified bundle solver for the CMB perturbation system.** This is the first PINN for the coupled Einstein–Boltzmann hierarchy. It is trained without any CLASS runs, and it comes with a residual-based a-posteriori error bound, extending 2508.08443 from 1 scalar ODE to ~15–60 coupled linear ODEs. The "no training data, yet a certified error" combination is something neither emulators nor ODE solvers offer.
2. **New physics where no solver exists yet** (spec Phase 5). Where the physics has no existing solver, a data-free emulator can still be trained, whereas CosmicNet/CosmoPower-style emulators cannot be. Caveat: DISCO-EB, ABCMB and SymBoltz all make adding equations fairly easy, and they can embed a neural free function (a "universal differential equation") and differentiate through it. The PINN advantage has to be demonstrated, for example as amortisation across a free-function family, rather than asserted.
3. **A methods study: what makes PINNs fail on Boltzmann hierarchies.** This covers stiffness vs TCA, phase error at high k, and constraint drift. It is publishable even if the accuracy targets are only partly met, but on its own it would be the weakest claim.

**Recommended framing:** lead with (1) and demonstrate (2) on one model. Benchmark honestly against DISCO-EB or ABCMB on accuracy, wall-clock time and amortised cost.

## 6. Changes to the spec suggested by this review

- **Ansatz:** add the Silk damping envelope, i.e. X = e^{−k²/k_D²(τ)}[A cos(k r_s) + B sin(k r_s)] + C. Tabulate k_D(τ) from CLASS along with r_s. Cite CosmicNet I for the physics-informed basis.
- **Optimiser:** add (energy) natural gradient as an alternative to Adam→L-BFGS for the final-precision stage.
- **Error bound:** add a deliverable for a residual-based a-posteriori bound (linear-ODE version, Liu et al.). This is what differentiates the paper from 2508.08443 and from emulators.
- **Benchmark:** add DISCO-EB (JAX, same ecosystem) and/or ABCMB as a second reference besides CLASS. Report runtime, gradients and amortised cost.
- **Phase 4 domain:** source functions need τ up to τ₀ for the ISW terms. Use an RSA-style late-time reduction so the late network carries only φ, ψ and the matter variables (DISCO-EB has RSA code we can check against).
- **Metrics:** use envelope-normalised error, not pointwise relative error, for oscillating and zero-crossing variables (ABCMB avoids comparing TE residuals for the same reason).

## 7. Still to check

- Bolt.jl paper (Li et al. 2023) for its exact accuracy numbers.
- Whether the Buenos Aires/Harvard group (Chantada, Protopapas et al.) has a newer preprint on the coupled system; they flagged it as their next step. Re-check arXiv before submission.
- CLASS II (Blas, Lesgourgues & Tram 2011) for the exact TCA switching criteria and the higher-order TCA terms.
