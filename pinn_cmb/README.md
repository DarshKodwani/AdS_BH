# pinn_cmb

Physics-informed neural network (PINN) solver for linear CMB (Einstein–Boltzmann) perturbations, validated against a stiff ODE reference and CLASS.

**Status:** Phase 0 (infrastructure and reference solver) is complete and passes its acceptance test against CLASS. Phase 1 (the PINN for the tight-coupled fluid) is next.

## Layout

| Path | Contents |
|---|---|
| `background.py` | CLASS → JAX-differentiable tables: a, ℋ, ρ_i, κ̇, c_b², R, r_s, k_D |
| `equations.py` | Newtonian-gauge Einstein–Boltzmann RHS (TCA and full hierarchy). Single source of truth for the PINN and the reference |
| `initial_conditions.py` | Constraint-consistent adiabatic ICs (CLASS scheme), plus M&B eq. 98 for checks |
| `reference.py` | scipy Radau with exact JAX Jacobian; TCA → full switch |
| `validate_class.py` | CLASS comparison harness (`python -m pinn_cmb.validate_class`) |
| `configs/` | YAML per phase |
| `tests/` | One test per equation, plus phase acceptance tests |
| `docs/project_spec.md` | The project spec (Rev 2) |
| `docs/literature_review.md` | Prior art and where the paper fits |
| `docs/phase0_results.md` | Phase 0 results and findings |

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r pinn_cmb/requirements.txt   # classy builds CLASS from source (~1-2 min)
cd pinn_cmb && pytest                        # 34 tests, ~90 s
cd .. && python -m pinn_cmb.validate_class   # Phase 0 table
```

This folder lives inside the `ads_bh` repo for now and can be moved to its own repository later.
