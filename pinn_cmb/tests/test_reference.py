"""Reference solver: internal consistency, and the Phase 0 acceptance test against CLASS."""

import numpy as np
import pytest

from pinn_cmb import reference as ref
from pinn_cmb import validate_class as vc

pytestmark = pytest.mark.slow


def _max_norm_diff(a, b):
    return np.max(np.abs(a - b)) / np.max(np.abs(b))


@pytest.mark.parametrize("k", [0.01, 0.05])
def test_constraint_preserved_through_recombination(bg, k):
    tau = np.geomspace(1.0, 2 * bg.tau_rec, 100)
    sol = ref.solve(k, bg, tau, mode="full")
    assert np.max(np.abs(sol.values["constraint_00"])) < 1e-4


@pytest.mark.parametrize("k", [0.01, 0.05])
def test_tight_coupling_converges_to_full_hierarchy(bg, k):
    """Zeroth-order TCA drops O(tau_c) terms, so its error up to the switch scales with the trigger."""
    errors = []
    for scale in (0.1, 0.01):
        trig = ref.TCATrigger(0.015 * scale, 0.01 * scale)
        t_sw = ref.tca_switch_time(k, bg, trig, 1e-2, 2 * bg.tau_rec)
        tau = np.geomspace(1.0, t_sw, 80)
        tca = ref.solve(k, bg, tau, mode="tca_only")
        full = ref.solve(k, bg, tau, mode="full")
        errors.append(max(_max_norm_diff(tca.values[v], full.values[v])
                          for v in ("phi", "delta_c", "delta_g", "theta_g")))
    assert errors[0] < 5e-4
    assert errors[1] < 1e-4 and errors[1] < errors[0] / 3


def test_solution_converged_in_tolerance(bg):
    tau = np.geomspace(1.0, 2 * bg.tau_rec, 60)
    a = ref.solve(0.05, bg, tau, rtol=1e-10)
    b = ref.solve(0.05, bg, tau, rtol=1e-8, atol=1e-11)
    for v in ("phi", "delta_c", "delta_g", "theta_b"):
        assert _max_norm_diff(b.values[v], a.values[v]) < 1e-6, v


def test_phase0_acceptance_against_class():
    """Spec Phase 0: reference vs CLASS < 1% on phi, delta_c, delta_g at k = 0.01, 0.05 to 2 tau_rec."""
    cfg = vc.load_config()
    for r in vc.run(cfg):
        for v in cfg["acceptance"]["variables"]:
            assert r["errors"][v] < cfg["acceptance"]["tolerance"], (r["k"], v, r["errors"][v])
        assert r["constraint_00"] < 1e-4
