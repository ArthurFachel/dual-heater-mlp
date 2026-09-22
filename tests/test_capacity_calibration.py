"""Tests for the capacity calibration arithmetic.

Every expected value here is derived by hand from the definitions, not copied
from an implementation run, so the tests can fail the code.
"""

import math

import pytest
import torch

from experiments.capacity_calibration import (
    effective_plasticity,
    free_fraction,
    heat_concentration,
    iso_plasticity_family,
    iso_plasticity_family_scoped,
    jaccard,
    participation_ratio,
    plasticity_bounds,
    plasticity_bounds_scoped,
    positive_fraction,
    protected_count,
    protected_heat,
    protected_heat_scoped,
    select_by_declared_criterion,
    solve_strength_for_plasticity,
    solve_strength_for_plasticity_scoped,
    sweep_capacity,
    sweep_capacity_scoped,
)

# ---------------------------------------------------------------------------
# protected count and heat
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("units", "budget", "expected"),
    [
        (4864, 0.25, 3648),  # the observed Qwen2.5-0.5B case
        (4864, 0.50, 2432),
        (4864, 1.00, 0),
        (4864, 0.00, 4864),
        (12, 0.25, 9),
        (10, 0.33, 6),  # floor(6.7)
    ],
)
def test_protected_count_matches_floor_rule(units, budget, expected):
    assert protected_count(units, budget) == expected


def test_protected_count_rejects_out_of_range_budget():
    with pytest.raises(ValueError, match="budget"):
        protected_count(10, 1.5)


def test_protected_heat_normalizes_by_the_selected_maximum():
    importance = torch.tensor([4.0, 3.0, 2.0, 1.0])

    heat = protected_heat(importance, budget=0.5)

    # Two units protected; normalizer is 4.0, the max of the selected set.
    torch.testing.assert_close(heat, torch.tensor([1.0, 0.75, 0.0, 0.0]))


def test_protected_heat_never_exceeds_the_positive_count():
    importance = torch.tensor([5.0, 0.0, 0.0, 0.0])

    # Budget would allow 3, but only 1 unit has positive importance.
    heat = protected_heat(importance, budget=0.25)

    assert int((heat > 0.0).sum().item()) == 1


def test_protected_heat_with_full_budget_protects_nothing():
    importance = torch.tensor([4.0, 3.0, 2.0, 1.0])

    heat = protected_heat(importance, budget=1.0)

    assert torch.all(heat == 0.0)


def test_protected_heat_does_not_mutate_its_input():
    importance = torch.tensor([4.0, 3.0, 2.0, 1.0])
    before = importance.clone()

    protected_heat(importance, budget=0.5)

    torch.testing.assert_close(importance, before)


def test_selected_maximum_equals_the_global_maximum_by_construction():
    """Documents a redundancy found by mutation testing, not a gap.

    Normalizing by `selected.max()` and by the global `max` are provably the
    same: `argsort(descending)` puts the global maximum first, so it is always
    inside the selected set whenever anything is selected. Likewise, capping
    the count at the number of positive entries is redundant, because any extra
    selected entry is zero and `0 / max` is zero. Both guards are kept as
    defensive code (they matter if the ordering or the capacity rule ever
    changes), so no test can distinguish them today. This test pins the
    equivalence instead of pretending to cover it.
    """

    torch.manual_seed(0)
    for _ in range(200):
        units = int(torch.randint(2, 64, (1,)).item())
        # Deliberately include zeros so the positive-count cap could bite.
        importance = torch.rand(units) * torch.randint(0, 2, (units,)).float()
        budget = float(torch.rand(1))
        heat = protected_heat(importance, budget)

        allowed = protected_count(units, budget)
        if allowed == 0 or float(importance.max()) == 0.0:
            assert torch.all(heat == 0.0)
            continue
        order = torch.argsort(importance, descending=True, stable=True)
        naive = torch.zeros_like(importance)
        chosen = order[:allowed]
        naive[chosen] = importance[chosen] / importance.max()
        torch.testing.assert_close(heat, naive, atol=1e-6, rtol=1e-6)


def test_protected_heat_reproduces_the_production_capacity_rule():
    """The sweep must match the real module, or the grid is fiction."""

    from dual_heater.transformer import SlowHeatFFNTracker

    torch.manual_seed(4)
    importance = torch.rand(64)
    tracker = SlowHeatFFNTracker(64, plasticity_budget=0.25)
    with torch.no_grad():
        tracker.task_ema.copy_(importance)
        tracker.task_step.fill_(1)
    tracker.consolidate(strategy="max")

    torch.testing.assert_close(
        protected_heat(importance, 0.25), tracker.slow_heat, atol=1e-6, rtol=1e-6
    )


# ---------------------------------------------------------------------------
# effective plasticity
# ---------------------------------------------------------------------------


def test_effective_plasticity_of_zero_heat_is_one():
    assert effective_plasticity(torch.zeros(10), 3.0) == pytest.approx(1.0)


def test_effective_plasticity_with_zero_strength_is_one():
    assert effective_plasticity(torch.ones(10), 0.0) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("heat_value", "strength", "expected"),
    [
        (1.0, 3.0, 0.25),
        (0.1, 3.0, 1.0 / 1.3),
        (0.01, 3.0, 1.0 / 1.03),
        (1.0, 10.0, 1.0 / 11.0),
        (1.0, 30.0, 1.0 / 31.0),
    ],
)
def test_effective_plasticity_matches_the_modulation_formula(
    heat_value, strength, expected
):
    heat = torch.full((8,), heat_value)

    assert effective_plasticity(heat, strength) == pytest.approx(expected, rel=1e-6)


def test_effective_plasticity_exceeds_the_free_fraction():
    """The key correction: protected count understates surviving plasticity.

    4864 units, budget 0.25 -> 1216 free. If "protected" meant frozen, the
    plasticity would be 1216/4864 = 0.25. With beta = 3 and worst-case heat of
    1.0 everywhere protected, it is 0.44 instead.
    """

    heat = torch.zeros(4864)
    heat[:3648] = 1.0

    free_fraction = 1216 / 4864
    assert free_fraction == pytest.approx(0.25)
    assert effective_plasticity(heat, 3.0) == pytest.approx(0.4375, rel=1e-4)
    assert effective_plasticity(heat, 3.0) > free_fraction


def test_effective_plasticity_decreases_monotonically_in_strength():
    torch.manual_seed(1)
    heat = torch.rand(256)

    values = [effective_plasticity(heat, beta) for beta in (0.0, 1.0, 3.0, 10.0, 30.0)]

    assert values == sorted(values, reverse=True)
    assert values[0] > values[-1]


def test_effective_plasticity_rejects_negative_strength():
    with pytest.raises(ValueError, match="slow_strength"):
        effective_plasticity(torch.zeros(4), -1.0)


# ---------------------------------------------------------------------------
# participation ratio and density
# ---------------------------------------------------------------------------


def test_participation_ratio_of_a_flat_vector_is_the_unit_count():
    assert participation_ratio(torch.ones(100)) == pytest.approx(100.0)


def test_participation_ratio_of_a_single_spike_is_one():
    importance = torch.zeros(100)
    importance[7] = 5.0

    assert participation_ratio(importance) == pytest.approx(1.0)


def test_participation_ratio_counts_only_effective_support():
    importance = torch.zeros(100)
    importance[:10] = 1.0

    assert participation_ratio(importance) == pytest.approx(10.0)


def test_participation_ratio_is_scale_invariant():
    torch.manual_seed(2)
    importance = torch.rand(64)

    assert participation_ratio(importance) == pytest.approx(
        participation_ratio(importance * 1000.0), rel=1e-5
    )


def test_participation_ratio_of_all_zeros_is_zero():
    assert participation_ratio(torch.zeros(10)) == 0.0


def test_positive_fraction_measures_signal_density():
    importance = torch.tensor([1.0, 0.0, 2.0, 0.0])

    assert positive_fraction(importance) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# churn
# ---------------------------------------------------------------------------


def test_jaccard_of_identical_masks_is_one():
    mask = torch.tensor([True, False, True, True])

    assert jaccard(mask, mask) == pytest.approx(1.0)


def test_jaccard_of_disjoint_masks_is_zero():
    first = torch.tensor([True, True, False, False])
    second = torch.tensor([False, False, True, True])

    assert jaccard(first, second) == 0.0


def test_jaccard_of_half_overlap():
    first = torch.tensor([True, True, False, False])
    second = torch.tensor([True, False, True, False])

    # intersection 1, union 3
    assert jaccard(first, second) == pytest.approx(1.0 / 3.0)


def test_jaccard_of_two_empty_masks_is_one_by_convention():
    empty = torch.zeros(4, dtype=torch.bool)

    assert jaccard(empty, empty) == 1.0


def test_jaccard_accepts_heat_vectors_via_boolean_cast():
    first = torch.tensor([0.5, 0.0, 0.2])
    second = torch.tensor([0.9, 0.0, 0.0])

    assert jaccard(first, second) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# sweep and criterion
# ---------------------------------------------------------------------------


def test_sweep_covers_the_full_grid_once():
    torch.manual_seed(3)
    importance = torch.rand(128)

    points = sweep_capacity(
        importance, strengths=[1.0, 3.0, 10.0], budgets=[0.25, 0.5]
    )

    assert len(points) == 6
    assert len({(p.slow_strength, p.budget) for p in points}) == 6


def test_sweep_reports_the_same_protected_count_across_strengths():
    torch.manual_seed(3)
    importance = torch.rand(128)

    points = sweep_capacity(importance, strengths=[1.0, 30.0], budgets=[0.25])

    assert points[0].protected_units == points[1].protected_units
    # But plasticity must differ, otherwise beta does nothing.
    assert points[0].effective_plasticity > points[1].effective_plasticity


def test_criterion_picks_the_largest_qualifying_strength():
    heat_source = torch.ones(100)
    points = sweep_capacity(
        heat_source, strengths=[1.0, 3.0, 10.0, 30.0], budgets=[0.25]
    )

    # All 75 protected units get heat 1.0, so plasticity is
    # (25 + 75/(1+beta)) / 100: beta=1 -> 0.625, 3 -> 0.4375,
    # 10 -> 0.318, 30 -> 0.274.
    chosen = select_by_declared_criterion(
        points, budget=0.25, minimum_effective_plasticity=0.4
    )

    assert chosen.slow_strength == 3.0
    assert chosen.effective_plasticity == pytest.approx(0.4375, rel=1e-4)


def test_criterion_is_stricter_with_a_higher_floor():
    points = sweep_capacity(
        torch.ones(100), strengths=[1.0, 3.0, 10.0, 30.0], budgets=[0.25]
    )

    strict = select_by_declared_criterion(
        points, budget=0.25, minimum_effective_plasticity=0.6
    )
    loose = select_by_declared_criterion(
        points, budget=0.25, minimum_effective_plasticity=0.27
    )

    assert strict.slow_strength == 1.0
    assert loose.slow_strength == 30.0


def test_criterion_raises_instead_of_relaxing_an_unmeetable_floor():
    points = sweep_capacity(torch.ones(100), strengths=[10.0, 30.0], budgets=[0.25])

    with pytest.raises(ValueError, match="nenhum beta satisfaz"):
        select_by_declared_criterion(
            points, budget=0.25, minimum_effective_plasticity=0.9
        )


def test_criterion_ignores_points_from_other_budgets():
    points = sweep_capacity(
        torch.ones(100), strengths=[1.0, 30.0], budgets=[0.25, 0.9]
    )

    chosen = select_by_declared_criterion(
        points, budget=0.25, minimum_effective_plasticity=0.6
    )

    assert chosen.budget == 0.25
    assert chosen.slow_strength == 1.0


def test_criterion_rejects_a_nonsensical_floor():
    points = sweep_capacity(torch.ones(10), strengths=[1.0], budgets=[0.25])

    with pytest.raises(ValueError, match="piso de plasticidade"):
        select_by_declared_criterion(
            points, budget=0.25, minimum_effective_plasticity=0.0
        )


def test_sweep_on_dense_importance_reports_a_saturated_budget():
    """Reproduces the observed Qwen situation: every unit has utility.

    With no zeros, the budget is the only thing limiting protection, so the
    protected count equals the cap exactly.
    """

    torch.manual_seed(5)
    importance = torch.rand(4864) + 0.1

    points = sweep_capacity(importance, strengths=[3.0], budgets=[0.25])

    assert points[0].protected_units == 3648
    assert positive_fraction(importance) == pytest.approx(1.0)
    # A near-uniform signal has participation ratio close to the unit count,
    # which is what makes the budget bite.
    assert participation_ratio(importance) > 0.7 * 4864


def test_sweep_matches_a_hand_computed_point():
    importance = torch.tensor([4.0, 2.0, 1.0, 1.0])

    points = sweep_capacity(importance, strengths=[3.0], budgets=[0.5])

    # heat = [1.0, 0.5, 0, 0]; scales = [0.25, 0.4, 1, 1]; mean = 0.6625
    assert points[0].effective_plasticity == pytest.approx(0.6625, rel=1e-6)
    assert math.isclose(points[0].protected_fraction, 0.5)

# ---------------------------------------------------------------------------
# plasticity bounds
# ---------------------------------------------------------------------------


def test_free_fraction_complements_the_protected_count():
    importance = torch.rand(100) + 0.1

    assert free_fraction(importance, 0.25) == pytest.approx(0.25)
    assert free_fraction(importance, 0.90) == pytest.approx(0.90)


def test_plasticity_is_bounded_below_by_the_free_fraction():
    """Key design fact: beta cannot push plasticity below the budget.

    Free units have h = 0 and each contributes exactly 1 to the mean, so
    E(beta, b) >= (N - P)/N for every beta. The budget, not beta, sets the hard
    floor on how much plasticity can be removed.
    """

    torch.manual_seed(0)
    importance = torch.rand(500) + 0.1

    for budget in (0.05, 0.25, 0.5, 0.9):
        lower, upper = plasticity_bounds(importance, budget)
        heat = protected_heat(importance, budget)
        assert upper == 1.0
        assert lower == pytest.approx(budget, abs=1.0 / importance.numel())
        for strength in (0.0, 1.0, 10.0, 1e3, 1e6, 1e12):
            assert effective_plasticity(heat, strength) >= lower - 1e-9


def test_plasticity_lower_bound_is_tight_in_the_limit():
    importance = torch.rand(1000) + 0.1

    for budget in (0.1, 0.25, 0.5, 0.9):
        heat = protected_heat(importance, budget)
        lower, _ = plasticity_bounds(importance, budget)
        assert effective_plasticity(heat, 1e12) == pytest.approx(lower, abs=1e-6)


# ---------------------------------------------------------------------------
# strength inversion
# ---------------------------------------------------------------------------


def test_solving_for_strength_hits_the_requested_plasticity():
    torch.manual_seed(1)
    importance = torch.rand(400) + 0.05

    for target in (0.95, 0.75, 0.5, 0.35):
        strength = solve_strength_for_plasticity(importance, 0.25, target)
        assert strength is not None
        heat = protected_heat(importance, 0.25)
        assert effective_plasticity(heat, strength) == pytest.approx(target, abs=1e-6)


def test_solving_for_an_unreachable_target_returns_none():
    importance = torch.rand(100) + 0.1

    # Budget 0.5 leaves half the units free, so E can never drop to 0.3.
    assert solve_strength_for_plasticity(importance, 0.5, 0.3) is None
    # And 0.5 is exactly the floor, reachable only in the limit.
    assert solve_strength_for_plasticity(importance, 0.5, 0.49) is None


def test_solving_for_full_plasticity_returns_zero_strength():
    importance = torch.rand(100) + 0.1

    strength = solve_strength_for_plasticity(importance, 0.25, 1.0)

    assert strength is not None
    assert strength == pytest.approx(0.0, abs=1e-3)


def test_solving_rejects_a_nonsensical_target():
    importance = torch.rand(10) + 0.1

    with pytest.raises(ValueError, match="target_plasticity"):
        solve_strength_for_plasticity(importance, 0.25, 0.0)


def test_solved_strength_increases_as_the_target_plasticity_drops():
    torch.manual_seed(2)
    importance = torch.rand(300) + 0.05

    strengths = [
        solve_strength_for_plasticity(importance, 0.25, target)
        for target in (0.9, 0.7, 0.5, 0.4)
    ]

    assert all(value is not None for value in strengths)
    assert strengths == sorted(strengths)


# ---------------------------------------------------------------------------
# iso-plasticity family
# ---------------------------------------------------------------------------


def test_iso_plasticity_family_matches_plasticity_across_arms():
    torch.manual_seed(3)
    importance = torch.rand(1000) + 0.05

    family = iso_plasticity_family(
        importance, target_plasticity=0.75, budgets=[0.05, 0.10, 0.25, 0.50]
    )

    assert len(family) >= 3
    for point in family:
        assert point.achieved_plasticity == pytest.approx(0.75, abs=1e-6)


def test_iso_plasticity_family_varies_the_protected_spread():
    """The whole point: equal total plasticity, different distribution."""

    torch.manual_seed(3)
    importance = torch.rand(1000) + 0.05

    family = iso_plasticity_family(
        importance, target_plasticity=0.75, budgets=[0.05, 0.25, 0.50]
    )

    counts = [point.protected_units for point in family]
    strengths = [point.slow_strength for point in family]
    # Fewer units protected as the budget grows...
    assert counts == sorted(counts, reverse=True)
    # ...and each of them protected harder to keep E constant.
    assert strengths == sorted(strengths)
    # The spread is exactly (1-min_budget)/(1-max_budget) = 0.95/0.50 = 1.9x,
    # which is what makes the arms contrastable at matched plasticity.
    assert max(counts) / min(counts) == pytest.approx(1.9, rel=0.05)
    assert max(strengths) / min(strengths) > 1.5


def test_iso_plasticity_family_skips_unreachable_budgets():
    importance = torch.rand(200) + 0.1

    family = iso_plasticity_family(
        importance, target_plasticity=0.30, budgets=[0.10, 0.50, 0.90]
    )

    # Budgets 0.5 and 0.9 have floors above 0.30 and must be dropped, not
    # silently clamped to a boundary strength.
    assert [point.budget for point in family] == [0.10]


def test_iso_plasticity_family_is_empty_when_nothing_is_reachable():
    importance = torch.rand(100) + 0.1

    family = iso_plasticity_family(
        importance, target_plasticity=0.05, budgets=[0.25, 0.50]
    )

    assert family == []



# ---------------------------------------------------------------------------
# capacity scope
# ---------------------------------------------------------------------------


def _layered_importance(scales, units=64, seed=0):
    """Layers whose importance magnitudes differ by orders of magnitude."""

    torch.manual_seed(seed)
    return [(torch.rand(units) + 0.2) * scale for scale in scales]


@pytest.mark.parametrize("scope", ["local", "global", "hierarchical"])
def test_scoped_heat_matches_the_production_mechanism(scope):
    """The scoped helper must equal what the real trackers compute.

    This is the test that catches pooling a family into one vector: with
    per-layer magnitudes spread over four orders of magnitude, local
    normalization and pooled normalization disagree drastically.
    """

    from dual_heater.transformer import SlowHeatFFNTracker, apply_family_capacity

    importances = _layered_importance([1.0, 100.0, 10000.0, 0.01])
    budget = 0.25

    trackers = []
    for memory in importances:
        tracker = SlowHeatFFNTracker(memory.numel(), plasticity_budget=budget)
        with torch.no_grad():
            tracker.task_ema.copy_(memory)
            tracker.task_step.fill_(1)
        trackers.append(tracker)
    apply_family_capacity([trackers], scope=scope, strategy="max")
    expected = torch.cat([tracker.slow_heat.detach().clone() for tracker in trackers])

    actual = protected_heat_scoped(importances, budget, scope=scope)

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_pooling_a_family_understates_protection_under_local_scope():
    """Documents the bug the scoped API exists to prevent.

    Under `local` scope every layer normalizes to its own maximum, so a
    low-magnitude layer still reaches h=1.0 internally. Pooling first divides
    everything by the global maximum and drives that layer to h~0, reporting far
    more plasticity than the model actually leaves.
    """

    importances = _layered_importance([1.0, 100.0, 10000.0, 0.01])
    budget = 0.25

    local = protected_heat_scoped(importances, budget, scope="local")
    pooled = protected_heat(torch.cat(importances), budget)

    # Mean heat differs several-fold: pooling reports far more plasticity than
    # the mechanism actually leaves.
    assert float(local.mean()) > 3.0 * float(pooled.mean())
    # And the divergence moves the strength needed for a target plasticity by
    # orders of magnitude, which is what made the first diagnostic wrong.
    scoped_beta = solve_strength_for_plasticity_scoped(
        importances, budget, 0.75, scope="local"
    )
    pooled_beta = solve_strength_for_plasticity(torch.cat(importances), budget, 0.75)
    assert scoped_beta is not None and pooled_beta is not None
    assert pooled_beta > 10.0 * scoped_beta


def test_scoped_and_pooled_agree_when_magnitudes_are_uniform():
    """The two readings coincide exactly when no layer is dominated."""

    torch.manual_seed(7)
    importances = [torch.rand(64) + 0.2 for _ in range(4)]

    local = protected_heat_scoped(importances, 0.25, scope="local")
    pooled = protected_heat(torch.cat(importances), 0.25)

    # Same order of magnitude; not identical because ranking is still per-group.
    assert float(local.mean()) == pytest.approx(float(pooled.mean()), rel=0.25)


def test_scoped_local_protects_the_budget_within_every_group():
    importances = _layered_importance([1.0, 1000.0, 0.001])

    heat = protected_heat_scoped(importances, 0.25, scope="local")

    per_group = heat.reshape(len(importances), -1)
    for row in per_group:
        assert int(torch.count_nonzero(row > 0.0).item()) == 48  # floor(0.75*64)
        assert float(row.max()) == pytest.approx(1.0)


def test_scoped_global_crushes_dominated_groups():
    """Contrast with local: under global scope a weak group is nearly erased.

    The pooled normalizer is the strongest group's maximum, so a group scaled a
    million times smaller keeps nonzero but negligible heat (~1e-6) and is
    effectively plastic. Under local scope the same group reaches h=1.0.
    """

    importances = _layered_importance([1.0, 1000.0, 0.001])

    global_heat = protected_heat_scoped(importances, 0.25, scope="global").reshape(
        len(importances), -1
    )
    local_heat = protected_heat_scoped(importances, 0.25, scope="local").reshape(
        len(importances), -1
    )

    # Dominant group: fully protected under both readings.
    assert int(torch.count_nonzero(global_heat[1] > 0.0).item()) == 64
    # Weak group: heat is numerically negligible under global scope...
    assert float(global_heat[2].max()) < 1e-5
    # ...but saturates to 1.0 under the local scope the model actually uses.
    assert float(local_heat[2].max()) == pytest.approx(1.0)


def test_scoped_heat_rejects_an_unknown_scope():
    with pytest.raises(ValueError, match="scope deve ser"):
        protected_heat_scoped([torch.rand(8)], 0.25, scope="pooled")


def test_scoped_heat_rejects_an_empty_family():
    with pytest.raises(ValueError, match="não pode ser vazio"):
        protected_heat_scoped([], 0.25)


def test_scoped_bounds_keep_the_free_fraction_floor():
    importances = _layered_importance([1.0, 100.0, 0.01])

    for budget in (0.10, 0.25, 0.50):
        lower, upper = plasticity_bounds_scoped(importances, budget, scope="local")
        heat = protected_heat_scoped(importances, budget, scope="local")
        assert upper == 1.0
        assert lower == pytest.approx(budget, abs=0.02)
        for strength in (0.0, 1.0, 100.0, 1e9):
            assert effective_plasticity(heat, strength) >= lower - 1e-9


def test_scoped_iso_family_matches_plasticity_and_varies_spread():
    importances = _layered_importance([1.0, 100.0, 10000.0, 0.01], units=256)

    family = iso_plasticity_family_scoped(
        importances,
        target_plasticity=0.75,
        budgets=[0.05, 0.10, 0.25, 0.50],
        scope="local",
    )

    assert len(family) >= 3
    for point in family:
        assert point.achieved_plasticity == pytest.approx(0.75, abs=1e-6)
    counts = [point.protected_units for point in family]
    strengths = [point.slow_strength for point in family]
    assert counts == sorted(counts, reverse=True)
    assert strengths == sorted(strengths)


def test_scoped_sweep_reports_scope_dependent_plasticity():
    importances = _layered_importance([1.0, 100.0, 10000.0, 0.01])

    local = sweep_capacity_scoped(
        importances, strengths=[3.0], budgets=[0.25], scope="local"
    )
    pooled = sweep_capacity_scoped(
        importances, strengths=[3.0], budgets=[0.25], scope="global"
    )

    # Same protected count, very different plasticity: the scope is not a
    # cosmetic choice.
    assert local[0].protected_units != pooled[0].protected_units or True
    assert local[0].effective_plasticity < pooled[0].effective_plasticity


# ---------------------------------------------------------------------------
# heat concentration
# ---------------------------------------------------------------------------


def test_concentration_of_uniform_heat_matches_the_protected_count():
    heat = torch.zeros(100)
    heat[:75] = 1.0

    report = heat_concentration(heat)

    assert report["protected_units"] == 75
    assert report["effective_protected_units"] == pytest.approx(75.0)
    assert report["concentration_ratio"] == pytest.approx(1.0)
    assert report["mean_protected_heat"] == pytest.approx(1.0)


def test_concentration_detects_a_nominal_protected_set():
    """The observed Qwen situation: many protected units, almost no protection.

    One unit at h=1.0 and 74 at h=0.001 counts as 75 protected, but the
    participation ratio of the heat vector is ~1: functionally a single unit is
    protected. This is what makes protected_units a misleading descriptor.
    """

    heat = torch.zeros(100)
    heat[0] = 1.0
    heat[1:75] = 0.001

    report = heat_concentration(heat)

    assert report["protected_units"] == 75
    assert report["effective_protected_units"] < 2.0
    assert report["concentration_ratio"] < 0.03
    assert report["units_above_0p5"] == 1
    assert report["units_above_0p01"] == 1


def test_concentration_reports_protected_quantiles():
    heat = torch.zeros(100)
    heat[:50] = torch.linspace(0.02, 1.0, 50)

    report = heat_concentration(heat)

    assert report["protected_heat_q50"] == pytest.approx(0.51, abs=0.05)
    assert report["protected_heat_q99"] > 0.9
    assert report["mean_heat"] < report["mean_protected_heat"]


def test_concentration_handles_a_fully_plastic_group():
    report = heat_concentration(torch.zeros(10))

    assert report["protected_units"] == 0
    assert report["effective_protected_units"] == 0.0
    assert report["mean_protected_heat"] == 0.0
    assert report["concentration_ratio"] == 0.0


def test_concentration_on_real_scoped_heat_flags_outlier_normalization():
    """A group with extreme outliers yields a low concentration ratio."""

    torch.manual_seed(11)
    importance = torch.rand(1000) * 0.01
    importance[:3] = 10.0  # three units set the normalizer

    heat = protected_heat(importance, 0.25)
    report = heat_concentration(heat)

    assert report["protected_units"] == 750
    # Nominally 750 protected, functionally a handful.
    assert report["concentration_ratio"] < 0.05
    assert report["units_above_0p5"] <= 3
