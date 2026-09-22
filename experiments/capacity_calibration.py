"""Capacity calibration arithmetic for Functional SlowHeat.

These functions are deliberately free of any model, dataset or accuracy
dependency: they map an importance vector plus candidate hyperparameters to
mechanism-level quantities. That separation is what lets a protocol declare a
selection criterion *before* any accuracy is observed.

Definitions
-----------
Given persistent importance memory ``m`` over ``N`` units, a plasticity budget
``b`` and a protection strength ``beta``:

``protected_heat(m, b)``
    Reproduces :meth:`_SlowHeatImportanceMixin._apply_capacity_budget` without
    mutating any module: the top ``floor((1-b)*N)`` positive entries keep
    ``m_i / max(selected)``, everything else is zero.

``effective_plasticity(h, beta)``
    ``mean_i 1 / (1 + beta * h_i)``. This is the quantity that actually governs
    how much the optimizer can still move, and it is NOT the same as the
    fraction of unprotected units: a unit with ``h = 0.01`` counts as protected
    but keeps 97% of its learning rate at ``beta = 3``. Reporting only the
    protected count overstates the loss of plasticity.

``participation_ratio(m)``
    ``(sum m)^2 / sum(m^2)``, the effective number of units carrying the
    importance signal. Compared against the free pool size ``b*N`` it says
    whether a budget is cutting into signal or only trimming noise.

Capacity scope is part of the mechanism
---------------------------------------
Every function taking a single ``importance`` vector answers a question about
ONE normalization group. Concatenating several layers into one vector and
calling these functions silently answers a different question than the model
asks, because ``protected_heat`` divides by a single maximum.

Under the production default ``capacity_scope="local"`` each layer is ranked and
normalized independently, so a layer whose importance magnitudes are orders of
magnitude smaller than its neighbours still gets ``h`` up to 1.0 inside itself.
Pooling first assigns that layer ``h ~ 0`` and drastically understates how much
protection the mechanism applies: on the observed Qwen importance the mean heat
differs by roughly 8x between the two readings, which shifts the ``beta`` that
reaches a target plasticity by about two orders of magnitude.

Use the ``*_scoped`` functions, which take the per-group importance vectors and
the scope, for anything that must match a real run.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor


def protected_count(unit_count: int, budget: float) -> int:
    """Number of units the budget allows to be protected."""

    if unit_count < 1:
        raise ValueError("unit_count deve ser >= 1")
    if not 0.0 <= budget <= 1.0:
        raise ValueError("budget deve estar em [0, 1]")
    return math.floor((1.0 - budget) * unit_count + 1e-12)


def protected_heat(
    importance: Tensor,
    budget: float,
    *,
    importance_eps: float = 1e-8,
) -> Tensor:
    """Recompute the protection vector for a candidate budget, without mutating.

    Mirrors the production capacity rule exactly so a sweep over ``budget``
    reports what the mechanism would really do, not an approximation.
    """

    if importance.ndim != 1:
        raise ValueError("importance deve ser um vetor 1-D")
    memory = importance.detach().to(dtype=torch.float32)
    heat = torch.zeros_like(memory)
    allowed = protected_count(memory.numel(), budget)
    positive = int(torch.count_nonzero(memory > 0.0).item())
    protected = min(allowed, positive)
    if protected == 0:
        return heat
    order = torch.argsort(memory, descending=True, stable=True)
    indices = order[:protected]
    selected = memory[indices]
    heat[indices] = selected / selected.max().clamp_min(importance_eps)
    return heat


def effective_plasticity(heat: Tensor, slow_strength: float) -> float:
    """Mean surviving learning-rate fraction under ``1 / (1 + beta * h)``."""

    if slow_strength < 0.0:
        raise ValueError("slow_strength deve ser >= 0")
    scales = 1.0 / (1.0 + slow_strength * heat.detach().to(dtype=torch.float32))
    return float(scales.mean().item())


def participation_ratio(importance: Tensor) -> float:
    """Effective number of units carrying the importance signal."""

    memory = importance.detach().to(dtype=torch.float32)
    total = float(memory.sum().item())
    squared = float((memory * memory).sum().item())
    if squared <= 0.0:
        return 0.0
    return total * total / squared


def positive_fraction(importance: Tensor) -> float:
    """Fraction of units with any recorded utility (signal density)."""

    memory = importance.detach()
    return float((memory > 0.0).to(dtype=torch.float32).mean().item())


def jaccard(first: Tensor, second: Tensor) -> float:
    """Overlap between two boolean protection masks."""

    left = first.detach().bool()
    right = second.detach().bool()
    union = int(torch.count_nonzero(left | right).item())
    if union == 0:
        return 1.0
    intersection = int(torch.count_nonzero(left & right).item())
    return intersection / union


def importance_profile(
    importance: Tensor,
    *,
    quantiles: Sequence[float] = (0.5, 0.9, 0.99, 0.999),
    top_counts: Sequence[int] = (1, 10, 100),
) -> dict[str, float]:
    """Shape of a raw importance vector, before any budget or normalization.

    ``heat_concentration`` describes the *protected* vector, which is already
    normalized by the selected maximum and truncated by the budget. When a
    layer's concentration ratio is anomalous, the question is whether the raw
    signal itself is degenerate, and that has to be read before the budget
    touches it.

    ``top_{n}_mass`` is the share of total importance held by the ``n`` largest
    units. A layer where ``top_1_mass`` is near 1 has a single unit carrying the
    layer; one where ``top_100_mass`` is near 1 has a small group. The two cases
    look identical in the participation ratio alone.
    """

    if importance.ndim != 1:
        raise ValueError("importance deve ser um vetor 1-D")
    # CPU for the same reason as `dominant_unit_overlap`, plus `torch.quantile`
    # carries a tensor-size limit on CUDA that a pooled vector can exceed.
    values = importance.detach().to(device="cpu", dtype=torch.float32)
    units = values.numel()
    if units == 0:
        raise ValueError("importance não pode ser vazia")
    total = float(values.sum().item())
    report: dict[str, float] = {
        "units": float(units),
        "total": total,
        "mean": float(values.mean().item()),
        "max": float(values.max().item()),
        "participation_ratio": participation_ratio(values),
        "positive_fraction": positive_fraction(values),
    }
    for quantile in quantiles:
        key = f"q{quantile:g}".replace(".", "p")
        report[key] = float(torch.quantile(values, quantile).item())
    ordered = torch.sort(values, descending=True).values
    for count in top_counts:
        taken = min(count, units)
        mass = float(ordered[:taken].sum().item())
        report[f"top_{count}_mass"] = mass / total if total > 0.0 else 0.0
    return report


def dominant_unit_overlap(
    first: Tensor,
    second: Tensor,
    *,
    k: int,
) -> dict[str, float]:
    """Agreement between the top-``k`` units of two importance vectors.

    Answers a question ``heat_concentration`` cannot: when a layer concentrates
    its importance on a handful of units, are they the *same* units across
    tasks? If they are, the concentration is a stable property of the layer and
    protecting it transfers; if they are not, each task claims a different
    handful and the protected set is rewritten at every boundary.

    ``overlap`` is the fraction of the top-``k`` set shared by both vectors and
    ``jaccard`` the symmetric version. Both are compared against ``chance``,
    ``k / N``, which is the expected ``overlap`` for independent rankings. An
    overlap near chance means the ranking carries no cross-task information.

    Ranking uses a stable descending sort, matching :func:`protected_heat`, so
    ties resolve by index in both vectors and cannot manufacture disagreement.
    """

    if first.ndim != 1 or second.ndim != 1:
        raise ValueError("as importâncias devem ser vetores 1-D")
    if first.numel() != second.numel():
        raise ValueError("as importâncias devem ter o mesmo número de unidades")
    units = first.numel()
    if units == 0:
        raise ValueError("as importâncias não podem ser vazias")
    if k < 1:
        raise ValueError("k deve ser >= 1")
    top = min(k, units)
    # Move to CPU first: callers pass live CUDA tensors, and mixing a CPU
    # membership mask with CUDA indices raises. Everything below is O(units)
    # bookkeeping, so the copy costs nothing worth optimizing.
    left = first.detach().to(device="cpu", dtype=torch.float32)
    right = second.detach().to(device="cpu", dtype=torch.float32)
    left_top = torch.argsort(left, descending=True, stable=True)[:top]
    right_top = torch.argsort(right, descending=True, stable=True)[:top]
    membership = torch.zeros(units, dtype=torch.bool)
    membership[left_top] = True
    intersection = int(membership[right_top].sum().item())
    union = 2 * top - intersection
    return {
        "k": float(top),
        "units": float(units),
        "intersection": float(intersection),
        "overlap": intersection / top,
        "jaccard": intersection / union if union else 1.0,
        "chance": top / units,
    }


@dataclass(frozen=True)
class CapacityPoint:
    """One (beta, budget) candidate evaluated on a fixed importance vector."""

    slow_strength: float
    budget: float
    protected_units: int
    protected_fraction: float
    effective_plasticity: float


def sweep_capacity(
    importance: Tensor,
    *,
    strengths: list[float],
    budgets: list[float],
    importance_eps: float = 1e-8,
) -> list[CapacityPoint]:
    """Evaluate every (beta, budget) pair analytically on one importance vector.

    No training is repeated: the importance memory is fixed, so the whole grid
    costs a few sorts. This is why the criterion can be evaluated over a wide
    grid without spending a run per candidate.
    """

    points: list[CapacityPoint] = []
    for budget in budgets:
        heat = protected_heat(importance, budget, importance_eps=importance_eps)
        protected = int(torch.count_nonzero(heat > 0.0).item())
        for strength in strengths:
            points.append(
                CapacityPoint(
                    slow_strength=strength,
                    budget=budget,
                    protected_units=protected,
                    protected_fraction=protected / heat.numel(),
                    effective_plasticity=effective_plasticity(heat, strength),
                )
            )
    return points


def select_by_declared_criterion(
    points: list[CapacityPoint],
    *,
    budget: float,
    minimum_effective_plasticity: float,
) -> CapacityPoint:
    """Pick the strongest protection that still meets a pre-declared floor.

    The criterion is "largest ``beta`` such that effective plasticity at the
    chosen ``budget`` stays at or above ``minimum_effective_plasticity``". It
    reads only mechanism quantities, never accuracy, so it can be fixed before
    the first run and audited afterwards.

    Raises when no candidate qualifies instead of silently relaxing the floor.
    """

    if not 0.0 < minimum_effective_plasticity <= 1.0:
        raise ValueError("o piso de plasticidade deve estar em (0, 1]")
    candidates = [
        point
        for point in points
        if point.budget == budget
        and point.effective_plasticity >= minimum_effective_plasticity
    ]
    if not candidates:
        raise ValueError(
            "nenhum beta satisfaz o piso declarado de plasticidade efetiva; "
            "aumente o budget ou reduza o piso ANTES de observar acurácia"
        )
    return max(candidates, key=lambda point: point.slow_strength)


def free_fraction(importance: Tensor, budget: float) -> float:
    """Fraction of units the budget leaves completely unprotected."""

    heat = protected_heat(importance, budget)
    return float((heat == 0.0).to(dtype=torch.float32).mean().item())


def plasticity_bounds(importance: Tensor, budget: float) -> tuple[float, float]:
    """Reachable interval of effective plasticity at a fixed budget.

    Effective plasticity is bounded below by the free fraction and above by 1:

        E(beta, b) = (1/N) * sum_i 1/(1 + beta*h_i)

    Free units have ``h_i = 0`` and contribute exactly 1 each, so

        E(beta, b) >= (N - P) / N   for every beta,

    where ``P`` is the protected count. The bound is tight: as ``beta -> inf``
    every protected term vanishes and ``E -> (N - P)/N``.

    Consequence for experiment design: ``beta`` and ``budget`` are NOT
    interchangeable. The budget sets a hard floor on how much plasticity can be
    removed, and ``beta`` only moves ``E`` inside ``[floor, 1]``. A target
    plasticity ``E*`` is therefore unreachable at any budget with
    ``free_fraction >= E*``.
    """

    lower = free_fraction(importance, budget)
    return lower, 1.0


def solve_strength_for_plasticity(
    importance: Tensor,
    budget: float,
    target_plasticity: float,
    *,
    maximum_strength: float = 1e6,
    tolerance: float = 1e-9,
    iterations: int = 200,
) -> float | None:
    """Invert ``E(beta, b) = target`` for ``beta`` by bisection.

    ``E`` is continuous and monotonically decreasing in ``beta`` with
    ``E(0) = 1``, so the root is unique when the target lies inside the
    reachable interval. Returns ``None`` when the target is unreachable at this
    budget (see :func:`plasticity_bounds`) rather than returning a boundary
    value that would silently misreport the configuration.
    """

    if not 0.0 < target_plasticity <= 1.0:
        raise ValueError("target_plasticity deve estar em (0, 1]")
    if maximum_strength <= 0.0:
        raise ValueError("maximum_strength deve ser > 0")
    heat = protected_heat(importance, budget)
    if effective_plasticity(heat, 0.0) < target_plasticity - tolerance:
        return None
    if effective_plasticity(heat, maximum_strength) > target_plasticity + tolerance:
        return None
    low, high = 0.0, maximum_strength
    for _ in range(iterations):
        middle = 0.5 * (low + high)
        if effective_plasticity(heat, middle) > target_plasticity:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


@dataclass(frozen=True)
class IsoPlasticityPoint:
    """One arm of an iso-plasticity family: matched ``E``, different spread."""

    target_plasticity: float
    budget: float
    slow_strength: float
    protected_units: int
    protected_fraction: float
    achieved_plasticity: float


def iso_plasticity_family(
    importance: Tensor,
    *,
    target_plasticity: float,
    budgets: list[float],
    maximum_strength: float = 1e6,
) -> list[IsoPlasticityPoint]:
    """Build configurations with equal effective plasticity but different spread.

    This is the cost-matched axis for an ablation. Holding ``E`` fixed removes
    the confound that stronger protection also means less learning capacity:
    every arm gives the optimizer the same total freedom, and only the
    *distribution* of that freedom changes. A low budget protects many units
    weakly; a high budget protects few units hard.

    Budgets whose floor already exceeds the target are skipped, because no
    ``beta`` can reach the target there.
    """

    points: list[IsoPlasticityPoint] = []
    for budget in budgets:
        strength = solve_strength_for_plasticity(
            importance,
            budget,
            target_plasticity,
            maximum_strength=maximum_strength,
        )
        if strength is None:
            continue
        heat = protected_heat(importance, budget)
        protected = int(torch.count_nonzero(heat > 0.0).item())
        points.append(
            IsoPlasticityPoint(
                target_plasticity=target_plasticity,
                budget=budget,
                slow_strength=strength,
                protected_units=protected,
                protected_fraction=protected / heat.numel(),
                achieved_plasticity=effective_plasticity(heat, strength),
            )
        )
    return points


# ---------------------------------------------------------------------------
# scope-aware variants
#
# These take the per-group importance vectors (one per layer/tracker) instead of
# a single pooled vector, so they reproduce what the model actually computes for
# a given `capacity_scope`.
# ---------------------------------------------------------------------------

CapacityScope = Literal["local", "global", "hierarchical"]


def protected_heat_scoped(
    importances: Sequence[Tensor],
    budget: float,
    *,
    scope: CapacityScope = "local",
    importance_eps: float = 1e-8,
) -> Tensor:
    """Protection vector for a whole family under an explicit capacity scope.

    ``local`` normalizes each group independently (the production default);
    ``global`` and ``hierarchical`` pool the family first. Returns the
    concatenated heat in input order so it can be fed to
    :func:`effective_plasticity`.

    This is the function to use when the number must match a real run. The
    pooled single-vector helpers answer a different question whenever groups
    have different importance magnitudes.
    """

    if not importances:
        raise ValueError("importances não pode ser vazio")
    if scope == "local":
        return torch.cat(
            [
                protected_heat(memory, budget, importance_eps=importance_eps)
                for memory in importances
            ]
        )
    if scope == "global":
        return protected_heat(
            torch.cat([memory.detach() for memory in importances]),
            budget,
            importance_eps=importance_eps,
        )
    if scope == "hierarchical":
        return _hierarchical_heat(importances, budget, importance_eps=importance_eps)
    raise ValueError("scope deve ser 'local', 'global' ou 'hierarchical'")


@torch.no_grad()
def _hierarchical_heat(
    importances: Sequence[Tensor],
    budget: float,
    *,
    importance_eps: float,
) -> Tensor:
    """Mirror ``apply_hierarchical_capacity``: global quota, local ranking."""

    memories = [memory.detach().to(dtype=torch.float32) for memory in importances]
    capacities = [
        int(torch.count_nonzero(memory > 0.0).item()) for memory in memories
    ]
    total_units = sum(memory.numel() for memory in memories)
    protected = min(
        math.floor((1.0 - budget) * total_units + 1e-12), sum(capacities)
    )
    heats = [torch.zeros_like(memory) for memory in memories]
    if protected == 0:
        return torch.cat(heats)
    weights = [float(memory.mean()) for memory in memories]
    weight_sum = sum(weights)
    if weight_sum <= 0.0:
        weights = [float(capacity) for capacity in capacities]
        weight_sum = sum(weights)
    ideals = [protected * weight / weight_sum for weight in weights]
    quotas = [
        min(capacity, math.floor(ideal))
        for capacity, ideal in zip(capacities, ideals, strict=True)
    ]
    remaining = protected - sum(quotas)
    priority = sorted(
        range(len(memories)),
        key=lambda index: (
            ideals[index] - math.floor(ideals[index]),
            weights[index],
            -index,
        ),
        reverse=True,
    )
    while remaining:
        progressed = False
        for index in priority:
            if quotas[index] < capacities[index]:
                quotas[index] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            raise RuntimeError("não foi possível distribuir o budget hierárquico")
    maxima = [
        memory[torch.argsort(memory, descending=True, stable=True)[:quota]].max()
        for memory, quota in zip(memories, quotas, strict=True)
        if quota
    ]
    normalizer = torch.stack(maxima).max().clamp_min(importance_eps)
    for heat, memory, quota in zip(heats, memories, quotas, strict=True):
        if quota == 0:
            continue
        selected = torch.argsort(memory, descending=True, stable=True)[:quota]
        heat[selected] = memory[selected] / normalizer
    return torch.cat(heats)


def free_fraction_scoped(
    importances: Sequence[Tensor],
    budget: float,
    *,
    scope: CapacityScope = "local",
) -> float:
    """Fraction of units left fully plastic under an explicit scope."""

    heat = protected_heat_scoped(importances, budget, scope=scope)
    return float((heat == 0.0).to(dtype=torch.float32).mean().item())


def plasticity_bounds_scoped(
    importances: Sequence[Tensor],
    budget: float,
    *,
    scope: CapacityScope = "local",
) -> tuple[float, float]:
    """Reachable plasticity interval under an explicit scope."""

    return free_fraction_scoped(importances, budget, scope=scope), 1.0


def solve_strength_for_plasticity_scoped(
    importances: Sequence[Tensor],
    budget: float,
    target_plasticity: float,
    *,
    scope: CapacityScope = "local",
    maximum_strength: float = 1e6,
    tolerance: float = 1e-9,
    iterations: int = 200,
) -> float | None:
    """Invert ``E(beta, b) = target`` under an explicit capacity scope."""

    if not 0.0 < target_plasticity <= 1.0:
        raise ValueError("target_plasticity deve estar em (0, 1]")
    if maximum_strength <= 0.0:
        raise ValueError("maximum_strength deve ser > 0")
    heat = protected_heat_scoped(importances, budget, scope=scope)
    if effective_plasticity(heat, 0.0) < target_plasticity - tolerance:
        return None
    if effective_plasticity(heat, maximum_strength) > target_plasticity + tolerance:
        return None
    low, high = 0.0, maximum_strength
    for _ in range(iterations):
        middle = 0.5 * (low + high)
        if effective_plasticity(heat, middle) > target_plasticity:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def iso_plasticity_family_scoped(
    importances: Sequence[Tensor],
    *,
    target_plasticity: float,
    budgets: list[float],
    scope: CapacityScope = "local",
    maximum_strength: float = 1e6,
) -> list[IsoPlasticityPoint]:
    """Iso-plasticity arms computed under the scope the model really uses."""

    points: list[IsoPlasticityPoint] = []
    for budget in budgets:
        strength = solve_strength_for_plasticity_scoped(
            importances,
            budget,
            target_plasticity,
            scope=scope,
            maximum_strength=maximum_strength,
        )
        if strength is None:
            continue
        heat = protected_heat_scoped(importances, budget, scope=scope)
        protected = int(torch.count_nonzero(heat > 0.0).item())
        points.append(
            IsoPlasticityPoint(
                target_plasticity=target_plasticity,
                budget=budget,
                slow_strength=strength,
                protected_units=protected,
                protected_fraction=protected / heat.numel(),
                achieved_plasticity=effective_plasticity(heat, strength),
            )
        )
    return points


def sweep_capacity_scoped(
    importances: Sequence[Tensor],
    *,
    strengths: list[float],
    budgets: list[float],
    scope: CapacityScope = "local",
    importance_eps: float = 1e-8,
) -> list[CapacityPoint]:
    """Analytic (beta, budget) grid under an explicit capacity scope."""

    points: list[CapacityPoint] = []
    for budget in budgets:
        heat = protected_heat_scoped(
            importances, budget, scope=scope, importance_eps=importance_eps
        )
        protected = int(torch.count_nonzero(heat > 0.0).item())
        for strength in strengths:
            points.append(
                CapacityPoint(
                    slow_strength=strength,
                    budget=budget,
                    protected_units=protected,
                    protected_fraction=protected / heat.numel(),
                    effective_plasticity=effective_plasticity(heat, strength),
                )
            )
    return points


def heat_concentration(heat: Tensor, *, thresholds: Sequence[float] = (0.5, 0.1, 0.01)) -> dict[str, float]:
    """Describe how protection mass is distributed across protected units.

    ``protected_units`` counts every unit with nonzero heat, but heat is
    normalized by the group maximum, so a few extreme units can push the rest
    of the protected set to negligible values. When that happens the protected
    count is a poor description of the mechanism: varying it barely changes
    behaviour because almost all protection sits on a handful of units.

    ``effective_protected_units`` is the participation ratio of the heat vector
    itself, ``(sum h)^2 / sum(h^2)``. Compare it against ``protected_units``:
    a large gap means the protected set is nominal rather than functional.
    """

    values = heat.detach().to(dtype=torch.float32)
    protected = int(torch.count_nonzero(values > 0.0).item())
    report: dict[str, float] = {
        "protected_units": float(protected),
        "effective_protected_units": participation_ratio(values),
        "mean_heat": float(values.mean().item()),
        "max_heat": float(values.max().item()),
        "mean_protected_heat": (
            float(values[values > 0.0].mean().item()) if protected else 0.0
        ),
    }
    if protected:
        selected = values[values > 0.0]
        for quantile in (0.5, 0.9, 0.99):
            report[f"protected_heat_q{int(quantile * 100)}"] = float(
                torch.quantile(selected, quantile).item()
            )
    for threshold in thresholds:
        key = f"units_above_{threshold}".replace(".", "p")
        report[key] = float(torch.count_nonzero(values >= threshold).item())
    report["concentration_ratio"] = (
        report["effective_protected_units"] / protected if protected else 0.0
    )
    return report
