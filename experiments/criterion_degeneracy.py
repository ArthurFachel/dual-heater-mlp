"""Cross-arm protection overlap: the second mandatory metric of Section F.

Section F of goals/protocol_importance_criterion_ablation.md declares two
obligatory diagnostics for the importance-criterion ablation:

1. the variance of the normalized ranking, and
2. the Jaccard overlap of the protected top-k sets between ``magnitude`` and
   ``random``, and between ``magnitude`` and ``functional``.

The frozen interpretation rule depends on the second: an overlap above
``DEGENERACY_OVERLAP_THRESHOLD`` between ``magnitude`` and ``random`` declares
the ``magnitude`` arm degenerate, and any tie it produces is reported as
inconclusive rather than as evidence that the criterion is irrelevant.

The 2026-09-28 confirmatory run emitted only (1): ``run_split_clinc150`` calls
``aggregate_ranking_degeneracy`` without a ``reference``, so ``top_k_overlap``
was never computed and the published artefacts cannot answer the falsifier's
second half. The overlap is a cross-arm quantity -- it compares two separate
runs -- so it cannot be produced from inside a single run at all.

This module recovers it from the saved checkpoints. ``slow_heat`` is the vector
the plasticity mask reads, so its support IS the protected set that was used,
not a reconstruction of it.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from experiments.artifacts import read_torch_checkpoint

#: Section F, frozen before the first seed.
DEGENERACY_OVERLAP_THRESHOLD = 0.8

#: Arm name -> (condition directory, method directory) under the run root.
ABLATION_ARM_PATHS: dict[str, tuple[str, str]] = {
    "random": ("slowheat_random_hard", "slowheat_ffn_attention"),
    "magnitude": ("slowheat_magnitude_hard", "slowheat_ffn_attention"),
    "functional": ("slowheat_hard", "slowheat_ffn_attention"),
}

#: The two edges of the ordered hypothesis, as declared in Section F.
OVERLAP_PAIRS: dict[str, tuple[str, str]] = {
    "magnitude_vs_random": ("magnitude", "random"),
    "magnitude_vs_functional": ("magnitude", "functional"),
}

_TRACKER_PREFIXES = ("ffn_trackers.", "attention_trackers.")
_SUFFIX = ".slow_heat"


def protected_masks(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Protected-unit boolean masks, keyed by tracker, from a model state dict.

    Only FFN and attention trackers are read: those are the two families the
    ablation's arms differ in, and the ones
    ``aggregate_ranking_degeneracy`` was called on by the runner.
    """

    masks: dict[str, Tensor] = {}
    for key, value in state.items():
        if not key.endswith(_SUFFIX):
            continue
        if not key.startswith(_TRACKER_PREFIXES):
            continue
        masks[key[: -len(_SUFFIX)]] = value.detach().flatten() > 0.0
    if not masks:
        raise ValueError(
            "estado não contém trackers de FFN/atenção com slow_heat"
        )
    return masks


def _jaccard(left: Tensor, right: Tensor) -> float:
    union = int(torch.count_nonzero(left | right).item())
    if union == 0:
        return 1.0
    return int(torch.count_nonzero(left & right).item()) / union


def arm_overlap(
    left_state: Mapping[str, Tensor],
    right_state: Mapping[str, Tensor],
) -> dict[str, Any]:
    """Jaccard overlap of the protected sets of two arms, per tracker.

    The mean is taken over trackers, not over units: ``_apply_capacity_budget``
    ranks inside each tracker, so a unit-weighted average would let one wide
    FFN tracker hide a degenerate narrow attention tracker.

    Capacity is reported, never assumed. BERT consolidates under a per-family
    scope, so two arms may protect different counts inside a given tracker
    while the family total matches; only the total is evidence about pairing.
    """

    left = protected_masks(left_state)
    right = protected_masks(right_state)
    if set(left) != set(right):
        raise ValueError("os dois braços devem expor os mesmos trackers")

    per_tracker: dict[str, float] = {}
    left_counts: dict[str, int] = {}
    right_counts: dict[str, int] = {}
    for name in sorted(left):
        if left[name].numel() != right[name].numel():
            raise ValueError(f"tracker {name} com tamanhos diferentes entre braços")
        per_tracker[name] = _jaccard(left[name], right[name])
        left_counts[name] = int(torch.count_nonzero(left[name]).item())
        right_counts[name] = int(torch.count_nonzero(right[name]).item())

    left_total = sum(left_counts.values())
    right_total = sum(right_counts.values())
    return {
        "tracker_count": len(per_tracker),
        "mean_jaccard": statistics.fmean(per_tracker.values()),
        "min_jaccard": min(per_tracker.values()),
        "max_jaccard": max(per_tracker.values()),
        "per_tracker": per_tracker,
        "left_protected_total": left_total,
        "right_protected_total": right_total,
        "left_protected_per_tracker": left_counts,
        "right_protected_per_tracker": right_counts,
        "capacity_matched": left_total == right_total,
    }


def load_arm_states(
    run_dir: str | Path,
    seed: int,
    arms: Mapping[str, tuple[str, str]] | None = None,
) -> dict[str, dict[str, Tensor]]:
    """Read one seed's per-arm model states from a completed ablation run."""

    root = Path(run_dir) / f"seed_{seed}"
    matrix = ABLATION_ARM_PATHS if arms is None else arms
    states: dict[str, dict[str, Tensor]] = {}
    for arm, (condition, method) in matrix.items():
        path = root / condition / method / "checkpoint.pt"
        if not path.is_file():
            raise FileNotFoundError(path)
        states[arm] = read_torch_checkpoint(path)["model"]
    return states


def summarize_degeneracy_overlap(
    by_seed: Mapping[int, Mapping[str, Mapping[str, Tensor]]],
) -> dict[str, Any]:
    """Apply Section F's frozen degeneracy rule across every seed of a run."""

    if not by_seed:
        raise ValueError("é preciso ao menos uma seed")
    seeds = sorted(by_seed)
    required = {"random", "magnitude", "functional"}
    for seed in seeds:
        missing = required - set(by_seed[seed])
        if missing:
            raise ValueError(f"seed {seed} sem os braços {sorted(missing)}")

    pairs: dict[str, dict[str, Any]] = {}
    capacity_matched = True
    for label, (left_arm, right_arm) in OVERLAP_PAIRS.items():
        values: dict[str, float] = {}
        for seed in seeds:
            report = arm_overlap(by_seed[seed][left_arm], by_seed[seed][right_arm])
            values[str(seed)] = report["mean_jaccard"]
            capacity_matched &= bool(report["capacity_matched"])
        series = list(values.values())
        pairs[label] = {
            "n": len(series),
            "mean": statistics.fmean(series),
            "min": min(series),
            "max": max(series),
            "by_seed": values,
        }

    observed = pairs["magnitude_vs_random"]["mean"]
    degenerate = observed > DEGENERACY_OVERLAP_THRESHOLD
    verdict = (
        "inconclusive: a sobreposição magnitude-vs-aleatório "
        f"({observed:.4f}) excede {DEGENERACY_OVERLAP_THRESHOLD}, "
        "o braço magnitude degenerou em máscara quase-aleatória"
        if degenerate
        else (
            "exercised: a sobreposição magnitude-vs-aleatório "
            f"({observed:.4f}) fica abaixo de {DEGENERACY_OVERLAP_THRESHOLD}, "
            "o braço magnitude selecionou unidades próprias"
        )
    )
    return {
        "schema_version": 1,
        "threshold": DEGENERACY_OVERLAP_THRESHOLD,
        "seeds": seeds,
        "pairs": pairs,
        "degenerate": degenerate,
        "capacity_matched_every_seed": capacity_matched,
        "verdict": verdict,
    }
