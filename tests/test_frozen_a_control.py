"""Decomposition of the `exact` arm into LoRA-FA and SlowHeat.

`exact` does two things at once: it freezes A (which is LoRA-FA,
arXiv 2308.03303) and it masks rows of B with a SlowHeat tracker. The
confirmed contrast `exact - lr_control` therefore measures the SUM of a
published technique and our own, with no way to tell how much comes from each.

`frozen_a_control` freezes A and does nothing else, so:

    frozen_a_control - vanilla   -> effect of LoRA-FA alone
    exact - frozen_a_control     -> effect of SlowHeat GIVEN LoRA-FA

Only the second is ours to claim.
"""

from __future__ import annotations

import pytest


def test_frozen_a_control_is_a_declared_method() -> None:
    from dual_heater.lora_slowheat import LoRASlowHeatConfig

    config = LoRASlowHeatConfig(method="frozen_a_control", task_count=2)
    assert config.method == "frozen_a_control"


def test_frozen_a_control_freezes_a_but_adds_no_mask() -> None:
    """It must be LoRA-FA exactly: A frozen, B trainable, zero masking."""

    from dual_heater.lora_slowheat import (
        TRACKED_METHODS,
        UNMASKED_METHODS,
    )

    # It carries no tracker and no mask binding: the whole point is that the
    # only difference from `exact` is the SlowHeat machinery.
    assert "frozen_a_control" not in TRACKED_METHODS
    assert "frozen_a_control" in UNMASKED_METHODS


def test_frozen_a_control_and_exact_freeze_a_identically() -> None:
    """The freezing path must be shared, not duplicated with a subtle drift."""

    import inspect

    from dual_heater import lora_slowheat

    source = inspect.getsource(lora_slowheat.build_lora_slowheat)
    # Both arms must reach the same freezing branch.
    assert 'config.method in FROZEN_A_METHODS' in source, (
        "freezing must be driven by a shared set, so the two arms cannot diverge"
    )
    assert lora_slowheat.FROZEN_A_METHODS == frozenset({"exact", "frozen_a_control"})


def test_exact_decomposition_seeds_are_registered_and_disjoint() -> None:
    """Section D4: the decomposition owns a seed band disjoint from prior work."""

    from experiments.qwen_lora_slowheat import EXACT_DECOMPOSITION_SEEDS
    from experiments.bert_slowheat_diagnostic import (
        IMPORTANCE_CRITERION_ABLATION_SEEDS,
    )
    from experiments.replay_selection_sweep import (
        REPLAY_SELECTOR_CONFIRMATORY_SEEDS,
    )
    from experiments.confirmatory_split_mnist import DERPP_CONFIRMATORY_SEEDS

    seeds = EXACT_DECOMPOSITION_SEEDS
    assert len(seeds) == 10 and len(set(seeds)) == 10

    for name, other in (
        ("criterion ablation", IMPORTANCE_CRITERION_ABLATION_SEEDS),
        ("replay selector", REPLAY_SELECTOR_CONFIRMATORY_SEEDS),
        ("derpp", DERPP_CONFIRMATORY_SEEDS),
    ):
        overlap = set(seeds) & set(other)
        assert not overlap, f"reuses {name} seeds: {sorted(overlap)}"
