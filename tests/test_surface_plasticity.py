"""Effective plasticity measured over the trainable SURFACE, not the mask.

``QwenLoRASlowHeat.effective_plasticity`` averages the mask over the parameters
that carry a mask binding. For an arm that also FREEZES parameters, those frozen
parameters have an update factor of exactly zero and carry no binding, so they
are invisible to that average.

The ``exact`` arm freezes ``A`` (LoRA-FA) and masks rows of ``B``, so its
binding-scoped ``E`` reported 0.85 while the mean update factor over the
reference arm's trainable surface was 0.532. The confirmatory contrast
``exact - lr_control`` was therefore run at 0.532 against 0.850: not matched.

``surface_plasticity`` is the quantity a pairing protocol must control. These
tests pin it, including the retrospective values for the arms already executed,
so the correction cannot silently regress.
"""

from __future__ import annotations

import pytest


# Measured in results/qwen_lora_confirmation/*/manifest.json, r=16.
VANILLA_TRAINABLE = 6_769_920
FROZEN_A_TRAINABLE = 4_214_016
HEAD_TRAINABLE = 896 * 150  # classification head, trainable and unmasked
MASKED_B = FROZEN_A_TRAINABLE - HEAD_TRAINABLE


def test_vanilla_surface_plasticity_is_one() -> None:
    from dual_heater.lora_slowheat import surface_plasticity

    assert surface_plasticity(
        masked_mean=1.0,
        masked_parameters=0,
        trainable_parameters=VANILLA_TRAINABLE,
        reference_parameters=VANILLA_TRAINABLE,
    ) == pytest.approx(1.0)


def test_frozen_parameters_lower_surface_plasticity_without_any_mask() -> None:
    """LoRA-FA removes plasticity by freezing, which no mask average can see."""

    from dual_heater.lora_slowheat import surface_plasticity

    value = surface_plasticity(
        masked_mean=1.0,
        masked_parameters=0,
        trainable_parameters=FROZEN_A_TRAINABLE,
        reference_parameters=VANILLA_TRAINABLE,
    )
    assert value == pytest.approx(0.622462, abs=1e-6)


def test_uniform_learning_rate_scaling_moves_surface_plasticity() -> None:
    from dual_heater.lora_slowheat import surface_plasticity

    assert surface_plasticity(
        masked_mean=1.0,
        masked_parameters=0,
        trainable_parameters=VANILLA_TRAINABLE,
        reference_parameters=VANILLA_TRAINABLE,
        lr_scale=0.85,
    ) == pytest.approx(0.85)


def test_exact_arm_surface_plasticity_is_far_below_its_reported_mask_average() -> None:
    """The regression this module exists to prevent.

    ``exact`` reported ``effective_plasticity = 0.850`` in every confirmatory
    manifest. Over the reference surface it was 0.532.
    """

    from dual_heater.lora_slowheat import surface_plasticity

    value = surface_plasticity(
        masked_mean=0.85,
        masked_parameters=MASKED_B,
        trainable_parameters=FROZEN_A_TRAINABLE,
        reference_parameters=VANILLA_TRAINABLE,
    )
    assert value == pytest.approx(0.532070, abs=1e-6)
    assert value < 0.85 - 0.3, "the gap is the confound, not a rounding detail"


def test_the_confirmed_contrast_was_not_plasticity_matched() -> None:
    """`exact - lr_control` compared 0.532 against 0.850.

    Kept as an executable statement of the defect so the write-up and the code
    cannot drift apart.
    """

    from dual_heater.lora_slowheat import surface_plasticity

    exact = surface_plasticity(
        masked_mean=0.85,
        masked_parameters=MASKED_B,
        trainable_parameters=FROZEN_A_TRAINABLE,
        reference_parameters=VANILLA_TRAINABLE,
    )
    lr_control = surface_plasticity(
        masked_mean=1.0,
        masked_parameters=0,
        trainable_parameters=VANILLA_TRAINABLE,
        reference_parameters=VANILLA_TRAINABLE,
        lr_scale=0.85,
    )
    assert abs(exact - lr_control) > 0.3


def test_plasticity_matched_arms_agree_to_the_declared_tolerance() -> None:
    """The three arms of goals/protocol_plasticity_matched.md.

    ``frozen_a_control`` removes plasticity by freezing 37.75% of the surface;
    ``lr_control`` removes the same MEAN amount uniformly through the learning
    rate. Matching them is the whole design.
    """

    from dual_heater.lora_slowheat import (
        FROZEN_A_LR_SCALE,
        surface_plasticity,
    )

    frozen = surface_plasticity(
        masked_mean=1.0,
        masked_parameters=0,
        trainable_parameters=FROZEN_A_TRAINABLE,
        reference_parameters=VANILLA_TRAINABLE,
    )
    control = surface_plasticity(
        masked_mean=1.0,
        masked_parameters=0,
        trainable_parameters=VANILLA_TRAINABLE,
        reference_parameters=VANILLA_TRAINABLE,
        lr_scale=FROZEN_A_LR_SCALE,
    )
    assert frozen == pytest.approx(control, abs=1e-9)
    assert FROZEN_A_LR_SCALE == pytest.approx(
        FROZEN_A_TRAINABLE / VANILLA_TRAINABLE, abs=1e-12
    )


def test_surface_plasticity_rejects_impossible_inputs() -> None:
    from dual_heater.lora_slowheat import surface_plasticity

    with pytest.raises(ValueError):
        surface_plasticity(
            masked_mean=1.0,
            masked_parameters=0,
            trainable_parameters=VANILLA_TRAINABLE,
            reference_parameters=0,
        )
    with pytest.raises(ValueError):
        # More masked parameters than trainable ones is a wiring bug.
        surface_plasticity(
            masked_mean=1.0,
            masked_parameters=VANILLA_TRAINABLE + 1,
            trainable_parameters=VANILLA_TRAINABLE,
            reference_parameters=VANILLA_TRAINABLE,
        )


def test_attach_surface_plasticity_uses_the_named_reference_arm() -> None:
    from dual_heater.lora_slowheat import attach_surface_plasticity

    results = [
        {
            "method": "vanilla",
            "trainable_parameters": VANILLA_TRAINABLE,
            "masked_parameters": 0,
            "masked_mean": 1.0,
            "learning_rate_scale": 1.0,
        },
        {
            "method": "lr_control",
            "trainable_parameters": VANILLA_TRAINABLE,
            "masked_parameters": 0,
            "masked_mean": 1.0,
            "learning_rate_scale": 0.6224617129892229,
        },
        {
            "method": "frozen_a_control",
            "trainable_parameters": FROZEN_A_TRAINABLE,
            "masked_parameters": 0,
            "masked_mean": 1.0,
            "learning_rate_scale": 1.0,
        },
    ]
    attach_surface_plasticity(results, reference_arm="vanilla")

    by_arm = {row["method"]: row["surface_plasticity"] for row in results}
    assert by_arm["vanilla"] == pytest.approx(1.0)
    assert by_arm["lr_control"] == pytest.approx(by_arm["frozen_a_control"], abs=1e-9)


def test_attach_surface_plasticity_fails_loudly_without_the_reference() -> None:
    """A missing reference must raise, never silently pick another arm."""

    from dual_heater.lora_slowheat import attach_surface_plasticity

    results = [
        {
            "method": "frozen_a_control",
            "trainable_parameters": FROZEN_A_TRAINABLE,
            "masked_parameters": 0,
            "masked_mean": 1.0,
            "learning_rate_scale": 1.0,
        }
    ]
    with pytest.raises(ValueError, match="vanilla"):
        attach_surface_plasticity(results, reference_arm="vanilla")
