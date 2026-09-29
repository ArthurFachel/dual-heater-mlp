"""Runner wiring for the plasticity-matched protocol.

goals/protocol_plasticity_matched.md runs three arms whose contrast is only
meaningful if two things hold in the artifacts:

1. ``lr_control`` must be able to sit at an ARBITRARY learning-rate scale. The
   existing runner ties it to ``--target-plasticity``, which only exists for
   masked arms; this protocol has no masked arm, so the scale must be its own
   flag or the falsifier silently runs at lr x 1.0 and the treatment arm is
   compared against an untreated baseline.

2. Every arm must record the three fields ``surface_plasticity`` is computed
   from. Without them the pairing cannot be VERIFIED from the manifest, which
   is exactly how the previous protocol reported 0.85 for an arm sitting at
   0.532.

These tests assert what the runner produces, not what the API accepts.
"""

from __future__ import annotations

import pytest


# Measured in results/qwen_lora_confirmation/*/manifest.json, r=16.
VANILLA_TRAINABLE = 6_769_920
FROZEN_A_TRAINABLE = 4_214_016
HEAD_TRAINABLE = 896 * 150  # classification head, trainable and unmasked
MASKED_B = FROZEN_A_TRAINABLE - HEAD_TRAINABLE


def test_run_config_carries_an_explicit_learning_rate_scale() -> None:
    from experiments.qwen_lora_slowheat import RunConfig

    config = RunConfig(learning_rate=1e-4, learning_rate_scale=0.6224617129892229)
    assert config.learning_rate_scale == pytest.approx(0.6224617129892229)


def test_learning_rate_scale_defaults_to_unset() -> None:
    """Unset must be distinguishable from 1.0, or precedence is undecidable."""

    from experiments.qwen_lora_slowheat import RunConfig

    assert RunConfig().learning_rate_scale is None


def test_lr_control_uses_the_explicit_scale_when_given() -> None:
    from experiments.qwen_lora_slowheat import RunConfig, resolve_effective_lr

    config = RunConfig(learning_rate=1e-4, learning_rate_scale=0.6224617129892229)
    assert resolve_effective_lr("lr_control", config) == pytest.approx(6.224617e-5)


def test_explicit_scale_overrides_target_plasticity() -> None:
    """A protocol with no masked arm must not inherit the masked arm's knob."""

    from experiments.qwen_lora_slowheat import RunConfig, resolve_effective_lr

    config = RunConfig(
        learning_rate=1e-4,
        learning_rate_scale=0.6224617129892229,
        target_plasticity=0.85,
    )
    assert resolve_effective_lr("lr_control", config) == pytest.approx(6.224617e-5)


def test_target_plasticity_still_drives_lr_control_when_no_scale_is_given() -> None:
    """Backwards compatibility: the confirmatory runs must stay reproducible."""

    from experiments.qwen_lora_slowheat import RunConfig, resolve_effective_lr

    config = RunConfig(learning_rate=1e-4, target_plasticity=0.85)
    assert resolve_effective_lr("lr_control", config) == pytest.approx(8.5e-5)


def test_the_scale_applies_only_to_lr_control() -> None:
    """Scaling any other arm would remove the contrast instead of pairing it."""

    from experiments.qwen_lora_slowheat import RunConfig, resolve_effective_lr

    config = RunConfig(learning_rate=1e-4, learning_rate_scale=0.5)
    for arm in ("vanilla", "frozen_a_control", "exact"):
        assert resolve_effective_lr(arm, config) == pytest.approx(1e-4)


def test_cli_exposes_the_learning_rate_scale() -> None:
    from experiments.qwen_lora_slowheat import build_parser

    parsed = build_parser().parse_args(
        ["--output", "/tmp/x", "--learning-rate-scale", "0.6224617129892229"]
    )
    assert parsed.learning_rate_scale == pytest.approx(0.6224617129892229)


def test_the_cli_scale_reaches_the_arm_the_runner_executes() -> None:
    """The last hop, and the one a parser test cannot see.

    A flag that parses but is never copied into the config leaves lr_control
    at lr x 1.0: the falsifier runs UNTREATED, the contrast is against an
    untreated baseline, and nothing in the manifest looks wrong.
    """

    from experiments.qwen_lora_slowheat import (
        build_parser,
        build_run_config,
        resolve_effective_lr,
    )

    parsed = build_parser().parse_args(
        ["--output", "/tmp/x", "--learning-rate-scale", "0.6224617129892229"]
    )
    config = build_run_config(parsed)
    assert config.learning_rate_scale == pytest.approx(0.6224617129892229)
    assert resolve_effective_lr("lr_control", config) == pytest.approx(6.224617e-5)
    # And the untreated value must NOT be what comes out.
    assert resolve_effective_lr("lr_control", config) != pytest.approx(
        config.learning_rate
    )


def test_mask_coverage_reports_mean_and_count_consistently() -> None:
    """``effective_plasticity`` must be the mean this pair reports."""

    import torch

    from dual_heater.lora_slowheat import QwenLoRASlowHeat

    class _Binding:
        def __init__(self, parameter, mask):
            self.parameter = parameter
            self._mask = mask

        def mask(self):
            return self._mask

    instrumentation = QwenLoRASlowHeat.__new__(QwenLoRASlowHeat)
    parameter = torch.zeros(4, 3)
    mask = torch.full((4, 1), 0.5)
    instrumentation.mask_bindings = lambda: [_Binding(parameter, mask)]  # type: ignore[method-assign]

    mean, count = instrumentation.mask_coverage()
    assert count == 12
    assert mean == pytest.approx(0.5)
    assert instrumentation.effective_plasticity() == pytest.approx(mean)


def test_mask_coverage_reports_zero_masked_parameters_for_unmasked_arms() -> None:
    from dual_heater.lora_slowheat import QwenLoRASlowHeat

    instrumentation = QwenLoRASlowHeat.__new__(QwenLoRASlowHeat)
    instrumentation.mask_bindings = lambda: []  # type: ignore[method-assign]

    mean, count = instrumentation.mask_coverage()
    assert count == 0
    assert mean == pytest.approx(1.0)


def test_plasticity_record_round_trips_through_surface_plasticity() -> None:
    """What the manifest stores must be enough to recompute the pairing."""

    from dual_heater.lora_slowheat import surface_plasticity
    from experiments.qwen_lora_slowheat import plasticity_record

    record = plasticity_record(
        masked_mean=0.85,
        masked_parameters=MASKED_B,
        effective_lr=1e-4,
        learning_rate=1e-4,
    )
    assert record["masked_parameters"] == MASKED_B
    assert record["masked_mean"] == pytest.approx(0.85)
    assert record["learning_rate_scale"] == pytest.approx(1.0)

    assert surface_plasticity(
        masked_mean=record["masked_mean"],
        masked_parameters=record["masked_parameters"],
        trainable_parameters=FROZEN_A_TRAINABLE,
        reference_parameters=VANILLA_TRAINABLE,
        lr_scale=record["learning_rate_scale"],
    ) == pytest.approx(0.532070, abs=1e-6)


def test_plasticity_record_captures_a_scaled_learning_rate() -> None:
    from experiments.qwen_lora_slowheat import plasticity_record

    record = plasticity_record(
        masked_mean=1.0,
        masked_parameters=0,
        effective_lr=6.224617129892229e-5,
        learning_rate=1e-4,
    )
    assert record["learning_rate_scale"] == pytest.approx(0.6224617129892229)
