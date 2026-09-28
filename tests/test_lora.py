import pytest
import torch
import torch.nn.functional as F

from dual_heater import DualHeatLoRALinear


@pytest.mark.parametrize(
    "kwargs",
    [
        {"r": 0},
        {"lora_alpha": 0.0},
        {"fast_decay": 1.0},
        {"fast_strength": -1.0},
        {"fast_decay_rate": -0.1},
        {"slow_strength": -1.0},
    ],
)
def test_lora_rejects_invalid_hyperparameters(kwargs):
    with pytest.raises(ValueError):
        DualHeatLoRALinear(8, 4, **kwargs)


def test_lora_detaches_external_base_parameters():
    base_weight = torch.nn.Parameter(torch.randn(4, 8))
    base_bias = torch.nn.Parameter(torch.randn(4))
    layer = DualHeatLoRALinear(
        8,
        4,
        r=2,
        fast_strength=0.0,
        base_weight=base_weight,
        base_bias=base_bias,
    )

    layer(torch.randn(2, 8)).sum().backward()

    assert not layer.base_weight.requires_grad
    assert layer.base_bias is not None
    assert not layer.base_bias.requires_grad
    assert base_weight.grad is None
    assert base_bias.grad is None


def test_lora_with_zero_delta_matches_frozen_base_when_inhibition_is_disabled():
    base_weight = torch.randn(4, 8)
    base_bias = torch.randn(4)
    layer = DualHeatLoRALinear(
        8,
        4,
        r=2,
        fast_strength=0.0,
        slow_strength=0.0,
        base_weight=base_weight,
        base_bias=base_bias,
    )
    inputs = torch.randn(3, 8)

    actual = layer(inputs)
    expected = F.linear(inputs, base_weight, base_bias)

    assert torch.equal(actual, expected)


def test_scaling_is_constant_when_alpha_tracks_rank() -> None:
    """r=16/alpha=16 and r=32/alpha=32 must produce the same adapter scale.

    QB-2 compares those two configurations. Since scaling is alpha/r, leaving
    alpha at 16 while doubling r would halve the effective scale, so the
    comparison would measure scale rather than capacity.
    """

    from dual_heater.lora import DualHeatLoRALinear

    small = DualHeatLoRALinear(64, 64, r=16, lora_alpha=16.0)
    large = DualHeatLoRALinear(64, 64, r=32, lora_alpha=32.0)

    assert small.scaling == 1.0
    assert large.scaling == 1.0
    assert small.scaling == large.scaling

    # And the failure mode QB-2 must avoid: alpha left behind at 16.
    unmatched = DualHeatLoRALinear(64, 64, r=32, lora_alpha=16.0)
    assert unmatched.scaling == 0.5
