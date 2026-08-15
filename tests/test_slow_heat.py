import pytest
import torch

from dual_heater import SlowHeatConv2d, SlowHeatLinear


def test_slow_heat_rejects_negative_strength():
    with pytest.raises(ValueError):
        SlowHeatLinear(8, 4, slow_strength=-1.0)


def test_slow_heat_linear_supports_sequence_inputs():
    layer = SlowHeatLinear(8, 4)
    layer.train()

    output = layer(torch.randn(2, 5, 8))
    output.sum().backward()

    assert output.shape == (2, 5, 4)
    assert layer.task_ema.shape == (4,)
    assert layer.weight.grad is not None
    assert torch.isfinite(layer.weight.grad).all()


def test_slow_heat_conv_keeps_consolidated_buffers_in_state_dict():
    layer = SlowHeatConv2d(3, 4, kernel_size=3, padding=1)
    layer(torch.randn(2, 3, 8, 8)).mean().backward()
    layer.consolidate()

    state = layer.state_dict()
    assert {"slow_heat", "task_ema", "task_step"} <= state.keys()


def test_max_consolidation_is_monotonic_and_resets_only_task_statistics():
    layer = SlowHeatLinear(2, 2)
    layer.task_ema.copy_(torch.tensor([2.0, 1.0]))
    layer.consolidate(strategy="max")
    first = layer.slow_heat.clone()

    layer.task_ema.copy_(torch.tensor([1.0, 3.0]))
    layer.consolidate(strategy="max")

    assert torch.equal(first, torch.tensor([2.0, 1.0]))
    assert torch.equal(layer.slow_heat, torch.tensor([2.0, 3.0]))
    assert torch.count_nonzero(layer.task_ema) == 0
    assert layer.task_step.item() == 0


def test_mean_consolidation_averages_task_importance_without_dilution_by_steps():
    layer = SlowHeatLinear(2, 2)
    layer.task_ema.copy_(torch.tensor([2.0, 4.0]))
    layer.consolidate(strategy="mean")
    layer.task_ema.copy_(torch.tensor([4.0, 2.0]))
    layer.consolidate(strategy="mean")

    assert torch.equal(layer.slow_heat, torch.tensor([3.0, 3.0]))
