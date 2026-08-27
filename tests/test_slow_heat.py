from copy import deepcopy

import pytest
import torch

from dual_heater import SlowHeatConv2d, SlowHeatLinear, SlowHeatMLP


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("field", ["slow_strength", "importance_eps"])
def test_slow_heat_rejects_non_finite_importance_parameters(field, value):
    with pytest.raises(ValueError, match="finitos"):
        SlowHeatLinear(4, 2, **{field: value})


def test_slow_heat_rejects_negative_strength():
    with pytest.raises(ValueError):
        SlowHeatLinear(8, 4, slow_strength=-1.0)


@pytest.mark.parametrize("dims", [(), (4,), (4, 0, 2)])
def test_slow_heat_mlp_rejects_invalid_dimensions(dims):
    with pytest.raises(ValueError, match="dimensões"):
        SlowHeatMLP(*dims)


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
    layer = SlowHeatLinear(2, 2, plasticity_budget=0.0)
    layer.task_ema.copy_(torch.tensor([2.0, 1.0]))
    layer.task_step.fill_(1)
    layer.consolidate(strategy="max")
    first = layer.importance_memory.clone()

    layer.task_ema.copy_(torch.tensor([1.0, 3.0]))
    layer.task_step.fill_(1)
    layer.consolidate(strategy="max")

    assert torch.equal(first, torch.tensor([2.0, 1.0]))
    assert torch.equal(layer.importance_memory, torch.tensor([2.0, 3.0]))
    assert torch.equal(layer.slow_heat, torch.tensor([2.0 / 3.0, 1.0]))
    assert torch.count_nonzero(layer.task_ema) == 0
    assert layer.task_step.item() == 0


def test_mean_consolidation_averages_task_importance_without_dilution_by_steps():
    layer = SlowHeatLinear(2, 2)
    layer.task_ema.copy_(torch.tensor([2.0, 4.0]))
    layer.task_step.fill_(1)
    layer.consolidate(strategy="mean")
    layer.task_ema.copy_(torch.tensor([4.0, 2.0]))
    layer.task_step.fill_(1)
    layer.consolidate(strategy="mean")

    assert torch.equal(layer.importance_memory, torch.tensor([3.0, 3.0]))


def test_functional_importance_is_invariant_to_reciprocal_relu_rescaling():
    first = torch.nn.Sequential(
        SlowHeatLinear(2, 2, gradient_masking=False),
        torch.nn.ReLU(),
        torch.nn.Linear(2, 1, bias=False),
    )
    with torch.no_grad():
        first[0].weight.abs_().add_(0.2)
        first[0].bias.fill_(0.5)
        first[2].weight.copy_(torch.tensor([[0.7, -0.4]]))
    second = deepcopy(first)
    with torch.no_grad():
        second[0].weight[0].mul_(10.0)
        second[0].bias[0].mul_(10.0)
        second[2].weight[:, 0].div_(10.0)

    inputs = torch.rand(16, 2) + 0.5
    first_output = first(inputs)
    second_output = second(inputs)
    first_output.square().mean().backward()
    second_output.square().mean().backward()

    assert torch.allclose(first_output, second_output, atol=1e-6, rtol=1e-6)
    assert torch.allclose(first[0].task_ema, second[0].task_ema, atol=1e-6, rtol=1e-6)


def test_functional_importance_does_not_protect_dead_relu_unit():
    layer = SlowHeatLinear(1, 2, gradient_masking=False)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[-10.0], [1.0]]))
        layer.bias.copy_(torch.tensor([-10.0, 0.0]))

    torch.relu(layer(torch.ones(8, 1))).sum().backward()

    assert layer.task_ema[0] == 0.0
    assert layer.task_ema[1] > 0.0


def test_capacity_budget_preserves_requested_plastic_fraction():
    layer = SlowHeatLinear(3, 4, plasticity_budget=0.5)
    layer.task_ema.copy_(torch.tensor([4.0, 3.0, 2.0, 1.0]))
    layer.task_step.fill_(1)

    layer.consolidate()

    assert torch.equal(layer.slow_heat > 0.0, torch.tensor([True, True, False, False]))
    assert layer.capacity_metrics() == {
        "protected_fraction": 0.5,
        "plastic_fraction": 0.5,
    }


def test_empty_consolidation_is_rejected_instead_of_diluting_memory():
    layer = SlowHeatLinear(2, 2)

    with pytest.raises(RuntimeError, match="sem backward"):
        layer.consolidate(strategy="mean")


def test_mlp_protects_output_layer_by_default():
    model = SlowHeatMLP(4, 3, 2)

    assert len(model.get_slow_layers()) == 2
    assert isinstance(model[-1], SlowHeatLinear)


def test_capacity_controller_releases_more_units_when_acquisition_is_weak():
    layer = SlowHeatLinear(2, 4, plasticity_budget=0.25)
    layer.importance_memory.copy_(torch.tensor([4.0, 3.0, 2.0, 1.0]))
    layer._apply_capacity_budget()

    budget = layer.adapt_capacity(
        acquisition_score=0.4,
        target_score=0.8,
        adaptation_rate=0.5,
    )

    assert budget == pytest.approx(0.45)
    assert layer.capacity_metrics()["plastic_fraction"] >= 0.45


def test_adapted_capacity_survives_state_dict_round_trip():
    layer = SlowHeatLinear(2, 4, plasticity_budget=0.25)
    layer.adapt_capacity(acquisition_score=0.4, target_score=0.8)
    restored = SlowHeatLinear(2, 4, plasticity_budget=0.9)

    restored.load_state_dict(layer.state_dict())

    assert restored.plasticity_budget == pytest.approx(layer.plasticity_budget)


def test_zero_functional_signal_remains_finite_in_half_precision():
    layer = SlowHeatLinear(2, 2, gradient_masking=False).half()
    output = layer(torch.zeros(4, 2, dtype=torch.float16))

    output.sum().backward()

    assert torch.isfinite(layer.task_ema).all()
    assert torch.count_nonzero(layer.task_ema) == 0
