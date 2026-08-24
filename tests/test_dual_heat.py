import pytest
import torch

from dual_heater import DualHeatLinear, DualHeatMLP


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fast_decay": -0.1},
        {"fast_decay": 1.0},
        {"fast_strength": -1.0},
        {"fast_decay_rate": -0.1},
        {"slow_strength": -1.0},
    ],
)
def test_dual_heat_rejects_invalid_strength_and_decay_values(kwargs):
    with pytest.raises(ValueError):
        DualHeatLinear(3, 2, **kwargs)


@pytest.mark.parametrize("dims", [(), (4,), (4, 0, 2)])
def test_dual_heat_mlp_rejects_invalid_dimensions(dims):
    with pytest.raises(ValueError, match="dimensões"):
        DualHeatMLP(*dims)


def test_dual_heat_linear_supports_sequence_inputs():
    layer = DualHeatLinear(8, 4)
    layer.train()

    output = layer(torch.randn(2, 5, 8))
    output.sum().backward()

    assert output.shape == (2, 5, 4)
    assert layer.fast_heat.shape == (4,)
    assert layer.slow_heat.shape == (4,)
    assert torch.isfinite(layer.weight.grad).all()


def test_sensitivity_importance_uses_output_gradient():
    layer = DualHeatLinear(
        3,
        2,
        fast_strength=0.0,
        slow_strength=1.0,
        importance="sensitivity",
    )
    layer.train()

    output = layer(torch.ones(4, 3))
    loss = (output[:, 0] * 10.0 + output[:, 1] * 0.1).mean()
    loss.backward()

    assert layer.slow_heat[0] > layer.slow_heat[1]
    assert torch.isfinite(layer.slow_heat).all()


def test_mlp_can_protect_output_layer_when_requested():
    model = DualHeatMLP(8, 6, 4, 3, protect_output=True)
    dual_layers = model.get_dual_layers()

    assert len(dual_layers) == 3
    assert isinstance(model[-1], DualHeatLinear)

    model.train()
    model(torch.randn(2, 8)).sum().backward()
    assert torch.isfinite(model[-1].weight.grad).all()


def test_sensitivity_importance_is_invariant_to_repeated_batch_size():
    torch.manual_seed(7)
    layer_single = DualHeatLinear(
        3,
        2,
        fast_strength=0.0,
        slow_strength=0.0,
        importance="sensitivity",
    )
    torch.manual_seed(7)
    layer_batch = DualHeatLinear(
        3,
        2,
        fast_strength=0.0,
        slow_strength=0.0,
        importance="sensitivity",
    )
    sample = torch.tensor([[1.0, -2.0, 0.5]])

    layer_single(sample).sum(dim=-1).mean().backward()
    layer_batch(sample.repeat(4, 1)).sum(dim=-1).mean().backward()

    assert isinstance(layer_single.slow_heat, torch.Tensor)
    assert isinstance(layer_batch.slow_heat, torch.Tensor)
    assert torch.allclose(layer_single.slow_heat, layer_batch.slow_heat)


def test_sensitivity_importance_is_consistent_with_gradient_accumulation():
    torch.manual_seed(11)
    layer_sequential = DualHeatLinear(
        3,
        2,
        fast_strength=0.0,
        slow_strength=0.0,
        importance="sensitivity",
    )
    torch.manual_seed(11)
    layer_accumulated = DualHeatLinear(
        3,
        2,
        fast_strength=0.0,
        slow_strength=0.0,
        importance="sensitivity",
    )
    first = torch.tensor([[1.0, -2.0, 0.5]])
    second = torch.tensor([[-0.5, 1.0, 2.0]])

    layer_sequential(first).sum().backward()
    layer_sequential.zero_grad()
    layer_sequential(second).sum().backward()

    first_loss = layer_accumulated(first).sum()
    second_loss = layer_accumulated(second).sum()
    (first_loss + second_loss).backward()

    assert isinstance(layer_sequential.slow_heat, torch.Tensor)
    assert isinstance(layer_accumulated.slow_heat, torch.Tensor)
    assert torch.allclose(layer_sequential.slow_heat, layer_accumulated.slow_heat)


def test_lateral_inhibition_is_an_explicit_train_only_regularizer():
    layer = DualHeatLinear(3, 2, fast_strength=2.0)
    inputs = torch.tensor([[1.0, -1.0, 0.5]])
    assert isinstance(layer.fast_heat, torch.Tensor)
    layer.fast_heat.copy_(torch.tensor([1.0, 3.0]))

    layer.train()
    training_output = layer(inputs)
    heat_after_training = layer.fast_heat.clone()

    layer.eval()
    evaluation_output = layer(inputs)

    assert not torch.equal(training_output, evaluation_output)
    assert torch.equal(layer.fast_heat, heat_after_training)
