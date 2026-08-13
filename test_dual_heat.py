import torch

from dual_heat_module import DualHeatLinear, DualHeatMLP


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
