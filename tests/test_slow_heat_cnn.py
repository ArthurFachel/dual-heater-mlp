from copy import deepcopy

import pytest
import torch

from dual_heater import SlowHeatCNN, SlowHeatConv2d, SlowHeatLinear
from dual_heater.optim import SlowHeatAdamW, SlowHeatSGD


def test_conv2d_matches_native_module_with_full_configuration():
    torch.manual_seed(12)
    native = torch.nn.Conv2d(
        4,
        6,
        kernel_size=(3, 2),
        stride=(2, 1),
        padding=(2, 1),
        dilation=(2, 1),
        groups=2,
        padding_mode="reflect",
    )
    protected = SlowHeatConv2d(
        4,
        6,
        kernel_size=(3, 2),
        stride=(2, 1),
        padding=(2, 1),
        dilation=(2, 1),
        groups=2,
        padding_mode="reflect",
    )
    with torch.no_grad():
        protected.weight.copy_(native.weight)
        protected.bias.copy_(native.bias)
    inputs = torch.randn(2, 4, 11, 9)

    assert torch.equal(protected(inputs), native(inputs))


def test_conv_importance_uses_valid_spatial_positions_only():
    layer = SlowHeatConv2d(
        2,
        2,
        kernel_size=1,
        gradient_masking=False,
        bias=False,
    )
    with torch.no_grad():
        layer.weight.copy_(
            torch.tensor(
                [
                    [[[1.0]], [[0.0]]],
                    [[[0.0]], [[1.0]]],
                ]
            )
        )
    inputs = torch.tensor(
        [[[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 1.0]]]]
    )
    validity = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])

    output = layer(inputs, validity_mask=validity)
    assert layer.task_step.item() == 0
    output.sum().backward()

    assert torch.allclose(layer.task_ema, torch.tensor([2.0, 0.0]))
    assert layer.task_step.item() == 1


def test_conv_functional_importance_is_invariant_to_reciprocal_rescaling():
    first = torch.nn.Sequential(
        SlowHeatConv2d(1, 2, kernel_size=1, gradient_masking=False),
        torch.nn.ReLU(),
        torch.nn.Conv2d(2, 1, kernel_size=1, bias=False),
    )
    with torch.no_grad():
        first[0].weight.abs_().add_(0.2)
        first[0].bias.fill_(0.5)
        first[2].weight.copy_(torch.tensor([[[[0.7]], [[-0.4]]]]))
    second = deepcopy(first)
    with torch.no_grad():
        second[0].weight[0].mul_(10.0)
        second[0].bias[0].mul_(10.0)
        second[2].weight[:, 0].div_(10.0)
    inputs = torch.rand(4, 1, 5, 5) + 0.5

    first_output = first(inputs)
    second_output = second(inputs)
    first_output.square().mean().backward()
    second_output.square().mean().backward()

    assert torch.allclose(first_output, second_output, atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        first[0].task_ema,
        second[0].task_ema,
        atol=1e-6,
        rtol=1e-6,
    )


def test_conv_functional_importance_stays_finite_in_half_precision():
    layer = SlowHeatConv2d(
        2,
        3,
        kernel_size=1,
        gradient_masking=False,
    ).half()

    layer(torch.randn(2, 2, 3, 3, dtype=torch.float16)).sum().backward()

    assert layer.task_ema.shape == (3,)
    assert torch.isfinite(layer.task_ema).all()


def test_grouped_convolution_mask_respects_global_channel_groups():
    source = SlowHeatConv2d(1, 4, kernel_size=1, slow_strength=9.0)
    target = SlowHeatConv2d(
        4,
        4,
        kernel_size=1,
        groups=2,
        slow_strength=9.0,
    )
    source.slow_heat.copy_(torch.tensor([1.0, 0.0, 0.5, 0.0]))
    target.slow_heat.zero_()
    optimizer = SlowHeatSGD(
        [*source.parameters(), *target.parameters()],
        lr=1.0,
    )
    optimizer.register_slow_heat_module(target, input_module=source)
    before = target.weight.detach().clone()
    target.weight.grad = torch.ones_like(target.weight)
    target.bias.grad = torch.ones_like(target.bias)

    optimizer.step()

    applied = (before - target.weight.detach()).squeeze(-1).squeeze(-1)
    expected = torch.tensor(
        [
            [0.1, 1.0],
            [0.1, 1.0],
            [1.0 / 5.5, 1.0],
            [1.0 / 5.5, 1.0],
        ]
    )
    assert torch.allclose(applied, expected)


def test_conv_to_linear_mask_repeats_each_nchw_channel_factor():
    source = SlowHeatConv2d(1, 2, kernel_size=1, slow_strength=9.0)
    target = SlowHeatLinear(8, 1, slow_strength=9.0)
    source.slow_heat.copy_(torch.tensor([1.0, 0.0]))
    target.slow_heat.zero_()
    optimizer = SlowHeatSGD(
        [*source.parameters(), *target.parameters()],
        lr=1.0,
    )
    optimizer.register_slow_heat_module(
        target,
        input_module=source,
        input_expansion=4,
    )
    before = target.weight.detach().clone()
    target.weight.grad = torch.ones_like(target.weight)
    target.bias.grad = torch.ones_like(target.bias)

    optimizer.step()

    expected = torch.tensor([[0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 1.0]])
    assert torch.allclose(before - target.weight.detach(), expected)


def test_slow_heat_cnn_registers_trains_and_consolidates_all_layers():
    model = SlowHeatCNN(
        3,
        5,
        channels=(4, 6),
        pooled_size=(2, 2),
        plasticity_budget=0.5,
    )
    optimizer = SlowHeatAdamW(model.parameters(), lr=1e-3)
    optimizer.register_slow_heat_model(model)

    loss = model(torch.randn(3, 3, 16, 16)).square().mean()
    loss.backward()
    optimizer.step()
    model.consolidate()

    assert model(torch.randn(2, 3, 20, 20)).shape == (2, 5)
    assert len(optimizer._plasticity_masks) == 6
    assert all(layer.consolidated_tasks.item() == 1 for layer in model.get_slow_layers())
    assert all(layer.task_step.item() == 0 for layer in model.get_slow_layers())


@pytest.mark.parametrize("optimizer_type", [torch.optim.AdamW, SlowHeatAdamW])
def test_zero_heat_cnn_step_is_well_defined(optimizer_type):
    model = SlowHeatCNN(1, 3, channels=(2, 2), pooled_size=1)
    optimizer = optimizer_type(model.parameters(), lr=1e-3, weight_decay=0.1)
    if isinstance(optimizer, SlowHeatAdamW):
        optimizer.register_slow_heat_model(model)

    model(torch.randn(2, 1, 8, 8)).sum().backward()
    optimizer.step()

    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())


def test_zero_heat_masked_cnn_matches_native_adamw_update():
    torch.manual_seed(31)
    native_model = SlowHeatCNN(1, 3, channels=(2, 2), pooled_size=1)
    protected_model = deepcopy(native_model)
    native_optimizer = torch.optim.AdamW(
        native_model.parameters(), lr=1e-3, weight_decay=0.1
    )
    protected_optimizer = SlowHeatAdamW(
        protected_model.parameters(), lr=1e-3, weight_decay=0.1
    )
    protected_optimizer.register_slow_heat_model(protected_model)
    inputs = torch.randn(2, 1, 8, 8)

    native_model(inputs).sum().backward()
    protected_model(inputs).sum().backward()
    native_optimizer.step()
    protected_optimizer.step()

    for native, protected in zip(
        native_model.parameters(), protected_model.parameters(), strict=True
    ):
        assert torch.equal(native, protected)
