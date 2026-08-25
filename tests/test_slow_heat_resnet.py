import torch

from dual_heater import CIFARResNet18, SlowHeatResNet18
from dual_heater.optim import SlowHeatAdamW, SlowHeatSGD


TINY_RESNET = {
    "stage_channels": (2, 4, 8, 16),
    "blocks_per_stage": (1, 1, 1, 1),
}


def _copy_trainable_parameters(source, destination):
    source_parameters = dict(source.named_parameters())
    with torch.no_grad():
        for name, parameter in destination.named_parameters():
            parameter.copy_(source_parameters[name])


def test_cifar_resnet_and_slowheat_variant_are_initially_equivalent():
    torch.manual_seed(41)
    native = CIFARResNet18(3, 5, **TINY_RESNET)
    protected = SlowHeatResNet18(3, 5, **TINY_RESNET)
    _copy_trainable_parameters(native, protected)
    inputs = torch.randn(2, 3, 16, 16)

    assert dict(native.named_parameters()).keys() == dict(
        protected.named_parameters()
    ).keys()
    assert torch.equal(native(inputs), protected(inputs))
    assert native.forward_features(inputs).shape == (2, 16)


def test_resnet_graph_registers_virtual_sources_and_groupnorm_affine_parameters():
    model = SlowHeatResNet18(3, 5, **TINY_RESNET)
    optimizer = SlowHeatSGD(model.parameters(), lr=1.0)

    optimizer.register_slow_heat_model(model)

    assert len(optimizer._plasticity_masks) == len(list(model.parameters()))
    kinds = [kind for _, _, kind in optimizer._plasticity_masks.values()]
    assert any("virtual_stem_output" in kind for kind in kinds)
    assert any("channel_affine_from_stem" in kind for kind in kinds)


def test_residual_tracker_protects_downstream_input_channels():
    model = SlowHeatResNet18(
        3,
        2,
        stage_channels=(3, 3, 3, 3),
        blocks_per_stage=(1, 1, 1, 1),
        slow_strength=9.0,
    )
    model.stem_tracker.slow_heat.copy_(torch.tensor([1.0, 0.0, 0.5]))
    target = model.stages[0][0].conv1
    optimizer = SlowHeatSGD(model.parameters(), lr=1.0)
    optimizer.register_slow_heat_model(model)
    before = target.weight.detach().clone()
    target.weight.grad = torch.ones_like(target.weight)

    optimizer.step()

    applied = before - target.weight.detach()
    expected = torch.tensor([0.1, 1.0, 1.0 / 5.5]).reshape(1, 3, 1, 1)
    assert torch.allclose(applied, expected.expand_as(applied))


def test_resnet_groupnorm_affine_follows_producer_channel_mask():
    model = SlowHeatResNet18(3, 2, **TINY_RESNET, slow_strength=9.0)
    model.stem.slow_heat.copy_(torch.tensor([1.0, 0.0]))
    optimizer = SlowHeatSGD(model.parameters(), lr=1.0)
    optimizer.register_slow_heat_model(model)
    before_weight = model.stem_norm.weight.detach().clone()
    before_bias = model.stem_norm.bias.detach().clone()
    model.stem_norm.weight.grad = torch.ones_like(model.stem_norm.weight)
    model.stem_norm.bias.grad = torch.ones_like(model.stem_norm.bias)

    optimizer.step()

    expected = torch.tensor([0.1, 1.0])
    assert torch.allclose(before_weight - model.stem_norm.weight.detach(), expected)
    assert torch.allclose(before_bias - model.stem_norm.bias.detach(), expected)


def test_zero_heat_resnet_matches_native_adamw_update_and_consolidates_trackers():
    torch.manual_seed(73)
    native = CIFARResNet18(3, 5, **TINY_RESNET)
    protected = SlowHeatResNet18(3, 5, **TINY_RESNET)
    _copy_trainable_parameters(native, protected)
    native_optimizer = torch.optim.AdamW(
        native.parameters(), lr=1e-3, weight_decay=0.1
    )
    protected_optimizer = SlowHeatAdamW(
        protected.parameters(), lr=1e-3, weight_decay=0.1
    )
    protected_optimizer.register_slow_heat_model(protected)
    inputs = torch.randn(2, 3, 16, 16)

    native(inputs).sum().backward()
    protected(inputs).sum().backward()
    native_optimizer.step()
    protected_optimizer.step()

    for native_parameter, protected_parameter in zip(
        native.parameters(), protected.parameters(), strict=True
    ):
        assert torch.equal(native_parameter, protected_parameter)

    protected.consolidate()
    assert all(
        state.consolidated_tasks.item() == 1
        for state in protected.get_slow_states()
    )
    assert all(state.task_step.item() == 0 for state in protected.get_slow_states())
