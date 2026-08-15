from copy import deepcopy

import pytest
import torch

from dual_heater import SlowHeatLinear
from dual_heater.optim import SlowHeatAdamW, SlowHeatSGD


def _single_adamw_step(optimizer_type, mask=None):
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = optimizer_type(
        [parameter],
        lr=0.1,
        betas=(0.0, 0.0),
        eps=1e-8,
        weight_decay=0.2,
    )
    if mask is not None:
        optimizer.register_plasticity_mask(parameter, torch.tensor([mask]))
    parameter.grad = torch.tensor([2.0])
    optimizer.step()
    return parameter.detach()


def test_adamw_unit_mask_matches_native_adamw():
    native = _single_adamw_step(torch.optim.AdamW)
    masked = _single_adamw_step(SlowHeatAdamW, mask=1.0)

    assert torch.equal(masked, native)


def test_adamw_zero_mask_blocks_gradient_and_weight_decay_update():
    masked = _single_adamw_step(SlowHeatAdamW, mask=0.0)

    assert torch.equal(masked, torch.tensor([1.0]))


def test_adamw_fractional_mask_scales_the_final_preconditioned_update():
    native = _single_adamw_step(torch.optim.AdamW)
    masked = _single_adamw_step(SlowHeatAdamW, mask=0.1)
    expected = torch.tensor([1.0]) + 0.1 * (native - torch.tensor([1.0]))

    assert torch.allclose(masked, expected)


def test_sgd_fractional_mask_scales_momentum_and_decay_update():
    native_parameter = torch.nn.Parameter(torch.tensor([1.0]))
    native = torch.optim.SGD(
        [native_parameter], lr=0.1, momentum=0.9, weight_decay=0.2
    )
    native_parameter.grad = torch.tensor([2.0])
    native.step()

    masked_parameter = torch.nn.Parameter(torch.tensor([1.0]))
    masked = SlowHeatSGD(
        [masked_parameter], lr=0.1, momentum=0.9, weight_decay=0.2
    )
    masked.register_plasticity_mask(masked_parameter, torch.tensor([0.1]))
    masked_parameter.grad = torch.tensor([2.0])
    masked.step()

    expected = torch.tensor([1.0]) + 0.1 * (
        native_parameter.detach() - torch.tensor([1.0])
    )
    assert torch.allclose(masked_parameter.detach(), expected)


def test_adamw_state_dict_round_trip_preserves_future_updates():
    first_parameter = torch.nn.Parameter(torch.tensor([1.0]))
    first = SlowHeatAdamW([first_parameter], lr=0.01, weight_decay=0.1)
    first.register_plasticity_mask(first_parameter, torch.tensor([0.5]))
    first_parameter.grad = torch.tensor([2.0])
    first.step()

    second_parameter = torch.nn.Parameter(first_parameter.detach().clone())
    second = SlowHeatAdamW([second_parameter], lr=0.01, weight_decay=0.1)
    second.load_state_dict(deepcopy(first.state_dict()))
    second.register_plasticity_mask(second_parameter, torch.tensor([0.5]))

    first_parameter.grad = torch.tensor([-1.0])
    second_parameter.grad = torch.tensor([-1.0])
    first.step()
    second.step()

    assert torch.equal(second_parameter.detach(), first_parameter.detach())


def test_registering_slow_heat_module_applies_mask_exactly_once():
    torch.manual_seed(5)
    native = torch.nn.Linear(3, 2)
    protected = SlowHeatLinear(3, 2, slow_strength=9.0)
    protected.weight.data.copy_(native.weight.data)
    protected.bias.data.copy_(native.bias.data)
    protected.slow_heat.fill_(1.0)

    native_optimizer = torch.optim.SGD(native.parameters(), lr=0.1)
    protected_optimizer = SlowHeatSGD(protected.parameters(), lr=0.1)
    protected_optimizer.register_slow_heat_module(protected)
    inputs = torch.tensor([[1.0, -2.0, 0.5]])

    native_before = native.weight.detach().clone()
    protected_before = protected.weight.detach().clone()
    native(inputs).sum().backward()
    protected(inputs).sum().backward()
    native_optimizer.step()
    protected_optimizer.step()

    native_delta = native.weight.detach() - native_before
    protected_delta = protected.weight.detach() - protected_before
    assert torch.allclose(protected_delta, 0.1 * native_delta)


def test_failed_module_registration_is_atomic():
    module = SlowHeatLinear(3, 2)
    optimizer = SlowHeatSGD([module.weight], lr=0.1)

    with pytest.raises(ValueError):
        optimizer.register_slow_heat_module(module)

    assert module.gradient_masking is True
    assert optimizer._plasticity_masks == {}


def test_loaded_masked_optimizer_fails_closed_until_masks_are_registered():
    first_module = SlowHeatLinear(3, 2)
    first = SlowHeatAdamW(first_module.parameters(), lr=0.01)
    first.register_slow_heat_module(first_module)
    state = deepcopy(first.state_dict())

    second_module = SlowHeatLinear(3, 2)
    second = SlowHeatAdamW(second_module.parameters(), lr=0.01)
    second.load_state_dict(state)
    second_module(torch.randn(1, 3)).sum().backward()

    with pytest.raises(RuntimeError, match="register"):
        second.step()

    second.register_slow_heat_module(second_module)
    second.step()


def test_legacy_checkpoint_without_mask_metadata_remains_compatible():
    first_module = SlowHeatLinear(3, 2)
    first = SlowHeatAdamW(first_module.parameters(), lr=0.01)
    state = deepcopy(first.state_dict())
    for key in [item for item in state if item.startswith("slowheat_")]:
        state.pop(key)

    second_module = SlowHeatLinear(3, 2)
    second = SlowHeatAdamW(second_module.parameters(), lr=0.01)
    second.load_state_dict(state)
    second.register_slow_heat_module(second_module)
    second_module(torch.randn(1, 3)).sum().backward()
    second.step()


def test_checkpoint_rejects_same_mask_count_with_wrong_registration_kind():
    first_module = SlowHeatLinear(3, 2)
    first = SlowHeatAdamW(first_module.parameters(), lr=0.01)
    first.register_slow_heat_module(first_module)
    state = deepcopy(first.state_dict())

    second_module = SlowHeatLinear(3, 2)
    second = SlowHeatAdamW(second_module.parameters(), lr=0.01)
    second.load_state_dict(state)
    second.register_plasticity_mask(second_module.weight, torch.tensor(1.0))
    second.register_plasticity_mask(second_module.bias, torch.tensor(1.0))
    second_module(torch.randn(1, 3)).sum().backward()

    with pytest.raises(RuntimeError, match="checkpoint"):
        second.step()


def test_optimizer_closure_updates_dynamic_mask_before_update_is_resolved():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    heat = torch.tensor([0.0])
    optimizer = SlowHeatSGD([parameter], lr=1.0)
    optimizer.register_plasticity_mask(
        parameter,
        lambda: 1.0 / (1.0 + 9.0 * heat),
    )

    def closure():
        heat.fill_(1.0)
        optimizer.zero_grad()
        parameter.sum().backward()
        return float(parameter.detach().item())

    optimizer.step(closure)

    assert torch.allclose(parameter.detach(), torch.tensor([0.9]))


def test_resaving_loaded_protected_checkpoint_preserves_expected_masks():
    module = SlowHeatLinear(3, 2)
    first = SlowHeatAdamW(module.parameters(), lr=0.01)
    first.register_slow_heat_module(module)
    protected_state = deepcopy(first.state_dict())

    restored_module = SlowHeatLinear(3, 2)
    restored = SlowHeatAdamW(restored_module.parameters(), lr=0.01)
    restored.load_state_dict(protected_state)
    resaved_state = restored.state_dict()

    assert resaved_state["slowheat_masks"] == protected_state["slowheat_masks"]


def test_loading_legacy_checkpoint_after_registration_preserves_masks():
    source_module = SlowHeatLinear(3, 2)
    source = torch.optim.AdamW(source_module.parameters(), lr=0.1)
    legacy_state = deepcopy(source.state_dict())

    module = SlowHeatLinear(3, 2, slow_strength=9.0)
    module.slow_heat.fill_(1.0)
    optimizer = SlowHeatAdamW(module.parameters(), lr=0.1)
    optimizer.register_slow_heat_module(module)
    optimizer.load_state_dict(legacy_state)

    before = module.weight.detach().clone()
    module(torch.ones(1, 3)).sum().backward()
    optimizer.step()
    protected_delta = (module.weight.detach() - before).abs().max()

    assert protected_delta < 0.02


def test_sgd_checkpoint_fails_closed_until_masks_are_registered():
    first_module = SlowHeatLinear(3, 2)
    first = SlowHeatSGD(first_module.parameters(), lr=0.1)
    first.register_slow_heat_module(first_module)
    state = deepcopy(first.state_dict())

    second_module = SlowHeatLinear(3, 2)
    second = SlowHeatSGD(second_module.parameters(), lr=0.1)
    second.load_state_dict(state)
    second_module(torch.ones(1, 3)).sum().backward()

    with pytest.raises(RuntimeError, match="checkpoint"):
        second.step()

    second.register_slow_heat_module(second_module)
    second.step()


@pytest.mark.parametrize(
    "mask, message",
    [
        (torch.tensor([float("nan")]), "finita"),
        (torch.tensor([float("inf")]), "finita"),
        (torch.tensor([-0.1]), r"\[0, 1\]"),
        (torch.tensor([1.1]), r"\[0, 1\]"),
        (torch.ones(2), "compatível"),
    ],
)
def test_generic_masks_reject_invalid_values_and_shapes(mask, message):
    parameter = torch.nn.Parameter(torch.ones(3))
    optimizer = SlowHeatSGD([parameter], lr=0.1)
    optimizer.register_plasticity_mask(parameter, mask)
    parameter.grad = torch.ones_like(parameter)

    with pytest.raises(ValueError, match=message):
        optimizer.step()


def test_factorized_mask_protects_rows_and_columns():
    source = SlowHeatLinear(2, 2, slow_strength=9.0)
    target = SlowHeatLinear(2, 2, slow_strength=9.0)
    source.slow_heat.copy_(torch.tensor([1.0, 0.0]))
    target.slow_heat.copy_(torch.tensor([0.0, 1.0]))
    optimizer = SlowHeatSGD(
        [*source.parameters(), *target.parameters()],
        lr=1.0,
    )
    optimizer.register_slow_heat_module(target, input_module=source)
    before = target.weight.detach().clone()
    target.weight.grad = torch.ones_like(target.weight)
    target.bias.grad = torch.ones_like(target.bias)

    optimizer.step()

    applied = before - target.weight.detach()
    expected = torch.tensor([[0.1, 1.0], [0.1, 0.1]])
    assert torch.allclose(applied, expected)


def test_follow_update_policy_masks_adam_moment_state():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
    optimizer = SlowHeatAdamW(
        [parameter],
        lr=0.1,
        weight_decay=0.0,
        state_policy="follow_update",
    )
    optimizer.register_plasticity_mask(parameter, torch.tensor([0.0, 1.0]))
    parameter.grad = torch.ones_like(parameter)

    optimizer.step()

    state = optimizer.state[parameter]
    assert parameter[0] == 1.0
    assert state["exp_avg"][0] == 0.0
    assert state["exp_avg_sq"][0] == 0.0
    assert state["exp_avg"][1] != 0.0
    assert state["exp_avg_sq"][1] != 0.0


def test_native_state_policy_keeps_moments_for_blocked_parameter():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = SlowHeatAdamW(
        [parameter],
        lr=0.1,
        weight_decay=0.0,
        state_policy="native",
    )
    optimizer.register_plasticity_mask(parameter, torch.tensor([0.0]))
    parameter.grad = torch.ones_like(parameter)

    optimizer.step()

    assert parameter.item() == 1.0
    assert optimizer.state[parameter]["exp_avg"].item() != 0.0


def test_checkpoint_rejects_different_optimizer_state_policy():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    first = SlowHeatAdamW([parameter], lr=0.1, state_policy="follow_update")
    state = deepcopy(first.state_dict())
    restored_parameter = torch.nn.Parameter(torch.tensor([1.0]))
    restored = SlowHeatAdamW(
        [restored_parameter],
        lr=0.1,
        state_policy="native",
    )

    with pytest.raises(ValueError, match="state_policy"):
        restored.load_state_dict(state)
