"""Uniform plasticity control (`plasticity_mask_mode="uniform_hard"`).

The control must give every covered entry the same update factor, equal to the
mean of the hard mask over the covered surface, so that it removes exactly the
same surface plasticity as the hard arms without selecting or freezing anything.
"""

from __future__ import annotations

import torch
from torch import nn

from dual_heater.optim import PlasticityMaskBinding, SlowHeatAdamW
from experiments.split_clinc150 import SplitCLINC150Config, UniformPlasticity


def _bindings(weight: nn.Parameter, bias: nn.Parameter, protected_rows: list[int]):
    row_mask = torch.ones(weight.shape[0])
    row_mask[protected_rows] = 0.0
    return [
        PlasticityMaskBinding(weight, lambda: row_mask[:, None], "slowheat_hard_weight"),
        PlasticityMaskBinding(bias, lambda: row_mask, "slowheat_hard_bias"),
    ]


def test_level_is_mean_of_hard_mask_over_covered_surface():
    layer = nn.Linear(4, 8)
    holder = UniformPlasticity(_bindings(layer.weight, layer.bias, [0, 1, 2, 3, 4, 5]))
    assert float(holder.level) == 1.0  # before any consolidation nothing is protected
    level = holder.refresh(stage=0)
    # 6 of 8 rows protected: weight 6*4 + bias 6 = 30 of 40 entries
    assert abs(level - 10 / 40) < 1e-9
    assert abs(float(holder.level) - 10 / 40) < 1e-7  # the optimizer reads holder.level
    assert holder.history[-1]["hard_protected_entries"] == 30
    assert holder.history[-1]["covered_entries"] == 40


def test_uniform_bindings_scale_every_entry_by_the_same_factor():
    torch.manual_seed(0)
    reference = nn.Linear(4, 8)
    treated = nn.Linear(4, 8)
    treated.load_state_dict(reference.state_dict())
    x, y = torch.randn(16, 4), torch.randn(16, 8)

    ref_opt = SlowHeatAdamW(reference.parameters(), lr=1e-2, weight_decay=0.0)
    opt = SlowHeatAdamW(treated.parameters(), lr=1e-2, weight_decay=0.0)
    holder = UniformPlasticity(_bindings(treated.weight, treated.bias, [0, 1, 2, 3, 4, 5]))
    holder.refresh(stage=0)
    opt.register_mask_bindings(holder.bindings())

    before = treated.weight.detach().clone()
    for model, optimizer in ((reference, ref_opt), (treated, opt)):
        optimizer.zero_grad()
        nn.functional.mse_loss(model(x), y).backward()
        optimizer.step()

    native = reference.weight.detach() - before
    applied = treated.weight.detach() - before
    # nothing frozen: every entry moves, protected rows included
    assert torch.all(applied[native != 0] != 0)
    # and every entry moves by exactly level * native delta
    assert torch.allclose(applied, 0.25 * native, atol=1e-7)  # 10 of 40 entries plastic


def test_uniform_kinds_are_distinct_from_hard_kinds():
    layer = nn.Linear(2, 2)
    holder = UniformPlasticity(_bindings(layer.weight, layer.bias, [0]))
    assert [b.kind for b in holder.bindings()] == ["slowheat_uniform_weight", "slowheat_uniform_bias"]


def test_config_accepts_uniform_hard_and_still_rejects_unknown_modes():
    SplitCLINC150Config(plasticity_mask_mode="uniform_hard").validate()
    try:
        SplitCLINC150Config(plasticity_mask_mode="nope").validate()
    except ValueError:
        return
    raise AssertionError("unknown mask mode must be rejected")
