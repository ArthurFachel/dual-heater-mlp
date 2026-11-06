"""uniform_soft: plasticity-matched control for the SOFT mask (exploratory, 2x2 run)."""

from __future__ import annotations

import torch
from torch import nn

from dual_heater.optim import PlasticityMaskBinding, SlowHeatAdamW
from experiments.split_clinc150 import SplitCLINC150Config, UniformPlasticity


def test_config_accepts_uniform_soft():
    SplitCLINC150Config(plasticity_mask_mode="uniform_soft").validate()


def test_uniform_level_matches_mean_soft_factor_and_scales_every_entry():
    torch.manual_seed(0)
    reference, treated = nn.Linear(4, 8), nn.Linear(4, 8)
    treated.load_state_dict(reference.state_dict())
    soft_rows = torch.tensor([1.0, 1.0, 0.5, 0.25, 0.1, 1.0, 0.2, 1.0])
    soft = [
        PlasticityMaskBinding(treated.weight, lambda: soft_rows[:, None], "slowheat_ffn_weight"),
        PlasticityMaskBinding(treated.bias, lambda: soft_rows, "slowheat_ffn_bias"),
    ]
    holder = UniformPlasticity(soft)
    level = holder.refresh(stage=0)
    expected = float((soft_rows.sum() * 4 + soft_rows.sum()) / 40)
    assert abs(level - expected) < 1e-9

    x, y = torch.randn(16, 4), torch.randn(16, 8)
    ref_opt = SlowHeatAdamW(reference.parameters(), lr=1e-2, weight_decay=0.0)
    opt = SlowHeatAdamW(treated.parameters(), lr=1e-2, weight_decay=0.0)
    opt.register_mask_bindings(holder.bindings())
    before = treated.weight.detach().clone()
    for model, optimizer in ((reference, ref_opt), (treated, opt)):
        optimizer.zero_grad()
        nn.functional.mse_loss(model(x), y).backward()
        optimizer.step()
    native = reference.weight.detach() - before
    applied = treated.weight.detach() - before
    assert torch.allclose(applied, expected * native, atol=1e-7)
