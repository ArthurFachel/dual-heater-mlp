"""F7/F8 CPU tests: each new uniform arm has its SlowHeat arm's surface plasticity, no selection."""
import pytest
import torch

from experiments.split_mnist import (
    HARD_MASK_METHODS,
    SplitMNISTConfig,
    UNIFORM_PLASTICITY_METHODS,
    _METHOD_SPECS,
    _build_optimizer,
    build_paired_models,
)

PAIRS = [
    ("slowheat_derpp_hidden_beta_30_budget_0.25", "uniform_derpp_hidden_beta_30_budget_0.25"),
    ("slowheat_replay_hidden_beta_30_budget_0.25_calibrated",
     "uniform_replay_hidden_beta_30_budget_0.25_calibrated"),
    ("hard_freeze", "uniform_hard_freeze"),
]


def _heat_up(model, seed=0):
    g = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(3):
        model(torch.randn(16, 784, generator=g)).pow(2).mean().backward()
        model.zero_grad(set_to_none=True)
    model.consolidate(strategy="max")


def _surface(optimizer):
    out = []
    for parameter, source, _ in optimizer._plasticity_masks.values():
        mask = source() if callable(source) else source
        out.append((parameter, torch.broadcast_to(mask.detach().float(), parameter.shape)))
    return out


@pytest.mark.parametrize("sh,un", PAIRS)
def test_spec_mirrors_slowheat_arm(sh, un):
    assert un in UNIFORM_PLASTICITY_METHODS
    assert _METHOD_SPECS[un] == _METHOD_SPECS[sh]
    assert (sh in HARD_MASK_METHODS) == (un in HARD_MASK_METHODS)


@pytest.mark.parametrize("sh,un", PAIRS)
def test_uniform_level_equals_slowheat_mask_mean(sh, un):
    config = SplitMNISTConfig(
        hidden_dims=(32, 16), train_per_class=20, validation_per_class=5,
        test_per_class=5, slow_strength=30.0, plasticity_budget=0.25,
        methods=("vanilla", sh, un), device="cpu",
    )
    models = build_paired_models(config)
    sh_model, un_model = models[sh], models[un]
    un_model.load_state_dict(sh_model.state_dict())
    _heat_up(sh_model)
    _heat_up(un_model)
    sh_opt = _build_optimizer(sh, sh_model, config)
    un_opt = _build_optimizer(un, un_model, config)
    sh_surface, un_surface = _surface(sh_opt), _surface(un_opt)
    assert [p.shape for p, _ in sh_surface] == [p.shape for p, _ in un_surface]
    kept = sum(float(m.sum()) for _, m in sh_surface)
    total = sum(p.numel() for p, _ in sh_surface)
    level = un_opt.uniform_plasticity.refresh()
    assert abs(level - kept / total) < 1e-6
    assert 0.0 < level < 1.0
    for _, mask in un_surface:
        assert torch.allclose(mask, torch.full_like(mask, level))
    values = torch.cat([m.flatten() for _, m in sh_surface])
    assert float(values.max() - values.min()) > 0.1
    if sh in HARD_MASK_METHODS:
        assert set(values.unique().tolist()) <= {0.0, 1.0}


def test_uniform_hard_step_moves_every_covered_entry():
    """Hard arm freezes some entries bit for bit; its uniform control freezes none."""
    sh, un = "hard_freeze", "uniform_hard_freeze"
    config = SplitMNISTConfig(
        hidden_dims=(32, 16), train_per_class=20, validation_per_class=5,
        test_per_class=5, slow_strength=30.0, plasticity_budget=0.25,
        methods=("vanilla", sh, un), device="cpu",
    )
    models = build_paired_models(config)
    frozen, frozen_with_grad = {}, {}
    for name in (sh, un):
        model = models[name]
        _heat_up(model)
        opt = _build_optimizer(name, model, config)
        before = [p.detach().clone() for p, _ in _surface(opt)]
        model(torch.randn(8, 784)).pow(2).mean().backward()
        grads = [p.grad.detach().clone() for p, _ in _surface(opt)]
        opt.step()
        same = [(p.detach() == b) for (p, _), b in zip(_surface(opt), before)]
        frozen[name] = sum(int(m.sum()) for m in same)
        # entries that stayed put although they had a nonzero gradient = masked out
        frozen_with_grad[name] = sum(int((m & (g != 0)).sum()) for m, g in zip(same, grads))
    assert frozen_with_grad[sh] > 0
    assert frozen_with_grad[un] == 0  # zero-grad entries (dead units) may stay put
