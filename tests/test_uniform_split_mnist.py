"""F4 CPU tests: the uniform arm has the same surface plasticity as SlowHeat, no selection."""
import torch

from experiments.split_mnist import (
    SplitMNISTConfig,
    UNIFORM_PLASTICITY_METHODS,
    _build_optimizer,
    build_paired_models,
)

SH = "slowheat_replay_hidden_beta_30_budget_0.25"
UN = "uniform_replay_hidden_beta_30_budget_0.25"


def _setup():
    config = SplitMNISTConfig(
        hidden_dims=(32, 16), train_per_class=20, validation_per_class=5,
        test_per_class=5, slow_strength=30.0, plasticity_budget=0.25,
        methods=("replay", SH, UN), device="cpu",
    )
    models = build_paired_models(config)
    return config, models


def _heat_up(model, layers_seed=0):
    """Real importance: a few backward passes on random inputs, then consolidate."""
    g = torch.Generator().manual_seed(layers_seed)
    model.train()
    for _ in range(3):
        x = torch.randn(16, 784, generator=g)
        model(x).pow(2).mean().backward()
        model.zero_grad(set_to_none=True)
    model.consolidate(strategy="max")


def _surface(optimizer):
    out = []
    for parameter, source, _ in optimizer._plasticity_masks.values():
        mask = source() if callable(source) else source
        out.append((parameter, torch.broadcast_to(mask.detach().float(), parameter.shape)))
    return out


def test_uniform_level_equals_slowheat_mask_mean():
    assert UN in UNIFORM_PLASTICITY_METHODS
    config, models = _setup()
    sh_model, un_model = models[SH], models[UN]
    un_model.load_state_dict(sh_model.state_dict())
    _heat_up(sh_model)
    _heat_up(un_model)
    sh_opt = _build_optimizer(SH, sh_model, config)
    un_opt = _build_optimizer(UN, un_model, config)
    sh_surface = _surface(sh_opt)
    un_surface = _surface(un_opt)
    kept = sum(float(m.sum()) for _, m in sh_surface)
    total = sum(p.numel() for p, _ in sh_surface)
    level = un_opt.uniform_plasticity.refresh()
    assert abs(level - kept / total) < 1e-6
    # same covered parameters, every covered entry gets the same factor
    assert len(sh_surface) == len(un_surface)
    for _, mask in un_surface:
        assert torch.allclose(mask, torch.full_like(mask, level))
    # the SlowHeat mask itself is not uniform (selection exists to be removed)
    values = torch.cat([m.flatten() for _, m in sh_surface])
    assert float(values.max() - values.min()) > 0.1
    assert 0.0 < level < 1.0


def test_level_tracks_consolidation_through_step_hook():
    config, models = _setup()
    model = models[UN]
    opt = _build_optimizer(UN, model, config)
    before = opt.uniform_plasticity.refresh()
    _heat_up(model, layers_seed=1)  # consolidation changes the SlowHeat mask
    x = torch.randn(8, 784)
    model(x).sum().backward()
    opt.step()
    after = float(opt.uniform_plasticity.level)
    assert after != before


def test_mutation_selection_leaks_is_caught():
    """Mutant: uniform arm keeps the original masks -> must fail the uniformity check."""
    config, models = _setup()
    model = models[SH]
    _heat_up(model)
    opt = _build_optimizer(SH, model, config)
    masks = [m for _, m in _surface(opt)]
    assert not all(torch.allclose(m, torch.full_like(m, float(m.mean()))) for m in masks)
