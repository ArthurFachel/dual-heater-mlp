"""T3c CPU tests: same-shape random control with a FIXED random order (R in the decomposition).

Each random_fixed arm mirrors its SlowHeat arm field by field. After every
consolidation the per-layer heat values are sorted and handed to the units of ONE
random order per layer, drawn once per run from the seed. Count, multiset of values
and hardness are preserved; only which units carry them changes, and that choice is
constant across boundaries (nested prefixes when the count grows).
"""
import pytest
import torch

from experiments.split_mnist import (
    HARD_MASK_METHODS,
    RANDOM_FIXED_METHODS,
    UNIFORM_PLASTICITY_METHODS,
    SplitMNISTConfig,
    _METHOD_SPECS,
    _method_spec,
    _build_optimizer,
    build_paired_models,
    reassign_to_fixed_random_units,
    run_split_mnist,
)
from tests.test_split_mnist import _tiny_tasks

PAIRS = [
    ("slowheat_replay_hidden_beta_30_budget_0.25", "random_fixed_replay_hidden_beta_30_budget_0.25"),
    ("slowheat_derpp_hidden_beta_30_budget_0.25", "random_fixed_derpp_hidden_beta_30_budget_0.25"),
    ("slowheat_replay_hidden_beta_30_budget_0.25_calibrated",
     "random_fixed_replay_hidden_beta_30_budget_0.25_calibrated"),
    ("slowheat_er_ace_hidden_beta_30_budget_0.25", "random_fixed_er_ace_hidden_beta_30_budget_0.25"),
    ("hard_freeze", "random_fixed_hard_freeze"),
]


def _heat_up(model, seed=0):
    g = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(3):
        model(torch.randn(16, 784, generator=g)).pow(2).mean().backward()
        model.zero_grad(set_to_none=True)
    model.consolidate(strategy="max")


def _config(*methods):
    return SplitMNISTConfig(
        hidden_dims=(32, 16), train_per_class=20, validation_per_class=5,
        test_per_class=5, slow_strength=30.0, plasticity_budget=0.25,
        methods=("vanilla", *methods), device="cpu",
    )


def _layers(model):
    return list(model.get_slow_layers())


@pytest.mark.parametrize("sh,rf", PAIRS)
def test_spec_mirrors_slowheat_arm(sh, rf):
    assert rf in RANDOM_FIXED_METHODS
    assert rf not in UNIFORM_PLASTICITY_METHODS
    assert _METHOD_SPECS[rf] == _method_spec(sh)
    assert (sh in HARD_MASK_METHODS) == (rf in HARD_MASK_METHODS)


def test_reassign_preserves_values_and_count_but_moves_units():
    models = build_paired_models(_config(PAIRS[0][0]))
    model = models[PAIRS[0][0]]
    _heat_up(model)
    before = [layer.slow_heat.detach().clone() for layer in _layers(model)]
    record = reassign_to_fixed_random_units(model, seed=123)
    after = [layer.slow_heat.detach().clone() for layer in _layers(model)]
    moved = False
    for b, a in zip(before, after):
        assert torch.equal(torch.sort(b).values, torch.sort(a).values)
        assert int((b > 0).sum()) == int((a > 0).sum())
        moved |= not torch.equal(b, a)
    assert moved
    assert record["protected_counts"] == [int((a > 0).sum()) for a in after]
    assert 0.0 <= record["overlap_with_own_selection"] < 1.0


def test_protected_units_are_prefix_of_one_fixed_order_across_boundaries():
    models = build_paired_models(_config(PAIRS[0][0]))
    model = models[PAIRS[0][0]]
    _heat_up(model, seed=0)
    first = reassign_to_fixed_random_units(model, seed=7)
    sets1 = [set(torch.nonzero(l.slow_heat > 0).flatten().tolist()) for l in _layers(model)]
    _heat_up(model, seed=99)  # second boundary, different importance
    second = reassign_to_fixed_random_units(model, seed=7)
    sets2 = [set(torch.nonzero(l.slow_heat > 0).flatten().tolist()) for l in _layers(model)]
    for s1, s2, c1, c2, d1, d2 in zip(sets1, sets2, first["protected_counts"],
                                     second["protected_counts"], first["digests"],
                                     second["digests"]):
        if c1 == c2:
            assert s1 == s2 and d1 == d2
        else:
            assert (s1 <= s2) if c1 < c2 else (s2 <= s1)


def test_order_depends_on_seed_only():
    cfg = _config(PAIRS[0][0])
    a = build_paired_models(cfg)[PAIRS[0][0]]
    b = build_paired_models(cfg)[PAIRS[0][0]]
    b.load_state_dict(a.state_dict())
    _heat_up(a)
    _heat_up(b)
    ra = reassign_to_fixed_random_units(a, seed=5)
    rb = reassign_to_fixed_random_units(b, seed=5)
    assert ra["digests"] == rb["digests"]
    c = build_paired_models(cfg)[PAIRS[0][0]]
    c.load_state_dict(a.state_dict())
    _heat_up(c)
    rc = reassign_to_fixed_random_units(c, seed=6)
    assert rc["digests"] != ra["digests"]


def test_hard_random_mask_is_binary_with_same_frozen_count():
    sh, rf = "hard_freeze", "random_fixed_hard_freeze"
    models = build_paired_models(_config(sh, rf))
    models[rf].load_state_dict(models[sh].state_dict())
    _heat_up(models[sh])
    _heat_up(models[rf])
    reassign_to_fixed_random_units(models[rf], seed=11)
    cfg = _config(sh, rf)
    masks = {}
    for name in (sh, rf):
        opt = _build_optimizer(name, models[name], cfg)
        values = []
        for parameter, source, _ in opt._plasticity_masks.values():
            mask = source() if callable(source) else source
            values.append(torch.broadcast_to(mask.detach().float(), parameter.shape).flatten())
        masks[name] = torch.cat(values)
    assert set(masks[rf].unique().tolist()) <= {0.0, 1.0}
    assert int((masks[sh] == 0).sum()) == int((masks[rf] == 0).sum())
    assert not torch.equal(masks[sh], masks[rf])


def test_runner_applies_reassignment_at_every_boundary_and_records_it():
    rf = PAIRS[0][1]
    config = _config(PAIRS[0][0], rf)
    results = run_split_mnist(config, _tiny_tasks(config))
    history = results[rf]["random_fixed_history"]
    assert len(history) == len(results[rf]["capacity_history"]) >= 2
    counts = [h["protected_counts"] for h in history]
    digests = [h["digests"] for h in history]
    for k in range(1, len(history)):
        for layer in range(len(counts[k])):
            if counts[k][layer] == counts[k - 1][layer]:
                assert digests[k][layer] == digests[k - 1][layer]
    assert results[PAIRS[0][0]].get("random_fixed_history", []) == []
