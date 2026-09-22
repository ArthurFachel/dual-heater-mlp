"""CPU tests for the Qwen2 Functional SlowHeat host.

Every model here is a tiny randomly-initialized Qwen2 (hidden 8, intermediate
12, 1-2 layers) so the whole file runs on CPU in seconds and never downloads a
checkpoint.
"""

from copy import deepcopy

import pytest
import torch

transformers = pytest.importorskip("transformers")

from dual_heater.optim import SlowHeatAdamW
from dual_heater.qwen import (
    QwenSlowHeatConfig,
    SlowHeatQwen2ForSequenceClassification,
)
from dual_heater.transformer import SlowHeatFFNTracker


def _qwen_config(*, layers: int = 1, labels: int = 4, **overrides):
    values = {
        "vocab_size": 64,
        "hidden_size": 8,
        "intermediate_size": 12,
        "num_hidden_layers": layers,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "num_labels": labels,
        "attention_dropout": 0.0,
        "tie_word_embeddings": False,
        "pad_token_id": 0,
    }
    values.update(overrides)
    return transformers.Qwen2Config(**values)


def _backward(model, input_ids, attention_mask, labels=None):
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels if labels is not None else torch.tensor([1]),
    )
    output.loss.backward()
    return output


def _tokens(model=None):
    return torch.tensor([[2, 7, 9, 3]]), torch.ones(1, 4, dtype=torch.long)


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


def test_default_config_tracks_only_the_gated_ffn():
    config = QwenSlowHeatConfig()

    assert config.track_ffn is True
    assert config.protect_score is False
    assert config.keep_score_plastic is True
    assert config.capacity_scope == "local"


def test_config_validates_budgets_and_strength():
    with pytest.raises(ValueError, match="ffn_plasticity_budget"):
        QwenSlowHeatConfig(ffn_plasticity_budget=1.5)
    with pytest.raises(ValueError, match="score_plasticity_budget"):
        QwenSlowHeatConfig(score_plasticity_budget=-0.1)
    with pytest.raises(ValueError, match="slow_strength"):
        QwenSlowHeatConfig(slow_strength=-1.0)
    with pytest.raises(ValueError, match="importance_decay"):
        QwenSlowHeatConfig(importance_decay=1.0)


def test_config_rejects_protecting_and_exempting_the_same_head():
    with pytest.raises(ValueError, match="mutuamente exclusivos"):
        QwenSlowHeatConfig(protect_score=True, keep_score_plastic=True)


def test_config_rejects_every_family_disabled():
    with pytest.raises(ValueError, match="ao menos uma família"):
        QwenSlowHeatConfig(track_ffn=False, protect_score=False)


# ---------------------------------------------------------------------------
# instrumentation placement
# ---------------------------------------------------------------------------


def test_one_ffn_tracker_per_layer_sized_by_intermediate():
    model = SlowHeatQwen2ForSequenceClassification(_qwen_config(layers=2))

    trackers = model.get_ffn_trackers()
    assert len(trackers) == 2
    assert all(isinstance(tracker, SlowHeatFFNTracker) for tracker in trackers)
    assert all(tracker.units == 12 for tracker in trackers)


def test_functional_importance_is_invariant_across_the_gated_product_chain():
    """`|z * dL/dz|` is the same at gate, up and their product.

    For an elementwise product z = a * u we have dL/da = dL/dz * u, so
    |a * dL/da| = |a*u * dL/dz| = |z * dL/dz|, and symmetrically for u. The
    estimator therefore ranks units identically whichever of the three tensors
    is observed. This is a real property of the estimator, not an accident of
    this seed, and it is what makes the gated MLP safe to instrument at a
    single point. The test pins it so a future change to `_reduce_contribution`
    or to the normalization cannot break it silently.
    """

    torch.manual_seed(7)
    model = SlowHeatQwen2ForSequenceClassification(_qwen_config())
    model.train()
    layer = model.model.layers[0]

    captured: dict[str, torch.Tensor] = {}

    def remember(key, tensor):
        captured[f"{key}_value"] = tensor.detach().clone()
        tensor.register_hook(
            lambda grad, name=key: captured.__setitem__(
                f"{name}_grad", grad.detach().clone()
            )
        )

    def pre_hook(_module, inputs):
        remember("gated", inputs[0])

    def post_hook(key):
        def hook(_module, _inputs, output):
            remember(key, output)

        return hook

    layer.mlp.down_proj.register_forward_pre_hook(pre_hook)
    layer.mlp.act_fn.register_forward_hook(post_hook("act"))
    layer.mlp.up_proj.register_forward_hook(post_hook("up"))

    input_ids, attention_mask = _tokens()
    _backward(model, input_ids, attention_mask)

    def expected_signal(key):
        contribution = (
            captured[f"{key}_value"].float().abs()
            * captured[f"{key}_grad"].float().abs()
        ).sum(dim=(0, 1))
        return contribution / contribution.mean().clamp_min(1e-8)

    gated = expected_signal("gated")
    # The signal must be non-degenerate, otherwise every comparison is vacuous.
    assert gated.std() > 0.1

    torch.testing.assert_close(expected_signal("act"), gated, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(expected_signal("up"), gated, atol=1e-5, rtol=1e-5)

    tracker = model.get_ffn_trackers()[0]
    assert tracker.task_step.item() == 1
    torch.testing.assert_close(tracker.task_ema, gated, atol=1e-5, rtol=1e-5)


def test_tracker_is_bound_to_the_gated_product_tensor():
    """The hook must sit on down_proj's input, the unit's actual value.

    The importance signal is invariant across the chain (see the test above),
    but the *placement* still matters: down_proj's input is the tensor the
    column mask on `down_proj.weight` multiplies, so observing it keeps the
    tracker and the mask addressing the same object. This asserts the binding
    structurally rather than through the (invariant) numbers.
    """

    model = SlowHeatQwen2ForSequenceClassification(_qwen_config(layers=2))

    for layer in model.model.layers:
        assert len(layer.mlp.down_proj._forward_pre_hooks) == 1
        assert len(layer.mlp.act_fn._forward_hooks) == 0
        assert len(layer.mlp.up_proj._forward_hooks) == 0
        assert len(layer.mlp.gate_proj._forward_hooks) == 0


def test_hook_forwards_the_attention_mask_to_the_tracker():
    """The host must hand the tracker the real validity mask, not None.

    End-to-end this is unobservable (causal attention plus last-token pooling
    already zeroes pad gradients), so the contract is asserted directly on the
    call. Without it a future change to pooling would silently start counting
    padding.
    """

    model = SlowHeatQwen2ForSequenceClassification(_qwen_config())
    model.train()
    tracker = model.get_ffn_trackers()[0]
    seen: list[torch.Tensor | None] = []
    original = tracker.observe

    def spy(hidden, validity_mask=None):
        seen.append(validity_mask)
        return original(hidden, validity_mask)

    tracker.observe = spy
    mask = torch.tensor([[1, 1, 1, 1, 0, 0]])
    _backward(model, torch.tensor([[2, 7, 9, 3, 0, 0]]), mask)

    assert len(seen) == 1
    assert seen[0] is not None
    torch.testing.assert_close(seen[0], mask.detach())


def test_validity_mask_actually_excludes_masked_tokens():
    """A masked token must contribute nothing to functional utility.

    This exercises the tracker directly because the end-to-end path cannot
    discriminate: see the padding test below.
    """

    tracker = SlowHeatFFNTracker(3, importance_decay=0.0)
    tracker.train()
    hidden = torch.tensor(
        [[[1.0, 2.0, 3.0], [100.0, 0.0, 0.0]]], requires_grad=True
    )
    mask = torch.tensor([[1.0, 0.0]])

    tracker.observe(hidden, mask)
    hidden.sum().backward()

    # Only the first token counts: contribution = |value| * |grad=1| = value.
    expected = torch.tensor([1.0, 2.0, 3.0])
    torch.testing.assert_close(
        tracker.task_ema, expected / expected.mean(), atol=1e-6, rtol=1e-6
    )
    # Unit 0 would dominate if the masked token had leaked in.
    assert tracker.task_ema.argmax().item() == 2


def test_padding_tokens_do_not_change_functional_utility():
    """End-to-end regression for right-padded batches.

    Note this is doubly guaranteed: causal attention plus last-non-pad-token
    pooling already drives the gradient at pad positions to exactly zero, so
    this test passes even without the validity mask. It is kept as a regression
    against a future change to pooling or attention, not as evidence that the
    mask works — `test_validity_mask_actually_excludes_masked_tokens` is that
    evidence.
    """

    torch.manual_seed(81)
    short = SlowHeatQwen2ForSequenceClassification(_qwen_config())
    padded = SlowHeatQwen2ForSequenceClassification(_qwen_config())
    padded.load_state_dict(deepcopy(short.state_dict()))
    short.train()
    padded.train()

    _backward(short, torch.tensor([[2, 7, 9, 3]]), torch.ones(1, 4, dtype=torch.long))
    _backward(
        padded,
        torch.tensor([[2, 7, 9, 3, 0, 0]]),
        torch.tensor([[1, 1, 1, 1, 0, 0]]),
    )

    torch.testing.assert_close(
        short.get_ffn_trackers()[0].task_ema,
        padded.get_ffn_trackers()[0].task_ema,
        atol=1e-5,
        rtol=1e-5,
    )


def test_eval_mode_records_no_importance():
    model = SlowHeatQwen2ForSequenceClassification(_qwen_config())
    model.eval()
    input_ids, attention_mask = _tokens()

    model(input_ids=input_ids, attention_mask=attention_mask)

    assert model.get_ffn_trackers()[0].task_step.item() == 0


def test_remove_instrumentation_drops_every_hook():
    model = SlowHeatQwen2ForSequenceClassification(_qwen_config(layers=2))
    model.train()
    model.remove_slowheat_instrumentation()

    input_ids, attention_mask = _tokens()
    _backward(model, input_ids, attention_mask)

    assert all(
        tracker.task_step.item() == 0 for tracker in model.get_ffn_trackers()
    )

    model.reinstall_slowheat_instrumentation()
    model.zero_grad()
    _backward(model, input_ids, attention_mask)
    assert all(
        tracker.task_step.item() == 1 for tracker in model.get_ffn_trackers()
    )


def test_gradient_checkpointing_is_rejected_while_training():
    model = SlowHeatQwen2ForSequenceClassification(_qwen_config())
    model.gradient_checkpointing_enable()
    model.train()
    input_ids, attention_mask = _tokens()

    with pytest.raises(RuntimeError, match="activation checkpointing"):
        model(input_ids=input_ids, attention_mask=attention_mask)


def test_forward_rejects_malformed_attention_mask():
    model = SlowHeatQwen2ForSequenceClassification(_qwen_config())
    input_ids, _ = _tokens()

    with pytest.raises(ValueError, match=r"\[B, T\]"):
        model(input_ids=input_ids, attention_mask=torch.ones(1, 4, 1))


# ---------------------------------------------------------------------------
# mask bindings
# ---------------------------------------------------------------------------


def test_bindings_cover_gate_up_and_down_exactly_once():
    model = SlowHeatQwen2ForSequenceClassification(_qwen_config(layers=2))
    bindings = model.mask_bindings()

    identifiers = [id(binding.parameter) for binding in bindings]
    assert len(identifiers) == len(set(identifiers))
    assert len(bindings) == 6

    expected = set()
    for layer in model.model.layers:
        expected.update(
            {
                id(layer.mlp.gate_proj.weight),
                id(layer.mlp.up_proj.weight),
                id(layer.mlp.down_proj.weight),
            }
        )
    assert set(identifiers) == expected


def test_producer_masks_are_rows_and_consumer_mask_is_columns():
    model = SlowHeatQwen2ForSequenceClassification(_qwen_config())
    tracker = model.get_ffn_trackers()[0]
    tracker.slow_heat.copy_(torch.linspace(0.0, 1.0, tracker.units))

    by_kind = {binding.kind: binding for binding in model.mask_bindings()}
    gate = next(b for k, b in by_kind.items() if k.endswith("_gate_producer_rows"))
    down = next(b for k, b in by_kind.items() if k.endswith("_down_consumer_columns"))

    gate_mask = gate.mask()
    down_mask = down.mask()
    # gate_proj.weight is [intermediate, hidden] -> row vector
    assert gate_mask.shape == (12, 1)
    # down_proj.weight is [hidden, intermediate] -> column vector
    assert down_mask.shape == (1, 12)
    torch.testing.assert_close(gate_mask.reshape(-1), down_mask.reshape(-1))


def test_every_mask_broadcasts_to_its_parameter_and_stays_in_unit_range():
    model = SlowHeatQwen2ForSequenceClassification(_qwen_config(layers=2))
    for tracker in model.get_slow_states():
        tracker.slow_heat.copy_(
            torch.linspace(0.0, 1.0, tracker.slow_heat.numel())
        )

    for binding in model.mask_bindings():
        mask = binding.mask() if callable(binding.mask) else binding.mask
        assert (
            torch.broadcast_shapes(mask.shape, binding.parameter.shape)
            == binding.parameter.shape
        )
        assert torch.isfinite(mask).all()
        assert torch.all((0.0 <= mask) & (mask <= 1.0))


def test_hard_masks_freeze_protected_units_exactly():
    torch.manual_seed(13)
    model = SlowHeatQwen2ForSequenceClassification(_qwen_config())
    model.train()
    tracker = model.get_ffn_trackers()[0]
    tracker.slow_heat.zero_()
    tracker.slow_heat[0] = 1.0  # protect unit 0 only

    gate = model.model.layers[0].mlp.gate_proj.weight
    down = model.model.layers[0].mlp.down_proj.weight
    gate_before = gate.detach().clone()
    down_before = down.detach().clone()

    optimizer = SlowHeatAdamW(model.parameters(), lr=0.05, weight_decay=0.1)
    model.register_plasticity_masks(optimizer, hard=True)
    input_ids, attention_mask = _tokens()
    _backward(model, input_ids, attention_mask)
    optimizer.step()

    # Protected producer row and consumer column must not move at all.
    torch.testing.assert_close(gate[0], gate_before[0], atol=0.0, rtol=0.0)
    torch.testing.assert_close(down[:, 0], down_before[:, 0], atol=0.0, rtol=0.0)
    # Plastic units must have moved, otherwise the test would pass vacuously.
    assert not torch.allclose(gate[1:], gate_before[1:])


def test_register_plasticity_masks_requires_a_slowheat_optimizer():
    model = SlowHeatQwen2ForSequenceClassification(_qwen_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    with pytest.raises(TypeError, match="register_mask_bindings"):
        model.register_plasticity_masks(optimizer)


# ---------------------------------------------------------------------------
# consolidation
# ---------------------------------------------------------------------------


def test_consolidate_reserves_the_plasticity_budget():
    torch.manual_seed(5)
    model = SlowHeatQwen2ForSequenceClassification(
        _qwen_config(), QwenSlowHeatConfig(ffn_plasticity_budget=0.25)
    )
    model.train()
    input_ids, attention_mask = _tokens()
    _backward(model, input_ids, attention_mask)

    model.consolidate()

    tracker = model.get_ffn_trackers()[0]
    protected = int((tracker.slow_heat > 0.0).sum().item())
    assert protected <= int(0.75 * tracker.units)
    assert tracker.task_step.item() == 0
    assert tracker.consolidated_tasks.item() == 1


def test_consolidate_without_backward_is_rejected():
    model = SlowHeatQwen2ForSequenceClassification(_qwen_config())

    with pytest.raises(RuntimeError, match="sem backward"):
        model.consolidate()


@pytest.mark.parametrize("scope", ["local", "global", "hierarchical"])
def test_every_capacity_scope_consolidates_two_layers(scope):
    torch.manual_seed(11)
    model = SlowHeatQwen2ForSequenceClassification(
        _qwen_config(layers=2), QwenSlowHeatConfig(capacity_scope=scope)
    )
    model.train()
    input_ids, attention_mask = _tokens()
    _backward(model, input_ids, attention_mask)

    model.consolidate()

    trackers = model.get_ffn_trackers()
    total_units = sum(tracker.units for tracker in trackers)
    total_protected = sum(
        int((tracker.slow_heat > 0.0).sum().item()) for tracker in trackers
    )
    assert total_protected <= int(0.75 * total_units)
    assert all(torch.isfinite(tracker.slow_heat).all() for tracker in trackers)


def _seed_lopsided_importance(model):
    """Give layer 0 large utility and layer 1 negligible utility.

    Under a local budget each layer keeps its own 75% quota. Under a global or
    hierarchical budget the pooled ranking shifts protection toward layer 0.
    The two regimes therefore produce different per-layer counts, which is what
    makes `capacity_scope` observable.
    """

    first, second = model.get_ffn_trackers()
    with torch.no_grad():
        first.task_ema.copy_(torch.linspace(10.0, 21.0, first.units))
        second.task_ema.copy_(torch.linspace(0.001, 0.012, second.units))
        first.task_step.fill_(1)
        second.task_step.fill_(1)


def _protected_per_layer(model):
    return [
        int((tracker.slow_heat > 0.0).sum().item())
        for tracker in model.get_ffn_trackers()
    ]


def test_local_scope_gives_each_layer_its_own_quota():
    model = SlowHeatQwen2ForSequenceClassification(
        _qwen_config(layers=2), QwenSlowHeatConfig(capacity_scope="local")
    )
    _seed_lopsided_importance(model)

    model.consolidate()

    # floor(0.75 * 12) = 9 protected units in each layer, independently.
    assert _protected_per_layer(model) == [9, 9]


@pytest.mark.parametrize("scope", ["global", "hierarchical"])
def test_pooled_scopes_shift_protection_toward_the_useful_layer(scope):
    model = SlowHeatQwen2ForSequenceClassification(
        _qwen_config(layers=2), QwenSlowHeatConfig(capacity_scope=scope)
    )
    _seed_lopsided_importance(model)

    model.consolidate()

    counts = _protected_per_layer(model)
    # 18 units protected overall, but concentrated where utility actually is.
    assert sum(counts) == 18
    assert counts[0] > counts[1]
    assert counts != [9, 9]


def test_pooled_scopes_reject_a_non_uniform_budget_within_a_family():
    model = SlowHeatQwen2ForSequenceClassification(
        _qwen_config(layers=2), QwenSlowHeatConfig(capacity_scope="global")
    )
    _seed_lopsided_importance(model)
    model.get_ffn_trackers()[1].plasticity_budget = 0.5

    with pytest.raises(RuntimeError, match="uniforme"):
        model.consolidate()


def test_max_consolidation_keeps_the_stronger_task_signal():
    model = SlowHeatQwen2ForSequenceClassification(_qwen_config())
    tracker = model.get_ffn_trackers()[0]
    tracker.importance_memory.copy_(torch.full((12,), 0.5))
    tracker.task_ema.copy_(torch.full((12,), 0.1))
    tracker.task_step.fill_(1)

    model.consolidate(strategy="max")

    torch.testing.assert_close(
        tracker.importance_memory, torch.full((12,), 0.5)
    )


# ---------------------------------------------------------------------------
# coverage accounting
# ---------------------------------------------------------------------------


def test_freeze_unbound_keeps_only_ffn_and_the_exempt_head_trainable():
    model = SlowHeatQwen2ForSequenceClassification(
        _qwen_config(layers=2),
        QwenSlowHeatConfig(freeze_unbound_parameters=True),
    )

    trainable = {
        name for name, p in model.named_parameters() if p.requires_grad
    }
    assert trainable == {
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.1.mlp.gate_proj.weight",
        "model.layers.1.mlp.up_proj.weight",
        "model.layers.1.mlp.down_proj.weight",
        "score.weight",
    }
    assert model.uncovered_trainable_parameters() == []
    assert model.exempt_parameter_names() == ["score.weight"]


def test_coverage_summary_reports_exempt_parameters_separately():
    model = SlowHeatQwen2ForSequenceClassification(
        _qwen_config(),
        QwenSlowHeatConfig(freeze_unbound_parameters=True),
    )

    summary = model.mask_coverage_summary()
    assert summary["binding_count"] == 3
    assert summary["exempt_parameter_count"] == model.score.weight.numel()
    assert (
        summary["masked_parameter_count"] + summary["exempt_parameter_count"]
        == summary["trainable_parameter_count"]
    )
    assert 0.0 < summary["masked_fraction"] < 1.0


def test_uncovered_parameters_are_reported_without_freezing():
    model = SlowHeatQwen2ForSequenceClassification(_qwen_config())

    uncovered = model.uncovered_trainable_parameters()
    assert "model.embed_tokens.weight" in uncovered
    assert "model.layers.0.self_attn.q_proj.weight" in uncovered
    assert "score.weight" not in uncovered  # declared exemption

    with pytest.raises(RuntimeError, match="sem máscara"):
        model.validate_trainable_mask_coverage()


def test_frozen_attention_does_not_drift_during_a_step():
    torch.manual_seed(23)
    model = SlowHeatQwen2ForSequenceClassification(
        _qwen_config(),
        QwenSlowHeatConfig(freeze_unbound_parameters=True),
    )
    model.train()
    query = model.model.layers[0].self_attn.q_proj.weight
    before = query.detach().clone()

    optimizer = SlowHeatAdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.05,
        weight_decay=0.1,
    )
    model.register_plasticity_masks(optimizer, hard=True)
    input_ids, attention_mask = _tokens()
    _backward(model, input_ids, attention_mask)
    optimizer.step()

    torch.testing.assert_close(query, before, atol=0.0, rtol=0.0)


# ---------------------------------------------------------------------------
# protocol identity and persistence
# ---------------------------------------------------------------------------


def test_protocol_is_persisted_on_the_hf_config():
    model = SlowHeatQwen2ForSequenceClassification(
        _qwen_config(), QwenSlowHeatConfig(slow_strength=10.0)
    )

    stored = model.config.dual_heater_slowheat
    assert stored["schema_version"] == model.slowheat_schema_version
    assert stored["config"]["slow_strength"] == 10.0


def test_conflicting_protocol_is_rejected_at_construction():
    config = _qwen_config()
    config.dual_heater_slowheat = {"schema_version": SlowHeatQwen2ForSequenceClassification.slowheat_schema_version, "config": {**QwenSlowHeatConfig(slow_strength=3.0).__dict__}}

    with pytest.raises(RuntimeError, match="diverge do protocolo"):
        SlowHeatQwen2ForSequenceClassification(
            config, QwenSlowHeatConfig(slow_strength=30.0)
        )


def test_signature_separates_hyperparameters_that_fp16_would_collapse():
    first = SlowHeatQwen2ForSequenceClassification(
        _qwen_config(), QwenSlowHeatConfig(importance_eps=1e-8)
    )
    second = SlowHeatQwen2ForSequenceClassification(
        _qwen_config(), QwenSlowHeatConfig(importance_eps=1e-9)
    )

    assert not torch.equal(first._slowheat_signature, second._slowheat_signature)


def test_load_state_dict_rejects_a_foreign_protocol():
    source = SlowHeatQwen2ForSequenceClassification(
        _qwen_config(), QwenSlowHeatConfig(slow_strength=3.0)
    )
    target = SlowHeatQwen2ForSequenceClassification(
        _qwen_config(), QwenSlowHeatConfig(slow_strength=30.0)
    )

    with pytest.raises(RuntimeError, match="incompatível"):
        target.load_state_dict(source.state_dict())


def test_round_trip_state_dict_preserves_heats():
    torch.manual_seed(3)
    source = SlowHeatQwen2ForSequenceClassification(_qwen_config())
    source.train()
    input_ids, attention_mask = _tokens()
    _backward(source, input_ids, attention_mask)
    source.consolidate()

    target = SlowHeatQwen2ForSequenceClassification(_qwen_config())
    target.load_state_dict(source.state_dict())

    torch.testing.assert_close(
        target.get_ffn_trackers()[0].slow_heat,
        source.get_ffn_trackers()[0].slow_heat,
    )


def test_heat_state_stays_fp32_after_a_half_precision_cast():
    """Pascal GPUs force fp16; heats must not lose the precision that ranks units."""

    model = SlowHeatQwen2ForSequenceClassification(_qwen_config())
    model.half()

    tracker = model.get_ffn_trackers()[0]
    assert tracker.slow_heat.dtype == torch.float32
    assert tracker.importance_memory.dtype == torch.float32
    assert tracker.task_ema.dtype == torch.float32


# ---------------------------------------------------------------------------
# two-task smoke
# ---------------------------------------------------------------------------


def test_two_task_sequence_runs_end_to_end_with_masked_optimizer():
    torch.manual_seed(29)
    model = SlowHeatQwen2ForSequenceClassification(
        _qwen_config(layers=2),
        QwenSlowHeatConfig(freeze_unbound_parameters=True),
    )
    model.train()
    optimizer = SlowHeatAdamW(
        [p for p in model.parameters() if p.requires_grad], lr=0.01
    )
    model.register_plasticity_masks(optimizer)

    for task in range(2):
        for _ in range(2):
            optimizer.zero_grad()
            _backward(
                model,
                torch.tensor([[2, 7, 9, 3]]),
                torch.ones(1, 4, dtype=torch.long),
                labels=torch.tensor([task]),
            )
            optimizer.step()
        model.consolidate()

    trackers = model.get_ffn_trackers()
    assert all(tracker.consolidated_tasks.item() == 2 for tracker in trackers)
    assert all(torch.isfinite(tracker.slow_heat).all() for tracker in trackers)
    assert all(
        torch.isfinite(p).all() for p in model.parameters() if p.requires_grad
    )
