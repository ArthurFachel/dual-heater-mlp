import weakref
from copy import deepcopy

import pytest
import torch

transformers = pytest.importorskip("transformers")

from dual_heater.bert import (
    BertSlowHeatConfig,
    ExactSlowHeatLoRAConfig,
    SlowHeatBertForSequenceClassification,
    build_exact_slowheat_lora,
    register_exact_lora_masks,
)
from dual_heater.fast_heat import FastHeatActivation, FastHeatConfig
from dual_heater.optim import SlowHeatAdamW, SlowHeatSGD
from dual_heater.transformer import (
    SlowHeatAttentionTracker,
    SlowHeatFFNTracker,
)
from experiments.live_telemetry import build_heat_snapshot


def _bert_config(*, heads: int = 2, labels: int = 4, layers: int = 1):
    return transformers.BertConfig(
        vocab_size=64,
        hidden_size=8,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        intermediate_size=12,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        classifier_dropout=0.0,
        num_labels=labels,
    )


def _backward(model, input_ids, attention_mask):
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=torch.tensor([1]),
    )
    output.loss.backward()


def test_extended_bert_config_defaults_preserve_historical_scope():
    config = BertSlowHeatConfig()

    assert config.track_ffn is True
    assert config.track_attention is True
    assert config.track_embeddings is False
    assert config.track_residual is False
    assert config.protect_layer_norm is False
    assert config.protect_pooler is False
    assert config.protect_classifier is False
    assert config.residual_plasticity_budget == 0.25
    assert config.pooler_plasticity_budget == 0.25


def test_extended_bert_config_validates_new_budgets():
    with pytest.raises(ValueError, match="residual_plasticity_budget"):
        BertSlowHeatConfig(residual_plasticity_budget=-0.1)
    with pytest.raises(ValueError, match="pooler_plasticity_budget"):
        BertSlowHeatConfig(pooler_plasticity_budget=1.1)


def test_bert_fastheat_is_post_gelu_inside_each_ffn():
    fast = FastHeatConfig(
        fast_decay=0.9,
        fast_strength=1.0,
        fast_threshold=0.0,
    )
    model = SlowHeatBertForSequenceClassification(
        _bert_config(),
        BertSlowHeatConfig(fast_heat=fast),
    )
    intermediate = model.bert.encoder.layer[0].intermediate

    assert isinstance(intermediate.intermediate_act_fn, FastHeatActivation)
    assert intermediate.intermediate_act_fn.gate.unit_count == 12
    assert model.get_fast_states() == [intermediate.intermediate_act_fn.gate]

    pre_activation = torch.linspace(-2.0, 2.0, 12).reshape(1, 1, 12)
    raw_post_gelu = intermediate.intermediate_act_fn.activation(pre_activation)
    with torch.no_grad():
        intermediate.intermediate_act_fn.gate.fast_heat.copy_(torch.arange(12.0))
    intermediate.intermediate_act_fn.eval()
    gated = intermediate.intermediate_act_fn(pre_activation)

    assert not torch.equal(gated, raw_post_gelu)
    torch.testing.assert_close(
        gated,
        intermediate.intermediate_act_fn.gate(raw_post_gelu),
    )


def test_ffn_tracker_excludes_padding_from_functional_utility():
    first = SlowHeatFFNTracker(3)
    second = SlowHeatFFNTracker(3)
    useful = torch.tensor([[[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]]])
    short = useful.clone().requires_grad_()
    padded = torch.cat((useful, torch.full((1, 2, 3), 1000.0)), dim=1).requires_grad_()

    first(short, torch.ones(1, 2)).sum().backward()
    second(padded, torch.tensor([[1.0, 1.0, 0.0, 0.0]])).sum().backward()

    assert torch.equal(first.task_ema, second.task_ema)
    assert first.task_ema.shape == (3,)


@pytest.mark.parametrize("combination", ["max", "mean", "sum"])
def test_attention_tracker_emits_one_value_per_head(combination):
    tracker = SlowHeatAttentionTracker(2, 3, combination=combination)
    tensors = [torch.randn(2, 4, 6, requires_grad=True) for _ in range(4)]
    tracker.observe(*tensors, torch.ones(2, 4))
    sum(tensor.square().sum() for tensor in tensors).backward()

    assert tracker.task_ema.shape == (2,)
    assert torch.isfinite(tracker.task_ema).all()
    assert tracker.task_step.item() == 1


def test_attention_tracker_releases_forward_tensors_after_backward():
    tracker = SlowHeatAttentionTracker(2, 3)
    query, key, value, output = (
        torch.randn(2, 4, 6, requires_grad=True) for _ in range(4)
    )
    tensors = [query, key, value, output]
    references = [weakref.ref(tensor) for tensor in tensors]
    tracker.observe(query, key, value, output, torch.ones(2, 4))

    loss = torch.stack([tensor.square().sum() for tensor in tensors]).sum()
    loss.backward()
    del loss, tensors, query, key, value, output

    assert all(reference() is None for reference in references)


def test_bert_padding_does_not_change_tracked_utility():
    torch.manual_seed(8)
    short = SlowHeatBertForSequenceClassification(_bert_config())
    padded = SlowHeatBertForSequenceClassification(_bert_config())
    padded.load_state_dict(deepcopy(short.state_dict()))
    short.train()
    padded.train()

    tokens = torch.tensor([[2, 7, 9, 3]])
    _backward(short, tokens, torch.ones(1, 4, dtype=torch.long))
    _backward(
        padded,
        torch.tensor([[2, 7, 9, 3, 0, 0]]),
        torch.tensor([[1, 1, 1, 1, 0, 0]]),
    )

    for first, second in zip(
        short.get_slow_states(), padded.get_slow_states(), strict=True
    ):
        assert torch.allclose(first.task_ema, second.task_ema, atol=1e-5, rtol=1e-5)


def test_freeze_unbound_protocol_leaves_only_masked_parameters_trainable():
    model = SlowHeatBertForSequenceClassification(
        _bert_config(),
        BertSlowHeatConfig(
            fast_heat=FastHeatConfig(),
            protect_classifier=True,
            freeze_unbound_parameters=True,
        ),
    )
    bound_ids = {id(binding.parameter) for binding in model.mask_bindings()}
    trainable = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    assert {id(parameter) for parameter in trainable.values()} == bound_ids
    assert model.uncovered_trainable_parameters() == []
    assert not model.bert.embeddings.word_embeddings.weight.requires_grad
    assert not model.bert.encoder.layer[0].attention.output.dense.bias.requires_grad
    assert not model.bert.encoder.layer[0].output.dense.bias.requires_grad
    assert not model.bert.encoder.layer[0].attention.output.LayerNorm.weight.requires_grad
    assert not model.bert.encoder.layer[0].output.LayerNorm.bias.requires_grad
    assert not model.bert.pooler.dense.weight.requires_grad
    assert model.classifier.weight.requires_grad

    optimizer = SlowHeatSGD(trainable.values(), lr=0.1)
    model.register_plasticity_masks(optimizer)

    model.bert.pooler.dense.weight.requires_grad_(True)
    with pytest.raises(RuntimeError, match="pooler.dense.weight"):
        model.validate_trainable_mask_coverage()


def test_bert_mask_bindings_cover_ffn_rows_columns_and_attention_blocks():
    model = SlowHeatBertForSequenceClassification(_bert_config())
    ffn = model.ffn_trackers[0]
    attention = model.attention_trackers[0]
    ffn.slow_heat.zero_()
    ffn.slow_heat[0] = 1.0
    attention.slow_heat.copy_(torch.tensor([1.0, 0.0]))
    bindings = {binding.kind: binding for binding in model.mask_bindings(hard=True)}

    ffn_rows = bindings["bert_layer_0_ffn_12_producer_rows"].mask()
    ffn_columns = bindings["bert_layer_0_ffn_12_consumer_columns"].mask()
    query_rows = bindings["bert_layer_0_attention_2x4_query_rows"].mask()
    output_columns = bindings["bert_layer_0_attention_2x4_output_columns"].mask()

    assert ffn_rows.shape == (12, 1)
    assert ffn_columns.shape == (1, 12)
    assert ffn_rows[0].item() == 0.0 and ffn_rows[1].item() == 1.0
    assert torch.equal(query_rows[:4], torch.zeros(4, 1))
    assert torch.equal(query_rows[4:], torch.ones(4, 1))
    assert output_columns.shape == (1, 8)


def test_global_family_capacity_is_allocated_across_all_ffn_layers():
    model = SlowHeatBertForSequenceClassification(
        _bert_config(layers=2),
        BertSlowHeatConfig(
            track_attention=False,
            ffn_plasticity_budget=0.5,
            capacity_scope="global",
        ),
    )
    first, second = model.ffn_trackers
    first.task_ema.copy_(torch.arange(24.0, 12.0, -1.0))
    second.task_ema.copy_(torch.arange(12.0, 0.0, -1.0))
    first.task_step.fill_(1)
    second.task_step.fill_(1)

    model.consolidate(strategy="max")

    assert torch.count_nonzero(first.slow_heat).item() == 12
    assert torch.count_nonzero(second.slow_heat).item() == 0
    assert sum(
        torch.count_nonzero(tracker.slow_heat).item()
        for tracker in model.ffn_trackers
    ) == 12


def test_hierarchical_capacity_allocates_layer_quotas_before_units():
    model = SlowHeatBertForSequenceClassification(
        _bert_config(layers=2),
        BertSlowHeatConfig(
            track_attention=False,
            ffn_plasticity_budget=0.25,
            capacity_scope="hierarchical",
        ),
    )
    first, second = model.ffn_trackers
    first.task_ema.fill_(3.0)
    second.task_ema.fill_(1.0)
    first.task_step.fill_(1)
    second.task_step.fill_(1)

    model.consolidate(strategy="max")

    assert torch.count_nonzero(first.slow_heat).item() == 12
    assert torch.count_nonzero(second.slow_heat).item() == 6
    assert sum(
        torch.count_nonzero(tracker.slow_heat).item()
        for tracker in model.ffn_trackers
    ) == 18


def test_global_topk_fastheat_only_gates_hottest_units_across_layers():
    model = SlowHeatBertForSequenceClassification(
        _bert_config(layers=2),
        BertSlowHeatConfig(
            track_attention=False,
            capacity_scope="global",
            fast_heat=FastHeatConfig(
                fast_strength=1.0,
                competition="global_topk",
                topk_fraction=0.25,
            ),
        ),
    )
    first, second = model.get_fast_states()
    first.fast_heat.copy_(torch.arange(12.0))
    second.fast_heat.zero_()

    model.eval()
    model(
        input_ids=torch.tensor([[2, 5, 3]]),
        attention_mask=torch.ones(1, 3, dtype=torch.long),
    )

    scales = torch.cat((first.current_scale(), second.current_scale()))
    assert torch.count_nonzero(scales < 1.0).item() == 6
    assert torch.equal(scales[:6], torch.ones(6))
    assert torch.all(scales[6:12] < 1.0)
    assert torch.equal(scales[12:], torch.ones(12))


def test_no_consolidation_slowheat_bert_matches_native_adamw():
    torch.manual_seed(12)
    native = transformers.BertForSequenceClassification(_bert_config())
    protected = SlowHeatBertForSequenceClassification(_bert_config())
    protected.load_state_dict(native.state_dict(), strict=False)
    native_optimizer = torch.optim.AdamW(native.parameters(), lr=1e-3)
    protected_optimizer = SlowHeatAdamW(protected.parameters(), lr=1e-3)
    protected.register_plasticity_masks(protected_optimizer)
    inputs = torch.tensor([[2, 5, 3]])
    mask = torch.ones_like(inputs)

    _backward(native, inputs, mask)
    _backward(protected, inputs, mask)
    native_optimizer.step()
    protected_optimizer.step()

    native_parameters = dict(native.named_parameters())
    for name, parameter in protected.named_parameters():
        assert torch.equal(parameter, native_parameters[name])


def test_bert_checkpoint_rejects_incompatible_head_topology():
    source = SlowHeatBertForSequenceClassification(_bert_config(heads=2))
    target = SlowHeatBertForSequenceClassification(_bert_config(heads=4))

    with pytest.raises((RuntimeError, ValueError), match="(size mismatch|topologia)"):
        target.load_state_dict(source.state_dict())


def test_bert_checkpoint_rejects_incompatible_slowheat_configuration():
    source = SlowHeatBertForSequenceClassification(
        _bert_config(), BertSlowHeatConfig(slow_strength=3.0)
    )
    target = SlowHeatBertForSequenceClassification(
        _bert_config(), BertSlowHeatConfig(slow_strength=10.0)
    )

    with pytest.raises(RuntimeError, match="configuração BERT"):
        target.load_state_dict(source.state_dict())


def test_bert_heat_snapshot_includes_real_fast_heat():
    model = SlowHeatBertForSequenceClassification(
        _bert_config(),
        BertSlowHeatConfig(fast_heat=FastHeatConfig()),
    )
    model.train()
    model(
        input_ids=torch.tensor([[2, 5, 3]]),
        attention_mask=torch.ones(1, 3, dtype=torch.long),
    ).logits.sum().backward()

    snapshot = build_heat_snapshot(
        model,
        context={"method": "dualheat", "stage": 0},
        run_id="run",
        session_id="session",
        sequence=1,
    )

    assert len(snapshot["ffn"]) == 1
    fast_heat = snapshot["ffn"][0]["fast_heat"]
    assert len(fast_heat) == 12  # intermediate_size do _bert_config
    assert max(abs(value) for value in fast_heat) > 0.0


def test_huggingface_roundtrip_restores_fastheat_protocol_and_rejects_mismatch(tmp_path):
    saved_config = BertSlowHeatConfig(
        fast_heat=FastHeatConfig(fast_strength=0.7),
        protect_classifier=True,
        freeze_unbound_parameters=True,
    )
    source = SlowHeatBertForSequenceClassification(_bert_config(), saved_config)
    with torch.no_grad():
        source.get_fast_states()[0].fast_heat.copy_(torch.arange(12.0))
    source.save_pretrained(tmp_path)

    restored = SlowHeatBertForSequenceClassification.from_pretrained(tmp_path)
    assert restored.slowheat_config == saved_config
    assert torch.equal(
        restored.get_fast_states()[0].fast_heat,
        source.get_fast_states()[0].fast_heat,
    )
    assert torch.equal(restored._slowheat_signature, restored._build_slowheat_signature())

    incompatible = BertSlowHeatConfig(
        fast_heat=FastHeatConfig(fast_strength=0.8),
        protect_classifier=True,
        freeze_unbound_parameters=True,
    )
    with pytest.raises(RuntimeError, match="config.json"):
        SlowHeatBertForSequenceClassification.from_pretrained(
            tmp_path,
            slowheat_config=incompatible,
        )


def test_huggingface_from_pretrained_loads_native_bert_checkpoint(tmp_path):
    native = transformers.BertForSequenceClassification(_bert_config())
    native.save_pretrained(tmp_path)

    model = SlowHeatBertForSequenceClassification.from_pretrained(
        tmp_path,
        slowheat_config=BertSlowHeatConfig(fast_heat=FastHeatConfig()),
    )

    assert len(model.ffn_trackers) == 1
    assert len(model.attention_trackers) == 1
    assert len(model.get_fast_states()) == 1
    assert torch.equal(
        model.get_fast_states()[0].fast_heat,
        torch.zeros(12),
    )
    assert torch.equal(model.classifier.weight, native.classifier.weight)


def test_slowheat_hooks_can_be_removed_before_releasing_a_gpu_model():
    model = SlowHeatBertForSequenceClassification(_bert_config())

    assert model._slowheat_hook_handles
    model.remove_slowheat_instrumentation()

    assert not model._slowheat_hook_handles


def test_exact_lora_freezes_a_and_hard_masks_protected_b_rows():
    pytest.importorskip("peft")
    torch.manual_seed(21)
    base = SlowHeatBertForSequenceClassification(
        _bert_config(), BertSlowHeatConfig(protect_classifier=False)
    )
    model = build_exact_slowheat_lora(
        base,
        ExactSlowHeatLoRAConfig(rank=2, alpha=2.0),
    )
    base.attention_trackers[0].slow_heat.copy_(torch.tensor([1.0, 0.0]))
    base.ffn_trackers[0].slow_heat.zero_()
    base.ffn_trackers[0].slow_heat[0] = 1.0
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = SlowHeatSGD(trainable, lr=0.1)
    register_exact_lora_masks(model, optimizer, hard=True)
    a_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if ".lora_A." in name
    }
    query_b = base.bert.encoder.layer[0].attention.self.query.lora_B["default"].weight
    query_before = query_b.detach().clone()

    _backward(
        model,
        torch.tensor([[2, 7, 3]]),
        torch.ones(1, 3, dtype=torch.long),
    )
    optimizer.step()

    for name, parameter in model.named_parameters():
        if name in a_before:
            assert torch.equal(parameter, a_before[name])
    assert torch.equal(query_b[:4], query_before[:4])
    assert not torch.equal(query_b[4:], query_before[4:])


def test_bert_scientific_state_survives_half_precision_cast():
    model = SlowHeatBertForSequenceClassification(
        _bert_config(),
        slowheat_config=BertSlowHeatConfig(fast_heat=FastHeatConfig()),
    )
    tracker = model.ffn_trackers[0]
    tracker.task_ema.add_(0.125)

    model.half()

    assert model.classifier.weight.dtype is torch.float16
    assert tracker.slow_heat.dtype is torch.float32
    assert tracker.task_ema.dtype is torch.float32
    assert tracker.importance_memory.dtype is torch.float32
    assert tracker.task_step.dtype is torch.int64
    assert tracker.consolidated_tasks.dtype is torch.int64
    assert tracker.task_ema[0].item() == pytest.approx(0.125)
    assert model.get_fast_states()[0].fast_heat.dtype is torch.float32


def test_half_precision_model_still_detects_incompatible_checkpoints(tmp_path):
    model = SlowHeatBertForSequenceClassification(
        _bert_config(),
        slowheat_config=BertSlowHeatConfig(slow_strength=5.0),
    ).half()
    other = SlowHeatBertForSequenceClassification(
        _bert_config(),
        slowheat_config=BertSlowHeatConfig(slow_strength=50.0),
    )

    with pytest.raises(RuntimeError, match="incompatível"):
        model.load_state_dict(other.state_dict())


def test_half_precision_does_not_collapse_distinct_slowheat_configs():
    model = SlowHeatBertForSequenceClassification(
        _bert_config(),
        slowheat_config=BertSlowHeatConfig(importance_eps=1e-8),
    ).half()
    other = SlowHeatBertForSequenceClassification(
        _bert_config(),
        slowheat_config=BertSlowHeatConfig(importance_eps=1e-9),
    )
    other_state = {
        name: value.half() if value.is_floating_point() else value
        for name, value in other.state_dict().items()
    }

    with pytest.raises(RuntimeError, match="incompatível"):
        model.load_state_dict(other_state)
