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
from dual_heater.optim import SlowHeatAdamW, SlowHeatSGD
from dual_heater.transformer import (
    SlowHeatAttentionTracker,
    SlowHeatFFNTracker,
)


def _bert_config(*, heads: int = 2, labels: int = 4):
    return transformers.BertConfig(
        vocab_size=64,
        hidden_size=8,
        num_hidden_layers=1,
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


def test_huggingface_from_pretrained_loads_native_bert_checkpoint(tmp_path):
    native = transformers.BertForSequenceClassification(_bert_config())
    native.save_pretrained(tmp_path)

    model = SlowHeatBertForSequenceClassification.from_pretrained(
        tmp_path,
        slowheat_config=BertSlowHeatConfig(),
    )

    assert len(model.ffn_trackers) == 1
    assert len(model.attention_trackers) == 1
    assert torch.equal(model.classifier.weight, native.classifier.weight)


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
