import pytest
import torch

transformers = pytest.importorskip("transformers")

from experiments.split_clinc150 import (
    BERT_BASE_MODEL,
    CLINC150_DOMAINS,
    SplitCLINC150Config,
    TextReplayBuffer,
    apply_frozen_slowheat_manifest,
    build_clinc150_tasks,
    run_split_clinc150,
    select_replay_examples,
    text_task_fingerprint,
)


class _LabelFeature:
    def __init__(self, names):
        self.names = names


class _FakeSplit(list):
    column_names = ("text", "intent")

    def __init__(self, rows, names):
        super().__init__(rows)
        self.features = {"intent": _LabelFeature(names)}


class _FakeTokenizer:
    def __call__(
        self,
        texts,
        *,
        truncation,
        padding,
        max_length,
        return_tensors,
    ):
        assert truncation and padding == "max_length" and return_tensors == "pt"
        input_ids = torch.zeros((len(texts), max_length), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        token_type_ids = torch.zeros_like(input_ids)
        for index, text in enumerate(texts):
            width = min(max_length, 2 + len(text) % 4)
            input_ids[index, :width] = torch.arange(1, width + 1)
            attention_mask[index, :width] = 1
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }


def _fake_dataset():
    names = [
        intent for intents in CLINC150_DOMAINS.values() for intent in intents
    ] + ["oos"]
    rows = []
    for label, name in enumerate(names):
        count = 2 if name != "oos" else 3
        rows.extend(
            {"text": f"{name}-{example}", "intent": label}
            for example in range(count)
        )
    return {
        name: _FakeSplit(list(rows), names)
        for name in ("train", "validation", "test")
    }


def test_clinc_builder_creates_official_domains_and_excludes_oos():
    tasks = build_clinc150_tasks(_fake_dataset(), _FakeTokenizer(), max_length=12)

    assert [task.domain for task in tasks] == list(CLINC150_DOMAINS)
    assert all(len(task.classes) == 15 for task in tasks)
    assert all(len(task.train.labels) == 30 for task in tasks)
    assert torch.equal(
        torch.unique(torch.cat([task.train.labels for task in tasks])),
        torch.arange(150),
    )


def test_text_replay_round_trip_preserves_int64_token_fields():
    tasks = build_clinc150_tasks(_fake_dataset(), _FakeTokenizer(), max_length=12)
    selected = select_replay_examples(tasks[0], per_class=1)
    source = TextReplayBuffer(max_length=12)
    source.append(selected, task_index=0)
    restored = TextReplayBuffer(max_length=12)
    restored.load_state_dict(source.state_dict())

    assert len(restored) == 15
    assert restored.memory_bytes == source.memory_bytes
    for name, tensor in source.state_dict().items():
        assert torch.equal(restored.state_dict()[name], tensor)
        assert tensor.dtype == torch.long


def test_text_task_fingerprint_changes_with_tokenized_data():
    first = build_clinc150_tasks(_fake_dataset(), _FakeTokenizer(), max_length=12)
    second = build_clinc150_tasks(_fake_dataset(), _FakeTokenizer(), max_length=12)
    assert text_task_fingerprint(first) == text_task_fingerprint(second)

    second[0].train.input_ids[0, 0] += 1
    assert text_task_fingerprint(first) != text_task_fingerprint(second)


def test_frozen_manifest_transfers_mini_hyperparameters_to_bert_base():
    manifest = {
        "schema_version": 1,
        "status": "frozen_before_test_evaluation",
        "test_evaluated": False,
        "selected": {
            "hyperparameters": {
                "slow_strength": 10.0,
                "ffn_plasticity_budget": 0.5,
                "attention_plasticity_budget": 0.25,
            }
        },
    }

    updated = apply_frozen_slowheat_manifest(
        SplitCLINC150Config(evaluate_test=False),
        manifest,
        model_name=BERT_BASE_MODEL,
    )

    assert updated.model_name == BERT_BASE_MODEL
    assert updated.slow_strength == 10.0
    assert updated.ffn_plasticity_budget == 0.5
    assert updated.attention_plasticity_budget == 0.25
    assert updated.evaluate_test is True


def test_tiny_clinc_runner_is_paired_and_stage_resumable(monkeypatch, tmp_path):
    tasks = build_clinc150_tasks(_fake_dataset(), _FakeTokenizer(), max_length=12)

    def local_pretrained(cls, _name, **kwargs):
        return cls(
            transformers.BertConfig(
                vocab_size=64,
                hidden_size=8,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=12,
                hidden_dropout_prob=0.0,
                attention_probs_dropout_prob=0.0,
                classifier_dropout=0.0,
                num_labels=kwargs["num_labels"],
            )
        )

    monkeypatch.setattr(
        transformers.BertForSequenceClassification,
        "from_pretrained",
        classmethod(local_pretrained),
    )
    config = SplitCLINC150Config(
        max_length=12,
        batch_size=30,
        replay_batch_size=5,
        replay_per_class=1,
        epochs_per_task=1,
        methods=("replay", "slowheat_replay"),
    )

    first = run_split_clinc150(config, tasks, output_dir=tmp_path)
    resumed = run_split_clinc150(config, tasks, output_dir=tmp_path, resume=True)

    assert first["replay"]["accuracy_matrix"] == resumed["replay"]["accuracy_matrix"]
    assert len(first["slowheat_replay"]["capacity_history"]) == 10
    assert first["replay"]["replay_memory_bytes"] > 0
