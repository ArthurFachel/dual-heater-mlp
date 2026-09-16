import json
import math
from dataclasses import replace

import pytest
import torch

transformers = pytest.importorskip("transformers")

from experiments import split_clinc150 as clinc_module
from experiments.artifacts import read_torch_checkpoint
from experiments.live_telemetry import read_events
from experiments.split_clinc150 import (
    BERT_BASE_MODEL,
    BERT_FULL_COVERAGE_VARIANTS,
    BERT_HEAT_VARIANTS,
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


def test_clinc_dualheat_has_matched_closed_slowheat_control():
    config = SplitCLINC150Config(methods=("slowheat_bound", "dualheat"))
    config.validate()

    slow = clinc_module._slowheat_config(config, "slowheat_bound")
    dual = clinc_module._slowheat_config(config, "dualheat")

    assert slow.fast_heat is None
    assert dual.fast_heat is not None
    assert dual.fast_heat.fast_strength == config.fast_strength
    assert slow.protect_classifier and dual.protect_classifier
    assert slow.freeze_unbound_parameters and dual.freeze_unbound_parameters


def test_bert_heat_variant_preset_is_a_matched_four_method_ablation():
    assert BERT_HEAT_VARIANTS == (
        "slowheat_bound",
        "slowheat_global",
        "slowheat_hierarchical",
        "dualheat_global_topk",
    )
    config = SplitCLINC150Config(methods=BERT_HEAT_VARIANTS)
    resolved = [clinc_module._slowheat_config(config, method) for method in config.methods]

    assert [item.capacity_scope for item in resolved] == [
        "local",
        "global",
        "hierarchical",
        "global",
    ]
    assert [item.fast_heat is not None for item in resolved] == [
        False,
        False,
        False,
        True,
    ]
    assert resolved[-1].fast_heat.competition == "global_topk"
    assert all(item.protect_classifier for item in resolved)
    assert all(item.freeze_unbound_parameters for item in resolved)


def test_full_coverage_variant_preset_contains_control_and_leave_one_out_methods():
    assert BERT_FULL_COVERAGE_VARIANTS == (
        "vanilla",
        "slowheat_full_coverage",
        "slowheat_all_minus_embeddings",
        "slowheat_all_minus_layernorm",
        "slowheat_all_minus_residual",
        "slowheat_all_minus_attention",
        "slowheat_all_minus_ffn",
        "slowheat_all_minus_pooler",
        "slowheat_all_minus_classifier",
        "slowheat_ffn_attention",
    )


def test_full_coverage_variants_change_exactly_the_named_family():
    config = SplitCLINC150Config(methods=BERT_FULL_COVERAGE_VARIANTS)
    full = clinc_module._slowheat_config(config, "slowheat_full_coverage")
    assert full.fast_heat is None
    assert full.freeze_unbound_parameters is False
    assert full.capacity_scope == "hierarchical"

    fields = {
        "embeddings": "track_embeddings",
        "layernorm": "protect_layer_norm",
        "residual": "track_residual",
        "attention": "track_attention",
        "ffn": "track_ffn",
        "pooler": "protect_pooler",
        "classifier": "protect_classifier",
    }
    for family, field in fields.items():
        candidate = clinc_module._slowheat_config(
            config, f"slowheat_all_minus_{family}"
        )
        for checked in fields.values():
            expected = False if checked == field else getattr(full, checked)
            assert getattr(candidate, checked) is expected

    baseline = clinc_module._slowheat_config(config, "slowheat_ffn_attention")
    assert baseline.track_ffn and baseline.track_attention
    assert not baseline.track_embeddings
    assert not baseline.track_residual
    assert not baseline.protect_layer_norm
    assert not baseline.protect_pooler
    assert not baseline.protect_classifier


def test_clinc_config_propagates_extended_capacity_budgets():
    config = SplitCLINC150Config(
        methods=("slowheat_full_coverage",),
        residual_plasticity_budget=0.5,
        pooler_plasticity_budget=0.75,
    )
    config.validate()
    resolved = clinc_module._slowheat_config(config, "slowheat_full_coverage")

    assert resolved.residual_plasticity_budget == 0.5
    assert resolved.pooler_plasticity_budget == 0.75


@pytest.mark.parametrize(
    ("method", "uncovered_fragment"),
    [
        ("slowheat_all_minus_embeddings", "word_embeddings.weight"),
        ("slowheat_all_minus_layernorm", "LayerNorm.weight"),
        ("slowheat_all_minus_attention", "attention.self.query.bias"),
        ("slowheat_all_minus_ffn", "intermediate.dense.bias"),
        ("slowheat_all_minus_pooler", "pooler.dense.bias"),
        ("slowheat_all_minus_classifier", "classifier.bias"),
    ],
)
def test_leave_one_out_method_exposes_the_named_unmasked_family(
    method, uncovered_fragment
):
    model_config = transformers.BertConfig(
        vocab_size=64,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=12,
        num_labels=4,
    )
    protocol = SplitCLINC150Config(methods=(method,))
    model = clinc_module.SlowHeatBertForSequenceClassification(
        model_config, clinc_module._slowheat_config(protocol, method)
    )

    assert any(
        uncovered_fragment in name
        for name in model.uncovered_trainable_parameters()
    )


def test_minus_residual_removes_column_and_residual_row_factors():
    config = SplitCLINC150Config(methods=("slowheat_all_minus_residual",))
    model = clinc_module.SlowHeatBertForSequenceClassification(
        transformers.BertConfig(
            vocab_size=64,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=12,
            num_labels=4,
        ),
        clinc_module._slowheat_config(config, "slowheat_all_minus_residual"),
    )
    kinds = {binding.kind for binding in model.mask_bindings()}

    assert not any("residual_rows" in kind for kind in kinds)
    assert not any("residual_columns" in kind for kind in kinds)
    assert any("attention_rows" in kind for kind in kinds)
    assert any("ffn_rows" in kind for kind in kinds)


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
    assert updated.evaluate_test is False
    assert (
        apply_frozen_slowheat_manifest(
            SplitCLINC150Config(evaluate_test=True),
            manifest,
            model_name=BERT_BASE_MODEL,
        ).evaluate_test
        is True
    )


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

    reference_dir = tmp_path / "reference"
    live_dir = tmp_path / "live"
    reference = run_split_clinc150(config, tasks, output_dir=reference_dir)
    first = run_split_clinc150(
        config,
        tasks,
        output_dir=live_dir,
        telemetry=True,
        telemetry_every=2,
    )
    resumed = run_split_clinc150(
        config,
        tasks,
        output_dir=live_dir,
        resume=True,
        telemetry=True,
        telemetry_every=2,
    )

    assert first["replay"]["accuracy_matrix"] == resumed["replay"]["accuracy_matrix"]
    assert first["replay"]["accuracy_matrix"] == reference["replay"]["accuracy_matrix"]
    assert first["slowheat_replay"]["training_losses"] == reference[
        "slowheat_replay"
    ]["training_losses"]
    for method in config.methods:
        reference_model = read_torch_checkpoint(
            reference_dir / method / "checkpoint.pt"
        )["model"]
        live_model = read_torch_checkpoint(live_dir / method / "checkpoint.pt")[
            "model"
        ]
        assert reference_model.keys() == live_model.keys()
        assert all(
            torch.equal(reference_model[name], live_model[name])
            for name in reference_model
        )
    assert len(first["slowheat_replay"]["capacity_history"]) == 10
    assert first["replay"]["replay_memory_bytes"] > 0
    events = read_events(live_dir / "telemetry/events.jsonl")
    event_types = {event["event"] for event in events}
    assert {
        "batch",
        "checkpoint",
        "consolidation",
        "epoch_end",
        "evaluation_end",
        "method_end",
        "run_end",
        "task_end",
    } <= event_types
    assert len([event for event in events if event["event"] == "batch"]) == 10
    assert len([event for event in events if event["event"] == "session_start"]) == 2
    method_ends = [event for event in events if event["event"] == "method_end"]
    assert all(event["telemetry_overhead_seconds"] >= 0.0 for event in method_ends)
    assert all(event["telemetry_overhead_ratio"] >= 0.0 for event in method_ends)
    heat_history = list((live_dir / "telemetry/heat").glob("*.json"))
    assert len(heat_history) == 20
    assert len([path for path in heat_history if "-epoch-" in path.name]) == 10

    def fail_model_build(*args, **kwargs):
        raise RuntimeError("falha de treino simulada")

    monkeypatch.setattr(clinc_module, "_build_model", fail_model_build)
    failed_output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="falha de treino simulada"):
        run_split_clinc150(
            config,
            tasks,
            output_dir=failed_output,
            telemetry=True,
        )
    failure_events = read_events(failed_output / "telemetry/events.jsonl")
    assert failure_events[-2]["event"] == "run_error"
    assert failure_events[-1]["event"] == "session_error"


def test_clinc_defaults_to_validation_only():
    assert SplitCLINC150Config().evaluate_test is False


def test_cli_evaluate_test_defaults_to_false():
    parser = clinc_module.build_parser()

    assert parser.parse_args([]).evaluate_test is False
    assert parser.parse_args(["--evaluate-test"]).evaluate_test is True
    assert parser.parse_args(["--no-evaluate-test"]).evaluate_test is False


def test_cli_full_coverage_variants_is_exclusive_with_other_method_selectors():
    parser = clinc_module.build_parser()
    args = parser.parse_args(["--full-coverage-variants"])
    assert args.full_coverage_variants is True

    with pytest.raises(SystemExit):
        parser.parse_args(["--full-coverage-variants", "--heat-variants"])
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--full-coverage-variants", "--methods", "slowheat_full_coverage"]
        )


def test_cli_rejects_test_access_without_frozen_manifest(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["split_clinc150", "--evaluate-test"])
    monkeypatch.setattr(
        clinc_module,
        "load_clinc150_tasks",
        lambda config: pytest.fail("dados não devem ser carregados"),
    )

    with pytest.raises(SystemExit):
        clinc_module.main()

    assert "--frozen-manifest" in capsys.readouterr().err


def _patch_tiny_bert(monkeypatch):
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


def test_tiny_full_coverage_run_reports_complete_mask_coverage(monkeypatch):
    _patch_tiny_bert(monkeypatch)
    tasks = build_clinc150_tasks(_fake_dataset(), _FakeTokenizer(), max_length=12)
    config = SplitCLINC150Config(
        max_length=12,
        batch_size=30,
        replay_batch_size=5,
        replay_per_class=1,
        epochs_per_task=1,
        methods=("slowheat_full_coverage",),
    )

    result = run_split_clinc150(config, tasks)["slowheat_full_coverage"]

    assert result["mask_coverage"]["masked_fraction"] == pytest.approx(1.0)


def test_incompatible_protocol_is_not_overwritten(monkeypatch, tmp_path):
    _patch_tiny_bert(monkeypatch)
    tasks = build_clinc150_tasks(_fake_dataset(), _FakeTokenizer(), max_length=12)
    config = SplitCLINC150Config(
        max_length=12,
        batch_size=30,
        replay_batch_size=5,
        replay_per_class=1,
        epochs_per_task=1,
        methods=("replay",),
    )
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    protocol_path = output_dir / "protocol.json"
    protocol_path.write_text('{"config": "outro protocolo"}', encoding="utf-8")
    original = protocol_path.read_bytes()

    with pytest.raises(RuntimeError, match="outro protocolo"):
        run_split_clinc150(config, tasks, output_dir=output_dir, resume=True)

    assert protocol_path.read_bytes() == original


def test_protocol_records_source_fingerprint(monkeypatch, tmp_path):
    _patch_tiny_bert(monkeypatch)
    tasks = build_clinc150_tasks(_fake_dataset(), _FakeTokenizer(), max_length=12)
    config = SplitCLINC150Config(
        max_length=12,
        batch_size=30,
        replay_batch_size=5,
        replay_per_class=1,
        epochs_per_task=1,
        methods=("replay",),
    )

    run_split_clinc150(config, tasks, output_dir=tmp_path / "run")

    protocol = json.loads(
        (tmp_path / "run" / "protocol.json").read_text(encoding="utf-8")
    )
    assert isinstance(protocol["source_sha256"], str)
    assert len(protocol["source_sha256"]) == 64


def _patch_tiny_bert_with_dropout(monkeypatch, dropout: float):
    def local_pretrained(cls, _name, **kwargs):
        return cls(
            transformers.BertConfig(
                vocab_size=64,
                hidden_size=8,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=12,
                hidden_dropout_prob=dropout,
                attention_probs_dropout_prob=dropout,
                classifier_dropout=dropout,
                num_labels=kwargs["num_labels"],
            )
        )

    monkeypatch.setattr(
        transformers.BertForSequenceClassification,
        "from_pretrained",
        classmethod(local_pretrained),
    )


class _StopAfterFirstCheckpoint(RuntimeError):
    pass


def test_resume_with_dropout_matches_uninterrupted_run(monkeypatch, tmp_path):
    _patch_tiny_bert_with_dropout(monkeypatch, 0.1)
    tasks = build_clinc150_tasks(_fake_dataset(), _FakeTokenizer(), max_length=12)
    config = SplitCLINC150Config(
        max_length=12,
        batch_size=30,
        replay_batch_size=5,
        replay_per_class=1,
        epochs_per_task=1,
        methods=("replay",),
    )

    reference = run_split_clinc150(config, tasks, output_dir=tmp_path / "reference")

    live_dir = tmp_path / "live"
    real_write = clinc_module.write_torch_atomic
    written = {"count": 0}

    def stopping_write(path, payload):
        real_write(path, payload)
        written["count"] += 1
        if written["count"] == 1:
            raise _StopAfterFirstCheckpoint

    monkeypatch.setattr(clinc_module, "write_torch_atomic", stopping_write)
    with pytest.raises(_StopAfterFirstCheckpoint):
        run_split_clinc150(config, tasks, output_dir=live_dir)

    monkeypatch.setattr(clinc_module, "write_torch_atomic", real_write)
    resumed = run_split_clinc150(config, tasks, output_dir=live_dir, resume=True)

    assert resumed["replay"]["training_losses"] == reference["replay"][
        "training_losses"
    ]
    assert (
        resumed["replay"]["validation_accuracy_matrix"]
        == reference["replay"]["validation_accuracy_matrix"]
    )


def test_checkpoint_without_rng_state_is_rejected(monkeypatch, tmp_path):
    _patch_tiny_bert(monkeypatch)
    tasks = build_clinc150_tasks(_fake_dataset(), _FakeTokenizer(), max_length=12)
    config = SplitCLINC150Config(
        max_length=12,
        batch_size=30,
        replay_batch_size=5,
        replay_per_class=1,
        epochs_per_task=1,
        methods=("replay",),
    )
    output_dir = tmp_path / "run"
    run_split_clinc150(config, tasks, output_dir=output_dir)

    checkpoint_path = output_dir / "replay" / "checkpoint.pt"
    payload = read_torch_checkpoint(checkpoint_path)
    payload["schema_version"] = 1
    payload.pop("host_rng_state", None)
    torch.save(payload, checkpoint_path)

    with pytest.raises(RuntimeError, match="incompatível"):
        run_split_clinc150(config, tasks, output_dir=output_dir, resume=True)


def _stage_learning_rates(config, task_steps, stage):
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([parameter], lr=config.learning_rate)
    scheduler = clinc_module.build_stage_scheduler(
        optimizer, config, task_steps=task_steps, stage=stage
    )
    learning_rates = []
    for _ in range(task_steps[stage]):
        learning_rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    return learning_rates


def test_task_scoped_scheduler_repeats_the_same_relative_schedule():
    config = SplitCLINC150Config(scheduler_scope="task")
    task_steps = [20, 20, 30]

    first = _stage_learning_rates(config, task_steps, 0)
    second = _stage_learning_rates(config, task_steps, 1)

    assert first == second
    assert first[0] == pytest.approx(second[0])
    assert max(first) == pytest.approx(config.learning_rate)
    assert first[-1] < first[len(first) // 2]


def test_stream_scoped_scheduler_spreads_decay_over_the_whole_stream():
    task_steps = [20, 20, 30]
    task_scope = _stage_learning_rates(
        SplitCLINC150Config(scheduler_scope="task"), task_steps, 0
    )
    stream_scope = _stage_learning_rates(
        SplitCLINC150Config(scheduler_scope="stream"), task_steps, 0
    )

    # A stream-wide horizon decays more slowly inside the first task.
    assert stream_scope[-1] > task_scope[-1]


def test_protocol_records_scheduler_scope(monkeypatch, tmp_path):
    _patch_tiny_bert(monkeypatch)
    tasks = build_clinc150_tasks(_fake_dataset(), _FakeTokenizer(), max_length=12)
    config = SplitCLINC150Config(
        max_length=12,
        batch_size=30,
        replay_batch_size=5,
        replay_per_class=1,
        epochs_per_task=1,
        methods=("replay",),
    )

    run_split_clinc150(config, tasks, output_dir=tmp_path / "run")

    protocol = json.loads(
        (tmp_path / "run" / "protocol.json").read_text(encoding="utf-8")
    )
    assert protocol["config"]["scheduler_scope"] == "task"


def _multi_seed_config():
    return SplitCLINC150Config(
        max_length=12,
        batch_size=30,
        replay_batch_size=5,
        replay_per_class=1,
        epochs_per_task=1,
        methods=("replay",),
    )


def test_multi_seed_aggregates_validation_when_test_split_is_closed(
    monkeypatch, tmp_path
):
    _patch_tiny_bert(monkeypatch)
    tasks = build_clinc150_tasks(_fake_dataset(), _FakeTokenizer(), max_length=12)
    config = _multi_seed_config()
    assert config.evaluate_test is False

    aggregate = clinc_module.run_split_clinc150_multi_seed(
        config,
        tasks,
        seeds=[1, 2],
        metadata={},
        output_dir=tmp_path / "run",
    )

    assert aggregate["endpoint_source"] == "validation"
    assert aggregate["evaluate_test"] is False
    summary = aggregate["methods"]["replay"]["final_average_accuracy"]
    assert math.isfinite(summary["mean"])


def test_multi_seed_confirmatory_mode_requires_the_test_split(monkeypatch, tmp_path):
    _patch_tiny_bert(monkeypatch)
    tasks = build_clinc150_tasks(_fake_dataset(), _FakeTokenizer(), max_length=12)

    with pytest.raises(ValueError, match="evaluate_test=True"):
        clinc_module.run_split_clinc150_multi_seed(
            _multi_seed_config(),
            tasks,
            seeds=[1],
            metadata={},
            output_dir=tmp_path / "run",
            confirmatory=True,
        )


def test_multi_seed_uses_test_endpoints_when_explicitly_opened(monkeypatch, tmp_path):
    _patch_tiny_bert(monkeypatch)
    tasks = build_clinc150_tasks(_fake_dataset(), _FakeTokenizer(), max_length=12)
    config = replace(_multi_seed_config(), evaluate_test=True)

    aggregate = clinc_module.run_split_clinc150_multi_seed(
        config,
        tasks,
        seeds=[1],
        metadata={},
        output_dir=tmp_path / "run",
        confirmatory=True,
    )

    assert aggregate["endpoint_source"] == "test"
    assert aggregate["evaluate_test"] is True
