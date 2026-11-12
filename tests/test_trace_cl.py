"""End-to-end wiring of the TRACE runner on a tiny random causal Qwen2 (CPU).

These tests do not check that any arm learns TRACE; a 2-layer random model
cannot. They check the failure modes this repo has already paid for: a knob
that never reaches the optimizer, a frozen factor that still moves, a manifest
that cannot be tied to its code, and a reader that takes the wrong row.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers import Qwen2Config, Qwen2ForCausalLM

from dual_heater.trace_data import (
    load_trace_tasks,
    special_token_ids,
    trace_data_identity,
)
from experiments.trace_cl import (
    ARMS,
    FROZEN_A_SURFACE,
    OFFICIAL_LORA_EPOCHS,
    TraceRunConfig,
    build_generation_config,
    build_parser,
    config_from_args,
    frozen_a_surface_scale,
    run_arm,
    run_arms,
)

TASKS = ("FOMC", "Py150", "20Minuten")
TARGETS = ("q_proj", "v_proj", "gate_proj", "up_proj", "down_proj")


class CharTokenizer:
    eos_token_id = 2
    bos_token_id = None

    def __call__(self, text, add_special_tokens=False, **_):
        return {"input_ids": [10 + ord(ch) % 50 for ch in text]}

    def batch_decode(self, rows, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return [
            "".join(chr(97 + (token - 10) % 26) for token in row if token >= 10)
            for row in rows.tolist()
        ]


def tiny_factory(config: TraceRunConfig):
    torch.manual_seed(config.seed)
    model_config = Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        tie_word_embeddings=True,
    )
    return Qwen2ForCausalLM(model_config)


@pytest.fixture()
def data_root(tmp_path: Path) -> Path:
    for task in TASKS:
        (tmp_path / task).mkdir()
        rows = [
            {"prompt": f"Q{i}: ", "answer": "hawkish" if i % 2 else "dovish"} for i in range(7)
        ]
        for split in ("train", "eval", "test"):
            (tmp_path / task / f"{split}.json").write_text(json.dumps(rows), encoding="utf-8")
    return tmp_path


def _config(**overrides) -> TraceRunConfig:
    values = {
        "tasks": TASKS,
        "epochs": (1, 2, 1),
        "device": "cpu",
        "micro_batch": 2,
        "grad_accum": 3,
        "eval_batch": 4,
        "max_prompt_len": 16,
        "max_ans_len": 8,
        "max_new_tokens": 4,
        "rank": 4,
        "target_modules": TARGETS,
        "learning_rate": 0.05,
    }
    values.update(overrides)
    return TraceRunConfig(**values)


def _run(method, data_root, *, inspect=None, **overrides):
    config = _config(**overrides)
    tasks = load_trace_tasks(data_root, task_names=config.tasks)
    return run_arm(
        method, tasks, config, CharTokenizer(), model_factory=tiny_factory, inspect=inspect
    )


def _lora(model, factor: str) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if f".lora_{factor}." in name
    }


# ---------------------------------------------------------------------------
# defaults are the official LoRA baseline
# ---------------------------------------------------------------------------


def test_defaults_match_train_lora_sh() -> None:
    config = TraceRunConfig()
    assert config.tasks == (
        "C-STANCE", "FOMC", "MeetingBank", "Py150",
        "ScienceQA", "NumGLUE-cm", "NumGLUE-ds", "20Minuten",
    )
    assert config.epochs == (5, 3, 7, 5, 3, 5, 5, 7)
    assert (config.rank, config.alpha, config.dropout) == (8, 32.0, 0.1)
    assert config.learning_rate == 1e-4
    assert config.adam_betas == (0.9, 0.95)
    assert config.weight_decay == 0.0
    assert (config.max_prompt_len, config.max_ans_len) == (1024, 512)
    assert config.temperature == 0.1 and config.decoding == "official"


def test_cli_without_flags_reproduces_the_defaults(tmp_path: Path) -> None:
    args = build_parser().parse_args(["--data-root", str(tmp_path), "--output", str(tmp_path)])
    config = config_from_args(args)
    assert config.payload() == TraceRunConfig(device="cuda:0").payload()


def test_cli_epochs_follow_the_selected_tasks(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        ["--data-root", str(tmp_path), "--output", str(tmp_path), "--tasks", "Py150", "FOMC"]
    )
    config = config_from_args(args)
    assert config.epochs == (OFFICIAL_LORA_EPOCHS["Py150"], OFFICIAL_LORA_EPOCHS["FOMC"])


def test_fp16_is_refused() -> None:
    with pytest.raises(ValueError):
        TraceRunConfig(dtype="float16").validate()


def test_generation_config_does_not_inherit_checkpoint_sampling() -> None:
    """R9: Qwen2.5-Instruct ships top_p=0.8, top_k=20, repetition_penalty=1.1."""

    special = special_token_ids(CharTokenizer())
    generation = build_generation_config(TraceRunConfig(), special)
    assert generation.do_sample is True
    assert generation.temperature == 0.1
    assert generation.top_p == 1.0
    assert generation.top_k == 0
    assert generation.repetition_penalty == 1.0
    assert generation.max_new_tokens == 512


# ---------------------------------------------------------------------------
# wiring: knobs reach the optimizer, frozen factors stay frozen
# ---------------------------------------------------------------------------


def test_optimizer_steps_flush_at_each_task_boundary(data_root: Path) -> None:
    """7 examples / micro 2 = 4 micro-batches per epoch; accum 3.

    Per task: ceil(epochs * 4 / 3). Epochs (1, 2, 1) -> 2 + 3 + 2 = 7 steps.
    Without the R3 flush the partial windows would merge across tasks.
    """

    result = _run("vanilla", data_root)
    assert result["optimizer_steps"] == 7


def test_vanilla_trains_only_lora_factors(data_root: Path) -> None:
    seen = {}

    def inspect(model, _instr, _optimizer, when):
        if when == "start":
            seen["names"] = [n for n, p in model.named_parameters() if p.requires_grad]

    result = _run("vanilla", data_root, inspect=inspect)
    assert seen["names"]
    assert all(".lora_A." in name or ".lora_B." in name for name in seen["names"])
    assert result["masked_parameters"] == 0


@pytest.mark.parametrize("method", ["frozen_a_control", "exact"])
def test_frozen_a_arms_never_move_a(method: str, data_root: Path) -> None:
    snapshots = {}

    def inspect(model, _instr, _optimizer, when):
        snapshots[when] = (_lora(model, "A"), _lora(model, "B"))

    _run(method, data_root, inspect=inspect)
    a_start, b_start = snapshots["start"]
    a_end, b_end = snapshots["end"]
    assert all(torch.equal(a_start[name], a_end[name]) for name in a_start)
    assert any(not torch.equal(b_start[name], b_end[name]) for name in b_start)


def test_vanilla_moves_a(data_root: Path) -> None:
    """Positive control for the test above: A does move when nothing freezes it."""

    snapshots = {}

    def inspect(model, _instr, _optimizer, when):
        snapshots[when] = _lora(model, "A")

    _run("vanilla", data_root, inspect=inspect)
    assert any(
        not torch.equal(snapshots["start"][name], snapshots["end"][name])
        for name in snapshots["start"]
    )


@pytest.mark.parametrize("method", ["exact", "rank", "leak", "slice"])
def test_masked_arms_register_masks_with_the_optimizer(method: str, data_root: Path) -> None:
    counts = {}

    def inspect(_model, _instr, optimizer, when):
        counts[when] = len(getattr(optimizer, "_plasticity_masks", {}))

    result = _run(method, data_root, inspect=inspect)
    assert counts["start"] > 0
    assert result["masked_parameters"] > 0
    assert 0.0 <= result["masked_mean"] < 1.0


def test_lr_control_frozen_a_surface_pairs_the_learning_rate(data_root: Path) -> None:
    seen = {}

    def inspect(model, _instr, optimizer, when):
        if when == "start":
            seen["lr"] = optimizer.param_groups[0]["lr"]
            seen["scale"] = frozen_a_surface_scale(model)

    result = _run("lr_control", data_root, inspect=inspect, learning_rate_scale=FROZEN_A_SURFACE)
    assert 0.0 < seen["scale"] < 1.0
    assert seen["lr"] == pytest.approx(0.05 * seen["scale"])
    assert result["learning_rate_scale"] == pytest.approx(seen["scale"])


def test_lr_scale_does_not_leak_into_other_arms(data_root: Path) -> None:
    seen = {}

    def inspect(_model, _instr, optimizer, when):
        if when == "start":
            seen["lr"] = optimizer.param_groups[0]["lr"]

    _run("vanilla", data_root, inspect=inspect, learning_rate_scale=0.3)
    assert seen["lr"] == pytest.approx(0.05)


def test_adam_betas_reach_the_optimizer(data_root: Path) -> None:
    seen = {}

    def inspect(_model, _instr, optimizer, when):
        if when == "start":
            seen["betas"] = optimizer.param_groups[0]["betas"]
            seen["wd"] = optimizer.param_groups[0]["weight_decay"]

    _run("vanilla", data_root, inspect=inspect)
    assert tuple(seen["betas"]) == (0.9, 0.95)
    assert seen["wd"] == 0.0


def test_surface_plasticity_pairs_frozen_a_with_lr_control(data_root: Path, tmp_path: Path) -> None:
    """E_surface(frozen_a_control) == E_surface(lr_control @ frozen_a_surface)."""

    config = _config(learning_rate_scale=FROZEN_A_SURFACE)
    tasks = load_trace_tasks(data_root, task_names=config.tasks)
    results = run_arms(
        ["vanilla", "frozen_a_control", "lr_control"],
        tasks,
        config,
        CharTokenizer(),
        tmp_path / "out",
        data_identity=trace_data_identity(data_root, tasks),
        model_factory=tiny_factory,
    )
    by_arm = {row["method"]: row for row in results}
    assert by_arm["vanilla"]["surface_plasticity"] == pytest.approx(1.0)
    assert by_arm["frozen_a_control"]["surface_plasticity"] == pytest.approx(
        by_arm["lr_control"]["surface_plasticity"], abs=1e-12
    )
    assert by_arm["frozen_a_control"]["surface_plasticity"] < 1.0


# ---------------------------------------------------------------------------
# evaluation and artifacts
# ---------------------------------------------------------------------------


def test_matrix_is_lower_triangular_and_metrics_use_the_last_row(data_root: Path) -> None:
    result = _run("vanilla", data_root)
    matrix = result["accuracy_matrix"]
    for stage, row in enumerate(matrix):
        assert all(value is not None for value in row[: stage + 1])
        assert all(value is None for value in row[stage + 1 :])
    assert result["per_task_final"] == pytest.approx(matrix[-1])
    assert result["overall_performance"] == pytest.approx(sum(matrix[-1]) / len(matrix))


def test_runs_are_reproducible_bit_for_bit(data_root: Path) -> None:
    first = _run("vanilla", data_root)
    second = _run("vanilla", data_root)
    assert first["accuracy_matrix"] == second["accuracy_matrix"]
    assert first["_predictions"] == second["_predictions"]


def test_tracked_arm_with_dead_importance_fails_loudly(data_root: Path, monkeypatch) -> None:
    """The fp16 failure mode: importance silently zero. Must raise, not record."""

    from dual_heater import slow_heat

    def dead_hook(self, preactivation, validity_mask=None):
        def hook(grad):
            self.task_step.add_(1)  # steps happen, evidence stays zero
            return grad

        return hook

    monkeypatch.setattr(slow_heat._SlowHeatImportanceMixin, "_functional_importance_hook", dead_hook)
    with pytest.raises(RuntimeError, match="sem importância"):
        _run("exact", data_root)


def test_manifest_with_another_identity_is_never_overwritten(data_root: Path, tmp_path: Path) -> None:
    config = _config()
    tasks = load_trace_tasks(data_root, task_names=config.tasks)
    identity = trace_data_identity(data_root, tasks)
    output = tmp_path / "out"
    run_arms(["vanilla"], tasks, config, CharTokenizer(), output,
             data_identity=identity, model_factory=tiny_factory)
    manifest = json.loads((output / "vanilla" / "manifest.json").read_text())
    assert manifest["identity"]["config"]["learning_rate"] == 0.05
    assert len(manifest["identity"]["source_sha256"]) == 64
    assert (output / "vanilla" / "predictions.json").is_file()
    progress = json.loads((output / "vanilla" / "progress.json").read_text())
    assert progress["complete"] is True
    assert [stage["task"] for stage in progress["stages"]] == list(TASKS)

    # same identity: skipped, not retrained
    again = run_arms(["vanilla"], tasks, config, CharTokenizer(), output,
                     data_identity=identity, model_factory=tiny_factory)
    assert again[0]["accuracy_matrix"] == manifest["result"]["accuracy_matrix"]

    changed = _config(learning_rate=0.01)
    with pytest.raises(RuntimeError, match="identidade diferente"):
        run_arms(["vanilla"], tasks, changed, CharTokenizer(), output,
                 data_identity=identity, model_factory=tiny_factory)


def test_generation_uses_the_fresh_config(data_root: Path, monkeypatch) -> None:
    """R9 must reach generate(): no checkpoint sampling knobs on the call."""

    calls: list[dict] = []
    original = Qwen2ForCausalLM.generate

    def spy(self, *args, **kwargs):
        calls.append(kwargs)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Qwen2ForCausalLM, "generate", spy)
    _run("vanilla", data_root, epochs=(1, 1, 1))
    assert calls
    for kwargs in calls:
        generation = kwargs.get("generation_config")
        assert generation is not None
        assert (generation.top_p, generation.top_k, generation.repetition_penalty) == (1.0, 0, 1.0)
        assert generation.temperature == 0.1
        assert not {"top_p", "top_k", "temperature", "do_sample"} & set(kwargs)


def test_training_forwards_carry_the_padding_mask(data_root: Path, monkeypatch) -> None:
    """Without the validity scope, pad positions feed the importance tracker."""

    from dual_heater.transformer import SlowHeatFFNTracker

    seen: list[bool] = []
    original = SlowHeatFFNTracker.observe

    def spy(self, hidden, validity_mask=None):
        if self.training:
            seen.append(validity_mask is not None)
        return original(self, hidden, validity_mask)

    monkeypatch.setattr(SlowHeatFFNTracker, "observe", spy)
    _run("exact", data_root, epochs=(1, 1, 1))
    assert seen and all(seen)


def test_label_tokenization_reaches_the_collator(data_root: Path, monkeypatch) -> None:
    """The default equals the collator's default, so only 'official' can prove the wire."""

    import experiments.trace_cl as runner

    seen: list[str] = []
    original = runner.collate_train

    def spy(*args, **kwargs):
        seen.append(kwargs.get("label_tokenization", "<missing>"))
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, "collate_train", spy)
    _run("vanilla", data_root, epochs=(1, 1, 1), label_tokenization="official")
    assert seen and set(seen) == {"official"}


def test_every_arm_runs_end_to_end(data_root: Path) -> None:
    for method in ARMS:
        result = _run(method, data_root, epochs=(1, 1, 1))
        assert result["method"] == method
        assert 0.0 <= result["overall_performance"] <= 1.0
