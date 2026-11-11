"""Generative CL runner for Fase 3b (SuperNI-15 from SAPT).

Pinned here, all of them silent failure modes on a real run:

* the loss reads ONLY answer tokens (prompt and padding labelled -100), even
  though Qwen's pad token IS its eos token;
* truncation keeps the START of the prompt (definition) and the output marker;
* after task t only tasks <= t are evaluated (no leakage of unseen tasks);
* AP / forgetting / BWT match SAPT's ``score.py`` arithmetic exactly;
* ``lr_control`` needs an explicit scale, and that scale reaches the optimizer
  and the manifest;
* ``frozen_a_control`` really leaves A untouched;
* E is recorded per arm, with the fields needed to recompute it offline;
* the fp16 path unscales before the masked step and skips non-finite steps.

Everything runs on a tiny random Qwen2 and a character tokenizer, CPU, fp32.
"""

from __future__ import annotations

import math

import pytest
import torch

from dual_heater.superni_data import SuperNIExample, SuperNITask

ALPHABET = "abcdefghijklmnopqrstuvwxyz :\n.ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


class CharTokenizer:
    """Character tokenizer whose pad id equals its eos id, like Qwen2.5."""

    eos_token_id = 1
    pad_token_id = 1

    def __init__(self) -> None:
        self._ids = {char: index + 3 for index, char in enumerate(ALPHABET)}
        self._chars = {index: char for char, index in self._ids.items()}
        self.vocab_size = len(ALPHABET) + 3

    def encode(self, text: str) -> list[int]:
        return [self._ids.get(char, 2) for char in text]

    def decode(self, ids) -> str:
        return "".join(self._chars.get(int(index), "") for index in ids)


def tiny_model_factory():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(
        vocab_size=CharTokenizer().vocab_size,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=256,
        tie_word_embeddings=True,
        eos_token_id=1,
        pad_token_id=1,
        bos_token_id=None,
    )
    config.torch_dtype = torch.float32
    torch.manual_seed(1234)
    return Qwen2ForCausalLM(config)


def _example(task: str, index: int, answer: str) -> SuperNIExample:
    return SuperNIExample(
        task_name=task,
        prompt=f"Copy the word.\n\nInput: {answer}\nOutput:",
        target=answer,
        references=(answer,),
        example_id=f"{task}-{index}",
    )


def tiny_tasks(count: int = 3) -> list[SuperNITask]:
    words = ["abc", "dog", "cat", "sun", "map", "kit", "red", "box"]
    tasks = []
    for task_index in range(count):
        name = f"t{task_index}"
        shift = task_index * 2
        train = tuple(
            _example(name, index, words[(index + shift) % len(words)])
            for index in range(6)
        )
        evaluation = tuple(
            _example(name, 100 + index, words[(index + shift + 3) % len(words)])
            for index in range(2)
        )
        tasks.append(SuperNITask(name=name, train=train, evaluation=evaluation))
    return tasks


def tiny_config(**overrides):
    from experiments.superni_cl import RunConfig

    values = dict(
        model_name="tiny",
        seed=0,
        batch_size=2,
        eval_batch_size=2,
        steps_per_task=3,
        learning_rate=5e-2,
        rank=2,
        alpha=4.0,
        max_source_length=64,
        max_target_length=8,
        max_new_tokens=6,
        device="cpu",
        learning_rate_scale=0.5,
    )
    values.update(overrides)
    return RunConfig(**values)


# ---------------------------------------------------------------------------
# tokenization and loss masking
# ---------------------------------------------------------------------------


def test_prompt_truncation_keeps_the_start_and_the_output_marker() -> None:
    from experiments.superni_cl import OUTPUT_MARKER, encode_prompt

    tokenizer = CharTokenizer()
    body = "Definition first. " + "x" * 200
    prompt = body + OUTPUT_MARKER
    ids, truncated = encode_prompt(tokenizer, prompt, max_source_length=40)
    marker = tokenizer.encode(OUTPUT_MARKER)
    assert truncated is True
    assert len(ids) == 40
    assert ids[: len("Definition first.")] == tokenizer.encode("Definition first.")
    assert ids[-len(marker) :] == marker

    short, short_truncated = encode_prompt(tokenizer, "Hi." + OUTPUT_MARKER, max_source_length=40)
    assert short_truncated is False
    assert tokenizer.decode(short) == "Hi." + OUTPUT_MARKER


def test_target_is_truncated_then_terminated_by_eos() -> None:
    from experiments.superni_cl import encode_target

    tokenizer = CharTokenizer()
    ids = encode_target(tokenizer, "abcdefghij", max_target_length=4)
    assert ids == tokenizer.encode(" abc") + [tokenizer.eos_token_id]
    assert encode_target(tokenizer, "ab", max_target_length=50) == tokenizer.encode(" ab") + [1]


def test_training_labels_cover_only_answer_tokens_even_when_pad_is_eos() -> None:
    from experiments.superni_cl import IGNORE_INDEX, collate_training, encode_example

    tokenizer = CharTokenizer()
    long_answer = encode_example(
        tokenizer, _example("t", 0, "abcd"), max_source_length=64, max_target_length=8
    )
    short_answer = encode_example(
        tokenizer, _example("t", 1, "a"), max_source_length=64, max_target_length=8
    )
    batch = collate_training([long_answer, short_answer], pad_token_id=1)
    labels = batch["labels"]
    attention = batch["attention_mask"]
    for row, feature in enumerate([long_answer, short_answer]):
        prompt_length = len(feature.prompt_ids)
        answer = feature.target_ids
        assert (labels[row, :prompt_length] == IGNORE_INDEX).all()
        assert labels[row, prompt_length : prompt_length + len(answer)].tolist() == answer
        assert (labels[row, prompt_length + len(answer) :] == IGNORE_INDEX).all()
        assert int(attention[row].sum()) == prompt_length + len(answer)
    # the eos that terminates the answer is a label; the pad eos after it is not
    row = 1
    end = len(short_answer.prompt_ids) + len(short_answer.target_ids)
    assert labels[row, end - 1] == tokenizer.eos_token_id
    assert batch["input_ids"][row, end] == tokenizer.pad_token_id
    assert labels[row, end] == IGNORE_INDEX


def test_model_loss_is_cross_entropy_over_answer_tokens_only() -> None:
    from experiments.superni_cl import collate_training, encode_example

    tokenizer = CharTokenizer()
    features = [
        encode_example(tokenizer, _example("t", i, w), max_source_length=64, max_target_length=8)
        for i, w in enumerate(["abcd", "x"])
    ]
    batch = collate_training(features, pad_token_id=1)
    model = tiny_model_factory()
    output = model(**batch)
    logits = output.logits[:, :-1].float()
    targets = batch["input_ids"][:, 1:]
    answer = torch.zeros_like(targets, dtype=torch.bool)
    for row, feature in enumerate(features):
        start = len(feature.prompt_ids) - 1
        answer[row, start : start + len(feature.target_ids)] = True
    manual = torch.nn.functional.cross_entropy(logits[answer], targets[answer])
    assert torch.allclose(output.loss, manual, atol=1e-6)


# ---------------------------------------------------------------------------
# generation and scoring
# ---------------------------------------------------------------------------


def test_cut_at_eos_stops_at_the_first_eos() -> None:
    from experiments.superni_cl import cut_at_eos

    assert cut_at_eos([5, 6, 1, 7, 1], eos_token_id=1) == [5, 6]
    assert cut_at_eos([5, 6], eos_token_id=1) == [5, 6]
    assert cut_at_eos([1, 5], eos_token_id=1) == []


def test_batched_left_padded_generation_matches_one_by_one() -> None:
    from experiments.superni_cl import generate_answers

    tokenizer = CharTokenizer()
    model = tiny_model_factory().eval()
    prompts = [tokenizer.encode("Copy: abc\nOutput:"), tokenizer.encode("Hi\nOutput:")]
    batched = generate_answers(
        model, prompts, pad_token_id=1, eos_token_id=1, max_new_tokens=6,
        batch_size=2, device="cpu",
    )
    single = [
        generate_answers(
            model, [prompt], pad_token_id=1, eos_token_id=1, max_new_tokens=6,
            batch_size=1, device="cpu",
        )[0]
        for prompt in prompts
    ]
    assert batched == single
    assert all(len(ids) <= 6 for ids in batched)


def test_score_is_rouge_l_max_over_references() -> None:
    from experiments.superni_cl import score_predictions

    scores = score_predictions(
        ["alt answer", "nothing", ""], [("gold", "alt answer"), ("gold",), ("gold",)]
    )
    assert scores["rouge_l"] == pytest.approx(1.0 / 3.0)
    assert scores["exact_match"] == pytest.approx(1.0 / 3.0)
    assert scores["per_example_rouge_l"] == [1.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# endpoints (SAPT score.py)
# ---------------------------------------------------------------------------


def _sapt_reference(scores_array):
    """Verbatim port of SAPT score.py::cal_continue_learning_metrics (no Fwt)."""

    task_num = len(scores_array)
    cl = sum(scores_array[-1]) / task_num
    fgt_list = []
    for t_idx in range(task_num - 1):
        history = [line[t_idx] for line in scores_array[:-1]]
        history_best = max(history)
        fgt_list.append(history_best - scores_array[-1][t_idx])
    fgt = sum(fgt_list) / len(fgt_list)
    bwt = sum(scores_array[-1][i] - scores_array[i][i] for i in range(task_num)) / task_num
    return {"Cl": cl, "Fgt": fgt, "Bwt": bwt}


MATRIX = [
    [0.50, None, None],
    [0.40, 0.70, None],
    [0.30, 0.65, 0.90],
]


def test_endpoints_match_hand_computed_values() -> None:
    from experiments.superni_cl import sapt_endpoints

    result = sapt_endpoints(MATRIX)
    assert result["ap"] == pytest.approx((0.30 + 0.65 + 0.90) / 3)
    # best-ever before the final row minus final: t0 0.50-0.30, t1 0.70-0.65
    assert result["per_task_forgetting"] == pytest.approx([0.20, 0.05])
    assert result["forgetting"] == pytest.approx(0.125)
    # SAPT divides BWT by T, last term is structurally zero
    assert result["bwt"] == pytest.approx(((0.30 - 0.50) + (0.65 - 0.70) + 0.0) / 3)


def test_endpoints_equal_sapt_score_py_with_zero_padding() -> None:
    from experiments.superni_cl import sapt_endpoints

    matrices = [
        MATRIX,
        [[0.2, None, None, None], [0.5, 0.1, None, None], [0.4, 0.3, 0.6, None], [0.6, 0.0, 0.2, 0.3]],
        [[0.9, None], [0.1, 0.4]],
    ]
    for matrix in matrices:
        padded = [[0.0 if value is None else value for value in row] for row in matrix]
        reference = _sapt_reference(padded)
        ours = sapt_endpoints(matrix)
        assert ours["ap"] == pytest.approx(reference["Cl"])
        assert ours["forgetting"] == pytest.approx(reference["Fgt"])
        assert ours["bwt"] == pytest.approx(reference["Bwt"])


def test_forgetting_best_excludes_the_final_row() -> None:
    """A score that only peaks at the end gives NEGATIVE forgetting, as in SAPT."""

    from experiments.superni_cl import sapt_endpoints

    result = sapt_endpoints([[0.2, None], [0.6, 0.5]])
    assert result["per_task_forgetting"] == pytest.approx([-0.4])


def test_endpoints_reject_seen_entries_left_empty() -> None:
    from experiments.superni_cl import sapt_endpoints

    with pytest.raises(ValueError):
        sapt_endpoints([[0.5, None], [None, 0.5]])


def test_acquisition_gate_is_diagonal_minus_zero_shot() -> None:
    from experiments.superni_cl import acquisition_gate

    gate = acquisition_gate(MATRIX, [0.40, 0.70, 0.10], margin=0.05)
    assert [row["delta"] for row in gate["per_task"]] == pytest.approx([0.10, 0.0, 0.80])
    assert [row["passes"] for row in gate["per_task"]] == [True, False, True]
    assert gate["all_pass"] is False
    assert gate["margin"] == 0.05


def test_dispersion_summary_reports_unmatched_and_matched_deltas() -> None:
    from experiments.superni_cl import dispersion_summary

    results = [
        {"method": "vanilla", "endpoints": {"ap": 0.40, "forgetting": 0.20}, "surface_plasticity": 1.0},
        {"method": "exact", "endpoints": {"ap": 0.46, "forgetting": 0.10}, "surface_plasticity": 0.50},
        {"method": "frozen_a_control", "endpoints": {"ap": 0.44, "forgetting": 0.12}, "surface_plasticity": 0.60},
        {"method": "lr_control", "endpoints": {"ap": 0.45, "forgetting": 0.08}, "surface_plasticity": 0.50},
    ]
    summary = dispersion_summary(results, min_vanilla_forgetting=0.05, matched_tolerance=0.01)
    assert summary["ap_range"] == pytest.approx(0.06)
    assert summary["forgetting_range"] == pytest.approx(0.12)
    assert summary["vanilla_forgetting"] == pytest.approx(0.20)
    assert summary["vanilla_forgetting_ok"] is True
    assert summary["d_unmatched"] == pytest.approx({"exact": 0.06, "frozen_a_control": 0.04})
    # only exact sits at lr_control's E; frozen_a_control is unmatched (0.60 vs 0.50)
    assert summary["d_matched"]["exact"] == pytest.approx(0.01)
    assert summary["d_matched"]["frozen_a_control"] is None
    assert summary["primary_varies"] is True


# ---------------------------------------------------------------------------
# arms: lr scale, frozen A, importance, E
# ---------------------------------------------------------------------------


def test_lr_control_needs_an_explicit_scale() -> None:
    from experiments.superni_cl import resolve_effective_lr

    config = tiny_config(learning_rate=1e-3, learning_rate_scale=None)
    assert resolve_effective_lr("vanilla", config) == 1e-3
    with pytest.raises(ValueError):
        resolve_effective_lr("lr_control", config)
    scaled = tiny_config(learning_rate=1e-3, learning_rate_scale=0.25)
    assert resolve_effective_lr("lr_control", scaled) == pytest.approx(2.5e-4)
    # only lr_control is scaled
    assert resolve_effective_lr("exact", scaled) == 1e-3
    assert resolve_effective_lr("frozen_a_control", scaled) == 1e-3


def test_cli_scale_reaches_the_config() -> None:
    from experiments.superni_cl import build_parser, build_run_config

    arguments = build_parser().parse_args(
        ["--output", "x", "--learning-rate-scale", "0.37", "--order", "2",
         "--train-per-task", "7", "--eval-per-task", "3", "--precision", "fp16",
         "--steps-per-task", "11"]
    )
    config = build_run_config(arguments)
    assert config.learning_rate_scale == pytest.approx(0.37)
    assert config.order == 2
    assert config.train_per_task == 7 and config.eval_per_task == 3
    assert config.precision == "fp16"
    assert config.steps_per_task == 11
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--output", "x", "--precision", "bf16"])


def test_validate_rejects_lr_control_without_scale_and_fp16_on_cpu() -> None:
    config = tiny_config(learning_rate_scale=None)
    with pytest.raises(ValueError):
        config.validate(arms=["vanilla", "lr_control"])
    config.validate(arms=["vanilla", "exact"])
    with pytest.raises(ValueError):
        tiny_config(precision="fp16", device="cpu").validate(arms=["vanilla"])


def test_lr_control_optimizer_trains_at_the_scaled_rate() -> None:
    from experiments.superni_cl import build_arm

    arm = build_arm("lr_control", tiny_config(learning_rate=0.2, learning_rate_scale=0.3),
                    model_factory=tiny_model_factory, task_count=2)
    assert all(group["lr"] == pytest.approx(0.06) for group in arm.optimizer.param_groups)


def _train_once(method: str):
    from experiments.superni_cl import build_arm, encode_task, train_task

    tokenizer = CharTokenizer()
    config = tiny_config()
    arm = build_arm(method, config, model_factory=tiny_model_factory, task_count=2)
    encoded = encode_task(tokenizer, tiny_tasks(1)[0], config)
    before = {name: p.detach().clone() for name, p in arm.model.named_parameters()}
    train_task(arm, encoded.train, config, stage=0, pad_token_id=1)
    return arm, before


def test_frozen_a_control_leaves_a_untouched_and_trains_b() -> None:
    arm, before = _train_once("frozen_a_control")
    a_names = [n for n, _ in arm.model.named_parameters() if ".lora_A." in n]
    b_names = [n for n, _ in arm.model.named_parameters() if ".lora_B." in n]
    assert a_names and b_names
    parameters = dict(arm.model.named_parameters())
    for name in a_names:
        assert not parameters[name].requires_grad
        assert torch.equal(parameters[name], before[name])
    assert any(not torch.equal(parameters[name], before[name]) for name in b_names)


def test_vanilla_trains_a_as_well() -> None:
    arm, before = _train_once("vanilla")
    parameters = dict(arm.model.named_parameters())
    a_names = [n for n in parameters if ".lora_A." in n]
    assert any(not torch.equal(parameters[n], before[n]) for n in a_names)
    # base weights never move
    base = [n for n in parameters if "lora_" not in n]
    assert all(torch.equal(parameters[n], before[n]) for n in base)


def test_importance_liveness_detects_a_dead_module() -> None:
    from experiments.superni_cl import importance_liveness

    arm, _ = _train_once("exact")
    live = importance_liveness(arm.instrumentation)
    assert live["applicable"] is True
    assert live["live"] is True
    assert live["modules_dead"] == 0
    assert live["layers_total"] == 2
    arm.instrumentation.modules[0].tracker.task_ema.zero_()
    dead = importance_liveness(arm.instrumentation)
    assert dead["live"] is False
    assert dead["modules_dead"] == 1
    assert dead["dead_modules"] == [arm.instrumentation.modules[0].name]
    assert dead["layers_dead"] == []  # the layer still has two live modules

    vanilla, _ = _train_once("vanilla")
    assert importance_liveness(vanilla.instrumentation) == {"applicable": False}


def test_exact_hard_mask_freezes_protected_rows_on_the_next_task() -> None:
    from experiments.superni_cl import build_arm, encode_task, train_task

    tokenizer = CharTokenizer()
    config = tiny_config(hard=True, plasticity_budget=0.5)
    arm = build_arm("exact", config, model_factory=tiny_model_factory, task_count=2)
    tasks = tiny_tasks(2)
    train_task(arm, encode_task(tokenizer, tasks[0], config).train, config, stage=0, pad_token_id=1)
    arm.instrumentation.consolidate()
    adapted = arm.instrumentation.modules[0]
    protected = adapted.tracker.slow_heat > 0
    assert protected.any() and (~protected).any()
    before = adapted.lora_b.weight.detach().clone()
    train_task(arm, encode_task(tokenizer, tasks[1], config).train, config, stage=1, pad_token_id=1)
    after = adapted.lora_b.weight.detach()
    assert torch.equal(after[protected], before[protected])
    assert not torch.equal(after[~protected], before[~protected])


# ---------------------------------------------------------------------------
# the loop: no leakage, E per arm, manifest
# ---------------------------------------------------------------------------


def test_only_seen_tasks_are_evaluated_after_each_stage(monkeypatch) -> None:
    import experiments.superni_cl as runner

    calls: list[tuple[int, str]] = []
    original = runner.evaluate_task

    def spy(model, encoded, config, *, tokenizer, stage=None):
        calls.append((stage, encoded.name))
        return original(model, encoded, config, tokenizer=tokenizer, stage=stage)

    monkeypatch.setattr(runner, "evaluate_task", spy)
    tokenizer = CharTokenizer()
    config = tiny_config(steps_per_task=1)
    encoded = [runner.encode_task(tokenizer, task, config) for task in tiny_tasks(3)]
    record = runner.run_arm("vanilla", encoded, config, tokenizer=tokenizer,
                            model_factory=tiny_model_factory)
    assert calls == [(0, "t0"), (1, "t0"), (1, "t1"), (2, "t0"), (2, "t1"), (2, "t2")]
    matrix = record["rouge_l_matrix"]
    for stage in range(3):
        for task in range(3):
            assert (matrix[stage][task] is None) == (task > stage)


def test_step_budget_is_exact_per_task() -> None:
    import experiments.superni_cl as runner

    tokenizer = CharTokenizer()
    config = tiny_config(steps_per_task=4, batch_size=4)  # 6 examples -> wraps
    encoded = [runner.encode_task(tokenizer, task, config) for task in tiny_tasks(2)]
    record = runner.run_arm("vanilla", encoded, config, tokenizer=tokenizer,
                            model_factory=tiny_model_factory)
    assert record["optimizer_steps"] == 8
    assert [stage["train_steps"] for stage in record["stages"]] == [4, 4]


def test_epoch_budget_covers_the_data_once_per_epoch() -> None:
    import experiments.superni_cl as runner

    tokenizer = CharTokenizer()
    config = tiny_config(steps_per_task=None, epochs_per_task=2, batch_size=4)
    encoded = [runner.encode_task(tokenizer, task, config) for task in tiny_tasks(2)]
    record = runner.run_arm("vanilla", encoded, config, tokenizer=tokenizer,
                            model_factory=tiny_model_factory)
    # 6 examples, batch 4 -> 2 batches per epoch, 2 epochs, 2 tasks
    assert record["optimizer_steps"] == 8


def test_experiment_records_e_zero_shot_and_gates_per_arm() -> None:
    import experiments.superni_cl as runner

    tokenizer = CharTokenizer()
    config = tiny_config(learning_rate_scale=0.4)
    manifest = runner.run_experiment(
        config, tiny_tasks(2), arms=list(runner.ARMS), tokenizer=tokenizer,
        model_factory=tiny_model_factory,
    )
    by_arm = {row["method"]: row for row in manifest["results"]}
    assert set(by_arm) == {"vanilla", "exact", "frozen_a_control", "lr_control", "random_fixed"}
    for row in by_arm.values():
        for key in ("masked_mean", "masked_parameters", "learning_rate_scale",
                    "surface_plasticity", "trainable_parameters"):
            assert key in row, (row["method"], key)
        assert len(row["rouge_l_matrix"]) == 2
        assert set(row["endpoints"]) >= {"ap", "forgetting", "bwt"}
        assert row["gates"]["acquisition"]["per_task"]
    vanilla = by_arm["vanilla"]
    assert vanilla["surface_plasticity"] == pytest.approx(1.0)
    assert by_arm["lr_control"]["surface_plasticity"] == pytest.approx(0.4)
    assert by_arm["lr_control"]["learning_rate_scale"] == pytest.approx(0.4)
    frozen = by_arm["frozen_a_control"]
    assert frozen["surface_plasticity"] == pytest.approx(
        frozen["trainable_parameters"] / vanilla["trainable_parameters"]
    )
    assert frozen["surface_plasticity"] < 1.0
    exact = by_arm["exact"]
    assert exact["masked_parameters"] > 0
    assert exact["masked_mean"] < 1.0
    assert exact["surface_plasticity"] < frozen["surface_plasticity"]
    # the E in force during each stage: identity on the first task
    assert exact["stages"][0]["mask_mean_in_force"] == pytest.approx(1.0)
    assert exact["stages"][1]["mask_mean_in_force"] < 1.0

    zero_shot = manifest["zero_shot"]["rouge_l"]
    assert len(zero_shot) == 2
    direct = runner.evaluate_task(
        tiny_model_factory(), runner.encode_task(tokenizer, tiny_tasks(2)[1], config),
        config, tokenizer=tokenizer,
    )
    assert zero_shot[1] == pytest.approx(direct["rouge_l"])

    gates = manifest["gates"]
    assert set(gates["G-aq"]) == set(by_arm)
    # read before consolidation, so a healthy fp32 run is live in every module
    assert gates["G-imp"]["exact"] is True
    assert all(stage["importance_liveness"]["live"] for stage in exact["stages"])
    assert gates["G-imp"]["vanilla"] is None
    shuffled = by_arm["random_fixed"]
    assert gates["G-imp"]["random_fixed"] is True
    assert shuffled["masked_parameters"] == exact["masked_parameters"]
    assert shuffled["trainable_parameters"] == frozen["trainable_parameters"]
    assert shuffled["stages"][0]["mask_mean_in_force"] == pytest.approx(1.0)
    assert shuffled["stages"][1]["mask_mean_in_force"] < 1.0
    assert all(stage["heat_reassignment"]["protected_rows"] > 0 for stage in shuffled["stages"])
    assert all(stage["heat_reassignment"] is None for stage in exact["stages"])
    assert "ap_range" in gates["G-disp"]
    assert manifest["protocol_hash"] == runner.protocol_hash(manifest["protocol"])
    assert manifest["protocol"]["arms"] == list(runner.ARMS)
    assert manifest["timings"]["seconds_per_train_step"] > 0
    assert manifest["timings"]["seconds_per_generated_example"] > 0


def test_protocol_hash_tracks_the_declared_protocol() -> None:
    from experiments.superni_cl import build_protocol, protocol_hash

    first = build_protocol(tiny_config(), task_names=["a", "b"], arms=["vanilla"])
    again = build_protocol(tiny_config(), task_names=["a", "b"], arms=["vanilla"])
    other = build_protocol(tiny_config(seed=1), task_names=["a", "b"], arms=["vanilla"])
    assert protocol_hash(first) == protocol_hash(again)
    assert protocol_hash(first) != protocol_hash(other)
    assert "device" not in first


# ---------------------------------------------------------------------------
# fp16 path logic (CPU, fp32 parameters)
# ---------------------------------------------------------------------------


def _masked_pair():
    from dual_heater.optim import PlasticityMaskBinding, SlowHeatAdamW

    torch.manual_seed(0)
    layer = torch.nn.Linear(4, 3, bias=False)
    mask = torch.tensor([[0.0], [1.0], [0.5]])
    optimizer = SlowHeatAdamW([layer.weight], lr=0.1, weight_decay=0.0)
    optimizer.register_mask_bindings([PlasticityMaskBinding(layer.weight, mask, "w")])
    return layer, optimizer


def test_scaled_step_unscales_before_the_masked_update() -> None:
    from experiments.superni_cl import scaled_optimizer_step

    inputs = torch.randn(5, 4, generator=torch.Generator().manual_seed(3))
    plain_layer, plain_optimizer = _masked_pair()
    plain_layer(inputs).pow(2).mean().backward()
    torch.nn.utils.clip_grad_norm_([plain_layer.weight], 0.05)
    plain_optimizer.step()

    layer, optimizer = _masked_pair()
    start = layer.weight.detach().clone()
    scaler = torch.amp.GradScaler("cpu", init_scale=2.0**12)
    scaler.scale(layer(inputs).pow(2).mean()).backward()
    stepped = scaled_optimizer_step(scaler, optimizer, [layer.weight], max_grad_norm=0.05)
    assert stepped is True
    # identical to the unscaled step: clipping saw the TRUE gradient norm
    assert torch.allclose(layer.weight, plain_layer.weight, atol=1e-6)
    assert torch.equal(layer.weight[0], start[0])  # mask 0 row frozen


def test_scaled_step_skips_non_finite_gradients() -> None:
    from experiments.superni_cl import scaled_optimizer_step

    layer, optimizer = _masked_pair()
    start = layer.weight.detach().clone()
    scaler = torch.amp.GradScaler("cpu", init_scale=2.0**12)
    scaler.scale(layer(torch.randn(2, 4)).sum()).backward()
    layer.weight.grad[1, 1] = float("inf")
    stepped = scaled_optimizer_step(scaler, optimizer, [layer.weight], max_grad_norm=1.0)
    assert stepped is False
    assert torch.equal(layer.weight, start)
    assert scaler.get_scale() < 2.0**12


def test_fp16_precision_keeps_lora_adapters_in_fp32() -> None:
    from experiments.superni_cl import configure_precision
    from dual_heater.lora_slowheat import LoRASlowHeatConfig, build_lora_slowheat

    model, _ = build_lora_slowheat(
        tiny_model_factory(), LoRASlowHeatConfig(method="vanilla", rank=2),
        task_type="CAUSAL_LM",
    )
    configure_precision(model, "fp16")
    lora = [(n, p) for n, p in model.named_parameters() if "lora_" in n]
    base = [(n, p) for n, p in model.named_parameters() if "lora_" not in n]
    assert lora and base
    assert all(p.dtype == torch.float32 for _, p in lora)
    assert all(p.dtype == torch.float16 for _, p in base)
    configure_precision(model, "fp32")
    assert all(p.dtype == torch.float32 for p in model.parameters())
    with pytest.raises(ValueError):
        configure_precision(model, "bf16")


def test_nan_is_not_a_score() -> None:
    from experiments.superni_cl import score_predictions

    assert math.isnan(score_predictions([], [])["rouge_l"])


def test_zero_shot_runs_first_on_the_untouched_base_model(monkeypatch) -> None:
    """A tiny random model generates the same junk whatever its weights, so
    the score alone cannot pin this: the evaluated weights are compared."""

    import experiments.superni_cl as runner

    def checksum(model) -> list[float]:
        return [float(p.detach().double().sum()) for p in model.parameters()]

    reference = checksum(tiny_model_factory())
    seen: list[tuple[int, bool, list[float] | None]] = []
    original = runner.evaluate_task

    def spy(model, encoded, config, *, tokenizer, stage=None):
        has_lora = any("lora_" in name for name, _ in model.named_parameters())
        seen.append((stage, has_lora, None if has_lora else checksum(model)))
        return original(model, encoded, config, tokenizer=tokenizer, stage=stage)

    monkeypatch.setattr(runner, "evaluate_task", spy)
    runner.run_experiment(
        tiny_config(steps_per_task=1), tiny_tasks(2), arms=["vanilla"],
        tokenizer=CharTokenizer(), model_factory=tiny_model_factory,
    )
    assert [stage for stage, _, _ in seen[:2]] == [-1, -1]
    assert all(stage >= 0 for stage, _, _ in seen[2:])
    for _, has_lora, weights in seen[:2]:
        assert has_lora is False
        assert weights == pytest.approx(reference)


def test_freeze_audit_is_recorded_per_stage() -> None:
    """The manifest itself must show hard-masked entries and frozen A did not move.

    This is what the GPU fp16 smoke is read against, so it is pinned on CPU.
    """

    import experiments.superni_cl as runner

    tokenizer = CharTokenizer()
    config = tiny_config(hard=True, plasticity_budget=0.5, steps_per_task=2)
    encoded = [runner.encode_task(tokenizer, task, config) for task in tiny_tasks(2)]
    exact = runner.run_arm("exact", encoded, config, tokenizer=tokenizer,
                           model_factory=tiny_model_factory)
    audit = exact["stages"][1]["freeze_audit"]
    assert audit["zero_mask_entries"] > 0
    assert audit["zero_mask_max_abs_change"] == 0.0
    assert audit["plastic_entries_changed_fraction"] > 0.0
    assert audit["frozen_parameters"] > 0
    assert audit["frozen_max_abs_change"] == 0.0
    # nothing is masked on the first task
    assert exact["stages"][0]["freeze_audit"]["zero_mask_entries"] == 0

    vanilla = runner.run_arm("vanilla", encoded, config, tokenizer=tokenizer,
                             model_factory=tiny_model_factory)
    plain = vanilla["stages"][1]["freeze_audit"]
    assert plain["frozen_parameters"] == 0
    assert plain["zero_mask_entries"] == 0


def test_answer_only_loss_equals_the_full_logits_loss() -> None:
    """lm_head runs only on answer positions (memory), the value must not change.

    Full-vocabulary logits for 4 x 1075 tokens do not fit an 11-12 GB card;
    this pins that the cheap path is the same loss, through PEFT as well.
    """

    from experiments.superni_cl import answer_only_loss, collate_training, encode_example
    from dual_heater.lora_slowheat import LoRASlowHeatConfig, build_lora_slowheat

    tokenizer = CharTokenizer()
    features = [
        encode_example(tokenizer, _example("t", i, w), max_source_length=64, max_target_length=8)
        for i, w in enumerate(["abcd", "x", "dog cat"])
    ]
    batch = collate_training(features, pad_token_id=1)
    plain = tiny_model_factory()
    assert torch.allclose(answer_only_loss(plain, batch), plain(**batch).loss, atol=1e-6)
    wrapped, _ = build_lora_slowheat(
        tiny_model_factory(), LoRASlowHeatConfig(method="vanilla", rank=2), task_type="CAUSAL_LM"
    )
    for name, parameter in wrapped.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(parameter, std=0.5)
    expected = wrapped(**batch).loss
    got = answer_only_loss(wrapped, batch)
    assert torch.allclose(got, expected, atol=1e-6)
    got.backward()
    assert all(
        p.grad is not None for n, p in wrapped.named_parameters() if "lora_" in n
    )


def test_generation_groups_prompts_by_length_and_keeps_the_input_order() -> None:
    """Sorting by length cuts padding; answers must still come back in input order."""

    from experiments.superni_cl import generate_answers

    tokenizer = CharTokenizer()
    model = tiny_model_factory().eval()
    long_a = tokenizer.encode("Copy the following long text: abcdefgh\nOutput:")
    long_b = tokenizer.encode("Copy the following long text: hgfedcba\nOutput:")
    short_a = tokenizer.encode("Hi\nOutput:")
    short_b = tokenizer.encode("Yo\nOutput:")
    prompts = [long_a, short_a, long_b, short_b]
    widths: list[int] = []
    original = model.generate

    def spy(*args, **kwargs):
        widths.append(int(kwargs["input_ids"].shape[1]))
        return original(*args, **kwargs)

    model.generate = spy
    batched = generate_answers(
        model, prompts, pad_token_id=1, eos_token_id=1, max_new_tokens=6, batch_size=2, device="cpu",
    )
    model.generate = original
    single = [
        generate_answers(
            model, [prompt], pad_token_id=1, eos_token_id=1, max_new_tokens=6, batch_size=1, device="cpu",
        )[0]
        for prompt in prompts
    ]
    assert batched == single
    assert sorted(widths) == sorted([len(long_a), len(short_a)]), widths


class _EchoModel:
    """generate() answers each row with its last real prompt token, then eos.

    Answers differ per prompt, so a scrambled output order is visible.
    """

    def generate(self, *, input_ids, attention_mask, generation_config):
        import torch as _t

        last = input_ids[:, -1:]
        eos = _t.full_like(last, generation_config.eos_token_id)
        return _t.cat([input_ids, last, eos], dim=1)


def test_sorted_generation_returns_each_answer_to_its_own_prompt() -> None:
    from experiments.superni_cl import generate_answers

    prompts = [[7, 7, 7, 7, 7, 11], [5, 12], [7, 7, 7, 7, 13], [5, 14], [9, 9, 9, 15]]
    answers = generate_answers(
        _EchoModel(), prompts, pad_token_id=0, eos_token_id=1, max_new_tokens=4, batch_size=2, device="cpu",
    )
    assert answers == [[11], [12], [13], [14], [15]]


def test_random_fixed_row_order_is_seeded_from_the_run_seed() -> None:
    import experiments.superni_cl as runner

    orders = []
    for seed in (5, 5, 6):
        arm = runner.build_arm("random_fixed", tiny_config(seed=seed),
                               model_factory=tiny_model_factory, task_count=2)
        assert arm.instrumentation.config.random_seed == seed + 500_000
        orders.append(arm.instrumentation.fixed_row_order(0))
    assert torch.equal(orders[0], orders[1])
    assert not torch.equal(orders[0], orders[2])
