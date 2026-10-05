"""Section 3.1 of goals/roadmap_icml_ijcnn.md: more than two tasks per sequence.

R-A/R-B's declared limit is "one host, one benchmark, TWO tasks per sequence".
Three places hard-code that two:

* ``run_diagnostic`` passes ``task_limit=2`` regardless of the base config;
* the CLI declares ``--task-limit`` with ``choices=[2]``;
* ``condition_endpoints`` reads ``validation_accuracy_matrix[1]`` and
  ``parameter_drift_history[1]``, i.e. the SECOND stage, not the LAST one.

The third is the dangerous one: with five tasks it would silently report
"retention" measured after task 2 while the run went on to task 5 -- a number
that looks right, is not, and nothing in the artefact would say so.

Every test here also pins the T=2 behaviour, because the published BERT
results were produced by that path and must stay bit-identical.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import experiments.bert_slowheat_diagnostic as diagnostic
from experiments.bert_slowheat_diagnostic_common import (
    condition_endpoints,
)
from experiments.split_clinc150 import SplitCLINC150Config


def _drift(stage: int, protected_rms: float):
    return {
        "stage": stage,
        "protected_count": 4,
        "protected_rms": protected_rms,
        "protected_max_abs": protected_rms * 2,
        "plastic_count": 8,
        "plastic_rms": protected_rms * 3,
        "plastic_max_abs": protected_rms * 4,
    }


def _result(matrix, drift_stages):
    return {
        "validation_accuracy_matrix": matrix,
        "validation_metrics": {
            "final_average_accuracy": 0.5,
            "average_forgetting": 0.3,
            "backward_transfer": -0.3,
            "forward_transfer": None,
            "per_task_forgetting": [0.3],
        },
        "parameter_drift_history": [
            _drift(stage, 1.0e-4 * (stage + 1)) for stage in range(drift_stages)
        ],
        "elapsed_seconds": 10.0,
        "tokens_processed": 1_000,
        "replay_memory_bytes": 2**20,
        "peak_memory": {
            "peak_memory_bytes": 2 * 2**20,
            "peak_cuda_reserved_bytes": 3 * 2**20,
        },
    }


def _five_task_result():
    matrix = [
        [0.90, None, None, None, None],
        [0.70, 0.88, None, None, None],
        [0.55, 0.60, 0.86, None, None],
        [0.40, 0.50, 0.58, 0.84, None],
        [0.20, 0.35, 0.45, 0.55, 0.82],
    ]
    return _result(matrix, drift_stages=5)


# ── endpoints must follow the sequence, not stage index 1 ──────────────


def test_retention_is_measured_after_the_last_task_not_after_the_second():
    endpoints = condition_endpoints(_five_task_result())

    assert endpoints["task1_acquisition"] == pytest.approx(0.90)
    # Stage 1 would give 0.70 -- the value that looks plausible and is wrong.
    assert endpoints["task1_retention"] == pytest.approx(0.20)
    assert endpoints["task1_forgetting"] == pytest.approx(0.70)


def test_endpoints_expose_the_sequence_length_so_runs_cannot_be_mixed():
    assert condition_endpoints(_five_task_result())["task_count"] == 5


def test_last_task_acquisition_is_the_diagonal_of_the_final_stage():
    endpoints = condition_endpoints(_five_task_result())

    assert endpoints["last_task_acquisition"] == pytest.approx(0.82)
    # task2_acquisition keeps its historical meaning: task 2 right after it was
    # trained. For T=2 the two coincide; for T=5 they must not.
    assert endpoints["task2_acquisition"] == pytest.approx(0.88)


def test_parameter_drift_is_read_from_the_final_stage():
    endpoints = condition_endpoints(_five_task_result())

    # stage 4 -> 5.0e-4; stage 1 would give 2.0e-4.
    assert endpoints["protected_rms_drift"] == pytest.approx(5.0e-4)
    assert endpoints["plastic_rms_drift"] == pytest.approx(1.5e-3)


def test_two_task_endpoints_are_unchanged():
    """The published BERT numbers came from this path; it must not move."""

    matrix = [[0.8, None], [0.6, 0.9]]
    endpoints = condition_endpoints(_result(matrix, drift_stages=2))

    assert endpoints["task_count"] == 2
    assert endpoints["task1_acquisition"] == pytest.approx(0.8)
    assert endpoints["task1_retention"] == pytest.approx(0.6)
    assert endpoints["task1_forgetting"] == pytest.approx(0.2)
    assert endpoints["task2_acquisition"] == pytest.approx(0.9)
    assert endpoints["last_task_acquisition"] == pytest.approx(0.9)
    assert endpoints["protected_rms_drift"] == pytest.approx(2.0e-4)


def test_endpoints_reject_a_single_task_sequence():
    """A one-task run has no retention to report; silence would be worse."""

    with pytest.raises(ValueError):
        condition_endpoints(_result([[0.9]], drift_stages=1))


def test_endpoints_reject_a_ragged_accuracy_matrix():
    matrix = [[0.9, None, None], [0.7, 0.8, None]]

    with pytest.raises(ValueError):
        condition_endpoints(_result(matrix, drift_stages=2))


# ── run_diagnostic must honour the configured sequence length ──────────


def _patch_runner(monkeypatch, sink):
    def fake_run(config, tasks, **kwargs):
        sink.append(config)
        return {config.methods[0]: _five_task_result()}

    monkeypatch.setattr(diagnostic, "run_split_clinc150", fake_run)
    monkeypatch.setattr(diagnostic, "write_environment_manifest", lambda *a, **k: None)


def test_run_diagnostic_propagates_an_explicit_task_limit(monkeypatch, tmp_path):
    seen: list = []
    _patch_runner(monkeypatch, seen)

    diagnostic.run_diagnostic(
        SplitCLINC150Config(device="cpu"),
        tasks=[object()] * 10,
        metadata={},
        seeds=[4_000_003],
        output_dir=tmp_path,
        conditions=diagnostic.criterion_ablation_conditions(),
        task_limit=5,
    )

    assert seen, "runner não foi chamado"
    assert all(config.task_limit == 5 for config in seen)


def test_run_diagnostic_still_defaults_to_two_tasks(monkeypatch, tmp_path):
    """Default stays at the published two-task diagnostic."""

    seen: list = []

    def fake_run(config, tasks, **kwargs):
        seen.append(config)
        return {config.methods[0]: _result([[0.8, None], [0.6, 0.9]], 2)}

    monkeypatch.setattr(diagnostic, "run_split_clinc150", fake_run)
    monkeypatch.setattr(diagnostic, "write_environment_manifest", lambda *a, **k: None)

    diagnostic.run_diagnostic(
        SplitCLINC150Config(device="cpu"),
        tasks=[object()] * 10,
        metadata={},
        seeds=[11],
        output_dir=tmp_path,
    )

    assert all(config.task_limit == 2 for config in seen)


def test_run_diagnostic_rejects_a_task_limit_longer_than_the_stream(
    monkeypatch, tmp_path
):
    """Asking for ten tasks with five loaded must fail loudly, not truncate."""

    seen: list = []
    _patch_runner(monkeypatch, seen)

    with pytest.raises(ValueError):
        diagnostic.run_diagnostic(
            SplitCLINC150Config(device="cpu"),
            tasks=[object()] * 5,
            metadata={},
            seeds=[4_000_003],
            output_dir=tmp_path,
            task_limit=10,
        )


# ── summary and CLI ────────────────────────────────────────────────────


def test_summary_reports_the_real_task_count(monkeypatch, tmp_path):
    names = [c.name for c in diagnostic.criterion_ablation_conditions()]
    raw = {4_000_003: {name: _five_task_result() for name in names}}

    summary = diagnostic.summarize_diagnostic(
        raw, diagnostic.criterion_ablation_conditions()
    )

    assert summary["task_count"] == 5


def test_summary_refuses_to_mix_sequence_lengths_across_arms():
    """Two arms with different T are not comparable, paired or otherwise."""

    conditions = diagnostic.criterion_ablation_conditions()
    names = [c.name for c in conditions]
    raw = {
        4_000_003: {
            name: (
                _five_task_result()
                if name != "vanilla"
                else _result([[0.8, None], [0.6, 0.9]], 2)
            )
            for name in names
        }
    }

    with pytest.raises(ValueError):
        diagnostic.summarize_diagnostic(raw, conditions)


def test_cli_accepts_the_full_declared_range_of_task_limits():
    parser = diagnostic.build_parser()

    assert parser.parse_args(["--task-limit", "2"]).task_limit == 2
    assert parser.parse_args(["--task-limit", "5"]).task_limit == 5
    assert parser.parse_args(["--task-limit", "10"]).task_limit == 10
    assert parser.parse_args([]).task_limit == 2


def test_cli_rejects_a_task_limit_outside_the_benchmark():
    parser = diagnostic.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--task-limit", "11"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--task-limit", "1"])


def test_main_wires_the_cli_task_limit_into_the_runner(monkeypatch):
    """Wiring guard: a flag that parses but never reaches the runner is worse
    than no flag, because the log would claim five tasks and the run would do
    two. Same class of bug as `importance_criterion` not reaching the config
    (goals/protocol_importance_criterion_ablation.md, Section H.1)."""

    seen: dict = {}

    def fake_run_diagnostic(config, tasks, **kwargs):
        seen["task_limit"] = kwargs.get("task_limit")
        seen["config_task_limit"] = config.task_limit
        return {}

    monkeypatch.setattr(diagnostic, "run_diagnostic", fake_run_diagnostic)
    monkeypatch.setattr(
        diagnostic, "load_clinc150_tasks", lambda config, **k: ([object()] * 10, {})
    )
    monkeypatch.setattr(
        "sys.argv",
        ["bert_slowheat_diagnostic", "--task-limit", "5", "--seeds", "11"],
    )

    diagnostic.main()

    assert seen["task_limit"] == 5, "o --task-limit do CLI não chegou ao runner"
    assert seen["config_task_limit"] == 5


def test_run_diagnostic_rejects_a_degenerate_task_limit(monkeypatch, tmp_path):
    """T < 2 has no retention endpoint; failing late inside the runner would
    waste the whole run before anyone noticed."""

    seen: list = []
    _patch_runner(monkeypatch, seen)

    for bad in (1, 0, -3):
        with pytest.raises(ValueError):
            diagnostic.run_diagnostic(
                SplitCLINC150Config(device="cpu"),
                tasks=[object()] * 10,
                metadata={},
                seeds=[11],
                output_dir=tmp_path,
                task_limit=bad,
            )
    assert not seen, "nenhum braço pode rodar com task_limit inválido"

    with pytest.raises(TypeError):
        diagnostic.run_diagnostic(
            SplitCLINC150Config(device="cpu"),
            tasks=[object()] * 10,
            metadata={},
            seeds=[11],
            output_dir=tmp_path,
            task_limit=True,
        )
