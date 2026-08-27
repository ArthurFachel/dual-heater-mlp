import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.dualheat_pairs import (
    METHOD_PAIRS,
    _holm_adjust,
    pair_protocol,
    paired_config,
    run_dualheat_pairs,
    summarize_pair_results,
)
from experiments.split_mnist import (
    MNISTTask,
    SplitMNISTConfig,
    _classes_for_task,
    _select_class_indices,
    build_paired_models,
    config_payload,
    run_split_mnist,
    run_split_mnist_epoch_sweep,
    run_split_mnist_multi_seed,
)


def test_class_selection_rejects_unavailable_sample_count():
    targets = torch.tensor([0, 0, 1])

    with pytest.raises(ValueError, match="2 exemplos, mas 3 foram solicitados"):
        _select_class_indices(targets, 0, count=3, seed=1)


def _tiny_tasks(config: SplitMNISTConfig) -> list[MNISTTask]:
    generator = torch.Generator().manual_seed(config.seed)
    centers = torch.randn(10, 784, generator=generator)
    tasks = []
    for task_index in range(config.task_count):
        classes = _classes_for_task(config, task_index)

        def split(
            samples: int,
            task_classes: tuple[int, ...] = tuple(classes),
        ) -> tuple[torch.Tensor, torch.Tensor]:
            inputs = []
            targets = []
            for label in task_classes:
                inputs.append(
                    centers[label]
                    + 0.1 * torch.randn(samples, 784, generator=generator)
                )
                targets.append(torch.full((samples,), label, dtype=torch.long))
            return torch.cat(inputs), torch.cat(targets)

        train_x, train_y = split(2)
        validation_x, validation_y = split(1)
        test_x, test_y = split(1)
        tasks.append(
            MNISTTask(
                classes=tuple(classes),
                train_x=train_x,
                train_y=train_y,
                validation_x=validation_x,
                validation_y=validation_y,
                test_x=test_x,
                test_y=test_y,
            )
        )
    return tasks


@pytest.fixture
def dualheat_pair_run(tmp_path, monkeypatch):
    config = replace(
        paired_config(), hidden_dims=(8,), batch_size=4, epochs_per_task=1,
        replay_per_class=1, replay_batch_size=2, bootstrap_resamples=50,
    )
    monkeypatch.setattr(
        "experiments.split_mnist.load_split_mnist",
        lambda current, **_: _tiny_tasks(current),
    )
    output = tmp_path / "pairs"
    report = run_dualheat_pairs(
        config=config, seeds=[2, 3], data_dir=tmp_path, output_dir=output,
        download=False, verbose=False,
    )
    return config, output, report


def test_dualheat_pairs_preserve_base_capabilities_and_initialization():
    from experiments.split_mnist import _method_spec

    config = replace(paired_config(), hidden_dims=(8,))
    models = build_paired_models(config)
    for pair in METHOD_PAIRS:
        base_spec = _method_spec(pair.reference)
        candidate_spec = _method_spec(pair.candidate)
        assert candidate_spec == replace(
            base_spec, slowheat=True, strength=30.0, budget=0.25, protect_output=False,
        )
        reference = dict(models[pair.reference].named_parameters())
        candidate = dict(models[pair.candidate].named_parameters())
        assert reference.keys() == candidate.keys()
        assert all(torch.equal(reference[key], candidate[key]) for key in reference)
        assert isinstance(models[pair.candidate][-1], torch.nn.Linear)


def test_dualheat_pair_suite_trains_and_reports_only_matched_contrasts(dualheat_pair_run):
    _, output, report = dualheat_pair_run
    assert len(report["pairs"]) == 4
    assert report["status"] == "exploratory_paired_suite"
    assert report["source_dir"] == "."
    assert report["source_dir_base"] == "report_directory"
    for pair, comparison in zip(METHOD_PAIRS, report["pairs"], strict=True):
        assert (comparison["reference"], comparison["candidate"]) == (
            pair.reference, pair.candidate,
        )
        differences = []
        for seed in report["seeds"]:
            raw = json.loads((output / f"seed_{seed}/results.json").read_text())
            differences.append(
                raw[pair.candidate]["metrics"]["final_average_accuracy"]
                - raw[pair.reference]["metrics"]["final_average_accuracy"]
            )
            assert raw[pair.candidate]["training_losses"][0] == pytest.approx(
                raw[pair.reference]["training_losses"][0]
            )
        accuracy = comparison["metrics"]["final_average_accuracy"]
        assert accuracy["mean_difference"] == pytest.approx(np.mean(differences))
        assert accuracy["n_pairs"] == 2
        assert accuracy["holm_adjusted_p"] >= accuracy["student_t"]["two_sided_p"]
    for name in ("pair_report.md", "pair_report.json", "pair_summary.csv", "pair_differences.csv"):
        assert (output / name).is_file()


def test_dualheat_report_still_locates_raw_results_after_moving_tree(dualheat_pair_run):
    _, source, original = dualheat_pair_run
    tree = source.parent / "portable tree"
    shutil.copytree(source, tree / "raw")
    report_dir = tree / "reports"
    report = summarize_pair_results(tree / "raw", output_dir=report_dir)
    assert report["source_dir"] == "../raw"
    assert report["pairs"] == original["pairs"]
    for name in ("pair_report.json", "pair_report.md"):
        assert str(source.parent) not in (report_dir / name).read_text()

    moved = source.parent / "another machine"
    shutil.move(str(tree), moved)
    saved = json.loads((moved / "reports/pair_report.json").read_text())
    relocated_source = moved / "reports" / Path(saved["source_dir"])
    assert (relocated_source / "seed_2/results.json").read_bytes() == (
        source / "seed_2/results.json"
    ).read_bytes()


def test_dualheat_pair_resume_reuses_seeds_and_rejects_changed_protocol(
    dualheat_pair_run, monkeypatch,
):
    config, output, first = dualheat_pair_run

    def unexpected_loader(*args, **kwargs):
        raise AssertionError("não deve retreinar seeds completas")

    monkeypatch.setattr("experiments.split_mnist.load_split_mnist", unexpected_loader)
    resumed = run_dualheat_pairs(
        config=config, seeds=[2, 3], data_dir=output, output_dir=output,
    )
    assert resumed["pairs"] == first["pairs"]
    with pytest.raises(ValueError, match="protocolo diferente"):
        run_dualheat_pairs(
            config=replace(config, epochs_per_task=2), seeds=[2, 3],
            data_dir=output, output_dir=output,
        )


@pytest.mark.parametrize("problem", ["missing_seed", "missing_pair", "config", "resources"])
def test_dualheat_reanalysis_rejects_incomplete_or_unfair_pairs(dualheat_pair_run, problem):
    _, output, _ = dualheat_pair_run
    result_path = output / "seed_2/results.json"
    if problem == "missing_seed":
        result_path.rename(output / "removed_result.json")
    elif problem == "config":
        path = output / "seed_2/config.json"
        config = json.loads(path.read_text())
        config["learning_rate"] *= 2
        path.write_text(json.dumps(config))
    else:
        raw = json.loads(result_path.read_text())
        if problem == "missing_pair":
            raw.pop(METHOD_PAIRS[0].candidate)
        else:
            raw[METHOD_PAIRS[0].candidate]["cost"]["learner_examples_processed"] += 1
        result_path.write_text(json.dumps(raw))
    destination = output / "invalid_report"
    with pytest.raises((ValueError, FileNotFoundError)):
        summarize_pair_results(output, output_dir=destination)
    assert not destination.exists()


def test_dualheat_single_seed_reanalysis_is_descriptive_and_read_only(dualheat_pair_run):
    _, output, _ = dualheat_pair_run
    manifest_path = output / "multi_seed_config.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["seeds"] = [2]
    manifest_path.write_text(json.dumps(manifest))
    source_before = (output / "seed_2/results.json").read_bytes()
    report = summarize_pair_results(output, output_dir=output / "one_seed")
    for comparison in report["pairs"]:
        accuracy = comparison["metrics"]["final_average_accuracy"]
        assert accuracy["inference_unavailable"] == "requires_at_least_two_paired_seeds"
        assert "student_t" not in accuracy
        assert "holm_adjusted_p" not in accuracy
    assert (output / "seed_2/results.json").read_bytes() == source_before


def test_dualheat_protocol_rejects_duplicate_and_reserved_seeds():
    from experiments.confirmatory_split_mnist import CONFIRMATORY_SEEDS

    with pytest.raises(ValueError, match="duplicatas"):
        pair_protocol(paired_config(), [2, 2])
    with pytest.raises(ValueError, match="reservadas"):
        pair_protocol(paired_config(), [CONFIRMATORY_SEEDS[0]])


def test_dualheat_holm_adjustment_preserves_order_and_monotonicity():
    assert _holm_adjust([0.04, 0.001, 0.03, 0.2]) == pytest.approx([0.09, 0.004, 0.09, 0.2])


def test_split_mnist_models_have_paired_trainable_initialization():
    config = SplitMNISTConfig(
        hidden_dims=(8,),
        methods=(
            "vanilla",
            "slowheat",
            "slowheat_replay_hidden_beta_3_budget_0.50",
            "slowheat_er_ace_hidden_beta_30_budget_0.25",
        ),
    )

    models = build_paired_models(config)

    vanilla = dict(models["vanilla"].named_parameters())
    slowheat = dict(models["slowheat"].named_parameters())
    assert vanilla.keys() == slowheat.keys()
    assert all(torch.equal(vanilla[name], slowheat[name]) for name in vanilla)
    hidden_only = models["slowheat_replay_hidden_beta_3_budget_0.50"]
    assert isinstance(hidden_only[-1], torch.nn.Linear)
    assert all(
        layer.plasticity_budget == 0.5 for layer in hidden_only.get_slow_layers()
    )
    slowheat_er_ace = models["slowheat_er_ace_hidden_beta_30_budget_0.25"]
    assert isinstance(slowheat_er_ace[-1], torch.nn.Linear)
    assert all(
        layer.slow_strength == 30.0
        and layer.plasticity_budget == 0.25
        for layer in slowheat_er_ace.get_slow_layers()
    )


def test_tiny_split_mnist_run_produces_curves_and_artifacts(tmp_path):
    config = SplitMNISTConfig(
        seed=3,
        hidden_dims=(8,),
        batch_size=4,
        epochs_per_task=1,
        replay_per_class=1,
        replay_batch_size=2,
        methods=(
            "vanilla",
            "slowheat_beta_10",
            "hard_freeze",
            "replay",
            "distillation",
        ),
    )

    results = run_split_mnist(config, _tiny_tasks(config), output_dir=tmp_path)

    assert results.keys() == {
        "vanilla",
        "slowheat_beta_10",
        "hard_freeze",
        "replay",
        "distillation",
    }
    for result in results.values():
        matrix = np.asarray(
            [
                [np.nan if value is None else value for value in row]
                for row in result["accuracy_matrix"]
            ]
        )
        assert matrix.shape == (5, 5)
        task_aware_matrix = np.asarray(
            [
                [np.nan if value is None else value for value in row]
                for row in result["task_aware_accuracy_matrix"]
            ]
        )
        assert task_aware_matrix.shape == (5, 5)
        assert np.isfinite(task_aware_matrix[np.tril_indices(5)]).all()
        assert len(result["stage_average_accuracy"]) == 5
        assert len(result["stage_average_forgetting"]) == 5
        assert np.isfinite(result["classifier_gap"])
    assert len(results["hard_freeze"]["capacity_history"]) == 5
    assert (tmp_path / "summary.csv").is_file()
    saved = json.loads((tmp_path / "results.json").read_text())
    assert saved.keys() == results.keys()


def test_requested_baselines_share_one_runner_and_report_costs(tmp_path):
    methods = (
        "replay",
        "derpp",
        "slowheat_derpp_hidden_beta_30_budget_0.25",
        "er_ace",
        "slowheat_er_ace_hidden_beta_30_budget_0.25",
        "agem",
        "ewc",
        "si",
        "lwf_calibrated",
        "replay_balanced",
        "slowheat_replay_hidden_beta_30_budget_0.25",
    )
    config = SplitMNISTConfig(
        seed=7,
        hidden_dims=(8,),
        batch_size=4,
        epochs_per_task=1,
        replay_per_class=1,
        replay_batch_size=2,
        methods=methods,
    )

    results = run_split_mnist(config, _tiny_tasks(config), output_dir=tmp_path)

    assert tuple(results) == methods
    for result in results.values():
        assert result["cost"]["learner_examples_processed"] > 0
        assert result["cost"]["estimated_total_flops"] > 0
        assert result["cost"]["optimizer_steps"] > 0
    slowheat_er_ace = results["slowheat_er_ace_hidden_beta_30_budget_0.25"]
    assert len(slowheat_er_ace["capacity_history"]) == config.task_count
    assert slowheat_er_ace["cost"]["replay_memory_bytes"] > 0
    assert slowheat_er_ace["cost"]["stored_logits_bytes"] == 0


def test_noncanonical_class_order_masks_unseen_global_logits():
    config = SplitMNISTConfig(
        seed=9,
        class_order=(8, 9, 6, 7, 4, 5, 2, 3, 0, 1),
        hidden_dims=(8,),
        batch_size=4,
        epochs_per_task=1,
        methods=("vanilla",),
    )

    result = run_split_mnist(config, _tiny_tasks(config))["vanilla"]

    assert np.asarray(result["accuracy_matrix"], dtype=object).shape == (5, 5)
    assert np.isfinite(result["metrics"]["final_average_accuracy"])


def test_domain_incremental_config_reuses_all_classes_per_task():
    config = SplitMNISTConfig(
        class_order=tuple(range(10)),
        classes_per_task=10,
        scenario="domain_incremental",
        domain_task_count=5,
    )

    config.validate()

    assert config.task_count == 5


def test_class_incremental_config_supports_nonuniform_task_sizes():
    config = SplitMNISTConfig(
        class_order=tuple(range(10)),
        classes_per_task=2,
        task_class_counts=(4, 3, 3),
    )

    config.validate()

    assert config.task_count == 3
    assert _classes_for_task(config, 0) == (0, 1, 2, 3)
    assert _classes_for_task(config, 1) == (4, 5, 6)
    assert _classes_for_task(config, 2) == (7, 8, 9)


def test_config_payload_preserves_legacy_protocols_and_nonuniform_counts():
    assert "task_class_counts" not in config_payload(SplitMNISTConfig())
    assert config_payload(
        SplitMNISTConfig(
            class_order=tuple(range(10)),
            classes_per_task=2,
            task_class_counts=(4, 3, 3),
        )
    )["task_class_counts"] == (4, 3, 3)


def test_multi_seed_runner_aggregates_means_and_paired_differences(
    tmp_path, monkeypatch
):
    config = SplitMNISTConfig(
        seed=2,
        hidden_dims=(8,),
        batch_size=4,
        epochs_per_task=1,
        methods=("vanilla", "slowheat_beta_100"),
    )

    monkeypatch.setattr(
        "experiments.split_mnist.load_split_mnist",
        lambda current, **_: _tiny_tasks(current),
    )
    aggregate = run_split_mnist_multi_seed(
        config,
        seeds=[2, 3],
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "results",
    )

    assert aggregate["seeds"] == [2, 3]
    assert aggregate["methods"].keys() == {"vanilla", "slowheat_beta_100"}
    summary = aggregate["methods"]["slowheat_beta_100"]["final_average_accuracy"]
    assert summary.keys() == {"mean", "std", "ci95_normal_half_width"}
    assert np.isfinite(list(summary.values())).all()
    assert "slowheat_beta_100" in aggregate["paired_differences_vs_vanilla"]
    assert (tmp_path / "results" / "aggregate.csv").is_file()
    assert (tmp_path / "results" / "aggregate.json").is_file()


def test_multi_seed_runner_allows_single_seed_as_descriptive_smoke_test(
    tmp_path, monkeypatch
):
    config = SplitMNISTConfig(
        seed=2,
        hidden_dims=(8,),
        batch_size=4,
        epochs_per_task=1,
        methods=("vanilla", "slowheat_beta_100"),
    )
    monkeypatch.setattr(
        "experiments.split_mnist.load_split_mnist",
        lambda current, **_: _tiny_tasks(current),
    )

    aggregate = run_split_mnist_multi_seed(
        config,
        seeds=[2],
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "results",
    )

    paired = aggregate["paired_differences_vs_vanilla"]["slowheat_beta_100"]
    assert paired["final_average_accuracy"]["confirmatory"] == {
        "available": False,
        "n_pairs": 1,
        "reason": "requires_at_least_two_paired_seeds",
    }
    assert paired["final_average_accuracy"]["std"] == 0.0


def test_multi_seed_resume_reuses_completed_matching_seeds(tmp_path, monkeypatch):
    config = SplitMNISTConfig(
        seed=2,
        hidden_dims=(8,),
        batch_size=4,
        epochs_per_task=1,
        methods=("vanilla", "replay"),
    )
    monkeypatch.setattr(
        "experiments.split_mnist.load_split_mnist",
        lambda current, **_: _tiny_tasks(current),
    )
    output_dir = tmp_path / "resume"
    first = run_split_mnist_multi_seed(
        config,
        seeds=[2, 3],
        data_dir=tmp_path / "data",
        output_dir=output_dir,
    )

    # Simulate artifacts written before replay/logit memory accounting existed.
    for seed in (2, 3):
        results_path = output_dir / f"seed_{seed}" / "results.json"
        legacy_results = json.loads(results_path.read_text(encoding="utf-8"))
        for result in legacy_results.values():
            result["cost"].pop("replay_memory_bytes", None)
            result["cost"].pop("stored_logits_bytes", None)
        results_path.write_text(
            json.dumps(legacy_results),
            encoding="utf-8",
        )

    def unexpected_loader(*args, **kwargs):
        raise AssertionError("loader não deve ser chamado para seeds completas")

    monkeypatch.setattr(
        "experiments.split_mnist.load_split_mnist", unexpected_loader
    )
    resumed = run_split_mnist_multi_seed(
        config,
        seeds=[2, 3],
        data_dir=tmp_path / "data",
        output_dir=output_dir,
        resume=True,
    )

    assert resumed["methods"] == first["methods"]
    assert resumed["methods"]["vanilla"]["replay_memory_bytes"]["mean"] == 0
    assert resumed["methods"]["replay"]["replay_memory_bytes"]["mean"] > 0
    assert (output_dir / "run_identity.json").is_file()
    assert list(output_dir.glob("seed_*/data_identity.json"))


def test_epoch_sweep_writes_long_form_metrics_and_replay_comparisons(
    tmp_path, monkeypatch
):
    config = SplitMNISTConfig(
        seed=2,
        hidden_dims=(8,),
        batch_size=4,
        replay_per_class=1,
        replay_batch_size=2,
        methods=("replay", "slowheat_replay_hidden_beta_3_budget_0.50"),
    )
    monkeypatch.setattr(
        "experiments.split_mnist.load_split_mnist",
        lambda current, **_: _tiny_tasks(current),
    )

    sweep = run_split_mnist_epoch_sweep(
        config,
        epochs=[1, 2],
        seeds=[2, 3],
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "sweep",
    )

    assert sweep["epochs"] == [1, 2]
    assert sweep["results"].keys() == {"1", "2"}
    for result in sweep["results"].values():
        assert "paired_differences_vs_replay" in result
    assert (tmp_path / "sweep" / "epoch_sweep.csv").is_file()
    assert (tmp_path / "sweep" / "epoch_sweep.json").is_file()


@pytest.mark.parametrize(
    "updates",
    [
        {"class_order": (0, 1, 2)},
        {"task_class_counts": (5, 4)},
        {"plasticity_budget": 1.1},
        {"methods": ("unknown",)},
        {"methods": ("slowheat_replay_hidden_beta_3_budget_1.20",)},
    ],
)
def test_split_mnist_config_rejects_invalid_protocol(updates):
    values = {**SplitMNISTConfig().__dict__, **updates}
    config = SplitMNISTConfig(**values)

    with pytest.raises(ValueError):
        config.validate()
