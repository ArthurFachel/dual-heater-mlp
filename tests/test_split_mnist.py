import json

import numpy as np
import pytest
import torch

from experiments.split_mnist import (
    MNISTTask,
    SplitMNISTConfig,
    _classes_for_task,
    build_paired_models,
    config_payload,
    run_split_mnist,
    run_split_mnist_epoch_sweep,
    run_split_mnist_multi_seed,
)


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
