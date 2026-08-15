import json

import numpy as np
import pytest
import torch

from experiments.synthetic_cl import (
    SyntheticConfig,
    build_paired_models,
    make_batch_schedule,
    run_experiment,
)


def test_paired_models_start_with_identical_trainable_parameters():
    config = SyntheticConfig(
        n_features=4,
        classes_per_task=2,
        task_count=2,
        hidden_dims=(6, 5),
        methods=("vanilla", "slowheat_max"),
    )

    models = build_paired_models(config)
    vanilla_parameters = dict(models["vanilla"].named_parameters())
    slowheat_parameters = dict(models["slowheat_max"].named_parameters())

    assert vanilla_parameters.keys() == slowheat_parameters.keys()
    for name, parameter in vanilla_parameters.items():
        assert torch.equal(parameter, slowheat_parameters[name])


def test_batch_schedule_is_deterministic_and_reused_by_method():
    first = make_batch_schedule(
        sample_count=12,
        batch_size=4,
        steps=5,
        seed=17,
    )
    second = make_batch_schedule(
        sample_count=12,
        batch_size=4,
        steps=5,
        seed=17,
    )

    assert len(first) == 5
    assert all(torch.equal(a, b) for a, b in zip(first, second))


def test_future_classifier_rows_receive_no_gradient_before_their_stage():
    config = SyntheticConfig(
        n_features=4,
        classes_per_task=2,
        task_count=2,
        hidden_dims=(6,),
        methods=("slowheat_max",),
    )
    model = build_paired_models(config)["slowheat_max"]
    logits = model(torch.randn(5, 4))
    targets = torch.tensor([0, 1, 0, 1, 0])

    torch.nn.functional.cross_entropy(logits[:, :2], targets).backward()

    output_weight = model[-1].weight
    assert torch.count_nonzero(output_weight.grad[2:]) == 0


def test_tiny_experiment_writes_complete_reproducibility_artifacts(tmp_path):
    config = SyntheticConfig(
        seed=3,
        n_features=4,
        classes_per_task=2,
        task_count=2,
        train_per_class=6,
        test_per_class=4,
        hidden_dims=(6, 5),
        batch_size=4,
        steps_per_task=1,
        methods=("vanilla", "slowheat_max"),
    )

    results = run_experiment(config, output_dir=tmp_path)

    assert results.keys() == {"vanilla", "slowheat_max"}
    for method in results.values():
        matrix = np.asarray(method["accuracy_matrix"], dtype=float)
        assert matrix.shape == (2, 2)
        assert np.isfinite(matrix[np.tril_indices(2)]).all()
        assert len(method["training_losses"]) == 2
        assert method["metrics"]["average_forgetting"] >= 0.0

    saved_config = json.loads((tmp_path / "config.json").read_text())
    saved_results = json.loads((tmp_path / "results.json").read_text())
    assert saved_config["seed"] == 3
    assert saved_results.keys() == results.keys()
    assert (tmp_path / "summary.csv").is_file()


def test_all_optimizer_and_consolidation_ablations_complete_a_cpu_step(tmp_path):
    methods = (
        "vanilla",
        "reduced_lr",
        "slowheat_max",
        "slowheat_mean",
        "slowheat_sum",
        "slowheat_none",
        "slowheat_max_native_state",
        "slowheat_max_unidirectional",
        "slowheat_max_unbudgeted",
        "slowheat_max_sgd",
        "slowheat_max_legacy_adamw",
    )
    config = SyntheticConfig(
        seed=9,
        n_features=3,
        classes_per_task=2,
        task_count=1,
        train_per_class=3,
        test_per_class=2,
        hidden_dims=(4,),
        batch_size=2,
        steps_per_task=1,
        methods=methods,
    )

    results = run_experiment(config, output_dir=tmp_path)

    assert results.keys() == set(methods)


@pytest.mark.parametrize(
    "methods",
    [(), ("vanilla", "vanilla")],
)
def test_config_rejects_empty_or_duplicate_methods(methods):
    config = SyntheticConfig(methods=methods)

    with pytest.raises(ValueError):
        config.validate()


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_config_rejects_non_finite_hyperparameters(value):
    config = SyntheticConfig(learning_rate=value)

    with pytest.raises(ValueError):
        config.validate()
