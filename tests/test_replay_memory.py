from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import nn

import experiments.replay_selection_sweep as sweep
import experiments.split_mnist as split
from experiments.contracts import ContinualTask
from experiments.replay_memory import ReplayBuffer, select_task_exemplars
from experiments.split_mnist import SplitMNISTConfig, config_payload, run_split_mnist


class IdentityFeatureModel(nn.Module):
    def __init__(self, dimensions: int = 2) -> None:
        super().__init__()
        self.classifier = nn.Linear(dimensions, dimensions, bias=False)
        with torch.no_grad():
            self.classifier.weight.copy_(torch.eye(dimensions))

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(inputs))


def _task(inputs: torch.Tensor, targets: torch.Tensor) -> ContinualTask:
    classes = tuple(int(value) for value in torch.unique(targets, sorted=True))
    return ContinualTask(
        classes=classes,
        train_x=inputs.to(torch.float32),
        train_y=targets.to(torch.long),
        validation_x=inputs[:1].to(torch.float32),
        validation_y=targets[:1].to(torch.long),
        test_x=inputs[:1].to(torch.float32),
        test_y=targets[:1].to(torch.long),
    )


def test_loss_selector_keeps_highest_loss_per_class():
    task = _task(
        torch.tensor([[4.0, 0.0], [0.0, 4.0], [0.0, 4.0], [4.0, 0.0]]),
        torch.tensor([0, 0, 1, 1]),
    )

    selected = select_task_exemplars(
        IdentityFeatureModel(), task, task_index=0, seen_classes=(0, 1),
        samples_per_class=1, strategy="loss", device="cpu",
    )

    assert selected.source_indices.tolist() == [1, 3]
    assert selected.targets.tolist() == [0, 1]
    assert selected.selector_forward_examples == 4


def test_representative_selector_uses_class_herding():
    inputs = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [-1.0, -1.0]]
    )
    task = _task(inputs, torch.tensor([0, 0, 0, 1, 1, 1]))

    selected = select_task_exemplars(
        IdentityFeatureModel(), task, task_index=2, seen_classes=(0, 1),
        samples_per_class=1, strategy="representative", device="cpu",
    )

    assert selected.source_indices.tolist() == [2, 5]
    assert selected.selector_distance_flops > 0


def test_hybrid_selector_is_deterministic_diverse_and_handles_zero_embeddings():
    inputs = torch.tensor(
        [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]
    )
    targets = torch.tensor([0, 0, 0, 0, 1, 1])
    task = _task(inputs, targets)
    kwargs = {
        "task_index": 0,
        "seen_classes": (0, 1),
        "samples_per_class": 2,
        "strategy": "hybrid",
        "device": "cpu",
    }

    first = select_task_exemplars(IdentityFeatureModel(), task, **kwargs)
    second = select_task_exemplars(IdentityFeatureModel(), task, **kwargs)

    assert torch.equal(first.source_indices, second.source_indices)
    assert len(set(first.source_indices[:2].tolist())) == 2
    assert torch.isfinite(first.scores).all()
    assert ((first.score_components >= 0.0) & (first.score_components <= 1.0)).all()


def test_first_selector_preserves_order_and_does_not_score_the_full_task():
    task = _task(torch.randn(6, 2), torch.tensor([0, 0, 0, 1, 1, 1]))

    selected = select_task_exemplars(
        IdentityFeatureModel(), task, task_index=0, seen_classes=(0, 1),
        samples_per_class=2, strategy="first", device="cpu",
    )

    assert selected.source_indices.tolist() == [0, 1, 3, 4]
    assert selected.selector_forward_examples == 0


def test_replay_buffer_round_trip_preserves_logits_metadata_and_domain_groups():
    model = IdentityFeatureModel()
    buffer = ReplayBuffer()
    for task_index in range(2):
        task = _task(
            torch.tensor([[2.0, 0.0], [1.0, 0.0], [0.0, 2.0], [0.0, 1.0]]),
            torch.tensor([0, 0, 1, 1]),
        )
        buffer.append(
            select_task_exemplars(
                model, task, task_index=task_index, seen_classes=(0, 1),
                samples_per_class=1, strategy="first", device="cpu",
                store_logits=True,
            )
        )

    restored = ReplayBuffer.from_state_dict(buffer.state_dict())

    assert len(restored) == 4
    assert restored.source_tasks.tolist() == [0, 0, 1, 1]
    assert torch.equal(restored.inputs, buffer.inputs)
    assert torch.equal(restored.logits, buffer.logits)
    assert restored.selection_history == buffer.selection_history
    assert restored.replay_memory_bytes > 0
    assert restored.stored_logits_bytes > 0


def _continual_tasks(config: SplitMNISTConfig) -> list[ContinualTask]:
    generator = torch.Generator().manual_seed(123)
    tasks = []
    for task_index in range(config.task_count):
        classes = config.class_order[task_index * 2 : task_index * 2 + 2]
        inputs = torch.randn(8, config.input_dim, generator=generator)
        targets = torch.tensor([classes[index % 2] for index in range(8)])
        tasks.append(
            ContinualTask(
                classes=classes,
                train_x=inputs,
                train_y=targets,
                validation_x=inputs[:4],
                validation_y=targets[:4],
                test_x=inputs[:4],
                test_y=targets[:4],
            )
        )
    return tasks


def test_ranked_replay_runs_with_mlp_and_cnn():
    mlp = SplitMNISTConfig(
        seed=4, class_order=(0, 1, 2, 3), classes_per_task=2,
        input_dim=4, hidden_dims=(4,), batch_size=4, epochs_per_task=1,
        replay_per_class=1, replay_batch_size=2, replay_selection="hybrid",
        methods=(
            "replay",
            "slowheat_replay_hidden_beta_30_budget_0.25",
            "derpp",
            "slowheat_derpp_hidden_beta_30_budget_0.25",
        ),
    )
    cnn = replace(
        mlp, input_dim=16, hidden_dims=(1,), backbone="cnn",
        image_shape=(1, 4, 4), cnn_channels=(2, 2), cnn_pooled_size=(1, 1),
        replay_selection="representative",
    )
    cnn_tasks = [
        replace(task, train_x=task.train_x.reshape(-1, 1, 4, 4),
                validation_x=task.validation_x.reshape(-1, 1, 4, 4),
                test_x=task.test_x.reshape(-1, 1, 4, 4))
        for task in _continual_tasks(cnn)
    ]

    mlp_results = run_split_mnist(mlp, _continual_tasks(mlp))
    cnn_results = run_split_mnist(cnn, cnn_tasks)

    for results in (mlp_results, cnn_results):
        assert tuple(results) == mlp.methods
        assert all(len(result["selection_history"]) == 2 for result in results.values())
        assert all(result["cost"]["selector_forward_examples"] > 0 for result in results.values())
        assert results["derpp"]["cost"]["stored_logits_bytes"] > 0
        assert (
            results["slowheat_derpp_hidden_beta_30_budget_0.25"]["cost"]
            ["stored_logits_bytes"]
            > 0
        )


def test_default_config_payload_omits_replay_selection():
    assert "replay_selection" not in config_payload(SplitMNISTConfig())
    assert config_payload(
        replace(SplitMNISTConfig(), replay_selection="loss")
    )["replay_selection"] == "loss"


def test_no_memory_methods_ignore_selector_and_report_zero_replay_cost():
    config = SplitMNISTConfig(
        seed=5, class_order=(0, 1, 2, 3), classes_per_task=2,
        input_dim=4, hidden_dims=(4,), batch_size=4, epochs_per_task=1,
        replay_selection="hybrid",
        methods=("vanilla", "slowheat_hidden_beta_30_budget_0.25"),
    )

    results = run_split_mnist(config, _continual_tasks(config))

    for result in results.values():
        assert result["selection_history"] == []
        assert result["cost"]["replay_memory_bytes"] == 0
        assert result["cost"]["selector_forward_examples"] == 0


@pytest.mark.parametrize(
    "method",
    ("replay", "derpp", "slowheat_derpp_hidden_beta_30_budget_0.25"),
)
def test_task_boundary_checkpoint_resume_matches_continuous_run(
    tmp_path, monkeypatch, method
):
    config = SplitMNISTConfig(
        seed=8, class_order=(0, 1, 2, 3), classes_per_task=2,
        input_dim=4, hidden_dims=(5,), batch_size=4, epochs_per_task=1,
        replay_per_class=1, replay_batch_size=2, replay_selection="hybrid",
        methods=(method,),
    )
    tasks = _continual_tasks(config)
    continuous = run_split_mnist(config, tasks, output_dir=tmp_path / "continuous")
    original = split.write_torch_atomic
    interrupted = False

    def interrupt_after_first_stage(path, payload):
        nonlocal interrupted
        original(path, payload)
        if payload["next_stage"] == 1 and not interrupted:
            interrupted = True
            raise RuntimeError("simulated interruption")

    monkeypatch.setattr(split, "write_torch_atomic", interrupt_after_first_stage)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_split_mnist(config, tasks, output_dir=tmp_path / "resumed")
    monkeypatch.setattr(split, "write_torch_atomic", original)

    resumed = run_split_mnist(
        config, tasks, output_dir=tmp_path / "resumed", resume=True
    )

    for key in (
        "accuracy_matrix",
        "task_aware_accuracy_matrix",
        "training_losses",
        "selection_history",
        "metrics",
    ):
        assert resumed[method][key] == continuous[method][key]
    assert (tmp_path / f"resumed/checkpoints/{method}.pt").is_file()
    if "derpp" in method:
        assert resumed[method]["cost"]["stored_logits_bytes"] > 0


def test_checkpoint_rejects_changed_selection(tmp_path):
    config = SplitMNISTConfig(
        seed=9, class_order=(0, 1), classes_per_task=2, input_dim=2,
        hidden_dims=(2,), batch_size=2, epochs_per_task=1,
        replay_per_class=1, replay_batch_size=1, methods=("replay",),
    )
    tasks = [_task(torch.randn(4, 2), torch.tensor([0, 0, 1, 1]))]
    run_split_mnist(config, tasks, output_dir=tmp_path)

    with pytest.raises(RuntimeError, match="checkpoint incompatível"):
        run_split_mnist(
            replace(config, replay_selection="loss"), tasks,
            output_dir=tmp_path, resume=True,
        )


def test_replay_selection_sweep_writes_paired_artifacts(tmp_path, monkeypatch):
    base = SplitMNISTConfig(
        seed=1, class_order=(0, 1, 2, 3), classes_per_task=2,
        input_dim=4, hidden_dims=(4,), batch_size=4, epochs_per_task=1,
        replay_per_class=1, replay_batch_size=2, bootstrap_resamples=20,
        methods=sweep.SWEEP_METHODS,
    )
    monkeypatch.setattr(
        sweep, "replay_selection_configs", lambda device="cpu": {"split_mnist": base}
    )
    monkeypatch.setattr(
        split,
        "load_split_mnist",
        lambda config, **_: _continual_tasks(config),
    )

    report = sweep.run_replay_selection_sweep(
        seeds=[2, 3], data_dir=tmp_path / "data", output_dir=tmp_path / "sweep",
        download=False, verbose=False, datasets=("split_mnist",),
    )

    assert report["status"] == "exploratory_not_independent_confirmation"
    comparisons = report["paired_differences_vs_first"]["split_mnist"]["replay"]
    assert set(comparisons) == {"loss", "representative", "hybrid"}
    assert all(
        "holm_adjusted_p" in comparison["final_average_accuracy"]
        for comparison in comparisons.values()
    )
    assert set(sweep.SWEEP_MEMORY_METHODS).issubset(
        report["paired_differences_vs_first"]["split_mnist"]
    )
    assert set(report["slowheat_vs_derpp"]["split_mnist"]) == set(
        sweep.REPLAY_SELECTION_STRATEGIES
    )
    assert set(report["memory_vs_no_memory"]["split_mnist"]) == set(
        sweep.SWEEP_MEMORY_METHODS
    )
    assert report["no_memory_references"] == sweep.NO_MEMORY_REFERENCES
    for name in (
        "sweep_index.json", "sweep_report.json", "sweep_summary.csv",
        "sweep_differences.csv",
    ):
        assert (tmp_path / "sweep" / name).is_file()


def test_replay_selection_sweep_rejects_confirmatory_seeds(tmp_path):
    from experiments.confirmatory_split_mnist import CONFIRMATORY_SEEDS

    with pytest.raises(ValueError, match="confirmatórias são reservadas"):
        sweep.run_replay_selection_sweep(
            seeds=[CONFIRMATORY_SEEDS[0]],
            data_dir=tmp_path / "data",
            output_dir=tmp_path / "sweep",
            download=False,
            verbose=False,
            datasets=("split_mnist",),
        )


def test_one_seed_smoke_covers_every_visual_stream(tmp_path, monkeypatch):
    mlp = SplitMNISTConfig(
        seed=1,
        class_order=(0, 1, 2, 3),
        classes_per_task=2,
        input_dim=4,
        hidden_dims=(4,),
        batch_size=4,
        epochs_per_task=1,
        replay_per_class=1,
        replay_batch_size=2,
        bootstrap_resamples=10,
        methods=sweep.SWEEP_METHODS,
    )
    cnn = replace(
        mlp,
        input_dim=16,
        hidden_dims=(1,),
        backbone="cnn",
        image_shape=(1, 4, 4),
        cnn_channels=(2, 2),
        cnn_pooled_size=(1, 1),
    )
    configs = {
        name: (cnn if name == "split_cifar10_cnn" else mlp)
        for name in sweep.SWEEP_DATASETS
    }

    def tiny_loader(config, **_):
        tasks = _continual_tasks(config)
        if config.backbone == "cnn":
            return [
                replace(
                    task,
                    train_x=task.train_x.reshape(-1, 1, 4, 4),
                    validation_x=task.validation_x.reshape(-1, 1, 4, 4),
                    test_x=task.test_x.reshape(-1, 1, 4, 4),
                )
                for task in tasks
            ]
        return tasks

    monkeypatch.setattr(sweep, "replay_selection_configs", lambda device="cpu": configs)
    monkeypatch.setattr(
        sweep,
        "_loaders",
        lambda: {name: tiny_loader for name in sweep.SWEEP_DATASETS},
    )

    report = sweep.run_replay_selection_sweep(
        seeds=[17],
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "all-streams",
        download=False,
        verbose=False,
    )

    assert tuple(report["datasets"]) == sweep.SWEEP_DATASETS
    assert set(report["memory_vs_no_memory"]) == set(sweep.SWEEP_DATASETS)
    for dataset in sweep.SWEEP_DATASETS:
        assert (
            tmp_path / "all-streams" / dataset / "hybrid" / "seed_17/results.json"
        ).is_file()
