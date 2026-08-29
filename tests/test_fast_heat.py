from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch
from torch import nn

from dual_heater import (
    FastHeatConfig,
    FastHeatGate,
    FunctionalDualHeatCNN,
    FunctionalDualHeatMLP,
    FunctionalDualHeatResNet18,
    FunctionalDualHeatVGG11,
    SlowHeatAdamW,
)
from experiments.contracts import ContinualTask
from experiments.functional_dualheat import (
    FUNCTIONAL_DUALHEAT_BENCHMARK_METHODS,
    select_fastheat_candidate,
)
from experiments.split_mnist import (
    SplitMNISTConfig,
    _build_optimizer,
    build_paired_models,
    config_payload,
    run_split_mnist,
)


def _expected_update(inputs: torch.Tensor, heat: torch.Tensor, config: FastHeatConfig):
    magnitude = inputs.abs().mean(dim=0)
    normalized = magnitude / magnitude.mean().clamp_min(config.eps)
    return torch.relu(
        config.fast_decay * heat
        + (1.0 - config.fast_decay) * (normalized - config.fast_threshold)
    )


def test_fastheat_linear_equation_uses_previous_state_then_updates():
    config = FastHeatConfig(0.9, 0.5, 0.25, 1e-8)
    gate = FastHeatGate(3, unit_dim=-1, config=config)
    previous = torch.tensor([1.0, 3.0, 2.0])
    gate.fast_heat.copy_(previous)
    inputs = torch.tensor([[1.0, -2.0, 3.0], [3.0, 2.0, 1.0]])

    output = gate(inputs)

    others = torch.tensor([2.5, 1.5, 2.0])
    expected_output = inputs / (1.0 + config.fast_strength * others)
    assert output == pytest.approx(expected_output)
    assert gate.fast_heat == pytest.approx(_expected_update(inputs, previous, config))


@pytest.mark.parametrize(
    ("shape", "unit_dim", "units"),
    [((2, 4, 3), -1, 3), ((2, 3, 4, 5), 1, 3)],
)
def test_fastheat_exact_equation_for_sequence_and_convolution(shape, unit_dim, units):
    config = FastHeatConfig(0.0, 2.0, 0.5, 1e-8)
    gate = FastHeatGate(units, unit_dim=unit_dim, config=config)
    inputs = torch.arange(1, torch.tensor(shape).prod().item() + 1).view(shape).float()
    reduce_dims = tuple(
        index for index in range(len(shape)) if index != unit_dim % len(shape)
    )
    magnitude = inputs.abs().mean(dim=reduce_dims)

    gate(inputs)

    expected = torch.relu(magnitude / magnitude.mean() - 0.5)
    assert gate.fast_heat == pytest.approx(expected)


def test_fastheat_eval_applies_frozen_gate_without_mutation():
    gate = FastHeatGate(
        3,
        unit_dim=-1,
        config=FastHeatConfig(0.9, 2.0, 0.5, 1e-8),
    )
    gate.fast_heat.copy_(torch.tensor([0.0, 1.0, 2.0]))
    before = gate.fast_heat.clone()
    inputs = torch.ones(2, 3)

    gate.eval()
    output = gate(inputs)

    assert not torch.equal(output, inputs)
    assert torch.equal(gate.fast_heat, before)


def test_fastheat_single_unit_zero_activations_and_reset():
    single = FastHeatGate(1, unit_dim=-1)
    inputs = torch.tensor([[2.0], [4.0]])
    assert torch.equal(single(inputs), inputs)
    zero_gate = FastHeatGate(2, unit_dim=-1)
    zero_gate(torch.zeros(3, 2))
    assert torch.equal(zero_gate.fast_heat, torch.zeros(2))
    single.reset_fast_heat()
    assert torch.equal(single.fast_heat, torch.zeros(1))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fast_decay": -0.1},
        {"fast_decay": 1.0},
        {"fast_strength": -1.0},
        {"fast_threshold": -1.0},
        {"eps": 0.0},
        {"fast_strength": float("nan")},
    ],
)
def test_fastheat_rejects_invalid_hyperparameters(kwargs):
    with pytest.raises(ValueError):
        FastHeatConfig(**kwargs)


def test_fastheat_state_dict_round_trip_and_non_parameter_state():
    source = FastHeatGate(3, unit_dim=-1)
    source(torch.tensor([[1.0, 2.0, 4.0]]))
    target = FastHeatGate(3, unit_dim=-1)
    target.load_state_dict(source.state_dict())

    assert torch.equal(target.fast_heat, source.fast_heat)
    assert list(source.parameters()) == []


def test_gamma_zero_fastheat_vanilla_and_dualheat_slowheat_equivalence():
    base = SplitMNISTConfig(
        input_dim=4,
        class_order=(0, 1),
        classes_per_task=2,
        hidden_dims=(5,),
        methods=("vanilla", "fastheat", "slowheat", "dualheat"),
        fast_strength=0.0,
    )
    models = build_paired_models(base)
    inputs = torch.randn(3, 4)
    for model in models.values():
        model.train()

    assert torch.equal(models["vanilla"](inputs), models["fastheat"](inputs))
    assert torch.equal(models["slowheat"](inputs), models["dualheat"](inputs))


def test_gate_counts_and_no_classifier_gate():
    mlp = FunctionalDualHeatMLP(4, 6, 5, 3)
    cnn = FunctionalDualHeatCNN(3, 10, channels=(2, 3))
    vgg = FunctionalDualHeatVGG11(
        3,
        10,
        channels=(2, 3, 4, 4, 5, 5, 6, 6),
    )
    resnet = FunctionalDualHeatResNet18(
        3,
        10,
        stage_channels=(4, 8, 12, 16),
    )

    assert len(mlp.get_fast_states()) == 2
    assert len(cnn.get_fast_states()) == 2
    assert len(vgg.get_fast_states()) == 8
    assert len(resnet.get_fast_states()) == 9
    assert not any(
        isinstance(module, FastHeatGate) for module in cnn.classifier.modules()
    )
    assert not any(
        isinstance(module, FastHeatGate) for module in vgg.classifier.modules()
    )
    assert not any(
        isinstance(module, FastHeatGate) for module in resnet.classifier.modules()
    )
    for stage in resnet.stages:
        for block in stage:
            assert isinstance(block.relu, nn.ReLU)
            assert any(
                isinstance(module, FastHeatGate)
                for module in block.output_activation.modules()
            )


def test_dualheat_and_slowheat_trainable_initialization_is_byte_identical():
    config = SplitMNISTConfig(
        input_dim=4,
        class_order=(0, 1),
        classes_per_task=2,
        hidden_dims=(5,),
        methods=("slowheat", "dualheat"),
    )
    models = build_paired_models(config)
    slow = dict(models["slowheat"].named_parameters())
    dual = dict(models["dualheat"].named_parameters())

    assert slow.keys() == dual.keys()
    assert all(torch.equal(slow[name], dual[name]) for name in slow)


def test_optimizer_and_manifest_contracts_for_fastheat_and_dualheat():
    config = SplitMNISTConfig(
        input_dim=4,
        class_order=(0, 1),
        classes_per_task=2,
        hidden_dims=(5,),
        methods=("fastheat", "dualheat"),
    )
    models = build_paired_models(config)

    fast_optimizer = _build_optimizer("fastheat", models["fastheat"], config)
    dual_optimizer = _build_optimizer("dualheat", models["dualheat"], config)
    payload = config_payload(config)

    assert type(fast_optimizer) is torch.optim.AdamW
    assert isinstance(dual_optimizer, SlowHeatAdamW)
    assert payload["fast_decay"] == config.fast_decay
    assert payload["fast_strength"] == config.fast_strength
    assert payload["fast_threshold"] == config.fast_threshold
    assert payload["fast_eps"] == config.fast_eps
    legacy_payload = config_payload(replace(config, methods=("slowheat",)))
    assert not any(key.startswith("fast_") for key in legacy_payload)


def test_functional_dualheat_reset_does_not_touch_slowheat():
    model = FunctionalDualHeatMLP(4, 5, 2)
    model(torch.randn(3, 4))
    slow_before = [layer.slow_heat.clone() for layer in model.get_slow_layers()]
    assert any(torch.count_nonzero(gate.fast_heat) for gate in model.get_fast_states())

    model.reset_fast_heat()

    assert all(
        torch.count_nonzero(gate.fast_heat) == 0 for gate in model.get_fast_states()
    )
    assert all(
        torch.equal(before, layer.slow_heat)
        for before, layer in zip(slow_before, model.get_slow_layers(), strict=True)
    )


def _one_visual_task() -> ContinualTask:
    generator = torch.Generator().manual_seed(11)
    train_x = torch.randn(4, 3, 32, 32, generator=generator)
    evaluation_x = torch.randn(2, 3, 32, 32, generator=generator)
    return ContinualTask(
        classes=(0, 1),
        train_x=train_x,
        train_y=torch.tensor([0, 0, 1, 1]),
        validation_x=evaluation_x,
        validation_y=torch.tensor([0, 1]),
        test_x=evaluation_x.clone(),
        test_y=torch.tensor([0, 1]),
    )


@pytest.mark.parametrize("architecture", ["vgg11", "resnet18"])
def test_thirteen_method_visual_smoke_has_finite_metrics_and_costs(architecture):
    architecture_kwargs = (
        {"vgg_channels": (2, 2, 2, 2, 2, 2, 2, 2)}
        if architecture == "vgg11"
        else {
            "resnet_stage_channels": (2, 2, 2, 2),
            "resnet_blocks_per_stage": (1, 1, 1, 1),
        }
    )
    config = SplitMNISTConfig(
        class_order=(0, 1),
        classes_per_task=2,
        input_dim=3 * 32 * 32,
        hidden_dims=(1,),
        backbone="cnn",
        image_shape=(3, 32, 32),
        cnn_architecture=architecture,
        cnn_pooled_size=(1, 1),
        batch_size=2,
        epochs_per_task=1,
        train_per_class=2,
        validation_per_class=1,
        test_per_class=1,
        replay_per_class=1,
        replay_batch_size=2,
        lpr_update_frequency=100,
        max_train_examples_per_task=2,
        methods=FUNCTIONAL_DUALHEAT_BENCHMARK_METHODS,
        **architecture_kwargs,
    )

    results = run_split_mnist(config, [_one_visual_task()])

    assert set(results) == set(FUNCTIONAL_DUALHEAT_BENCHMARK_METHODS)
    for result in results.values():
        assert torch.isfinite(torch.tensor(result["metrics"]["final_average_accuracy"]))
        assert math.isfinite(result["cost"]["estimated_total_flops"])
        assert math.isfinite(result["cost"]["method_state_bytes"])
    assert results["vanilla"]["cost"]["method_state_bytes"] == 0
    assert results["fastheat"]["cost"]["fastheat_state_bytes"] > 0
    assert results["dualheat"]["cost"]["fastheat_overhead_flops"] > 0
    assert results["dualheat"]["cost"]["slowheat_state_bytes"] > 0
    for reference, candidate in (
        ("slowheat", "dualheat"),
        ("slowheat_lpr", "dualheat_lpr"),
        ("slowheat_classifier_expander", "dualheat_classifier_expander"),
        ("slowheat_scroll", "dualheat_scroll"),
    ):
        assert (
            results[reference]["completed_epochs"]
            == results[candidate]["completed_epochs"]
        )
        assert (
            results[reference]["cost"]["current_examples"]
            == results[candidate]["cost"]["current_examples"]
        )
        assert (
            results[reference]["cost"]["replay_examples"]
            == results[candidate]["cost"]["replay_examples"]
        )


def _two_mlp_tasks() -> list[ContinualTask]:
    tasks = []
    for label in (0, 1):
        inputs = torch.full((2, 4), float(label + 1))
        targets = torch.full((2,), label, dtype=torch.long)
        tasks.append(
            ContinualTask(
                classes=(label,),
                train_x=inputs,
                train_y=targets,
                validation_x=inputs.clone(),
                validation_y=targets.clone(),
                test_x=inputs.clone(),
                test_y=targets.clone(),
            )
        )
    return tasks


def test_fastheat_stage_resume_and_config_mismatch_rejection(tmp_path):
    config = SplitMNISTConfig(
        class_order=(0, 1),
        classes_per_task=1,
        input_dim=4,
        hidden_dims=(5,),
        batch_size=2,
        epochs_per_task=1,
        train_per_class=2,
        validation_per_class=1,
        test_per_class=1,
        methods=("fastheat",),
    )
    tasks = _two_mlp_tasks()
    first = run_split_mnist(config, tasks, output_dir=tmp_path)
    resumed = run_split_mnist(config, tasks, output_dir=tmp_path, resume=True)
    checkpoint = torch.load(
        tmp_path / "checkpoints" / "fastheat.pt",
        map_location="cpu",
        weights_only=True,
    )

    assert (
        first["fastheat"]["accuracy_matrix"] == resumed["fastheat"]["accuracy_matrix"]
    )
    fast_buffers = [
        value for key, value in checkpoint["model"].items() if key.endswith("fast_heat")
    ]
    assert fast_buffers and any(torch.count_nonzero(value) for value in fast_buffers)
    with pytest.raises(RuntimeError, match="checkpoint incompatível"):
        run_split_mnist(
            replace(config, fast_strength=2.0),
            tasks,
            output_dir=tmp_path,
            resume=True,
        )


def test_pilot_tie_break_prefers_weaker_intervention():
    tied = [
        {
            "fast_decay": decay,
            "fast_strength": strength,
            "fast_threshold": threshold,
            "mean_paired_validation_difference": 0.1,
        }
        for decay in (0.90, 0.97)
        for strength in (0.5, 2.0)
        for threshold in (0.0, 0.5)
    ]

    selected = select_fastheat_candidate(tied)

    assert selected["fast_strength"] == 0.5
    assert selected["fast_decay"] == 0.90
    assert selected["fast_threshold"] == 0.5
