"""Wiring guard: the shadow-step instrument must actually reach the runner.

`goals/protocol_penalty_reevaluation.md` §E. A metric that exists as a function
but is never called records nothing, and the pilot would produce a manifest with
no `plasticity_ratio` in it — indistinguishable from "the mechanism does not
matter". These tests run the real runner on a tiny synthetic problem and read
what it emitted.

Unit tests of the ratio formula live in `test_plasticity_generalization.py` and
CANNOT detect an unwired instrument; that is what this file is for.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))


def _tiny_config(method: str, interval: int):
    """Smallest config that still exercises penalty + consolidation.

    Three tasks: stage 0 has no anchor, so the penalty is inactive there by
    construction and no sample is taken; the later stages are where an
    anchored penalty must show up. `epochs_per_task=2` gives more than one
    sampled step per stage.
    """

    from experiments.split_mnist import SplitMNISTConfig

    return SplitMNISTConfig(
        methods=(method,),
        class_order=tuple(range(6)),
        epochs_per_task=2,
        ewc_lambda=25.0,
        mas_lambda=25.0,
        si_lambda=25.0,
        plasticity_sampling_interval=interval,
    )


def _run(config):
    from test_split_mnist import _tiny_tasks

    from experiments.split_mnist import run_split_mnist

    return run_split_mnist(config, _tiny_tasks(config))


def test_the_penalty_scale_is_one_shared_constant_per_method() -> None:
    """The shadow step must use the SAME factor the training loss uses.

    Two independent copies of `0.5 * ewc_lambda` drift, and then the measured
    plasticity belongs to a penalty that was never applied.
    """

    from experiments.split_mnist import _penalty_scale

    config = _tiny_config("ewc", 1)
    assert _penalty_scale("ewc", config) == pytest.approx(0.5 * config.ewc_lambda)
    assert _penalty_scale("si", config) == pytest.approx(config.si_lambda)
    assert _penalty_scale("mas", config) == pytest.approx(config.mas_lambda)
    assert _penalty_scale("vanilla", config) == 0.0


@pytest.mark.parametrize("method", ["ewc", "si", "mas"])
def test_the_runner_records_plasticity_samples_for_every_penalty_method(
    method: str,
) -> None:
    """§E. The field must be in the artifact, with the declared keys."""

    results = _run(_tiny_config(method, 1))
    samples = results[method]["plasticity_samples"]

    assert samples, f"{method}: nenhuma amostra de plasticidade registrada"
    for sample in samples:
        assert set(sample) >= {
            "plasticity_ratio",
            "norm_ratio",
            "direction_cosine",
            "stage",
            "step",
        }


def test_samples_carry_the_step_index_so_runs_can_be_aligned() -> None:
    """Duas taxas de amostragem têm de ser alinháveis pelo índice do passo.

    A amostragem a 1/k mede os passos cujo índice é múltiplo de k, e 1/50 é
    portanto um SUBCONJUNTO de 1/5 — mas não nas mesmas posições da lista,
    porque no estágio 0 o contador anda sem gravar (ainda não há âncora). Sem
    o índice gravado, comparar por posição alinha amostras de estágios
    diferentes, que foi exatamente o erro que este teste passou a impedir.

    Consequência prática: uma run densa entrega de graça a série esparsa
    pré-registrada, sem emendar a taxa declarada.
    """

    dense = _run(_tiny_config("ewc", 1))["ewc"]["plasticity_samples"]
    sparse = _run(_tiny_config("ewc", 3))["ewc"]["plasticity_samples"]

    assert dense and sparse
    dense_by_step = {s["step"]: s for s in dense}

    for sample in sparse:
        assert sample["step"] % 3 == 0, "amostra fora da grade declarada"
        twin = dense_by_step.get(sample["step"])
        assert twin is not None, f"passo {sample['step']} ausente da run densa"
        assert twin["stage"] == sample["stage"]
        assert twin["norm_ratio"] == pytest.approx(sample["norm_ratio"], abs=1e-12), (
            f"passo {sample['step']}: a mesma medição diverge entre as runs"
        )


@pytest.mark.parametrize("method", ["ewc", "si", "mas"])
def test_the_penalty_visibly_changes_the_measured_step(method: str) -> None:
    """An active penalty must CHANGE the update, or it is not wired.

    Deliberately NOT asserted as `ratio < 1`. Under an additive penalty the
    per-element ratio is not bounded above by 1 (see
    `test_the_ratio_is_unbounded_above_under_an_adaptive_optimizer`), so the
    evidence that the penalty reached the measured step is that the step
    MOVED — in magnitude or in direction — not that it shrank.

    Asserting only "the key exists" would pass against a penalty that never
    entered the loss.
    """

    results = _run(_tiny_config(method, 1))
    samples = results[method]["plasticity_samples"]
    assert samples, f"{method}: nenhuma amostra registrada"

    moved = [
        sample
        for sample in samples
        if sample["plasticity_ratio"] is not None
        and (
            abs(sample["plasticity_ratio"] - 1.0) > 1e-3
            or abs(sample["norm_ratio"] - 1.0) > 1e-6
        )
    ]
    assert moved, (
        f"{method}: nenhuma amostra difere do passo não penalizado — a "
        f"penalidade não chegou ao passo medido"
    )


@pytest.mark.parametrize("method", ["ewc", "si", "mas"])
def test_the_first_step_after_an_anchor_has_ratio_exactly_one(method: str) -> None:
    """Zero drift means zero penalty gradient, so the two flows coincide.

    Right after consolidation `θ − θ* = 0`, the quadratic's gradient is
    exactly zero and the penalized step IS the unpenalized step. A sample of
    exactly 1.0 there is the construction, not a dead mechanism — pinned so a
    future reader cannot mistake it for one.
    """

    results = _run(_tiny_config(method, 1))
    samples = results[method]["plasticity_samples"]

    firsts = {}
    for sample in samples:
        firsts.setdefault(sample["stage"], sample)
    assert firsts, f"{method}: nenhuma amostra registrada"
    for stage, sample in firsts.items():
        assert sample["plasticity_ratio"] == pytest.approx(1.0, abs=1e-6), (
            f"{method}, stage {stage}: primeira amostra após a âncora deveria "
            f"ser exatamente 1,0"
        )


def test_vanilla_records_no_samples_because_it_has_no_penalty() -> None:
    """Measuring `vanilla` would cost 3x a step to always report exactly 1.0."""

    results = _run(_tiny_config("vanilla", 1))
    assert results["vanilla"]["plasticity_samples"] == []


def test_the_sampling_interval_is_respected() -> None:
    """G7: the rate is declared, not incidental, and goes in the manifest."""

    dense = _run(_tiny_config("ewc", 1))
    sparse = _run(_tiny_config("ewc", 1000))

    assert len(dense["ewc"]["plasticity_samples"]) > len(
        sparse["ewc"]["plasticity_samples"]
    )


def test_measuring_does_not_change_the_trained_model() -> None:
    """The measurement must not perturb the run it reports.

    Same seed with sampling on and off must produce identical losses. If they
    diverge, the shadow step leaked into the trajectory and every endpoint of
    the pilot belongs to a different run than the one described. This is the
    test that justifies turning sampling on for a confirmatory run at all.
    """

    measured = _run(_tiny_config("ewc", 1))
    untouched = _run(_tiny_config("ewc", 0))

    assert untouched["ewc"]["plasticity_samples"] == []

    measured_losses = measured["ewc"]["training_losses"]
    untouched_losses = untouched["ewc"]["training_losses"]
    assert len(measured_losses) == len(untouched_losses)
    for stage, (with_sampling, without) in enumerate(
        zip(measured_losses, untouched_losses, strict=True)
    ):
        assert with_sampling == pytest.approx(without, abs=1e-9), (
            f"stage {stage}: a medição alterou a trajetória que ela reporta"
        )


def test_the_sampling_interval_stays_out_of_frozen_config_payloads() -> None:
    """A new field must not move the sha256 of a protocol frozen before it.

    Same convention as `mas_lambda`: the field enters the payload only when
    sampling is actually on. Never repair this by updating an expected hash.
    """

    from experiments.split_mnist import config_payload

    off = _tiny_config("ewc", 0)
    assert "plasticity_sampling_interval" not in config_payload(off)

    on = _tiny_config("ewc", 50)
    assert config_payload(on)["plasticity_sampling_interval"] == 50


def test_the_shadow_step_uses_the_masked_logits_the_training_loss_uses() -> None:
    """The counterfactual must differ from the real loss ONLY by the penalty.

    If the shadow step recomputes an unmasked cross-entropy while training uses
    `_mask_unseen_logits`, the ratio measures the masking too, and the number
    is not the plasticity of the penalty.
    """

    import torch.nn as nn

    from experiments.split_mnist import _build_shadow_losses, _mask_unseen_logits

    torch.manual_seed(0)
    model = nn.Linear(4, 6, bias=False)
    inputs = torch.randn(8, 4)
    targets = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    seen = (0, 1)

    penalized, unpenalized = _build_shadow_losses(
        model=model,
        inputs=inputs,
        targets=targets,
        seen_classes=seen,
        importance={n: torch.ones_like(p) for n, p in model.named_parameters()},
        anchors={n: torch.zeros_like(p) for n, p in model.named_parameters()},
        scale=10.0,
    )

    # The unpenalized branch must equal the MASKED cross-entropy exactly.
    expected = torch.nn.functional.cross_entropy(
        _mask_unseen_logits(model(inputs), seen), targets
    )
    assert float(unpenalized()) == pytest.approx(float(expected), abs=1e-6)
    # And the penalized branch must sit strictly above it.
    assert float(penalized()) > float(unpenalized())
