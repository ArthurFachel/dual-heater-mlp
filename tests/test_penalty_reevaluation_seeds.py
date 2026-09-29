"""Seed band and design guards for the penalty re-evaluation pilot.

`goals/protocol_penalty_reevaluation.md`, sections F, G and H. A seed reused
from a band already spent breaks the paired design's independence from prior
analysis, and a family sized past the significance floor cannot clear its own
bar no matter how large the effect is.

This file is a pre-registration guard. When it goes red, the fix is never to
update the expected value: the protocol is frozen and the code is what moved.
"""

from __future__ import annotations

import pytest


def test_seed_band_has_twelve_distinct_seeds() -> None:
    from experiments.confirmatory_split_mnist import PENALTY_REEVALUATION_SEEDS

    assert len(PENALTY_REEVALUATION_SEEDS) == 12
    assert len(set(PENALTY_REEVALUATION_SEEDS)) == 12


def test_seed_band_is_disjoint_from_every_band_already_spent() -> None:
    """§G. Every band listed in the protocol's table, checked by import."""

    from experiments.bert_slowheat_diagnostic import (
        IMPORTANCE_CRITERION_ABLATION_SEEDS,
    )
    from experiments.confirmatory_split_mnist import (
        DERPP_CONFIRMATORY_SEEDS,
        PENALTY_REEVALUATION_SEEDS,
    )
    from experiments.qwen_lora_slowheat import (
        EXACT_DECOMPOSITION_SEEDS,
        PLASTICITY_MATCHED_SEEDS,
    )
    from experiments.replay_selection_sweep import (
        REPLAY_SELECTOR_CONFIRMATORY_SEEDS,
    )

    seeds = set(PENALTY_REEVALUATION_SEEDS)
    for name, other in (
        ("derpp", DERPP_CONFIRMATORY_SEEDS),
        ("replay selector", REPLAY_SELECTOR_CONFIRMATORY_SEEDS),
        ("criterion ablation", IMPORTANCE_CRITERION_ABLATION_SEEDS),
        ("exact decomposition", EXACT_DECOMPOSITION_SEEDS),
        ("plasticity matched", PLASTICITY_MATCHED_SEEDS),
    ):
        overlap = seeds & set(other)
        assert not overlap, f"reusa seeds de {name}: {sorted(overlap)}"


def test_seed_band_is_disjoint_from_the_lora_confirmation_seeds() -> None:
    """The LoRA confirmation band, hardcoded from its artifacts (700.001+)."""

    from experiments.confirmatory_split_mnist import PENALTY_REEVALUATION_SEEDS

    confirmation = {
        700_001, 725_009, 750_019, 775_037, 800_053,
        825_059, 850_061, 875_089, 900_089, 925_097,
    }
    assert not set(PENALTY_REEVALUATION_SEEDS) & confirmation


def test_the_calibration_seed_is_outside_the_confirmatory_band() -> None:
    """§I. Its endpoints are n=1; pooling them with the band would be fraud."""

    from experiments.confirmatory_split_mnist import (
        PENALTY_REEVALUATION_CALIBRATION_SEED,
        PENALTY_REEVALUATION_SEEDS,
    )

    assert PENALTY_REEVALUATION_CALIBRATION_SEED not in PENALTY_REEVALUATION_SEEDS


def test_the_design_clears_its_own_significance_floor() -> None:
    """§H. With n seeds and a family of m under Holm, the smallest attainable
    p is `m * 2 / 2^n`. A design whose floor exceeds 0.05 cannot produce a
    significant result regardless of effect size, so this is checked BEFORE
    any GPU time is spent rather than discovered after.
    """

    from experiments.confirmatory_split_mnist import PENALTY_REEVALUATION_SEEDS

    n = len(PENALTY_REEVALUATION_SEEDS)
    m = 6  # the confirmatory family declared in §F
    floor = m * 2 / 2**n

    assert floor < 0.05, f"piso {floor:.4f} não limpa 0,05 com n={n}, m={m}"
    assert floor == pytest.approx(0.0029296875)

    # Room to spare: the design survives two dissenting seeds and still clears.
    assert m * 2 / 2 ** (n - 2) < 0.05


def test_the_protocol_document_exists_and_is_not_authorized() -> None:
    """The document is the deliverable of the gate task; running is separate.

    If this ever needs changing, the run was authorized, and the change belongs
    in the protocol's §K change log in the same commit.
    """

    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    document = root / "goals" / "protocol_penalty_reevaluation.md"
    assert document.exists(), "protocolo do piloto ausente"

    text = document.read_text(encoding="utf-8")
    assert "Estado de execução: NÃO AUTORIZADO" in text
    assert "7000003, 7025011" in text, "banda do documento saiu de sincronia"


def test_the_amendment_is_recorded_in_both_protocols() -> None:
    """G1 → G1': a primária mudou, e a troca tem de estar nos dois change logs.

    Uma emenda registrada só no protocolo do instrumento deixaria o piloto
    construindo `lr_scale` a partir de uma métrica que não é mais a primária.
    Uma emenda registrada só no piloto esconderia a revogação de G1.
    """

    import pathlib

    goals = pathlib.Path(__file__).resolve().parents[1] / "goals"

    instrument = (goals / "protocol_plasticity_generalization.md").read_text(
        encoding="utf-8"
    )
    # A primária nova, a antiga preservada, e o motivo medido.
    assert "G1'" in instrument, "a emenda não aparece no protocolo do instrumento"
    assert "REVOGADA" in instrument, "G1 não foi marcada como revogada"
    assert "1,7759" in instrument, "o número que motivou a emenda não está registrado"
    assert "§B.0" in instrument, "a métrica original não foi preservada"
    assert "Nenhuma acurácia foi lida" in instrument or (
        "Nenhum endpoint de acurácia foi lido" in instrument
    ), "a emenda precisa declarar que nenhuma acurácia foi consultada"

    pilot = (goals / "protocol_penalty_reevaluation.md").read_text(encoding="utf-8")
    assert "razão de normas" in pilot, "o piloto não reflete a métrica nova"
    assert "não-pareável" in pilot, "a regra G8 não chegou ao piloto"


def test_the_calibration_seed_produced_a_recorded_measurement() -> None:
    """A emenda cita números; eles têm de existir num artefato, não na memória.

    Se o JSON da calibração sumir, a justificativa da emenda vira alegação sem
    lastro — que é exatamente o que um pré-registro não pode ter.
    """

    import json
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    artifact = (
        root
        / "results"
        / "penalty_calibration"
        / "calibration_seed_7999991.json"
    )
    if not artifact.exists():
        pytest.skip("artefato da calibração não está nesta cópia de trabalho")

    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["seed"] == 7_999_991
    # Os três métodos de penalidade mediram algo; vanilla não mede nada.
    assert payload["per_arm"]["vanilla"]["samples"] == 0
    for method in ("ewc", "si", "mas"):
        record = payload["per_arm"][method]
        assert record["samples"] > 0, f"{method} não registrou amostras"
        assert "norm_ratio_mean" in record, f"{method} sem a métrica primária nova"


def test_the_confirmatory_family_is_six_paired_contrasts() -> None:
    """§F. The family is fixed here so it cannot be narrowed after the run.

    Narrowing a family after seeing which arms won is exactly the selection
    effect Holm exists to prevent; pinning the size makes that edit visible.
    """

    from experiments.confirmatory_split_mnist import PENALTY_REEVALUATION_SEEDS

    methods = ("ewc", "si", "mas")
    hosts = ("split_mnist", "split_cifar100")
    family = tuple(
        f"{method} - lr_control_{method} @ {host}"
        for host in hosts
        for method in methods
    )

    assert len(family) == 6
    assert len(set(family)) == 6
    # The floor computation in §H depends on this size and this n together.
    assert len(family) * 2 / 2 ** len(PENALTY_REEVALUATION_SEEDS) < 0.05
