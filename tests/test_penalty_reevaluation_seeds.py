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


def test_the_published_penalty_strengths_match_hsu_2018_class_il() -> None:
    """§C.2. Os λ vêm de Hsu et al. 2018, não de escolha nossa.

    Fonte: `scripts/split_MNIST_incremental_class.sh` do repositório
    GT-RIPL/Continual-Learning-Benchmark, que acompanha *Re-evaluating
    Continual Learning Scenarios* (NeurIPS CL Workshop 2018). Split-MNIST
    class-incremental, MLP400, Adam lr=1e-3, batch 128 — o mesmo cenário deste
    piloto.

    **Convenção de fator, e é por isso que os números aqui não batem de olho
    com os do script.** Hsu soma `reg_coef * Σ Ω (θ−θ*)²`, sem o meio. Este
    repo passa por `ewc_penalty`, que já aplica `1/2`, e o runner soma
    `0.5 * ewc_lambda` para EWC e `si_lambda`/`mas_lambda` sem o meio para SI e
    MAS. Logo o fator efetivo `k` na loss é o que precisa bater, não o λ.
    """

    from experiments.penalty_reevaluation import PUBLISHED_PENALTY_STRENGTHS
    from experiments.split_mnist import SplitMNISTConfig, _penalty_scale

    # reg_coef publicado por Hsu et al. para Split-MNIST class-incremental.
    hsu_class_il = {"ewc": 100.0, "si": 600.0, "mas": 1.0}

    config = SplitMNISTConfig(**PUBLISHED_PENALTY_STRENGTHS)
    for method, published in hsu_class_il.items():
        assert _penalty_scale(method, config) == pytest.approx(published), (
            f"{method}: fator efetivo não bate com o publicado"
        )


def test_the_published_strengths_use_the_online_ewc_variant() -> None:
    """Hsu publica DOIS números para EWC; a escolha não é arbitrária.

    `EWC_mnist` (reg_coef 600) recomputa o Fisher por tarefa e guarda um termo
    por tarefa (`online_reg = False`). `EWC_online_mnist` (reg_coef 100) mantém
    UM Fisher acumulado (`online_reg = True`).

    `consolidate_importance` deste repo faz `decay * importance + estimate`
    sobre um único dicionário — é a variante ONLINE. Logo o número comparável é
    100, não 600, e `ewc_decay = 1.0` é a soma acumulada clássica.
    """

    from experiments.penalty_reevaluation import PUBLISHED_PENALTY_STRENGTHS

    # decay=1.0 é o EWC online sem esquecimento, que é o que o runner faz.
    assert PUBLISHED_PENALTY_STRENGTHS["ewc_decay"] == 1.0
    assert PUBLISHED_PENALTY_STRENGTHS["mas_decay"] == 1.0
    # 0.5 * 200 = 100 = reg_coef do EWC_online_mnist.
    assert PUBLISHED_PENALTY_STRENGTHS["ewc_lambda"] == pytest.approx(200.0)


def test_the_published_strengths_do_not_touch_the_dataclass_defaults() -> None:
    """Mexer nos defaults moveria o sha256 de um pré-registro congelado.

    `ewc_lambda` e `si_lambda` entram em `config_payload` mesmo quando o método
    não é usado, então alterar o default de `SplitMNISTConfig` mudaria o hash
    de `experiments/confirmatory_split_mnist.py`. Os valores publicados vivem
    numa constante consumida só por este piloto.
    """

    from experiments.split_mnist import SplitMNISTConfig

    default = SplitMNISTConfig()
    assert default.ewc_lambda == 100.0, "default alterado: o hash congelado se move"
    assert default.si_lambda == 1.0, "default alterado: o hash congelado se move"
    assert default.mas_lambda == 1.0, "default alterado: o hash congelado se move"


def test_si_was_three_hundred_times_weaker_than_published() -> None:
    """O achado que motivou a adoção: SI estava funcionalmente inerte.

    Com `si_lambda = 1.0` o fator efetivo é 1 contra os 600 publicados para
    class-IL. A calibração mediu `E = 0,9997` para SI — ou seja, o método
    removia 0,03% da plasticidade e entraria no piloto como vanilla com outro
    nome. Este teste fixa a razão para que o motivo da mudança fique legível.
    """

    from experiments.penalty_reevaluation import PUBLISHED_PENALTY_STRENGTHS
    from experiments.split_mnist import SplitMNISTConfig, _penalty_scale

    stale = _penalty_scale("si", SplitMNISTConfig())
    published = _penalty_scale("si", SplitMNISTConfig(**PUBLISHED_PENALTY_STRENGTHS))

    assert stale == pytest.approx(1.0)
    assert published / stale == pytest.approx(600.0)


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


def test_pass1_measures_every_anchored_step() -> None:
    """§E, emendado: a taxa é 1/1 e não é ajustável por conveniência.

    A 1/50 o `E` de uma seed vinha de 3 amostras, com IC95% de largura 0,171
    sobre um efeito de 0,012 — e o veredito G8 do MAS invertia entre a série
    amostrada e a medição exata. A taxa faz parte do pré-registro.
    """

    from scripts.run_penalty_pass1 import DECLARED_INTERVAL

    assert DECLARED_INTERVAL == 1


def test_pass1_refuses_a_grid_that_misses_declared_steps() -> None:
    """Uma grade que não contém a declarada perderia passos silenciosamente."""

    from scripts.run_penalty_pass1 import validate_intervals

    validate_intervals(dense_interval=1)
    for bad in (0, -1, 2, 5, 50):
        with pytest.raises(ValueError):
            validate_intervals(dense_interval=bad)


def test_pass1_resume_rejects_artifacts_from_another_configuration() -> None:
    """Chavear o resume em "o arquivo existe" mistura proveniências.

    Os 12 artefatos gravados a 1/5 antes da emenda do §E seriam reaproveitados
    por um guarda ingênuo, e o agregado misturaria duas taxas de amostragem sem
    nada falhar. O guarda compara o que o artefato declara ter usado.
    """

    from experiments.penalty_reevaluation import PUBLISHED_PENALTY_STRENGTHS
    from scripts.run_penalty_pass1 import should_reuse

    good = {
        "dense_interval": 1,
        "declared_interval": 1,
        "penalty_strengths": dict(PUBLISHED_PENALTY_STRENGTHS),
    }
    assert should_reuse(good, dense_interval=1)

    # O artefato da run descartada: medido a 1/5 sob a taxa declarada antiga.
    stale = {
        "dense_interval": 5,
        "declared_interval": 50,
        "penalty_strengths": dict(PUBLISHED_PENALTY_STRENGTHS),
    }
    assert not should_reuse(stale, dense_interval=1)

    # Forças de penalidade diferentes: outro experimento.
    other_strengths = dict(good)
    other_strengths["penalty_strengths"] = {"ewc_lambda": 1.0}
    assert not should_reuse(other_strengths, dense_interval=1)

    # Artefato sem os campos de proveniência: não é reaproveitável.
    assert not should_reuse({"seed": 1}, dense_interval=1)
    assert not should_reuse(None, dense_interval=1)


def test_cifar_sharding_covers_every_seed_exactly_once() -> None:
    """Uma seed perdida ou duplicada entre shards invalida a banda.

    O sharding roda em 3 GPUs em paralelo; se a partição não for exata, a
    passada 1 do CIFAR mede um conjunto de seeds diferente do declarado no §G
    e nada falha em tempo de execução.
    """

    from experiments.confirmatory_split_mnist import PENALTY_REEVALUATION_SEEDS
    from scripts.run_penalty_pass1_cifar import shard_seeds

    seeds = list(PENALTY_REEVALUATION_SEEDS)
    for shards in (1, 2, 3, 4, 5):
        covered: list[int] = []
        for shard in range(shards):
            covered.extend(shard_seeds(seeds, shard=shard, shards=shards))
        assert sorted(covered) == sorted(seeds), f"{shards} shards: cobertura errada"
        assert len(covered) == len(set(covered)), f"{shards} shards: seed duplicada"


def test_cifar_sharding_keeps_a_whole_seed_on_one_device() -> None:
    """Partir as arms de uma seed entre devices quebraria o pareamento.

    As arms compartilham inicialização e fluxo de dados dentro de uma seed, e é
    isso que valida a diferença pareada. O sharding é por seed, nunca por arm.
    """

    from experiments.confirmatory_split_mnist import PENALTY_REEVALUATION_SEEDS
    from scripts.run_penalty_pass1_cifar import ARMS, shard_seeds

    seeds = list(PENALTY_REEVALUATION_SEEDS)
    sizes = [len(shard_seeds(seeds, shard=i, shards=3)) for i in range(3)]
    assert sum(sizes) == 12
    assert sorted(sizes) == [4, 4, 4], "12 seeds em 3 shards devem dar 4/4/4"
    # As arms nunca entram no cálculo do shard.
    assert ARMS == ("vanilla", "ewc", "si", "mas")


def test_cifar_sharding_rejects_an_out_of_range_shard() -> None:
    from scripts.run_penalty_pass1_cifar import shard_seeds

    with pytest.raises(ValueError):
        shard_seeds([1, 2, 3], shard=3, shards=3)
    with pytest.raises(ValueError):
        shard_seeds([1, 2, 3], shard=0, shards=0)
    with pytest.raises(ValueError):
        shard_seeds([1, 2, 3], shard=-1, shards=3)


def test_cifar_resume_rejects_artifacts_from_another_host_or_config() -> None:
    """O agregado não pode misturar hosts nem configurações."""

    from experiments.penalty_reevaluation import PUBLISHED_PENALTY_STRENGTHS
    from scripts.run_penalty_pass1_cifar import should_reuse

    good = {
        "host": "split_cifar100",
        "declared_interval": 1,
        "penalty_strengths": dict(PUBLISHED_PENALTY_STRENGTHS),
    }
    assert should_reuse(good)

    wrong_host = dict(good, host="split_mnist")
    assert not should_reuse(wrong_host)

    wrong_interval = dict(good, declared_interval=50)
    assert not should_reuse(wrong_interval)

    wrong_strengths = dict(good, penalty_strengths={"ewc_lambda": 1.0})
    assert not should_reuse(wrong_strengths)

    assert not should_reuse({"seed": 1})
    assert not should_reuse(None)
