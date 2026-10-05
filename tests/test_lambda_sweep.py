"""Banda de seeds, grade de λ e guardas do protocolo L3.

Pré-registro: `goals/protocol_lambda_sweep.md`.

O L3 pergunta se existe uma força de penalidade onde o controle pareado tem
poder (§D.2: `E <= 0,90`) sem que o otimizador divirja. Três coisas precisam
ser verdadeiras antes da primeira seed: a banda é disjunta de tudo já gasto
(T5), a grade é a declarada (T3), e o multiplicador de λ realmente chega na
config — um sweep que ignora o próprio eixo mede ruído entre seeds com
manifests de aparência perfeita.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

# --- T5: a banda -----------------------------------------------------------


def test_lambda_sweep_band_has_ten_distinct_seeds() -> None:
    from experiments.confirmatory_split_mnist import LAMBDA_SWEEP_SEEDS

    assert len(LAMBDA_SWEEP_SEEDS) == 10
    assert len(set(LAMBDA_SWEEP_SEEDS)) == 10


def test_lambda_sweep_band_is_disjoint_from_every_band_already_spent() -> None:
    """§D.1. Toda banda já gasta, verificada por import — não por inspeção."""

    from experiments.bert_slowheat_diagnostic import (
        IMPORTANCE_CRITERION_ABLATION_SEEDS,
    )
    from experiments.confirmatory_split_mnist import (
        CONFIRMATORY_SEEDS,
        DECLARED_EXPLORATORY_SEEDS,
        DERPP_CONFIRMATORY_SEEDS,
        LAMBDA_SWEEP_SEEDS,
        PENALTY_REEVALUATION_SEEDS,
        SGD_PLASTICITY_SEEDS,
    )
    from experiments.qwen_lora_slowheat import (
        EXACT_DECOMPOSITION_SEEDS,
        PLASTICITY_MATCHED_SEEDS,
    )
    from experiments.replay_selection_sweep import (
        REPLAY_SELECTOR_CONFIRMATORY_SEEDS,
    )

    seeds = set(LAMBDA_SWEEP_SEEDS)
    for name, band in (
        ("confirmatory", CONFIRMATORY_SEEDS),
        ("exploratory", DECLARED_EXPLORATORY_SEEDS),
        ("derpp", DERPP_CONFIRMATORY_SEEDS),
        ("replay selector", REPLAY_SELECTOR_CONFIRMATORY_SEEDS),
        ("criterion ablation", IMPORTANCE_CRITERION_ABLATION_SEEDS),
        ("exact decomposition", EXACT_DECOMPOSITION_SEEDS),
        ("plasticity matched", PLASTICITY_MATCHED_SEEDS),
        ("penalty reevaluation", PENALTY_REEVALUATION_SEEDS),
        ("sgd plasticity (L2)", SGD_PLASTICITY_SEEDS),
    ):
        overlap = seeds & set(band)
        assert not overlap, f"banda L3 colide com {name}: {sorted(overlap)}"


def test_lambda_sweep_band_matches_the_frozen_protocol() -> None:
    """Os valores do §D.1, literais. Mudar a banda muda o pré-registro."""

    from experiments.confirmatory_split_mnist import LAMBDA_SWEEP_SEEDS

    assert tuple(LAMBDA_SWEEP_SEEDS) == (
        9000011, 9025013, 9050033, 9075041, 9100051,
        9125059, 9150067, 9175073, 9200089, 9225091,
    )


def test_calibration_seed_is_outside_the_declared_band() -> None:
    """A seed que escolheu a grade não pode entrar na run que ela governa."""

    from experiments.confirmatory_split_mnist import LAMBDA_SWEEP_SEEDS
    from scripts.calibrate_lambda_sweep import CALIBRATION_SEED

    assert CALIBRATION_SEED not in set(LAMBDA_SWEEP_SEEDS)


# --- T3: a grade -----------------------------------------------------------


def test_declared_multiplier_axis_matches_the_frozen_protocol() -> None:
    """T3: `1, 3, 10, 30, 100`. O `300` foi excluído e não pode voltar sozinho."""

    from scripts.run_lambda_sweep import MULTIPLIER_AXIS

    assert MULTIPLIER_AXIS == (1.0, 3.0, 10.0, 30.0, 100.0)
    assert 300.0 not in MULTIPLIER_AXIS


def test_multiplier_scales_every_lambda_and_leaves_decay_alone() -> None:
    """T3: multiplicador preserva a razão entre métodos publicada por Hsu.

    Escalar os `decay` junto mudaria a estrutura do método, não a força —
    `ewc_decay` é o fator de acumulação do Fisher entre tarefas.
    """

    from experiments.penalty_reevaluation import PUBLISHED_PENALTY_STRENGTHS
    from scripts.run_lambda_sweep import strengths_for

    scaled = strengths_for(10.0)
    assert scaled["ewc_lambda"] == PUBLISHED_PENALTY_STRENGTHS["ewc_lambda"] * 10
    assert scaled["si_lambda"] == PUBLISHED_PENALTY_STRENGTHS["si_lambda"] * 10
    assert scaled["mas_lambda"] == PUBLISHED_PENALTY_STRENGTHS["mas_lambda"] * 10
    assert scaled["ewc_decay"] == PUBLISHED_PENALTY_STRENGTHS["ewc_decay"]
    assert scaled["mas_decay"] == PUBLISHED_PENALTY_STRENGTHS["mas_decay"]


def test_multiplier_one_reproduces_the_published_strengths_exactly() -> None:
    """O ponto 1× tem de ser o L2, não uma aproximação dele."""

    from experiments.penalty_reevaluation import PUBLISHED_PENALTY_STRENGTHS
    from scripts.run_lambda_sweep import strengths_for

    assert strengths_for(1.0) == dict(PUBLISHED_PENALTY_STRENGTHS)


def test_config_carries_the_multiplier_not_a_default() -> None:
    """O bug que este projeto já teve duas vezes: o eixo não chega no runner.

    Se `multiplier` fosse ignorado, os cinco pontos rodariam na mesma força e o
    sweep mediria ruído entre seeds — com manifests de aparência normal.
    """

    from scripts.run_lambda_sweep import build_config

    weak = build_config(seed=1, multiplier=1.0)
    strong = build_config(seed=1, multiplier=100.0)

    assert strong.ewc_lambda == weak.ewc_lambda * 100
    assert strong.si_lambda == weak.si_lambda * 100
    assert strong.mas_lambda == weak.mas_lambda * 100


def test_config_pins_pure_sgd_at_the_declared_learning_rate() -> None:
    """T1 e T2: SGD puro a `3e-3`. Sob AdamW a pergunta do L3 não faz sentido."""

    from scripts.run_lambda_sweep import build_config

    config = build_config(seed=1, multiplier=1.0)
    config.validate()
    assert config.optimizer == "sgd"
    assert config.learning_rate == pytest.approx(3e-3)


def test_config_always_requests_exhaustive_sampling() -> None:
    """T6. A série amostrada já inverteu uma classificação G8 uma vez."""

    from scripts.run_lambda_sweep import build_config

    assert build_config(seed=1, multiplier=30.0).plasticity_sampling_interval == 1


# --- T8: divergência é dado, não lixo --------------------------------------


def test_arm_with_any_non_finite_sample_is_marked_divergent() -> None:
    """T8: uma única amostra não-finita contamina a célula.

    A calibração mediu `E = 2,3e9` com 29 de 128 amostras não-finitas. A média
    dos 99 finitos restantes é um número, mas não é plasticidade — é o que
    sobrou de um otimizador explodindo.
    """

    from scripts.run_lambda_sweep import summarize_metric

    summary = summarize_metric([0.95, 0.96, float("inf"), 0.94])
    assert summary["n_non_finite"] == 1
    assert summary["divergent"] is True


def test_a_clean_arm_is_not_marked_divergent() -> None:
    from scripts.run_lambda_sweep import summarize_metric

    summary = summarize_metric([0.95, 0.96, 0.94])
    assert summary["n_non_finite"] == 0
    assert summary["divergent"] is False
    assert summary["mean"] == pytest.approx(0.95)


def test_nan_counts_as_non_finite_not_as_a_missing_sample() -> None:
    """NaN e None são coisas diferentes: um é divergência, o outro é ausência."""

    from scripts.run_lambda_sweep import summarize_metric

    assert summarize_metric([0.9, float("nan")])["n_non_finite"] == 1
    assert summarize_metric([0.9, None])["n_non_finite"] == 0


def test_entirely_divergent_arm_has_no_mean() -> None:
    """Inventar uma média para um arm sem nenhuma amostra sã seria inventar dado."""

    from scripts.run_lambda_sweep import summarize_metric

    summary = summarize_metric([float("inf"), float("nan")])
    assert summary["mean"] is None
    assert summary["divergent"] is True


# --- §D.2: o limiar de poder, declarado antes das seeds --------------------


def test_power_threshold_matches_the_frozen_protocol() -> None:
    """§D.2: `E <= 0,90`. Mudar isso depois de ver os dados é escolher o resultado."""

    from scripts.run_lambda_sweep import POWER_THRESHOLD

    assert POWER_THRESHOLD == 0.90


def test_a_cell_has_power_only_when_low_and_non_divergent() -> None:
    """§D.2 tem DUAS condições, e a conjunção é o ponto.

    Um `E` baixo num arm que divergiu é o caso que o §D.2 existe para excluir:
    a calibração viu `ewc` a 100x com média 1,5e19 e 111 amostras não-finitas.
    """

    from scripts.run_lambda_sweep import has_power

    assert has_power(mean_e=0.75, divergent=False) is True
    assert has_power(mean_e=0.95, divergent=False) is False
    assert has_power(mean_e=0.75, divergent=True) is False
    assert has_power(mean_e=None, divergent=True) is False


def test_the_threshold_boundary_is_inclusive() -> None:
    """`E = 0,90` exato tem poder: o §D.2 diz `<=`, não `<`."""

    from scripts.run_lambda_sweep import has_power

    assert has_power(mean_e=0.90, divergent=False) is True
    assert has_power(mean_e=0.9000001, divergent=False) is False


# --- proveniência ----------------------------------------------------------


def test_record_declares_everything_needed_to_audit_it(tmp_path: Path) -> None:
    """Um manifest que não diz sua própria força não é auditável offline."""

    from scripts.run_lambda_sweep import build_record_header

    record = build_record_header(seed=9000011, multiplier=30.0, elapsed=1.0)
    for field in (
        "seed", "host", "optimizer", "learning_rate",
        "multiplier", "penalty_strengths", "declared_interval",
    ):
        assert field in record, f"campo {field} ausente do manifest"

    assert record["optimizer"] == "sgd"
    assert record["multiplier"] == 30.0
    assert record["penalty_strengths"]["ewc_lambda"] == 200.0 * 30

    # Tem de sobreviver a um round-trip JSON: é assim que ele será lido.
    (tmp_path / "r.json").write_text(json.dumps(record), encoding="utf-8")
    assert json.loads((tmp_path / "r.json").read_text())["multiplier"] == 30.0


def test_reuse_guard_rejects_a_different_multiplier() -> None:
    """Resumir por "o arquivo existe" misturaria pontos do eixo em silêncio."""

    from scripts.run_lambda_sweep import build_record_header, should_reuse

    record = build_record_header(seed=9000011, multiplier=30.0, elapsed=1.0)
    assert should_reuse(record, multiplier=30.0) is True
    assert should_reuse(record, multiplier=100.0) is False


def test_reuse_guard_rejects_adamw_artifacts() -> None:
    """Um manifest das passadas 1 não pode ser lido como se fosse do L3."""

    from scripts.run_lambda_sweep import should_reuse

    assert should_reuse({"optimizer": "adamw", "multiplier": 30.0}, multiplier=30.0) is False
    assert should_reuse("not a dict", multiplier=30.0) is False


def test_destination_separates_the_axis_points(tmp_path: Path) -> None:
    """Dois pontos no mesmo caminho: o último sobrescreve o primeiro em silêncio."""

    from scripts.run_lambda_sweep import destination_for

    a = destination_for(root=tmp_path, multiplier=30.0, seed=9000011)
    b = destination_for(root=tmp_path, multiplier=100.0, seed=9000011)
    assert a != b


# --- T9: a passada é mecanismo-only ---------------------------------------


def test_runner_reads_no_accuracy_field() -> None:
    """T9, verificado no fonte: nenhum endpoint de acurácia é tocado.

    Mesmo teste do L2. O risco não é ler de propósito, é ler sem perceber ao
    reusar um helper que devolve a matriz inteira.
    """

    source = Path("scripts/run_lambda_sweep.py").read_text(encoding="utf-8")
    for forbidden in (
        "accuracy_matrix",
        "final_average_accuracy",
        "average_forgetting",
        "backward_transfer",
    ):
        assert forbidden not in source, f"o runner referencia {forbidden}"


def test_sign_test_is_exact_and_two_sided_with_ten_seeds() -> None:
    """T10 e §D.1: com n=10 unânime o `p` tem de ser 2^-9, abaixo do Holm de 0,01."""

    from experiments.confirmatory_statistics import exact_two_sided_sign_test

    result = exact_two_sided_sign_test([-0.05] * 10)
    p = result["p_value"] if isinstance(result, dict) else result
    assert p == pytest.approx(2.0 ** -9, rel=1e-9)
    assert p < 0.01


def test_power_floor_claim_in_the_protocol_is_arithmetically_true() -> None:
    """§D.1 afirma `2^-9 = 0,00195`. Uma afirmação numérica no pré-registro
    que ninguém verifica é uma afirmação não verificada."""

    assert math.isclose(2.0 ** -9, 0.001953125, rel_tol=1e-12)
