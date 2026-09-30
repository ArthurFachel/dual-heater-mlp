"""Runner do L2: o eixo de learning rate sob SGD puro.

Pré-registro: `goals/protocol_sgd_plasticity.md`, S4 emendado em `80bd322`
ANTES desta run. O eixo existe porque `lr = 1e-2` faz o `si` divergir
numericamente (5,5e18) enquanto `3e-3` e `1e-3` ficam sãos — a fronteira é
dado, não ruído a esconder.

Testa o que o runner decide, não a aritmética que já tem teste em
`tests/test_sgd_plasticity.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_declared_axis_matches_the_frozen_protocol() -> None:
    """S4 emendado: exatamente três pontos, nesta ordem."""

    from scripts.run_sgd_plasticity import LEARNING_RATE_AXIS

    assert LEARNING_RATE_AXIS == (1e-2, 3e-3, 1e-3)


def test_config_carries_the_axis_value_not_a_default() -> None:
    """A mutação clássica: o valor do eixo não chega no config do arm.

    Se `build_config` ignorasse o argumento, os três pontos rodariam no mesmo
    learning rate e o eixo mediria ruído entre seeds, com manifests de
    aparência perfeitamente normal.
    """

    from scripts.run_sgd_plasticity import LEARNING_RATE_AXIS, build_config

    for learning_rate in LEARNING_RATE_AXIS:
        config = build_config(seed=8_000_011, learning_rate=learning_rate)
        assert config.learning_rate == pytest.approx(learning_rate)
        assert config.optimizer == "sgd"


def test_config_always_requests_exhaustive_sampling() -> None:
    """S7. Uma série amostrada já inverteu uma classificação G8 uma vez."""

    from scripts.run_sgd_plasticity import build_config

    assert build_config(seed=1, learning_rate=1e-3).plasticity_sampling_interval == 1


def test_config_uses_the_published_strengths_unchanged() -> None:
    """S3: mudar força junto com otimizador confundiria os dois eixos."""

    from experiments.penalty_reevaluation import PUBLISHED_PENALTY_STRENGTHS
    from scripts.run_sgd_plasticity import build_config

    config = build_config(seed=1, learning_rate=1e-3)
    for field, value in PUBLISHED_PENALTY_STRENGTHS.items():
        assert getattr(config, field) == value


def test_destination_separates_the_axis_points(tmp_path: Path) -> None:
    """Dois pontos do eixo no mesmo caminho = um sobrescreve o outro.

    É o modo de falha do pitfall de colisão de diretório: a run que termina
    por último vence em silêncio e o agregado mistura learning rates.
    """

    from scripts.run_sgd_plasticity import destination_for

    paths = {
        destination_for(root=tmp_path, learning_rate=lr, seed=8_000_011)
        for lr in (1e-2, 3e-3, 1e-3)
    }
    assert len(paths) == 3


def test_reuse_guard_rejects_a_different_learning_rate(tmp_path: Path) -> None:
    """Resume chaveado em existência preservaria artefato de outro lr."""

    from experiments.penalty_reevaluation import PUBLISHED_PENALTY_STRENGTHS
    from scripts.run_sgd_plasticity import should_reuse

    record = {
        "optimizer": "sgd",
        "learning_rate": 3e-3,
        "declared_interval": 1,
        "penalty_strengths": dict(PUBLISHED_PENALTY_STRENGTHS),
    }
    assert should_reuse(record, learning_rate=3e-3) is True
    assert should_reuse(record, learning_rate=1e-2) is False


def test_reuse_guard_rejects_adamw_artifacts() -> None:
    from experiments.penalty_reevaluation import PUBLISHED_PENALTY_STRENGTHS
    from scripts.run_sgd_plasticity import should_reuse

    record = {
        "optimizer": "adamw",
        "learning_rate": 3e-3,
        "declared_interval": 1,
        "penalty_strengths": dict(PUBLISHED_PENALTY_STRENGTHS),
    }
    assert should_reuse(record, learning_rate=3e-3) is False


def test_summaries_drop_non_finite_samples_and_count_them() -> None:
    """A divergência do `si` em 1e-2 é DADO, não uma amostra a descartar em silêncio.

    Um `norm_ratio` de 5,5e18 não pode entrar numa média — ela deixaria de
    significar qualquer coisa. Mas o número de amostras que divergiram tem de
    aparecer no manifest, ou o relatório reporta uma média sobre 102 de 128
    passos como se fossem 128.
    """

    from scripts.run_sgd_plasticity import summarize_metric

    values = [1.0, 2.0, float("inf"), float("nan"), 3.0]
    summary = summarize_metric(values)

    assert summary["n"] == 3
    assert summary["n_non_finite"] == 2
    assert summary["mean"] == pytest.approx(2.0)


def test_summary_of_an_entirely_divergent_arm_is_not_a_number() -> None:
    """Se TUDO divergiu, não há média a reportar — e dizer isso é obrigatório."""

    from scripts.run_sgd_plasticity import summarize_metric

    summary = summarize_metric([float("inf"), float("nan")])
    assert summary["n"] == 0
    assert summary["n_non_finite"] == 2
    assert summary["mean"] is None


def test_prediction_agreement_ignores_non_finite_samples() -> None:
    """C3 sobre um `E` infinito não é acerto nem erro: é ausência de medida."""

    from scripts.run_sgd_plasticity import prediction_agreement

    samples = [
        {"norm_ratio": 0.9, "predicted_above_one": False},
        {"norm_ratio": 1.1, "predicted_above_one": True},
        {"norm_ratio": float("inf"), "predicted_above_one": True},
        {"norm_ratio": None, "predicted_above_one": True},
    ]
    result = prediction_agreement(samples)
    assert result["n"] == 2
    assert result["agreed"] == 2
    assert result["rate"] == pytest.approx(1.0)


def test_prediction_agreement_catches_a_disagreement() -> None:
    """Se o teste não distinguir acerto de erro, C3 passa com qualquer coisa."""

    from scripts.run_sgd_plasticity import prediction_agreement

    samples = [
        {"norm_ratio": 0.9, "predicted_above_one": True},   # discorda
        {"norm_ratio": 1.1, "predicted_above_one": True},   # concorda
    ]
    result = prediction_agreement(samples)
    assert result["n"] == 2
    assert result["agreed"] == 1
    assert result["rate"] == pytest.approx(0.5)


def test_calibration_seed_is_outside_the_declared_band() -> None:
    """Gastar uma seed da banda em calibração contaminaria o pré-registro."""

    from experiments.confirmatory_split_mnist import SGD_PLASTICITY_SEEDS
    from scripts.run_sgd_plasticity import CALIBRATION_SEED

    assert CALIBRATION_SEED not in set(SGD_PLASTICITY_SEEDS)


def test_runner_reads_no_accuracy_field() -> None:
    """S9, verificado no código: o runner não pode tocar em acurácia.

    Um runner que lê a matriz de acurácia numa passada mecanismo-only queima o
    pré-registro inteiro, e o modo de falha é silencioso — o número entra num
    log e ninguém nota.
    """

    source = Path("scripts/run_sgd_plasticity.py").read_text(encoding="utf-8")
    for forbidden in (
        "accuracy_matrix",
        "final_average_accuracy",
        "average_forgetting",
        "backward_transfer",
    ):
        assert forbidden not in source, (
            f"o runner do L2 referencia `{forbidden}`, o que viola S9"
        )


def test_record_declares_everything_needed_to_audit_it(tmp_path: Path) -> None:
    """Um manifest sem a configuração que o gerou não é auditável offline."""

    from scripts.run_sgd_plasticity import build_record_header

    header = build_record_header(seed=8_000_011, learning_rate=3e-3, elapsed=12.5)
    for field in (
        "seed",
        "host",
        "optimizer",
        "learning_rate",
        "declared_interval",
        "penalty_strengths",
        "elapsed_seconds",
    ):
        assert field in header
    assert header["optimizer"] == "sgd"
    assert header["learning_rate"] == pytest.approx(3e-3)
