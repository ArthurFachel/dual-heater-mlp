"""Testes do agregador do L2.

O agregador decide se C1–C5 foram confirmadas ou falsificadas. Um julgamento
errado aqui vira uma frase errada no artigo, e nenhum manifest denuncia isso —
os números crus continuam certos.

Os casos cobrem o que a calibração mostrou ser possível: um arm que diverge num
ponto do eixo e é são nos outros, e um `E` unanimemente abaixo de 1.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.analyze_sgd_plasticity import (
    EXPECTED_SEEDS,
    analyze,
    check_monotonic,
    summarize_arm,
)
from scripts.run_sgd_plasticity import LEARNING_RATE_AXIS

STRENGTHS = {
    "ewc_lambda": 200.0,
    "ewc_decay": 1.0,
    "si_lambda": 600.0,
    "mas_lambda": 1.0,
    "mas_decay": 1.0,
}


def _arm_payload(
    *,
    norm: float | None,
    non_finite: int = 0,
    cosine: float = 0.98,
    scale: float = 0.18,
    residual: float = 1.2e-5,
    agreed: int = 128,
    counted: int = 128,
) -> dict:
    return {
        "norm_ratio": {
            "n": 128 - non_finite,
            "n_non_finite": non_finite,
            "mean": norm,
            "stdev": 0.01,
            "min": norm,
            "max": norm,
        },
        "direction_cosine": {"n": 128, "n_non_finite": 0, "mean": cosine},
        "penalty_scale": {"n": 128, "n_non_finite": 0, "mean": scale},
        "penalty_alignment": {"n": 128, "n_non_finite": 0, "mean": -0.05},
        "identity_residual": {"n": 128, "n_non_finite": 0, "mean": residual, "max": residual},
        "prediction_agreement": {"n": counted, "agreed": agreed, "rate": agreed / counted if counted else None},
        "n_samples": 128,
    }


def _write_point(
    root: Path,
    learning_rate: float,
    arms: dict[str, dict],
    *,
    seeds: int = EXPECTED_SEEDS,
) -> None:
    directory = root / f"lr_{learning_rate:g}"
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(seeds):
        payload = {
            "seed": 8_000_011 + index * 25_000,
            "host": "split_mnist",
            "optimizer": "sgd",
            "learning_rate": learning_rate,
            "declared_interval": 1,
            "penalty_strengths": STRENGTHS,
            "elapsed_seconds": 55.0,
            "arms": arms,
        }
        (directory / f"seed_{payload['seed']}.json").write_text(json.dumps(payload))


def _records(payload: dict) -> list[dict]:
    return [{"arms": {"ewc": payload}} for _ in range(EXPECTED_SEEDS)]


def test_c1_is_falsified_when_every_seed_is_below_one() -> None:
    """O desfecho que a calibração antecipou: E < 1 unânime."""

    summary = summarize_arm(_records(_arm_payload(norm=0.95)), "ewc")
    assert summary["c1_above_one"] is False
    assert summary["below_one"] == EXPECTED_SEEDS
    assert summary["p_exact"] == pytest.approx(2 / 2**EXPECTED_SEEDS)


def test_c1_is_confirmed_only_when_unanimously_above_one() -> None:
    summary = summarize_arm(_records(_arm_payload(norm=1.02)), "ewc")
    assert summary["c1_above_one"] is True
    assert summary["above_one"] == EXPECTED_SEEDS


def test_c1_is_not_confirmed_by_a_mere_majority() -> None:
    """11 de 12 acima de 1 NÃO confirma: C1 foi declarada como persistência."""

    records = _records(_arm_payload(norm=1.02))
    records[0] = {"arms": {"ewc": _arm_payload(norm=0.99)}}
    summary = summarize_arm(records, "ewc")
    assert summary["c1_above_one"] is False
    assert summary["above_one"] == EXPECTED_SEEDS - 1


def test_c2_is_unjudgeable_on_a_divergent_arm() -> None:
    """Um arm que explodiu não testa a identidade — testa o otimizador.

    Reportar C2 como falsificada ali culparia o instrumento por uma
    divergência numérica que não é dele.
    """

    summary = summarize_arm(
        _records(_arm_payload(norm=5.4e16, non_finite=26, residual=13.4)), "ewc"
    )
    assert summary["divergent"] is True
    assert summary["c2_identity_holds"] is None


def test_c2_holds_below_the_tolerance() -> None:
    summary = summarize_arm(_records(_arm_payload(norm=0.95, residual=1.2e-5)), "ewc")
    assert summary["c2_identity_holds"] is True


def test_c2_fails_above_the_tolerance() -> None:
    summary = summarize_arm(_records(_arm_payload(norm=0.95, residual=1e-2)), "ewc")
    assert summary["c2_identity_holds"] is False


def test_c3_requires_every_single_step_to_agree() -> None:
    """C3 foi declarada como `todos os passos`, não `quase todos`."""

    perfect = summarize_arm(_records(_arm_payload(norm=0.95, agreed=128, counted=128)), "ewc")
    assert perfect["c3_prediction_exact"] is True

    one_off = summarize_arm(_records(_arm_payload(norm=0.95, agreed=127, counted=128)), "ewc")
    assert one_off["c3_prediction_exact"] is False
    assert one_off["prediction_rate"] == pytest.approx(127 / 128)


def test_c4_uses_the_declared_threshold() -> None:
    assert summarize_arm(_records(_arm_payload(norm=0.95, scale=0.6)), "ewc")["c4_penalty_dominates"] is True
    assert summarize_arm(_records(_arm_payload(norm=0.95, scale=0.4)), "ewc")["c4_penalty_dominates"] is False


def test_non_finite_counts_are_summed_not_dropped() -> None:
    summary = summarize_arm(_records(_arm_payload(norm=1.0, non_finite=26)), "ewc")
    assert summary["n_non_finite"] == 26 * EXPECTED_SEEDS


def test_c5_detects_monotonic_growth_as_lr_falls(tmp_path: Path) -> None:
    """O padrão da calibração: E sobe quando o lr cai."""

    summaries = {
        1e-2: [{"arm": "ewc", "E_mean": 0.87, "divergent": False}],
        3e-3: [{"arm": "ewc", "E_mean": 0.95, "divergent": False}],
        1e-3: [{"arm": "ewc", "E_mean": 0.97, "divergent": False}],
    }
    result = check_monotonic(summaries, "ewc")
    assert result["monotonic"] is True
    assert result["n_points"] == 3


def test_c5_detects_a_non_monotonic_arm() -> None:
    summaries = {
        1e-2: [{"arm": "ewc", "E_mean": 0.87, "divergent": False}],
        3e-3: [{"arm": "ewc", "E_mean": 0.99, "divergent": False}],
        1e-3: [{"arm": "ewc", "E_mean": 0.93, "divergent": False}],
    }
    assert check_monotonic(summaries, "ewc")["monotonic"] is False


def test_c5_excludes_divergent_points(tmp_path: Path) -> None:
    """Um E de 1e16 destruiria qualquer afirmação de monotonicidade.

    O ponto divergente sai da curva, e o número de pontos restantes é
    reportado para que ninguém leia a conclusão como se fossem três.
    """

    summaries = {
        1e-2: [{"arm": "si", "E_mean": 5.4e16, "divergent": True}],
        3e-3: [{"arm": "si", "E_mean": 0.973, "divergent": False}],
        1e-3: [{"arm": "si", "E_mean": 0.993, "divergent": False}],
    }
    result = check_monotonic(summaries, "si")
    assert result["n_points"] == 2
    assert result["monotonic"] is True


def test_c5_is_unjudgeable_with_a_single_usable_point() -> None:
    summaries = {
        1e-2: [{"arm": "si", "E_mean": 5.4e16, "divergent": True}],
        3e-3: [{"arm": "si", "E_mean": 0.973, "divergent": False}],
    }
    assert check_monotonic(summaries, "si")["monotonic"] is None


def test_integrity_flags_a_short_axis_point(tmp_path: Path) -> None:
    for learning_rate in LEARNING_RATE_AXIS:
        _write_point(
            tmp_path,
            learning_rate,
            {arm: _arm_payload(norm=0.95) for arm in ("ewc", "si", "mas")},
            seeds=EXPECTED_SEEDS if learning_rate != 1e-3 else 7,
        )
    report = analyze(tmp_path)
    assert any("lr=0.001" in item for item in report["integrity_failures"])


def test_analyze_covers_every_axis_point(tmp_path: Path) -> None:
    for learning_rate in LEARNING_RATE_AXIS:
        _write_point(
            tmp_path,
            learning_rate,
            {arm: _arm_payload(norm=0.95) for arm in ("ewc", "si", "mas")},
        )
    report = analyze(tmp_path)
    assert report["integrity_failures"] == []
    assert set(report["by_learning_rate"]) == set(LEARNING_RATE_AXIS)
    assert len(report["monotonicity"]) == 3
