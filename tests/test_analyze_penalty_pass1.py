"""Testes do agregador da passada 1 e da regra G8.

G8 decide quais contrastes existem na família confirmatória. Um agregador que
classifica errado constrói um `lr_control` impossível (`lr_scale > 1`, que é um
AUMENTO de learning rate) ou descarta um contraste válido. As duas falhas são
silenciosas no manifest.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.analyze_penalty_pass1 import (
    EXPECTED_SEEDS,
    analyze,
    arm_metrics,
    check_integrity,
    summarize_arm,
)

STRENGTHS = {
    "ewc_lambda": 200.0,
    "ewc_decay": 1.0,
    "si_lambda": 600.0,
    "mas_lambda": 1.0,
    "mas_decay": 1.0,
}


def _metrics(norm: float, *, cos: float = 0.99, cos_min: float = 0.9) -> dict:
    return {
        "norm_ratio": {"n": 128, "mean": norm, "stdev": 0.01, "min": norm, "max": norm},
        "plasticity_ratio": {"n": 128, "mean": 1.2, "stdev": 0.3, "min": 1.0, "max": 4.0},
        "direction_cosine": {"n": 128, "mean": cos, "stdev": 0.01, "min": cos_min, "max": 1.0},
    }


def _write_flat(root: Path, norms: dict[str, list[float]], *, host: str = "split_cifar100") -> Path:
    """Schema do runner do CIFAR: métricas direto sob o arm."""

    root.mkdir(parents=True, exist_ok=True)
    for index in range(len(next(iter(norms.values())))):
        payload = {
            "seed": 7_000_003 + index * 25_000,
            "host": host,
            "declared_interval": 1,
            "penalty_strengths": STRENGTHS,
            "arms": {arm: _metrics(values[index]) for arm, values in norms.items()},
        }
        (root / f"seed_{payload['seed']}.json").write_text(json.dumps(payload))
    return root


def _write_nested(root: Path, norms: dict[str, list[float]]) -> Path:
    """Schema do runner do MNIST: métricas aninhadas em declared/dense."""

    root.mkdir(parents=True, exist_ok=True)
    for index in range(len(next(iter(norms.values())))):
        payload = {
            "seed": 7_000_003 + index * 25_000,
            "declared_interval": 1,
            "dense_interval": 1,
            "penalty_strengths": STRENGTHS,
            "arms": {
                arm: {"declared": _metrics(values[index]), "dense": _metrics(values[index])}
                for arm, values in norms.items()
            },
        }
        (root / f"seed_{payload['seed']}.json").write_text(json.dumps(payload))
    return root


def _uniform(value: float) -> list[float]:
    return [value] * EXPECTED_SEEDS


def _all_arms(ewc: float, si: float, mas: float) -> dict[str, list[float]]:
    return {"ewc": _uniform(ewc), "si": _uniform(si), "mas": _uniform(mas)}


def test_g8_marks_an_arm_above_one_as_unpairable(tmp_path: Path) -> None:
    """E > 1 significa lr*E > lr: aumento de learning rate, não controle."""

    report = analyze(_write_flat(tmp_path, _all_arms(1.0013, 1.0124, 1.0194)))
    assert report["pairable_arms"] == []
    for row in report["arms"]:
        assert row["pairable"] is False
        assert row["lr_scale"] is None


def test_g8_keeps_an_arm_below_one(tmp_path: Path) -> None:
    report = analyze(_write_flat(tmp_path, _all_arms(0.9958, 1.0116, 1.0103)))
    assert report["pairable_arms"] == ["ewc"]
    ewc = next(row for row in report["arms"] if row["arm"] == "ewc")
    assert ewc["lr_scale"] == pytest.approx(0.9958)


def test_g8_boundary_exactly_one_is_pairable_but_vacuous(tmp_path: Path) -> None:
    """E == 1 não é > 1, logo passa por G8 — e o controle é idêntico a vanilla.

    Pinar a fronteira importa: um `>` trocado por `>=` mudaria a classificação
    de um arm que medisse exatamente 1.
    """

    report = analyze(_write_flat(tmp_path, _all_arms(1.0, 1.0, 1.0)))
    assert report["pairable_arms"] == ["ewc", "si", "mas"]


def test_lr_scale_is_the_measured_mean_not_a_rounded_value(tmp_path: Path) -> None:
    """O escalar congelado tem de ser o número medido, com todos os dígitos."""

    norms = {"ewc": [0.99 + i * 0.0001 for i in range(EXPECTED_SEEDS)],
             "si": _uniform(1.02), "mas": _uniform(1.02)}
    report = analyze(_write_flat(tmp_path, norms))
    ewc = next(row for row in report["arms"] if row["arm"] == "ewc")
    expected = sum(norms["ewc"]) / EXPECTED_SEEDS
    assert ewc["lr_scale"] == pytest.approx(expected, abs=1e-12)


def test_both_manifest_schemas_agree(tmp_path: Path) -> None:
    """O host MNIST aninha em declared/dense; o CIFAR não. Mesmo veredito."""

    norms = _all_arms(0.9958, 1.0116, 1.0103)
    flat = analyze(_write_flat(tmp_path / "flat", norms))
    nested = analyze(_write_nested(tmp_path / "nested", norms))
    assert flat["pairable_arms"] == nested["pairable_arms"]
    for left, right in zip(flat["arms"], nested["arms"], strict=True):
        assert left["E_mean"] == pytest.approx(right["E_mean"])


def test_arm_metrics_reads_the_declared_series(tmp_path: Path) -> None:
    """Ler `dense` em vez de `declared` mudaria a fonte declarada no protocolo."""

    run = {
        "arms": {
            "ewc": {
                "declared": _metrics(0.5),
                "dense": _metrics(0.9),
            }
        }
    }
    assert arm_metrics(run, "ewc")["norm_ratio"]["mean"] == pytest.approx(0.5)


def test_integrity_rejects_a_sampled_series(tmp_path: Path) -> None:
    """Amostragem não exaustiva já inverteu uma classificação G8 uma vez."""

    root = _write_flat(tmp_path, _all_arms(0.99, 1.02, 1.02))
    path = next(root.glob("seed_*.json"))
    payload = json.loads(path.read_text())
    payload["declared_interval"] = 50
    path.write_text(json.dumps(payload))

    failures = check_integrity([json.loads(p.read_text()) for p in sorted(root.glob("seed_*.json"))])
    assert any("amostragem não exaustiva" in item for item in failures)


def test_integrity_rejects_divergent_penalty_strengths(tmp_path: Path) -> None:
    """Arms rodados a forças diferentes não agregam numa média."""

    root = _write_flat(tmp_path, _all_arms(0.99, 1.02, 1.02))
    path = next(root.glob("seed_*.json"))
    payload = json.loads(path.read_text())
    payload["penalty_strengths"] = {**STRENGTHS, "si_lambda": 1.0}
    path.write_text(json.dumps(payload))

    failures = check_integrity([json.loads(p.read_text()) for p in sorted(root.glob("seed_*.json"))])
    assert any("forças de penalidade divergem" in item for item in failures)


def test_integrity_rejects_a_short_run(tmp_path: Path) -> None:
    norms = {arm: [1.01] * 9 for arm in ("ewc", "si", "mas")}
    root = _write_flat(tmp_path, norms)
    failures = check_integrity([json.loads(p.read_text()) for p in sorted(root.glob("seed_*.json"))])
    assert any("esperava 12 seeds" in item for item in failures)


def test_integrity_rejects_mixed_hosts(tmp_path: Path) -> None:
    root = _write_flat(tmp_path, _all_arms(0.99, 1.02, 1.02))
    path = next(root.glob("seed_*.json"))
    payload = json.loads(path.read_text())
    payload["host"] = "split_mnist"
    path.write_text(json.dumps(payload))

    failures = check_integrity([json.loads(p.read_text()) for p in sorted(root.glob("seed_*.json"))])
    assert any("hosts diferentes" in item for item in failures)


def test_sign_test_is_over_deviations_from_one_not_from_zero(tmp_path: Path) -> None:
    """A hipótese nula é E == 1 (sem efeito), não E == 0.

    Todos os E acima de 1 têm de dar 12 sinais positivos e p no piso; um teste
    sobre os valores brutos daria 12 positivos para QUALQUER E > 0 e nunca
    distinguiria nada.
    """

    runs = [
        {
            "seed": i,
            "declared_interval": 1,
            "penalty_strengths": STRENGTHS,
            "arms": {"ewc": _metrics(0.5)},
        }
        for i in range(EXPECTED_SEEDS)
    ]
    summary = summarize_arm(runs, "ewc")
    assert summary["below_one"] == EXPECTED_SEEDS
    assert summary["above_one"] == 0
    assert summary["p_exact"] == pytest.approx(2 / 2**EXPECTED_SEEDS)


def test_direction_cosine_min_is_the_worst_step_not_the_mean(tmp_path: Path) -> None:
    """A média do cosseno esconde rotação; o mínimo por passo é o que revela.

    No CIFAR o MAS tem cos médio 0,449 e mínimo −0,876: o update chega a
    apontar na direção oposta. Reportar só a média perderia isso.
    """

    runs = [
        {
            "seed": i,
            "declared_interval": 1,
            "penalty_strengths": STRENGTHS,
            "arms": {"mas": _metrics(1.02, cos=0.45, cos_min=-0.88 if i == 3 else 0.2)},
        }
        for i in range(EXPECTED_SEEDS)
    ]
    summary = summarize_arm(runs, "mas")
    assert summary["direction_cosine_min"] == pytest.approx(-0.88)
    assert summary["direction_cosine_mean"] == pytest.approx(0.45)


def test_cli_exit_code_signals_integrity(tmp_path: Path) -> None:
    from scripts.analyze_penalty_pass1 import main

    healthy = _write_flat(tmp_path / "ok", _all_arms(0.99, 1.02, 1.02))
    assert main(["--root", str(healthy)]) == 0

    short = _write_flat(tmp_path / "short", {arm: [1.01] * 3 for arm in ("ewc", "si", "mas")})
    assert main(["--root", str(short)]) == 1
