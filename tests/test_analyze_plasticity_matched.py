"""Testes do agregador do contraste de plasticidade pareada.

O que precisa morder: o pareamento por seed (não por posição na lista), a
direção do contraste, a família de Holm com tamanho exatamente 2, e a recusa
de agregar uma run que reprovou integridade.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.analyze_plasticity_matched import (
    ALPHA_PRIMARY,
    analyze,
    main,
    paired_contrast,
)
from scripts.verify_plasticity_matched import TARGET_E


def _matrix(n: int, value: float) -> list[list[float | None]]:
    return [[value if j <= i else None for j in range(n)] for i in range(n)]


def _arm(method: str, *, forgetting: float, faa: float, tokens: int, **over) -> dict:
    record = {
        "method": method,
        "average_forgetting": forgetting,
        "final_average_accuracy": faa,
        "train_tokens": tokens,
        "surface_plasticity": 1.0 if method == "vanilla" else TARGET_E,
        "learning_rate_scale": TARGET_E if method == "lr_control" else 1.0,
        "trainable_parameters": 4_214_016 if method == "frozen_a_control" else 6_769_920,
        "accuracy_matrix": _matrix(3, 0.8),
    }
    record.update(over)
    return record


def _write(tmp_path: Path, per_seed: list[tuple[float, float]]) -> Path:
    """per_seed: lista de (forgetting_frozen, forgetting_lr) por seed."""

    for index, (frozen, lr) in enumerate(per_seed):
        seed_dir = tmp_path / f"seed_{6_000_003 + index}"
        seed_dir.mkdir(parents=True)
        tokens = 200_000 + index
        arms = [
            _arm("vanilla", forgetting=0.40, faa=0.60, tokens=tokens),
            _arm("lr_control", forgetting=lr, faa=0.55, tokens=tokens),
            _arm("frozen_a_control", forgetting=frozen, faa=0.58, tokens=tokens),
        ]
        (seed_dir / "manifest.json").write_text(json.dumps({"results": arms}))
    return tmp_path


def test_contrast_is_treatment_minus_comparator(tmp_path: Path) -> None:
    """Direção: frozen − lr. Invertê-la troca o sinal da conclusão."""

    root = _write(tmp_path, [(0.30, 0.50)] * 10)
    contrast = paired_contrast(
        {s: arms for s, arms in _load(root).items()}, "average_forgetting"
    )
    assert contrast["mean"] == pytest.approx(-0.20)
    assert contrast["negative"] == 10


def _load(root: Path) -> dict:
    from scripts.analyze_plasticity_matched import load_seed_results

    return load_seed_results(root)


def test_pairing_is_by_seed_not_by_position(tmp_path: Path) -> None:
    """Valores distintos por seed: parear errado move a média, não a contagem."""

    # frozen sobe por seed, lr desce. A diferença correta varia de -0.4 a +0.4;
    # a média pareada é 0. Um pareamento por posição embaralhada daria outro
    # número, então a asserção é sobre a MÉDIA, não sobre n.
    frozen = [0.10 * i for i in range(10)]
    lr = [0.10 * (9 - i) for i in range(10)]
    root = _write(tmp_path, list(zip(frozen, lr, strict=True)))
    contrast = paired_contrast(_load(root), "average_forgetting")
    assert contrast["n"] == 10
    assert contrast["mean"] == pytest.approx(0.0, abs=1e-12)
    expected = [f - l for f, l in zip(frozen, lr, strict=True)]
    assert sorted(contrast["per_seed"].values()) == pytest.approx(sorted(expected))


def test_family_has_exactly_two_members(tmp_path: Path) -> None:
    """P6 congelou a família em 2. Crescer a família invalida a correção."""

    report = analyze(_write(tmp_path, [(0.30, 0.50)] * 10))
    assert len(report["family"]) == 2
    assert [c["role"] for c in report["family"]] == ["primário", "secundário"]


def test_unanimous_effect_hits_the_holm_floor(tmp_path: Path) -> None:
    report = analyze(_write(tmp_path, [(0.30, 0.50)] * 10))
    primary = report["family"][0]
    assert primary["p_raw"] == pytest.approx(2 / 1024)
    # Holm sobre 2: o menor p é multiplicado por 2.
    assert primary["p_holm"] == pytest.approx(4 / 1024)
    assert report["verdict"]["significant"]
    assert report["verdict"]["branch"] == "A"


def test_tie_is_branch_b_not_an_error(tmp_path: Path) -> None:
    """Empate 5/5 é desfecho declarado (galho B), não falha do analisador."""

    per_seed = [(0.40, 0.30)] * 5 + [(0.30, 0.40)] * 5
    report = analyze(_write(tmp_path, per_seed))
    primary = report["family"][0]
    assert primary["p_raw"] == pytest.approx(1.0)
    assert not report["verdict"]["significant"]
    assert report["verdict"]["branch"] == "B"


def test_significant_but_wrong_direction_is_not_success(tmp_path: Path) -> None:
    """P9 é conjuntivo: p pequeno com mediana positiva NÃO é o galho A."""

    report = analyze(_write(tmp_path, [(0.50, 0.30)] * 10))
    primary = report["family"][0]
    assert primary["p_holm"] < ALPHA_PRIMARY
    assert not report["verdict"]["median_negative"]
    assert not report["verdict"]["significant"]


def test_vanilla_stays_out_of_the_family(tmp_path: Path) -> None:
    """P10: vanilla é descritivo. Se entrasse em Holm, m seria 3+."""

    report = analyze(_write(tmp_path, [(0.30, 0.50)] * 10))
    assert all("vanilla" not in c["contrast"] for c in report["family"])
    assert any(row["arm"] == "vanilla" for row in report["arms"])


def test_arm_table_reports_absolute_values(tmp_path: Path) -> None:
    """Contraste sem valores absolutos ao lado não é reportável."""

    report = analyze(_write(tmp_path, [(0.30, 0.50)] * 10))
    rows = {row["arm"]: row for row in report["arms"]}
    assert set(rows) == {"vanilla", "lr_control", "frozen_a_control"}
    assert rows["frozen_a_control"]["forgetting_mean"] == pytest.approx(0.30)
    assert rows["lr_control"]["forgetting_mean"] == pytest.approx(0.50)
    for row in rows.values():
        assert "acquisition_last_mean" in row
        assert "retention_first_mean" in row


def test_cli_refuses_a_run_that_failed_integrity(tmp_path: Path, capsys) -> None:
    root = _write(tmp_path, [(0.30, 0.50)] * 9)  # 9 seeds: H1 falha
    assert main(["--root", str(root)]) == 1
    assert "INTEGRIDADE FALHOU" in capsys.readouterr().out


def test_cli_runs_on_a_healthy_tree(tmp_path: Path, capsys) -> None:
    root = _write(tmp_path, [(0.30, 0.50)] * 10)
    assert main(["--root", str(root)]) == 0
    out = capsys.readouterr().out
    assert "VEREDITO P9" in out
    assert "frozen_a_control" in out
