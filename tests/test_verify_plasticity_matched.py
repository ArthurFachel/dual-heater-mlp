"""Testes do verificador de integridade da run de plasticidade pareada.

Um verificador que nunca reprova nada certifica qualquer run. Cada checagem
H1-H8 tem aqui um caso que a faz falhar; se um desses testes passa com a
checagem removida, a checagem não existe de fato.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.verify_plasticity_matched import TARGET_E, check_plasticity_matched


def _arm(
    method: str,
    *,
    surface_plasticity: float,
    learning_rate_scale: float,
    train_tokens: int = 200_000,
    trainable_parameters: int = 6_769_920,
) -> dict[str, object]:
    return {
        "method": method,
        "surface_plasticity": surface_plasticity,
        "learning_rate_scale": learning_rate_scale,
        "train_tokens": train_tokens,
        "trainable_parameters": trainable_parameters,
        "average_forgetting": 0.4,
        "final_average_accuracy": 0.6,
    }


def _healthy_arms(tokens: int = 200_000) -> list[dict[str, object]]:
    return [
        _arm("vanilla", surface_plasticity=1.0, learning_rate_scale=1.0, train_tokens=tokens),
        _arm(
            "lr_control",
            surface_plasticity=TARGET_E,
            learning_rate_scale=TARGET_E,
            train_tokens=tokens,
        ),
        _arm(
            "frozen_a_control",
            surface_plasticity=TARGET_E,
            learning_rate_scale=1.0,
            train_tokens=tokens,
            trainable_parameters=4_214_016,
        ),
    ]


def _write_run(root: Path, *, seeds: int = 10, mutate=None) -> Path:
    for index in range(seeds):
        arms = _healthy_arms(tokens=200_000 + index)
        if mutate is not None:
            arms = mutate(arms, index)
        seed_dir = root / f"seed_{6_000_003 + index}"
        seed_dir.mkdir(parents=True)
        (seed_dir / "manifest.json").write_text(json.dumps({"results": arms}))
    return root


def test_healthy_run_passes(tmp_path: Path) -> None:
    assert check_plasticity_matched(_write_run(tmp_path)) == []


def test_h1_missing_seeds(tmp_path: Path) -> None:
    failures = check_plasticity_matched(_write_run(tmp_path, seeds=9))
    assert any(item.startswith("H1") for item in failures)


def test_h2_missing_manifest(tmp_path: Path) -> None:
    _write_run(tmp_path)
    (tmp_path / "seed_6000003" / "manifest.json").unlink()
    failures = check_plasticity_matched(tmp_path)
    assert any(item.startswith("H2") for item in failures)


def test_h3_undeclared_arm(tmp_path: Path) -> None:
    def mutate(arms, index):
        if index == 0:
            arms = [*arms, _arm("exact", surface_plasticity=TARGET_E, learning_rate_scale=1.0)]
        return arms

    failures = check_plasticity_matched(_write_run(tmp_path, mutate=mutate))
    assert any(item.startswith("H3") for item in failures)


def test_h4_arms_not_matched_to_each_other(tmp_path: Path) -> None:
    def mutate(arms, index):
        if index == 3:
            arms[1] = {**arms[1], "surface_plasticity": 0.70}
        return arms

    failures = check_plasticity_matched(_write_run(tmp_path, mutate=mutate))
    assert any(item.startswith("H4") for item in failures)


def test_h5_matched_but_off_target(tmp_path: Path) -> None:
    """Os dois braços iguais entre si mas no E errado: H4 passa, H5 tem de pegar."""

    def mutate(arms, index):
        arms[1] = {**arms[1], "surface_plasticity": 0.5, "learning_rate_scale": TARGET_E}
        arms[2] = {**arms[2], "surface_plasticity": 0.5}
        return arms

    failures = check_plasticity_matched(_write_run(tmp_path, mutate=mutate))
    assert not any(item.startswith("H4") for item in failures)
    assert any(item.startswith("H5") for item in failures)


def test_h6_vanilla_is_not_the_reference(tmp_path: Path) -> None:
    def mutate(arms, index):
        arms[0] = {**arms[0], "surface_plasticity": 0.99}
        return arms

    failures = check_plasticity_matched(_write_run(tmp_path, mutate=mutate))
    assert any(item.startswith("H6") for item in failures)


def test_h7_tokens_unpaired_within_a_seed(tmp_path: Path) -> None:
    def mutate(arms, index):
        if index == 5:
            arms[2] = {**arms[2], "train_tokens": 1}
        return arms

    failures = check_plasticity_matched(_write_run(tmp_path, mutate=mutate))
    assert any(item.startswith("H7") for item in failures)


def test_h8_treatment_never_reached_the_control(tmp_path: Path) -> None:
    """O bug histórico: learning_rate_scale não chega no lr_control."""

    def mutate(arms, index):
        arms[1] = {**arms[1], "learning_rate_scale": 1.0}
        return arms

    failures = check_plasticity_matched(_write_run(tmp_path, mutate=mutate))
    assert any(item.startswith("H8") for item in failures)


def test_h8_frozen_arm_must_not_also_be_lr_scaled(tmp_path: Path) -> None:
    def mutate(arms, index):
        arms[2] = {**arms[2], "learning_rate_scale": TARGET_E}
        return arms

    failures = check_plasticity_matched(_write_run(tmp_path, mutate=mutate))
    assert any(item.startswith("H8") for item in failures)


def test_h8_equal_trainable_counts_is_the_r3_fingerprint(tmp_path: Path) -> None:
    """E igual com superfície treinável igual é a assinatura do bug R3."""

    def mutate(arms, index):
        arms[2] = {**arms[2], "trainable_parameters": arms[1]["trainable_parameters"]}
        return arms

    failures = check_plasticity_matched(_write_run(tmp_path, mutate=mutate))
    assert any(item.startswith("H8") for item in failures)


def test_empty_root_fails_rather_than_passing_vacuously(tmp_path: Path) -> None:
    failures = check_plasticity_matched(tmp_path)
    assert failures, "um diretório vazio não pode ser reportado como run íntegra"


@pytest.mark.parametrize("seeds", [1, 9, 11])
def test_seed_count_is_exact_not_minimum(tmp_path: Path, seeds: int) -> None:
    failures = check_plasticity_matched(_write_run(tmp_path, seeds=seeds))
    assert any(item.startswith("H1") for item in failures)
