"""ROUGE-L e Exact Match pinados.

A métrica decide se o experimento do SuperNI tem dispersão. Se ela estiver
errada, o endpoint é lixo e nada rio abaixo denuncia — o mesmo padrão do R-E,
onde a métrica estava calculada errada e cinco gates passaram.
"""

from __future__ import annotations

import pytest

from dual_heater.superni_metrics import (
    exact_match,
    lcs_length,
    normalize_answer,
    rouge_l,
    tokenize,
)


def test_tokenize_lowercases_and_strips_punctuation() -> None:
    assert tokenize("Hello, World!") == ["hello", "world"]


def test_tokenize_returns_empty_for_punctuation_only() -> None:
    assert tokenize("!!! ... ???") == []


def test_lcs_is_subsequence_not_substring() -> None:
    """A diferença que separa ROUGE-L de overlap contíguo."""
    assert lcs_length(["a", "b", "c"], ["a", "x", "c"]) == 2


def test_lcs_respects_order() -> None:
    """['b','a'] contra ['a','b'] tem LCS 1, não 2 — ordem importa."""
    assert lcs_length(["b", "a"], ["a", "b"]) == 1


def test_lcs_is_symmetric() -> None:
    left, right = ["a", "b", "c", "d"], ["b", "d"]
    assert lcs_length(left, right) == lcs_length(right, left)


def test_rouge_l_is_one_for_identical_text() -> None:
    assert rouge_l("the cat sat", "the cat sat") == pytest.approx(1.0)


def test_rouge_l_is_zero_for_disjoint_text() -> None:
    assert rouge_l("alpha beta", "gamma delta") == pytest.approx(0.0)


def test_rouge_l_f_measure_balances_precision_and_recall() -> None:
    """LCS=2, pred=4 tokens, ref=2 tokens.

    precision = 2/4 = 0.5, recall = 2/2 = 1.0, F = 2*0.5*1/(1.5) = 0.6667.
    """
    assert rouge_l("the cat sat down", "the cat") == pytest.approx(2 / 3, rel=1e-6)


def test_rouge_l_penalizes_verbosity() -> None:
    """Despejar texto deve pontuar MENOS que responder exato.

    Sem isto, um modelo degenerado que gera parágrafos ganharia recall de graça.
    """
    concise = rouge_l("paris", "paris")
    verbose = rouge_l("well i think the answer is probably paris or similar", "paris")
    assert verbose < concise


def test_rouge_l_is_zero_when_both_sides_are_empty() -> None:
    """Dois vazios NÃO são acerto perfeito.

    Premiar isso com 1.0 daria nota máxima a um modelo que não gera nada — o
    modo de falha mais provável de um 0.5B em tarefas não vistas.
    """
    assert rouge_l("", "") == pytest.approx(0.0)
    assert rouge_l("!!!", "???") == pytest.approx(0.0)


def test_rouge_l_is_zero_when_prediction_is_empty() -> None:
    assert rouge_l("", "the answer") == pytest.approx(0.0)


def test_rouge_l_ignores_case_and_punctuation() -> None:
    assert rouge_l("The Cat, sat.", "the cat sat") == pytest.approx(1.0)


def test_exact_match_requires_full_equality() -> None:
    assert exact_match("yes", "yes") == pytest.approx(1.0)
    assert exact_match("yes", "no") == pytest.approx(0.0)


def test_exact_match_normalizes_before_comparing() -> None:
    assert exact_match("Yes.", "yes") == pytest.approx(1.0)


def test_exact_match_is_stricter_than_rouge() -> None:
    """Resposta parcial: ROUGE dá crédito, EM não. É o ponto de ter os dois."""
    prediction, reference = "the cat", "the cat sat"
    assert rouge_l(prediction, reference) > 0.0
    assert exact_match(prediction, reference) == pytest.approx(0.0)


def test_normalize_answer_collapses_whitespace() -> None:
    assert normalize_answer("  a   b  ") == "a b"
