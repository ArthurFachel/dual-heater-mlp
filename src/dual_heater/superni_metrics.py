"""ROUGE-L e Exact Match — as métricas do SuperNI.

Implementadas aqui em vez de via `rouge_score` por dois motivos:

1. O pacote não está no ambiente, e instalar dependência exige autorização.
2. Uma métrica que decide o endpoint de um experimento deve estar pinada por
   teste no repo, não herdada de um pacote cuja versão pode mudar sob os pés.

**Desvio declarado:** a implementação oficial do SuperNI usa
`rouge_score.rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)`, que
aplica a tokenização do Google (lowercase + remoção de não-alfanuméricos). Esta
implementação faz o mesmo pré-processamento, mas **não foi validada contra o
pacote oficial**. Para uma run confirmatória, validar antes; para calibração de
dispersão, a diferença não muda a decisão.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

__all__ = ["exact_match", "lcs_length", "normalize_answer", "rouge_l", "tokenize"]

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase, troca não-alfanuméricos por espaço, divide.

    Mesma normalização do `rouge_score` com `use_stemmer=False`. Sem stemming:
    o SuperNI oficial também não usa.
    """

    return [token for token in _NON_ALNUM.sub(" ", text.lower()).split() if token]


def lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    """Comprimento da maior subsequência comum.

    Programação dinâmica O(n*m) em tempo, O(min(n,m)) em espaço. As saídas do
    SuperNI são curtas, então não vale complicar.
    """

    if not left or not right:
        return 0
    if len(left) < len(right):
        left, right = right, left
    previous = [0] * (len(right) + 1)
    for item in left:
        current = [0]
        for index, other in enumerate(right):
            if item == other:
                current.append(previous[index] + 1)
            else:
                current.append(max(current[index], previous[index + 1]))
        previous = current
    return previous[-1]


def rouge_l(prediction: str, reference: str) -> float:
    """F-measure do ROUGE-L entre uma predição e uma referência.

    Retorna 0.0 quando qualquer lado fica vazio após a tokenização — inclusive
    quando AMBOS ficam vazios. Dois vazios não são um acerto perfeito: são
    ausência de evidência, e pontuá-los como 1.0 premiaria um modelo que não
    gera nada.
    """

    predicted = tokenize(prediction)
    expected = tokenize(reference)
    if not predicted or not expected:
        return 0.0
    overlap = lcs_length(predicted, expected)
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2.0 * precision * recall / (precision + recall)


def normalize_answer(text: str) -> str:
    """Normalização para Exact Match: tokeniza e rejunta."""

    return " ".join(tokenize(text))


def exact_match(prediction: str, reference: str) -> float:
    """1.0 se as formas normalizadas coincidem, senão 0.0.

    Usado pelas categorias de classificação do SuperNI (entailment,
    coreference, dialogue act, answerability).
    """

    return float(normalize_answer(prediction) == normalize_answer(reference))
