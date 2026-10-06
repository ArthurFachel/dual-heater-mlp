"""Sequência de tarefas do SuperNI para continual learning.

**Aviso de escopo, que precisa estar no artigo se isto virar resultado:** o
SuperNI NÃO é, na origem, um benchmark de continual learning. É um benchmark de
generalização cross-task — treina num subconjunto e avalia em tarefas não
vistas. A sequência de CL montada AQUI é nossa e nenhum número publicado é
diretamente comparável a ela.

**Correção (05/10):** uma versão anterior deste aviso dizia que "não existe
sequência canônica de tarefas para CL" no SuperNI. Está errado. O SAPT (ACL
2024, ``circle-hit/SAPT``, ``CL_Benchmark/SuperNI/``) fixa 15 tarefas do
SuperNI com ordens declaradas, e o CITB (Findings EMNLP 2023) fixa os streams
InstrDialog/InstrDialog++. Comparação externa é possível contra esses
protocolos, não contra esta sequência.

Consequência prática: esta sequência serve para medir **dispersão e custo**
(calibração), não para alegar "re-avaliamos ganhos publicados".

Dataset: `Muennighoff/natural-instructions`, campos `task_name`, `id`,
`definition`, `inputs`, `targets`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "SuperNIExample",
    "SuperNITask",
    "build_prompt",
    "group_examples_by_task",
    "truncate_to_budget",
]


@dataclass(frozen=True)
class SuperNIExample:
    """Um par instrução/resposta já formatado para o host."""

    task_name: str
    prompt: str
    target: str


@dataclass(frozen=True)
class SuperNITask:
    """Uma tarefa da sequência, com seus splits de treino e avaliação."""

    name: str
    train: tuple[SuperNIExample, ...] = field(default_factory=tuple)
    evaluation: tuple[SuperNIExample, ...] = field(default_factory=tuple)


def build_prompt(*, definition: str, inputs: str) -> str:
    """Monta o prompt no formato do SuperNI oficial (zero-shot, sem exemplos).

    A ordem é definição primeiro, entrada depois: é como o Tk-Instruct foi
    treinado, e inverter muda a tarefa sem avisar.
    """

    return f"{definition.strip()}\n\nInput: {inputs.strip()}\nOutput:"


def truncate_to_budget(text: str, *, max_characters: int) -> str:
    """Corta pelo FIM, preservando o começo do texto.

    O corte é por caracteres, não tokens, de propósito: serve para derrubar
    passagens gigantes antes da tokenização, barato e determinístico. O
    truncamento real por tokens é do tokenizer, com `max_length`.

    Preservar o começo importa porque a definição da tarefa vem primeiro; cortar
    pelo início removeria a instrução e deixaria o modelo adivinhando.
    """

    if max_characters <= 0:
        raise ValueError("max_characters deve ser positivo")
    if len(text) <= max_characters:
        return text
    return text[:max_characters]


def group_examples_by_task(
    rows: Iterable[dict[str, Any]],
    *,
    task_names: Sequence[str],
    train_per_task: int,
    eval_per_task: int,
    max_prompt_characters: int = 4000,
) -> list[SuperNITask]:
    """Agrupa linhas cruas do dataset nas tarefas pedidas, na ORDEM pedida.

    A ordem de `task_names` é a ordem da sequência de CL e é preservada
    exatamente: em continual learning a ordem das tarefas é a variável
    independente, então deixá-la depender da ordem de iteração do dataset
    tornaria a run irreprodutível sem nada no artefato denunciando.

    Os primeiros `train_per_task` exemplos de cada tarefa vão para treino e os
    `eval_per_task` seguintes para avaliação — **disjuntos por construção**.
    Tarefas sem exemplos suficientes são devolvidas com o que houver; a decisão
    de rejeitar ou não é de quem chama.
    """

    if train_per_task < 0 or eval_per_task < 0:
        raise ValueError("contagens de exemplos não podem ser negativas")

    wanted = set(task_names)
    buckets: dict[str, list[SuperNIExample]] = {name: [] for name in task_names}
    needed = train_per_task + eval_per_task

    for row in rows:
        name = row["task_name"]
        if name not in wanted or len(buckets[name]) >= needed:
            continue
        prompt = truncate_to_budget(
            build_prompt(definition=row["definition"], inputs=row["inputs"]),
            max_characters=max_prompt_characters,
        )
        buckets[name].append(
            SuperNIExample(
                task_name=name, prompt=prompt, target=str(row["targets"]).strip()
            )
        )

    return [
        SuperNITask(
            name=name,
            train=tuple(buckets[name][:train_per_task]),
            evaluation=tuple(buckets[name][train_per_task:needed]),
        )
        for name in task_names
    ]
