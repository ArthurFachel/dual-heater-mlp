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

Fase 3b: :func:`load_sapt_superni` lê o layout JSON do SAPT
(``CL_Benchmark/SuperNI/<tarefa>/{train,dev,test}.json``, chaves
``Definition`` e ``Instances``) nas ordens publicadas em
``gen_script_superni_llama.py`` (:data:`SAPT_ORDERS`). Ali a sequência É a do
SAPT, então a comparação externa é possível no protocolo, não nos números
(o orçamento de treino é nosso).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "SAPT_ORDERS",
    "SAPT_SPLITS",
    "SuperNIExample",
    "SuperNITask",
    "build_prompt",
    "group_examples_by_task",
    "load_sapt_superni",
    "load_sapt_task",
    "truncate_to_budget",
]

#: Ordens 1 e 2 do SAPT (circle-hit/SAPT@52a52b9, gen_script_superni_llama.py).
SAPT_ORDERS: dict[int, tuple[str, ...]] = {
    1: (
        "task1572_samsum_summary",
        "task363_sst2_polarity_classification",
        "task1290_xsum_summarization",
        "task181_outcome_extraction",
        "task002_quoref_answer_generation",
        "task1510_evalution_relation_extraction",
        "task639_multi_woz_user_utterance_generation",
        "task1729_personachat_generate_next",
        "task073_commonsenseqa_answer_generation",
        "task1590_diplomacy_text_generation",
        "task748_glucose_reverse_cause_event_detection",
        "task511_reddit_tifu_long_text_summarization",
        "task591_sciq_answer_generation",
        "task1687_sentiment140_classification",
        "task875_emotion_classification",
    ),
    2: (
        "task748_glucose_reverse_cause_event_detection",
        "task073_commonsenseqa_answer_generation",
        "task1590_diplomacy_text_generation",
        "task639_multi_woz_user_utterance_generation",
        "task1572_samsum_summary",
        "task1687_sentiment140_classification",
        "task591_sciq_answer_generation",
        "task363_sst2_polarity_classification",
        "task1510_evalution_relation_extraction",
        "task1729_personachat_generate_next",
        "task181_outcome_extraction",
        "task511_reddit_tifu_long_text_summarization",
        "task002_quoref_answer_generation",
        "task1290_xsum_summarization",
        "task875_emotion_classification",
    ),
}

#: Splits que a avaliação pode ler. ``train`` nunca é avaliação.
SAPT_SPLITS: tuple[str, ...] = ("dev", "test")


@dataclass(frozen=True)
class SuperNIExample:
    """Um par instrução/resposta já formatado para o host."""

    task_name: str
    prompt: str
    target: str
    #: Todas as referências aceitas (o SuperNI pontua o máximo sobre elas).
    #: Vazio no loader antigo; quem pontua usa ``(target,)`` nesse caso.
    references: tuple[str, ...] = ()
    example_id: str = ""


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


def _references(output: Any) -> tuple[str, ...]:
    if isinstance(output, str):
        values = [output]
    elif isinstance(output, (list, tuple)):
        values = [str(value) for value in output]
    else:
        raise TypeError(f"campo 'output' com tipo inesperado: {type(output).__name__}")
    references = tuple(value.strip() for value in values)
    if not references:
        raise ValueError("instância sem referência")
    return references


def _read_split(
    directory: Path, split: str, *, task_name: str, limit: int | None
) -> tuple[SuperNIExample, ...]:
    path = directory / f"{split}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    definition = payload["Definition"]
    if isinstance(definition, list):
        # O SuperNI oficial usa Definition[0]; as tarefas do SAPT têm uma só.
        definition = definition[0]
    instances = payload["Instances"]
    if limit is not None:
        if limit < 0:
            raise ValueError("caps de exemplos não podem ser negativos")
        instances = instances[:limit]
    examples = []
    for instance in instances:
        references = _references(instance["output"])
        examples.append(
            SuperNIExample(
                task_name=task_name,
                prompt=build_prompt(definition=definition, inputs=instance["input"]),
                # Treino usa a PRIMEIRA referência, deterministicamente.
                target=references[0],
                references=references,
                example_id=str(instance.get("id", "")),
            )
        )
    return tuple(examples)


def load_sapt_task(
    root: str | Path,
    name: str,
    *,
    eval_split: str = "test",
    train_per_task: int | None = None,
    eval_per_task: int | None = None,
) -> SuperNITask:
    """Uma tarefa do layout SAPT: treino de ``train.json``, avaliação de ``eval_split``.

    Os caps pegam os PRIMEIROS exemplos de cada arquivo (ordem do arquivo,
    determinística). Um id presente no treino e na avaliação é vazamento e
    levanta ``ValueError`` em vez de inflar a retenção em silêncio.
    """

    if eval_split not in SAPT_SPLITS:
        raise ValueError(f"eval_split deve ser um de {SAPT_SPLITS}")
    directory = Path(root) / name
    if not directory.is_dir():
        raise FileNotFoundError(f"tarefa SAPT ausente: {directory}")
    train = _read_split(directory, "train", task_name=name, limit=train_per_task)
    evaluation = _read_split(directory, eval_split, task_name=name, limit=eval_per_task)
    train_ids = {example.example_id for example in train if example.example_id}
    leaked = sorted(
        example.example_id for example in evaluation if example.example_id in train_ids
    )
    if leaked:
        raise ValueError(
            f"{name}: vazamento treino/avaliação, ids repetidos {leaked[:5]}"
        )
    return SuperNITask(name=name, train=train, evaluation=evaluation)


def load_sapt_superni(
    root: str | Path,
    *,
    order: int = 1,
    task_names: Sequence[str] | None = None,
    tasks: int | None = None,
    eval_split: str = "test",
    train_per_task: int | None = None,
    eval_per_task: int | None = None,
) -> list[SuperNITask]:
    """A sequência SuperNI-15 do SAPT, na ordem declarada.

    ``task_names`` substitui a ordem publicada (testes, extensões); ``tasks``
    corta para as primeiras N tarefas da ordem (smoke/calibração).
    """

    if task_names is None:
        if order not in SAPT_ORDERS:
            raise ValueError(f"ordem SAPT desconhecida: {order}")
        names = list(SAPT_ORDERS[order])
    else:
        names = list(task_names)
    if len(set(names)) != len(names):
        raise ValueError("a sequência não pode repetir tarefas")
    if tasks is not None:
        if tasks < 1:
            raise ValueError("tasks deve ser >= 1")
        names = names[:tasks]
    return [
        load_sapt_task(
            root,
            name,
            eval_split=eval_split,
            train_per_task=train_per_task,
            eval_per_task=eval_per_task,
        )
        for name in names
    ]
