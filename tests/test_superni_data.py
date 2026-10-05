"""Loader do SuperNI: ordem, disjunção e truncamento.

Os três modos de falha que este arquivo vigia, todos silenciosos:

1. Ordem das tarefas vinda da iteração do dataset em vez do pedido — em CL a
   ordem É a variável independente.
2. Treino e avaliação se sobrepondo — infla retenção sem nada denunciar.
3. Truncamento cortando a definição da tarefa em vez da passagem.
"""

from __future__ import annotations

import pytest

from dual_heater.superni_data import (
    build_prompt,
    group_examples_by_task,
    truncate_to_budget,
)


def _row(task: str, index: int, *, definition: str = "Do the thing.") -> dict:
    return {
        "task_name": task,
        "id": f"{task}-{index}",
        "definition": definition,
        "inputs": f"input {index}",
        "targets": f"target {index}",
    }


def test_prompt_puts_definition_before_input() -> None:
    prompt = build_prompt(definition="Classify.", inputs="hello")
    assert prompt.index("Classify.") < prompt.index("hello")
    assert prompt.endswith("Output:")


def test_prompt_strips_surrounding_whitespace() -> None:
    assert build_prompt(definition="  A.  ", inputs="  b  ") == "A.\n\nInput: b\nOutput:"


def test_truncate_keeps_the_beginning() -> None:
    """Cortar pelo início removeria a instrução da tarefa."""
    assert truncate_to_budget("abcdef", max_characters=3) == "abc"


def test_truncate_is_a_noop_below_the_budget() -> None:
    assert truncate_to_budget("abc", max_characters=10) == "abc"


def test_truncate_rejects_a_nonpositive_budget() -> None:
    with pytest.raises(ValueError):
        truncate_to_budget("abc", max_characters=0)


def test_tasks_come_back_in_the_requested_order() -> None:
    """A ordem do dataset é B,A; a pedida é A,B. Vence a pedida.

    As contagens são DESIGUAIS de propósito (B tem 9 linhas, A tem 3): com
    contagens iguais, uma ordenação por frequência produziria a mesma saída e o
    teste passaria sem testar nada.
    """
    rows = [_row("B", i) for i in range(9)] + [_row("A", i) for i in range(3)]
    tasks = group_examples_by_task(
        rows, task_names=["A", "B"], train_per_task=2, eval_per_task=1
    )
    assert [task.name for task in tasks] == ["A", "B"]


def test_requested_order_wins_over_dataset_order_for_three_tasks() -> None:
    """Três tarefas cujos buckets FINAIS têm tamanhos diferentes.

    O detalhe que torna o teste capaz de detectar: `B` tem só 1 linha, abaixo
    de `needed`, então os buckets terminam com 3/3/1 em vez de empatados. Com
    buckets empatados (todos truncados em `needed`) uma ordenação por
    frequência é estável e devolve a ordem de inserção, que é a ordem pedida —
    e o teste passaria sem distinguir nada.
    """
    rows = (
        [_row("C", i) for i in range(9)]
        + [_row("A", i) for i in range(6)]
        + [_row("B", 0)]
    )
    tasks = group_examples_by_task(
        rows, task_names=["B", "C", "A"], train_per_task=2, eval_per_task=1
    )
    assert [task.name for task in tasks] == ["B", "C", "A"]
    # O bucket curto continua curto: a ordem não foi comprada truncando dados.
    assert len(tasks[0].train) == 1


def test_train_and_eval_splits_are_disjoint() -> None:
    rows = [_row("A", i) for i in range(10)]
    task = group_examples_by_task(
        rows, task_names=["A"], train_per_task=3, eval_per_task=2
    )[0]
    train_ids = {example.prompt for example in task.train}
    eval_ids = {example.prompt for example in task.evaluation}
    assert len(task.train) == 3
    assert len(task.evaluation) == 2
    assert train_ids.isdisjoint(eval_ids)


def test_examples_from_other_tasks_are_ignored() -> None:
    rows = [_row("A", 0), _row("Z", 0), _row("A", 1)]
    task = group_examples_by_task(
        rows, task_names=["A"], train_per_task=2, eval_per_task=0
    )[0]
    assert all(example.task_name == "A" for example in task.train)
    assert len(task.train) == 2


def test_collection_stops_at_the_requested_count() -> None:
    """Não carregar o dataset inteiro na memória por tarefa."""
    rows = [_row("A", i) for i in range(1000)]
    task = group_examples_by_task(
        rows, task_names=["A"], train_per_task=2, eval_per_task=1
    )[0]
    assert len(task.train) + len(task.evaluation) == 3


def test_a_task_with_too_few_examples_comes_back_short() -> None:
    """Devolve o que há; quem chama decide se rejeita."""
    rows = [_row("A", 0)]
    task = group_examples_by_task(
        rows, task_names=["A"], train_per_task=5, eval_per_task=5
    )[0]
    assert len(task.train) == 1
    assert len(task.evaluation) == 0


def test_a_missing_task_still_appears_empty() -> None:
    """Sumir com a tarefa mudaria o comprimento da sequência em silêncio."""
    tasks = group_examples_by_task(
        [_row("A", 0)], task_names=["A", "MISSING"], train_per_task=1, eval_per_task=0
    )
    assert [task.name for task in tasks] == ["A", "MISSING"]
    assert tasks[1].train == ()


def test_long_prompts_are_truncated_to_the_budget() -> None:
    rows = [_row("A", 0, definition="x" * 10_000)]
    task = group_examples_by_task(
        rows,
        task_names=["A"],
        train_per_task=1,
        eval_per_task=0,
        max_prompt_characters=100,
    )[0]
    assert len(task.train[0].prompt) == 100


def test_targets_are_stripped() -> None:
    row = _row("A", 0)
    row["targets"] = "  spaced  "
    task = group_examples_by_task(
        [row], task_names=["A"], train_per_task=1, eval_per_task=0
    )[0]
    assert task.train[0].target == "spaced"


def test_negative_counts_are_rejected() -> None:
    with pytest.raises(ValueError):
        group_examples_by_task([], task_names=["A"], train_per_task=-1, eval_per_task=1)
