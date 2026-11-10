"""Loader do layout JSON do SAPT (CL_Benchmark/SuperNI/<tarefa>/{train,dev,test}.json).

O que este arquivo vigia, todos modos de falha silenciosos:

1. Ordem da sequência diferente da declarada no SAPT (a ordem é a variável
   independente de uma run de CL).
2. Avaliação lendo o split errado ou se sobrepondo ao treino (infla retenção).
3. Referências múltiplas perdidas: o ROUGE-L do SuperNI é o máximo sobre as
   referências, e ficar só com a primeira derruba a pontuação sem aviso.
4. Caps de smoke/calibração aplicados no split errado.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dual_heater.superni_data import (
    SAPT_ORDERS,
    SuperNIExample,
    build_prompt,
    load_sapt_superni,
    load_sapt_task,
)


def _write_task(
    root: Path,
    name: str,
    *,
    sizes: dict[str, int],
    definition: str = "Answer the thing.",
    multi_reference: bool = False,
) -> None:
    directory = root / name
    directory.mkdir(parents=True)
    for split, count in sizes.items():
        instances = []
        for index in range(count):
            output: list[str] | str
            if multi_reference:
                output = [f"{split} answer {index}", f"alt {index}"]
            else:
                output = [f"{split} answer {index}"]
            instances.append(
                {
                    "id": f"{name}-{split}-{index}",
                    "input": f"  {split} input {index}  ",
                    "output": output,
                }
            )
        payload = {"Definition": [definition], "Instances": instances}
        (directory / f"{split}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_sapt_orders_match_the_published_script() -> None:
    order_1 = SAPT_ORDERS[1]
    order_2 = SAPT_ORDERS[2]
    assert len(order_1) == 15 and len(set(order_1)) == 15
    assert sorted(order_1) == sorted(order_2)
    assert order_1[:3] == (
        "task1572_samsum_summary",
        "task363_sst2_polarity_classification",
        "task1290_xsum_summarization",
    )
    assert order_1[-1] == "task875_emotion_classification"
    assert order_2[0] == "task748_glucose_reverse_cause_event_detection"
    assert order_2[-2:] == (
        "task1290_xsum_summarization",
        "task875_emotion_classification",
    )


def test_task_loads_definition_first_prompt_and_stripped_target(tmp_path: Path) -> None:
    _write_task(tmp_path, "tA", sizes={"train": 3, "dev": 2, "test": 2})
    task = load_sapt_task(tmp_path, "tA")
    first = task.train[0]
    assert isinstance(first, SuperNIExample)
    assert first.task_name == "tA"
    assert first.prompt == build_prompt(definition="Answer the thing.", inputs="train input 0")
    assert first.target == "train answer 0"
    assert first.example_id == "tA-train-0"


def test_eval_split_is_test_by_default_and_dev_on_request(tmp_path: Path) -> None:
    _write_task(tmp_path, "tA", sizes={"train": 3, "dev": 2, "test": 4})
    default = load_sapt_task(tmp_path, "tA")
    dev = load_sapt_task(tmp_path, "tA", eval_split="dev")
    assert len(default.evaluation) == 4
    assert all("-test-" in example.example_id for example in default.evaluation)
    assert len(dev.evaluation) == 2
    assert all("-dev-" in example.example_id for example in dev.evaluation)
    with pytest.raises(ValueError):
        load_sapt_task(tmp_path, "tA", eval_split="train")


def test_all_references_are_kept_for_evaluation(tmp_path: Path) -> None:
    _write_task(tmp_path, "tA", sizes={"train": 1, "test": 1}, multi_reference=True)
    task = load_sapt_task(tmp_path, "tA")
    example = task.evaluation[0]
    assert example.target == "test answer 0"
    assert example.references == ("test answer 0", "alt 0")


def test_string_output_parses_and_shared_ids_are_leakage(tmp_path: Path) -> None:
    directory = tmp_path / "tS"
    directory.mkdir()
    payload = {
        "Definition": ["Do it."],
        "Instances": [{"id": "x0", "input": "in", "output": "  out  "}],
    }
    for split in ("train", "test"):
        (directory / f"{split}.json").write_text(json.dumps(payload), encoding="utf-8")
    other = {
        "Definition": ["Do it."],
        "Instances": [{"id": "y0", "input": "in", "output": "  other  "}],
    }
    (directory / "dev.json").write_text(json.dumps(other), encoding="utf-8")
    task = load_sapt_task(tmp_path, "tS", eval_split="dev")
    assert task.train[0].target == "out"
    assert task.evaluation[0].references == ("other",)
    # identical ids in train and test are leakage and must be refused
    with pytest.raises(ValueError, match="vazamento|leak"):
        load_sapt_task(tmp_path, "tS")


def test_caps_apply_to_their_own_split(tmp_path: Path) -> None:
    _write_task(tmp_path, "tA", sizes={"train": 10, "test": 6})
    task = load_sapt_task(tmp_path, "tA", train_per_task=4, eval_per_task=2)
    assert [example.example_id for example in task.train] == [
        f"tA-train-{index}" for index in range(4)
    ]
    assert [example.example_id for example in task.evaluation] == [
        "tA-test-0",
        "tA-test-1",
    ]


def test_train_and_eval_ids_are_disjoint(tmp_path: Path) -> None:
    _write_task(tmp_path, "tA", sizes={"train": 5, "test": 5})
    task = load_sapt_task(tmp_path, "tA")
    train_ids = {example.example_id for example in task.train}
    eval_ids = {example.example_id for example in task.evaluation}
    assert train_ids.isdisjoint(eval_ids)


def test_sequence_follows_the_declared_order(tmp_path: Path) -> None:
    for name in ("tC", "tA", "tB"):
        _write_task(tmp_path, name, sizes={"train": 2, "test": 1})
    tasks = load_sapt_superni(tmp_path, task_names=["tB", "tC", "tA"])
    assert [task.name for task in tasks] == ["tB", "tC", "tA"]


def test_sequence_uses_sapt_order_and_can_cut_to_first_n(tmp_path: Path) -> None:
    for name in SAPT_ORDERS[2][:3]:
        _write_task(tmp_path, name, sizes={"train": 2, "test": 1})
    tasks = load_sapt_superni(tmp_path, order=2, tasks=3)
    assert [task.name for task in tasks] == list(SAPT_ORDERS[2][:3])
    with pytest.raises(ValueError):
        load_sapt_superni(tmp_path, order=3)


def test_missing_task_directory_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_sapt_superni(tmp_path, task_names=["nope"])


def test_repeated_task_names_are_rejected(tmp_path: Path) -> None:
    _write_task(tmp_path, "tA", sizes={"train": 2, "test": 1})
    with pytest.raises(ValueError):
        load_sapt_superni(tmp_path, task_names=["tA", "tA"])
