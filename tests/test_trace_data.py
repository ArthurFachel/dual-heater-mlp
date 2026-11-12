"""TRACE data loading and the official collation.

The collator decides which tokens carry loss and what the model sees at
inference. If it drifts from upstream's ``DataCollator``, a run trains on a
different objective than the paper's while every metric still looks normal.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from dual_heater.trace_data import (
    SpecialTokens,
    TraceExample,
    collate_inference,
    collate_train,
    load_trace_tasks,
    read_split,
    special_token_ids,
    tokenize_official,
    trace_data_identity,
)

SPECIAL = SpecialTokens(bos=1, eos=2, pad=2)


class CharTokenizer:
    """One token per character, ids >= 10. Deterministic and boundary-free."""

    def __call__(self, text: str, add_special_tokens: bool = False, **_):
        assert add_special_tokens is False, "upstream tokenizes with add_special_tokens=False"
        return {"input_ids": [10 + (ord(ch) % 200) for ch in text]}


class MergingTokenizer:
    """Merges the pair 'ab' into one token, to expose the label-boundary quirk."""

    def __call__(self, text: str, add_special_tokens: bool = False, **_):
        ids, index = [], 0
        while index < len(text):
            if text[index : index + 2] == "ab":
                ids.append(999)
                index += 2
            else:
                ids.append(10 + ord(text[index]) % 200)
                index += 1
        return {"input_ids": ids}


def _write_task(root: Path, task: str, train: list[dict], test: list[dict]) -> None:
    directory = root / task
    directory.mkdir(parents=True)
    (directory / "train.json").write_text(json.dumps(train), encoding="utf-8")
    (directory / "eval.json").write_text(json.dumps(test), encoding="utf-8")
    (directory / "test.json").write_text(json.dumps(test), encoding="utf-8")


# ---------------------------------------------------------------------------
# tokenize_official
# ---------------------------------------------------------------------------


def test_eos_then_bos_are_added_when_room() -> None:
    ids = tokenize_official(CharTokenizer(), "ab", 10, SPECIAL, add_bos_token=True, add_eos_token=True)
    assert ids[0] == SPECIAL.bos and ids[-1] == SPECIAL.eos and len(ids) == 4


def test_left_truncation_keeps_the_end() -> None:
    tok = CharTokenizer()
    ids = tokenize_official(tok, "abcdef", 3, SPECIAL, add_bos_token=True, add_eos_token=True)
    assert ids == tok("def")["input_ids"]  # no room for BOS/EOS once full


def test_eos_fits_but_bos_does_not() -> None:
    """Upstream appends EOS first; if that fills the cutoff, BOS is dropped."""

    tok = CharTokenizer()
    ids = tokenize_official(tok, "abc", 4, SPECIAL, add_bos_token=True, add_eos_token=True)
    assert ids == tok("abc")["input_ids"] + [SPECIAL.eos]


def test_special_token_rule_for_tokenizers_without_bos() -> None:
    class QwenLike:
        eos_token_id = 151643
        bos_token_id = None

    special = special_token_ids(QwenLike())
    assert special == SpecialTokens(bos=151643, eos=151643, pad=151643)


# ---------------------------------------------------------------------------
# collate_train
# ---------------------------------------------------------------------------


def test_labels_cover_exactly_answer_plus_eos() -> None:
    tok = CharTokenizer()
    batch = collate_train(
        tok, [TraceExample("Q:", "yes")], SPECIAL, max_prompt_len=50, max_ans_len=50
    )
    labels = batch["labels"][0]
    supervised = labels[labels != -100].tolist()
    assert supervised == tok("yes")["input_ids"] + [SPECIAL.eos]


def test_left_padding_to_a_multiple_of_eight() -> None:
    batch = collate_train(
        CharTokenizer(),
        [TraceExample("Q:", "a"), TraceExample("Question:", "abc")],
        SPECIAL,
        max_prompt_len=50,
        max_ans_len=50,
    )
    assert batch["input_ids"].shape[1] % 8 == 0
    # left padding: the first column is padding in the short row, real in neither
    assert batch["attention_mask"][0, 0] == 0
    assert batch["attention_mask"][0, -1] == 1
    assert torch.equal(batch["attention_mask"].sum(1), torch.tensor([1 + 2 + 1 + 1, 1 + 9 + 3 + 1]))


def test_padding_never_carries_loss() -> None:
    batch = collate_train(
        CharTokenizer(),
        [TraceExample("Q:", "a"), TraceExample("Question: long", "abc")],
        SPECIAL,
        max_prompt_len=50,
        max_ans_len=50,
    )
    supervised = batch["labels"] != -100
    assert not bool((supervised & (batch["attention_mask"] == 0)).any())


def test_label_boundary_quirk_is_reproduced() -> None:
    """Upstream masks by the STANDALONE answer length.

    'xa' + 'b' tokenizes jointly as [x, ab] (2 tokens) but the answer 'b'
    alone is 1 token + EOS = 2, so the supervised span covers 'ab' + EOS
    and not just 'b' + EOS. Pinned so nobody "fixes" it silently.
    """

    tok = MergingTokenizer()
    batch = collate_train(
        tok, [TraceExample("xa", "b")], SPECIAL, max_prompt_len=50, max_ans_len=50,
        label_tokenization="official",
    )
    supervised = batch["labels"][0][batch["labels"][0] != -100].tolist()
    assert supervised == [999, SPECIAL.eos]


def test_split_labels_are_exactly_the_answer_tokens() -> None:
    """R10: no merge across the boundary; the context equals the inference prompt."""

    tok = MergingTokenizer()
    batch = collate_train(tok, [TraceExample("xa", "b")], SPECIAL, max_prompt_len=50, max_ans_len=50)
    row = batch["input_ids"][0][batch["attention_mask"][0] == 1].tolist()
    supervised = batch["labels"][0][batch["labels"][0] != -100].tolist()
    assert supervised == tok("b")["input_ids"] + [SPECIAL.eos]
    inference = collate_inference(tok, [TraceExample("xa", "b")], SPECIAL, max_prompt_len=50)
    prompt = inference["input_ids"][0][inference["attention_mask"][0] == 1].tolist()
    assert row[: len(prompt)] == prompt


def test_split_truncates_from_the_left_and_keeps_the_answer() -> None:
    tok = CharTokenizer()
    batch = collate_train(
        tok, [TraceExample("0123456789", "ab")], SPECIAL, max_prompt_len=3, max_ans_len=2
    )
    row = batch["input_ids"][0][batch["attention_mask"][0] == 1].tolist()
    assert row == tok("89ab")["input_ids"] + [SPECIAL.eos]  # 5 = limit, no BOS room
    supervised = batch["labels"][0][batch["labels"][0] != -100].tolist()
    assert supervised == tok("ab")["input_ids"] + [SPECIAL.eos]


def test_unknown_label_tokenization_is_rejected() -> None:
    with pytest.raises(ValueError):
        collate_train(
            CharTokenizer(), [TraceExample("a", "b")], SPECIAL,
            max_prompt_len=4, max_ans_len=4, label_tokenization="joint",
        )


def test_inference_prompt_has_bos_and_no_eos() -> None:
    tok = CharTokenizer()
    batch = collate_inference(tok, [TraceExample("Q:", "ignored")], SPECIAL, max_prompt_len=50)
    real = batch["input_ids"][0][batch["attention_mask"][0] == 1].tolist()
    assert real == [SPECIAL.bos, *tok("Q:")["input_ids"]]


def test_inference_truncates_from_the_left_to_keep_the_question_end() -> None:
    tok = CharTokenizer()
    batch = collate_inference(tok, [TraceExample("0123456789", "")], SPECIAL, max_prompt_len=4)
    real = batch["input_ids"][0][batch["attention_mask"][0] == 1].tolist()
    assert real == tok("6789")["input_ids"]


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def test_missing_task_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_split(tmp_path, "FOMC", "train")


def test_rows_without_prompt_or_answer_are_rejected(tmp_path: Path) -> None:
    _write_task(tmp_path, "FOMC", [{"prompt": "p"}], [{"prompt": "p", "answer": "a"}])
    with pytest.raises(ValueError):
        read_split(tmp_path, "FOMC", "train")


def test_non_list_file_is_a_type_error(tmp_path: Path) -> None:
    _write_task(tmp_path, "FOMC", [], [])
    (tmp_path / "FOMC" / "train.json").write_text("{}", encoding="utf-8")
    with pytest.raises(TypeError):
        read_split(tmp_path, "FOMC", "train")


def test_null_answer_becomes_empty_like_upstream(tmp_path: Path) -> None:
    _write_task(tmp_path, "FOMC", [{"prompt": "p", "answer": None}], [])
    assert read_split(tmp_path, "FOMC", "train")[0].answer == ""


def test_order_is_preserved_and_subsets_are_seeded(tmp_path: Path) -> None:
    rows = [{"prompt": f"p{i}", "answer": f"a{i}"} for i in range(50)]
    for task in ("FOMC", "C-STANCE"):
        _write_task(tmp_path, task, rows, rows)
    first = load_trace_tasks(
        tmp_path, task_names=["FOMC", "C-STANCE"], train_per_task=10, test_per_task=5, data_seed=3
    )
    again = load_trace_tasks(
        tmp_path, task_names=["FOMC", "C-STANCE"], train_per_task=10, test_per_task=5, data_seed=3
    )
    other = load_trace_tasks(
        tmp_path, task_names=["FOMC", "C-STANCE"], train_per_task=10, test_per_task=5, data_seed=4
    )
    assert [task.name for task in first] == ["FOMC", "C-STANCE"]
    assert first == again
    assert first[0].train != other[0].train
    # order preserving: indices of the chosen rows are increasing
    indices = [int(example.prompt[1:]) for example in first[0].train]
    assert indices == sorted(indices)


def test_a_task_subset_does_not_depend_on_the_other_tasks(tmp_path: Path) -> None:
    rows = [{"prompt": f"p{i}", "answer": f"a{i}"} for i in range(50)]
    for task in ("FOMC", "C-STANCE"):
        _write_task(tmp_path, task, rows, rows)
    both = load_trace_tasks(tmp_path, task_names=["C-STANCE", "FOMC"], train_per_task=7)
    alone = load_trace_tasks(tmp_path, task_names=["FOMC"], train_per_task=7)
    assert both[1].train == alone[0].train


def test_unknown_or_repeated_tasks_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        load_trace_tasks(tmp_path, task_names=["SuperNI"])
    with pytest.raises(ValueError):
        load_trace_tasks(tmp_path, task_names=["FOMC", "FOMC"])


def test_identity_reports_train_test_overlap(tmp_path: Path) -> None:
    _write_task(
        tmp_path,
        "FOMC",
        [{"prompt": "same", "answer": "a"}, {"prompt": "x", "answer": "b"}],
        [{"prompt": "same", "answer": "a"}],
    )
    tasks = load_trace_tasks(tmp_path, task_names=["FOMC"])
    identity = trace_data_identity(tmp_path, tasks)
    assert identity["train_test_prompt_overlap"] == {"FOMC": 1}
    assert identity["unique_train_prompts"] == {"FOMC": 2}
    assert len(identity["materialized_sha256"]) == 64
    assert set(identity["source_files_sha256"]) == {"FOMC/train.json", "FOMC/test.json"}
