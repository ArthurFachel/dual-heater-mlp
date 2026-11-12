"""TRACE data loading and the official decoder-only collation.

Data layout (official release, ``LLM-CL-Benchmark_5000``)::

    <root>/<TASK>/train.json   [{"prompt": str, "answer": str}, ...]
    <root>/<TASK>/eval.json
    <root>/<TASK>/test.json

The data is distributed as a Google Drive archive linked from the TRACE README
(``BeyonderXX/TRACE``). It is NOT fetched by this module: downloading a dataset
needs explicit authorization in this repository. :func:`load_trace_tasks`
fails closed when a task directory is missing.

Collation reproduces ``utils/data/data_collator.py`` of the official repo:

* the tokenizer adds no special tokens; BOS/EOS are added by hand;
* truncation keeps the LAST ``cutoff`` tokens (``truncation_side="left"``);
* EOS is appended only if the truncated sequence is shorter than ``cutoff``,
  then BOS is prepended under the same condition (in that order);
* the training target is ``tokenize(prompt + answer)``, and the label mask
  keeps the last ``len(tokenize(answer) + [EOS])`` positions. When the answer's
  standalone tokenization differs from its tokenization inside
  ``prompt + answer`` (a BPE merge across the boundary), the mask is off by
  those tokens. This is upstream behaviour, kept on purpose and pinned by test;
* left padding with ``pad_token_id``, padded width rounded up to a multiple of
  ``pad_to_multiple_of`` (8 upstream).

Label tokenization (deviation R10, default ``"split"``)
------------------------------------------------------
``"official"`` reproduces the upstream mask above. With Qwen's BPE it is
WRONG on the token that decides the score: the prompt ends in ``":"`` and the
answer starts with a letter, and ``tokenize(prompt + answer)`` merges them into
``":C"``. Measured on the official 5000 release with Qwen2.5 tokenizers: the
supervised span differs from the answer's tokens in 100% of FOMC and ScienceQA
train examples (57% MeetingBank, 25% Py150, 18% 20Minuten). The model is then
trained to emit ``":C"`` after ``"Stance"``, while at inference the prompt
already ends in ``":"`` and it must emit ``"C"``, a continuation it never saw.

``"split"`` tokenizes prompt and answer separately and concatenates them, so
the training context of every answer token is exactly the inference prompt
(when nothing is truncated) and the supervised span is exactly
``tokens(answer) + [EOS]``. Truncation still drops from the LEFT and BOS is
still prepended only when there is room, as upstream.

Upstream sets ``pad_token = eos_token`` and, when the tokenizer has no BOS
(Qwen), ``bos_token = eos_token``. :func:`special_token_ids` applies the same
rule.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import Tensor

from .trace_metrics import TRACE_TASKS

__all__ = [
    "LABEL_TOKENIZATIONS",
    "SPLITS",
    "SpecialTokens",
    "TraceExample",
    "TraceTask",
    "collate_inference",
    "collate_train",
    "load_trace_tasks",
    "read_split",
    "special_token_ids",
    "tokenize_official",
    "trace_data_identity",
]

SPLITS: tuple[str, ...] = ("train", "eval", "test")


class _Tokenizer(Protocol):
    def __call__(self, text: str, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class TraceExample:
    prompt: str
    answer: str


@dataclass(frozen=True)
class TraceTask:
    name: str
    train: tuple[TraceExample, ...] = field(default_factory=tuple)
    test: tuple[TraceExample, ...] = field(default_factory=tuple)
    #: Number of rows in the source files before subsampling.
    source_sizes: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class SpecialTokens:
    bos: int
    eos: int
    pad: int


def special_token_ids(tokenizer: Any) -> SpecialTokens:
    """Upstream ``load_hf_tokenizer`` rule for non-LLaMA tokenizers."""

    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("o tokenizer precisa de eos_token")
    bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else eos
    # Upstream: tokenizer.pad_token = tokenizer.eos_token, unconditionally.
    return SpecialTokens(bos=int(bos), eos=int(eos), pad=int(eos))


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------


def read_split(root: str | Path, task: str, split: str) -> list[TraceExample]:
    if split not in SPLITS:
        raise ValueError(f"split deve ser um de {SPLITS}")
    path = Path(root) / task / f"{split}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} não existe. Os dados do TRACE não são baixados por este "
            "módulo; veja o docstring para a origem oficial."
        )
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise TypeError(f"{path}: esperava uma lista JSON")
    examples: list[TraceExample] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or "prompt" not in row or "answer" not in row:
            raise ValueError(f"{path}[{index}]: linha sem 'prompt'/'answer'")
        prompt, answer = row["prompt"], row["answer"]
        # Upstream get_answer() maps a null answer to "", get_prompt() keeps None.
        if prompt is None:
            raise ValueError(f"{path}[{index}]: prompt nulo")
        examples.append(TraceExample(prompt=str(prompt), answer="" if answer is None else str(answer)))
    return examples


def _subsample(
    examples: Sequence[TraceExample], limit: int | None, *, rng: random.Random
) -> tuple[TraceExample, ...]:
    """Seeded subset that PRESERVES file order.

    Order is preserved because the official evaluation iterates the test file
    sequentially; a subset should be a subset of that order, not a reshuffle.
    """

    if limit is None or limit >= len(examples):
        return tuple(examples)
    if limit < 1:
        raise ValueError("limite de exemplos deve ser >= 1")
    chosen = sorted(rng.sample(range(len(examples)), limit))
    return tuple(examples[index] for index in chosen)


def load_trace_tasks(
    root: str | Path,
    *,
    task_names: Sequence[str] = TRACE_TASKS,
    train_per_task: int | None = None,
    test_per_task: int | None = None,
    data_seed: int = 0,
) -> list[TraceTask]:
    """Load the sequence in the ORDER given; subsample deterministically.

    ``data_seed`` is separate from the training seed on purpose: every arm and
    every training seed of a protocol should see the same examples, so that
    seed-to-seed variance is optimization variance, not data variance.
    Each task gets its own RNG derived from ``(data_seed, task)`` so dropping
    or reordering a task never changes another task's subset.
    """

    unknown = [name for name in task_names if name not in TRACE_TASKS]
    if unknown:
        raise ValueError(f"tarefas fora do TRACE: {unknown}")
    if len(set(task_names)) != len(task_names):
        raise ValueError("tarefa repetida na sequência")

    tasks: list[TraceTask] = []
    for name in task_names:
        train = read_split(root, name, "train")
        test = read_split(root, name, "test")
        rng_train = random.Random(f"{data_seed}:{name}:train")
        rng_test = random.Random(f"{data_seed}:{name}:test")
        tasks.append(
            TraceTask(
                name=name,
                train=_subsample(train, train_per_task, rng=rng_train),
                test=_subsample(test, test_per_task, rng=rng_test),
                source_sizes={"train": len(train), "test": len(test)},
            )
        )
    return tasks


def trace_data_identity(root: str | Path, tasks: Sequence[TraceTask]) -> dict[str, Any]:
    """Hashes of the source files AND of the materialized subsets.

    The file hash ties a run to a specific release of the archive; the subset
    hash ties it to the exact examples consumed. Both are needed: the same
    files with a different ``data_seed`` give a different subset.
    """

    files: dict[str, str] = {}
    for task in tasks:
        for split in ("train", "test"):
            path = Path(root) / task.name / f"{split}.json"
            files[f"{task.name}/{split}.json"] = hashlib.sha256(path.read_bytes()).hexdigest()

    digest = hashlib.sha256()
    overlap: dict[str, int] = {}
    unique_train: dict[str, int] = {}
    for task in tasks:
        train_prompts = {example.prompt for example in task.train}
        unique_train[task.name] = len(train_prompts)
        overlap[task.name] = sum(1 for example in task.test if example.prompt in train_prompts)
        for split, examples in (("train", task.train), ("test", task.test)):
            for example in examples:
                payload = json.dumps(
                    [task.name, split, example.prompt, example.answer],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
                digest.update(len(payload).to_bytes(8, "big"))
                digest.update(payload)
    return {
        "source_files_sha256": files,
        "materialized_sha256": digest.hexdigest(),
        "train_test_prompt_overlap": overlap,
        # The official 5000 release repeats prompts (NumGLUE-cm: 282 unique of
        # 5000). Epoch counts are therefore NOT a measure of distinct data.
        "unique_train_prompts": unique_train,
        "counts": {
            task.name: {"train": len(task.train), "test": len(task.test), **{
                f"source_{key}": value for key, value in task.source_sizes.items()
            }}
            for task in tasks
        },
    }


# ---------------------------------------------------------------------------
# official collation
# ---------------------------------------------------------------------------


def tokenize_official(
    tokenizer: _Tokenizer,
    text: str,
    cutoff_len: int,
    special: SpecialTokens,
    *,
    add_bos_token: bool,
    add_eos_token: bool,
) -> list[int]:
    """``DataCollator.tokenize`` with ``truncation_side='left'``."""

    if cutoff_len < 1:
        raise ValueError("cutoff_len deve ser >= 1")
    ids = list(tokenizer(text, add_special_tokens=False)["input_ids"])
    if len(ids) > cutoff_len:
        ids = ids[-cutoff_len:]
    if len(ids) < cutoff_len and add_eos_token:
        ids.append(special.eos)
    if len(ids) < cutoff_len and add_bos_token:
        ids = [special.bos, *ids]
    return ids


def _pad_width(longest: int, multiple: int) -> int:
    return (longest + multiple - 1) // multiple * multiple


LABEL_TOKENIZATIONS: tuple[str, ...] = ("split", "official")


def _split_sequence(
    tokenizer: _Tokenizer,
    example: TraceExample,
    limit: int,
    special: SpecialTokens,
) -> tuple[list[int], int]:
    """``prompt_ids + answer_ids + [EOS]``, left-truncated, BOS if room."""

    prompt_ids = list(tokenizer(example.prompt, add_special_tokens=False)["input_ids"])
    answer_ids = list(tokenizer(example.answer, add_special_tokens=False)["input_ids"])
    ids = prompt_ids + answer_ids + [special.eos]
    label_len = len(answer_ids) + 1
    if len(ids) > limit:
        ids = ids[-limit:]
    if len(ids) < limit:
        ids = [special.bos, *ids]
    return ids, min(label_len, len(ids))


def collate_train(
    tokenizer: _Tokenizer,
    examples: Sequence[TraceExample],
    special: SpecialTokens,
    *,
    max_prompt_len: int,
    max_ans_len: int,
    pad_to_multiple_of: int = 8,
    label_tokenization: str = "split",
) -> dict[str, Tensor]:
    if label_tokenization not in LABEL_TOKENIZATIONS:
        raise ValueError(f"label_tokenization deve ser um de {LABEL_TOKENIZATIONS}")
    limit = max_prompt_len + max_ans_len
    sequences: list[list[int]] = []
    label_lens: list[int] = []
    for example in examples:
        if label_tokenization == "split":
            source, label_len = _split_sequence(tokenizer, example, limit, special)
            sequences.append(source)
            label_lens.append(label_len)
            continue
        label = tokenize_official(
            tokenizer, example.answer, limit, special, add_bos_token=False, add_eos_token=True
        )
        source = tokenize_official(
            tokenizer,
            example.prompt + example.answer,
            limit,
            special,
            add_bos_token=True,
            add_eos_token=True,
        )
        sequences.append(source)
        label_lens.append(len(label))
    width = _pad_width(max(len(seq) for seq in sequences), pad_to_multiple_of)

    input_ids, attention, labels = [], [], []
    for seq, label_len in zip(sequences, label_lens):
        pad = width - len(seq)
        input_ids.append([special.pad] * pad + seq)
        attention.append([0] * pad + [1] * len(seq))
        padded = [special.pad] * pad + seq
        # Upstream: [-100] * (width - label_len) + labels[-label_len:]
        labels.append([-100] * (width - label_len) + padded[-label_len:])
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def collate_inference(
    tokenizer: _Tokenizer,
    examples: Sequence[TraceExample],
    special: SpecialTokens,
    *,
    max_prompt_len: int,
    pad_to_multiple_of: int = 8,
) -> dict[str, Tensor]:
    sequences = [
        tokenize_official(
            tokenizer,
            example.prompt,
            max_prompt_len,
            special,
            add_bos_token=True,
            add_eos_token=False,
        )
        for example in examples
    ]
    width = _pad_width(max(len(seq) for seq in sequences), pad_to_multiple_of)
    input_ids, attention = [], []
    for seq in sequences:
        pad = width - len(seq)
        input_ids.append([special.pad] * pad + seq)
        attention.append([0] * pad + [1] * len(seq))
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention, dtype=torch.long),
    }
