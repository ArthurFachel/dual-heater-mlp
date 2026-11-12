#!/usr/bin/env python3
"""Audit the official TRACE release before any run is designed on it.

Measures, per task, on the files actually on disk:

* counts, unique train prompts, train/test and eval/test overlap;
* token lengths and truncation under the official collator limits;
* how often the official label mask is misaligned (BPE merge across the
  prompt/answer boundary, see ``dual_heater.trace_data``);
* metric FLOORS with the official scorer: empty prediction, majority answer,
  copy of the prompt, and the oracle (gold as prediction). A preregistered
  near-chance gate needs these numbers; "chance" is not zero for every task.

CPU only. No model is loaded, only the tokenizer.

Usage::

    HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. \\
        .venv/bin/python scripts/audit_trace_data.py --data-root data/TRACE \\
        --output docs/audits/trace_data_audit.json
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from dual_heater.trace_data import (
    collate_inference,
    collate_train,
    read_split,
    special_token_ids,
    tokenize_official,
)
from dual_heater.trace_metrics import TRACE_TASKS, evaluate_task, primary_score

sys.setrecursionlimit(10_000)


def _floors(task: str, train, test) -> dict[str, float]:
    sources = [example.prompt for example in test]
    truths = [example.answer for example in test]
    n = len(test)

    def score(predictions: list[str]) -> float:
        return primary_score(task, evaluate_task(task, sources, predictions, truths))

    majority = collections.Counter(example.answer for example in train).most_common(1)[0][0]
    return {
        "oracle": score(list(truths)),
        "empty": score([""] * n),
        "majority_train_answer": score([majority] * n),
        "copy_prompt": score(list(sources)),
    }


def _label_misalignment(tokenizer, examples, special, args, mode: str) -> dict[str, float]:
    """Through the REAL collator, per example:

    * ``span``: supervised tokens != tokens(answer) + [EOS];
    * ``context``: the tokens before the supervised span are not the tokens the
      model sees at inference (only checked when the prompt is not truncated).
    """

    span_bad = 0
    context_bad = 0
    context_checked = 0
    for example in examples:
        batch = collate_train(
            tokenizer, [example], special, max_prompt_len=args.max_prompt_len,
            max_ans_len=args.max_ans_len, pad_to_multiple_of=1, label_tokenization=mode,
        )
        ids = batch["input_ids"][0].tolist()
        labels = batch["labels"][0].tolist()
        start = next(i for i, value in enumerate(labels) if value != -100)
        supervised = labels[start:]
        answer = list(tokenizer(example.answer, add_special_tokens=False)["input_ids"]) + [special.eos]
        if supervised != answer:
            span_bad += 1
        prompt = collate_inference(
            tokenizer, [example], special, max_prompt_len=args.max_prompt_len, pad_to_multiple_of=1
        )["input_ids"][0].tolist()
        full_prompt = tokenize_official(
            tokenizer, example.prompt, 10**9, special, add_bos_token=True, add_eos_token=False
        )
        if prompt == full_prompt and len(ids) == len(prompt) + len(answer):
            context_checked += 1
            if ids[:start] != prompt:
                context_bad += 1
    return {
        "span_mismatch": span_bad / len(examples),
        # None, not 0.0: "nothing was checkable" must not read as "all correct".
        "context_mismatch": context_bad / context_checked if context_checked else None,
        "context_checked": context_checked,
    }


def _fmt(mask: dict[str, Any]) -> str:
    value = mask["context_mismatch"]
    return "n/a" if value is None else f"{value:.3f}({mask['context_checked']})"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--max-prompt-len", type=int, default=1024)
    parser.add_argument("--max-ans-len", type=int, default=512)
    parser.add_argument(
        "--tasks", nargs="+", default=list(TRACE_TASKS), choices=TRACE_TASKS,
        help="audita só estas; tarefas já presentes em --output são preservadas",
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    special = special_token_ids(tokenizer)
    limit = args.max_prompt_len + args.max_ans_len

    report: dict[str, Any] = {
        "data_root": str(args.data_root.resolve()),
        "tokenizer": args.tokenizer,
        "limits": {"max_prompt_len": args.max_prompt_len, "max_ans_len": args.max_ans_len},
        "tasks": {},
    }
    if args.output.is_file():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        if previous.get("limits") == report["limits"] and previous.get("tokenizer") == args.tokenizer:
            report["tasks"].update(previous.get("tasks", {}))
    for task in args.tasks:
        train = read_split(args.data_root, task, "train")
        evaluation = read_split(args.data_root, task, "eval")
        test = read_split(args.data_root, task, "test")
        train_prompts = collections.Counter(example.prompt for example in train)
        test_prompts = {example.prompt for example in test}
        answers_by_prompt: dict[str, set[str]] = collections.defaultdict(set)
        for example in train:
            answers_by_prompt[example.prompt].add(example.answer)

        joint = [
            len(tokenize_official(tokenizer, e.prompt + e.answer, 10**9, special,
                                  add_bos_token=True, add_eos_token=True))
            for e in train
        ]
        prompts = [
            len(tokenize_official(tokenizer, e.prompt, 10**9, special,
                                  add_bos_token=True, add_eos_token=False))
            for e in test
        ]
        answers = [
            len(tokenize_official(tokenizer, e.answer, 10**9, special,
                                  add_bos_token=False, add_eos_token=True))
            for e in test
        ]
        report["tasks"][task] = {
            "files_sha256": {
                split: hashlib.sha256((args.data_root / task / f"{split}.json").read_bytes()).hexdigest()
                for split in ("train", "eval", "test")
            },
            "counts": {"train": len(train), "eval": len(evaluation), "test": len(test)},
            "unique_train_prompts": len(train_prompts),
            "max_train_prompt_repeats": max(train_prompts.values()),
            "train_prompts_with_conflicting_answers": sum(
                1 for values in answers_by_prompt.values() if len(values) > 1
            ),
            "test_prompts_in_train": sum(1 for e in test if e.prompt in train_prompts),
            "eval_prompts_in_test": sum(1 for e in evaluation if e.prompt in test_prompts),
            "eval_equals_test_as_multiset": collections.Counter(
                (e.prompt, e.answer) for e in evaluation
            ) == collections.Counter((e.prompt, e.answer) for e in test),
            "train_seq_tokens": {
                "mean": statistics.fmean(joint), "max": max(joint),
                "truncated_fraction": sum(v > limit for v in joint) / len(joint),
            },
            "test_prompt_tokens": {
                "mean": statistics.fmean(prompts), "max": max(prompts),
                "truncated_fraction": sum(v > args.max_prompt_len for v in prompts) / len(prompts),
            },
            "test_answer_tokens_with_eos": {
                "mean": statistics.fmean(answers), "max": max(answers),
                "longer_than_max_ans_len": sum(v > args.max_ans_len for v in answers) / len(answers),
            },
            "label_mask": {
                mode: _label_misalignment(tokenizer, train, special, args, mode)
                for mode in ("official", "split")
            },
            "floors": _floors(task, train, test),
        }
        row = report["tasks"][task]
        print(
            f"{task:12s} uniq {row['unique_train_prompts']:5d} leak {row['test_prompts_in_train']:4d} "
            f"span off/split {row['label_mask']['official']['span_mismatch']:.3f}/"
            f"{row['label_mask']['split']['span_mismatch']:.3f} "
            f"ctx off/split {_fmt(row['label_mask']['official'])}/"
            f"{_fmt(row['label_mask']['split'])} "
            + " ".join(f"{k} {v:.3f}" for k, v in row["floors"].items()),
            flush=True,
        )

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if all(task in report["tasks"] for task in TRACE_TASKS):
        generations = 0
        for stage in range(len(TRACE_TASKS)):
            generations += sum(
                report["tasks"][t]["counts"]["test"] for t in TRACE_TASKS[: stage + 1]
            )
        report["evaluation_generations_full_protocol"] = generations
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"gerações de avaliação no protocolo completo: {generations:,}")
    print(f"relatório: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
