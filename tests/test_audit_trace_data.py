"""The TRACE audit script measures what it claims, on a synthetic release.

Its floors feed a preregistered near-chance gate, so a floor computed wrong
would move the gate silently.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest

from dual_heater.trace_data import SpecialTokens, TraceExample

_SPEC = importlib.util.spec_from_file_location(
    "audit_trace_data", Path(__file__).resolve().parents[1] / "scripts" / "audit_trace_data.py"
)
assert _SPEC is not None and _SPEC.loader is not None
audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit)

SPECIAL = SpecialTokens(bos=1, eos=2, pad=2)


class MergingTokenizer:
    """Merges ':' + a following letter, like Qwen's BPE on ':C'."""

    def __call__(self, text, add_special_tokens=False, **_):
        ids, index = [], 0
        while index < len(text):
            if text[index] == ":" and index + 1 < len(text) and text[index + 1].isalpha():
                ids.append(900 + ord(text[index + 1]) % 50)
                index += 2
            else:
                ids.append(10 + ord(text[index]) % 200)
                index += 1
        return {"input_ids": ids}


ARGS = argparse.Namespace(max_prompt_len=64, max_ans_len=16)


def test_official_mask_is_misaligned_when_bpe_merges_the_boundary() -> None:
    examples = [TraceExample("Stance:", "C"), TraceExample("Stance:", "A")]
    official = audit._label_misalignment(MergingTokenizer(), examples, SPECIAL, ARGS, "official")
    split = audit._label_misalignment(MergingTokenizer(), examples, SPECIAL, ARGS, "split")
    assert official["span_mismatch"] == 1.0
    assert split["span_mismatch"] == 0.0
    assert split["context_mismatch"] == 0.0
    assert split["context_checked"] == 2


def test_uncheckable_context_is_none_not_zero() -> None:
    """When the joint sequence length never lines up, nothing was verified."""

    examples = [TraceExample("Stance:", "C")]
    official = audit._label_misalignment(MergingTokenizer(), examples, SPECIAL, ARGS, "official")
    assert official["context_checked"] == 0
    assert official["context_mismatch"] is None


def test_floors_for_accuracy_tasks() -> None:
    train = [TraceExample("q", "A"), TraceExample("q2", "A"), TraceExample("q3", "B")]
    test = [TraceExample("x", "A"), TraceExample("y", "B"), TraceExample("z", "C"), TraceExample("w", "A")]
    floors = audit._floors("FOMC", train, test)
    assert floors["oracle"] == 1.0
    assert floors["empty"] == 0.0
    assert floors["majority_train_answer"] == pytest.approx(0.5)
    assert floors["copy_prompt"] == 0.0


def test_sari_floor_of_an_empty_prediction_is_not_zero() -> None:
    """The finding that forces a per-task gate: 20Minuten rewards emitting nothing."""

    source = "Die Stadt Zürich hat heute beschlossen, die Steuern zu senken, sagte der Präsident."
    test = [TraceExample(source, "Zürich senkt die Steuern.")]
    floors = audit._floors("20Minuten", test, test)
    assert floors["empty"] > 0.4


def test_main_merges_partial_runs(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "data"
    for task in ("FOMC", "C-STANCE"):
        (root / task).mkdir(parents=True)
        rows = [{"prompt": f"p{i}:", "answer": "A" if i % 2 else "B"} for i in range(4)]
        for split in ("train", "eval", "test"):
            (root / task / f"{split}.json").write_text(json.dumps(rows), encoding="utf-8")

    class Tok(MergingTokenizer):
        eos_token_id = 2
        bos_token_id = None

    import transformers

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *_a, **_k: Tok())
    output = tmp_path / "audit.json"
    for task in ("FOMC", "C-STANCE"):
        monkeypatch.setattr(
            "sys.argv",
            ["audit", "--data-root", str(root), "--output", str(output), "--tasks", task],
        )
        assert audit.main() == 0
    report = json.loads(output.read_text())
    assert set(report["tasks"]) == {"FOMC", "C-STANCE"}
    assert report["tasks"]["FOMC"]["label_mask"]["official"]["span_mismatch"] == 1.0
    assert "evaluation_generations_full_protocol" not in report
