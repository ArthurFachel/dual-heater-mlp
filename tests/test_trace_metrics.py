"""TRACE metrics pinned against values produced by the OFFICIAL upstream code.

Every golden number below was computed by running the real upstream sources
(TRACE metrics.py + rouge 1.0.1 PyPI wheel + nltk 3.5 bleu_score + fuzzywuzzy
difflib path + HF datasets 2.14 sari core) on the same inputs, not by hand.
A differential check over 400 random cases per metric found zero mismatches.
If one of these tests fails, the port has drifted from what the TRACE paper
computed, and OP/BWT from this repo stop being comparable to anything.
"""

from __future__ import annotations

import pytest

from dual_heater.trace_metrics import (
    TRACE_PRIMARY_METRIC,
    TRACE_TASKS,
    caculate_accuracy,
    caculate_bleu,
    caculate_fuzz,
    caculate_rouge,
    caculate_sari,
    evaluate_task,
    fuzz_ratio,
    postprocess_py150,
    primary_score,
    resolve_scienceqa,
    rouge_l_f,
    sari_corpus,
    trace_cl_metrics,
)

# ---------------------------------------------------------------------------
# golden values from the upstream code
# ---------------------------------------------------------------------------


def test_rouge_matches_upstream_including_the_argument_swap() -> None:
    value = caculate_rouge(
        ["the cat sat on the mat. it was happy"],
        ["a cat sat. the mat was red and the dog ran"],
    )
    assert value == pytest.approx(0.555555550617284, abs=1e-15)


def test_rouge_argument_swap_is_load_bearing() -> None:
    """Summary-level ROUGE-L is asymmetric; un-swapping changes the number."""

    pred, ref = "the cat sat on the mat. it was happy", "a cat sat. the mat was red and the dog ran"
    swapped = caculate_rouge([pred], [ref])
    assert swapped == pytest.approx(rouge_l_f(ref, pred))
    assert swapped != pytest.approx(rouge_l_f(pred, ref))


def test_rouge_counts_unique_words_like_rouge_1_0_1() -> None:
    """exclusive=True: 'the the the cat' vs 'the cat' scores ~1, not 0.67."""

    assert caculate_rouge(["the the the cat"], ["the cat"]) == pytest.approx(0.999999995, abs=1e-15)


@pytest.mark.parametrize(
    ("gram", "expected"),
    [(1, 0.8333333333333334), (4, 7.262123179505913e-78)],
)
def test_bleu_matches_nltk_3_5_method0(gram: int, expected: float) -> None:
    value = caculate_bleu(["the cat sat on the mat"], ["the cat is on the mat"], gram)
    assert value == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(
    ("gram", "expected"),
    [(1, 0.1353352832366127), (4, 2.4655814830110698e-232)],
)
def test_bleu_brevity_penalty_and_missing_orders(gram: int, expected: float) -> None:
    value = caculate_bleu(["cat"], ["the cat sat"], gram)
    assert value == pytest.approx(expected, rel=1e-12)


def test_fuzz_matches_fuzzywuzzy_ratio() -> None:
    assert caculate_fuzz(["def foo(x): return x+1"], ["def foo(y): return y + 1"]) == 87.0


def test_accuracy_is_raw_equality_and_empties_count_as_misses() -> None:
    """' A' != 'A', '' never scores, case matters: 2 hits out of 5."""

    value = caculate_accuracy(["A", " A", "", "B", "a"], ["A", "A", "", "B", "A"])
    assert value == pytest.approx(0.4)


def test_sari_matches_hf_implementation() -> None:
    value = caculate_sari(
        ["Die Stadt ist sehr groß und alt, sagt man."],
        ["Die Stadt ist groß."],
        ["Die Stadt ist alt."],
    )["sari"]
    assert value == pytest.approx(71.94444444444444, rel=1e-12)


def test_sari_is_case_insensitive_like_upstream() -> None:
    """HF sari lowercases before 13a tokenization; upstream gives exactly 100."""

    value = caculate_sari(
        ["Die Stadt ist sehr groß und alt."], ["die stadt ist alt."], ["Die Stadt ist alt."]
    )["sari"]
    assert value == pytest.approx(100.0)


def test_sari_doc_example_and_perfect_copy() -> None:
    """The two examples published in the HF metric card."""

    assert sari_corpus(
        ["About 95 species are currently accepted ."],
        ["About 95 you now get in ."],
        [[
            "About 95 species are currently known .",
            "About 95 species are now accepted .",
            "95 species are now accepted .",
        ]],
    ) == pytest.approx(26.953601953601954, rel=1e-12)
    assert sari_corpus(
        ["Die Stadt ist groß ."], ["Die Stadt ist groß ."], [["Die Stadt ist groß ."]]
    ) == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# official quirks that change numbers
# ---------------------------------------------------------------------------


def test_empty_predictions_stay_in_the_denominator() -> None:
    """Skipping empties AND shrinking n would reward a model that emits nothing."""

    assert caculate_fuzz(["abc", ""], ["abc", "abc"]) == pytest.approx(50.0)
    assert caculate_rouge(["a b", ""], ["a b", "a b"]) == pytest.approx(0.5, abs=1e-7)
    assert caculate_bleu(["a b c", ""], ["a b c", "a b c"], 1) == pytest.approx(0.5)


def test_fuzz_decorator_order() -> None:
    assert fuzz_ratio("", "") == 100  # equivalence is checked before emptiness
    assert fuzz_ratio("abc", "") == 0
    assert fuzz_ratio(None, "abc") == 0


def test_py150_literal_restoration() -> None:
    code = "x = <NUM_LIT> + <NUM_LIT:3> ; s = <STR_LIT:hi> + <STR_LIT> + <CHAR_LIT>"
    assert postprocess_py150(code) == "x = 0 + 3 ; s = hi +  + "


def test_scienceqa_takes_first_char_and_reasoning_from_index_two() -> None:
    resolved = resolve_scienceqa(["B\nBecause plants need light.", ""])
    assert resolved["answers"] == ["B", ""]
    assert resolved["reasonings"] == ["Because plants need light.", ""]


def test_scienceqa_primary_is_answer_accuracy_not_rouge() -> None:
    evaluation = evaluate_task(
        "ScienceQA",
        ["q1", "q2"],
        ["B\nwrong reasoning", "C\nexact"],
        ["B\nBecause x.", "A\nexact"],
    )
    assert evaluation["accuracy"] == pytest.approx(0.5)
    assert primary_score("ScienceQA", evaluation) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# declared deviations
# ---------------------------------------------------------------------------


def test_d1_sentence_less_prediction_scores_zero_instead_of_crashing() -> None:
    assert rouge_l_f("...", "the cat") == 0.0
    assert caculate_rouge(["...", "the cat"], ["the cat", "the cat"]) == pytest.approx(0.5, abs=1e-7)


def test_d2_empty_scienceqa_generation_does_not_crash() -> None:
    evaluation = evaluate_task("ScienceQA", ["q"], [""], ["A\nr"])
    assert evaluation["accuracy"] == 0.0


def test_d6_long_sentence_does_not_overflow_the_stack() -> None:
    """Upstream's recursive LCS dies near 1000 words; MeetingBank has ~2853."""

    long = " ".join(f"w{i % 97}" for i in range(4000))
    assert 0.0 < rouge_l_f(long, long) <= 1.0


# ---------------------------------------------------------------------------
# dispatch, primary metric, scale
# ---------------------------------------------------------------------------


def test_every_task_has_an_evaluator_and_a_primary_metric() -> None:
    assert set(TRACE_PRIMARY_METRIC) == set(TRACE_TASKS)
    for task in TRACE_TASKS:
        prediction = "A\nx" if task == "ScienceQA" else "x"
        evaluation = evaluate_task(task, ["src x"], [prediction], [prediction])
        assert 0.0 <= primary_score(task, evaluation) <= 1.0


def test_unknown_task_raises_instead_of_returning_empty() -> None:
    with pytest.raises(KeyError):
        evaluate_task("SuperNI", ["s"], ["p"], ["p"])


@pytest.mark.parametrize(("task", "raw"), [("Py150", 87.0), ("20Minuten", 71.9)])
def test_d5_percent_metrics_are_divided_by_100(task: str, raw: float) -> None:
    key = TRACE_PRIMARY_METRIC[task]
    assert primary_score(task, {key: raw}) == pytest.approx(raw / 100)


def test_primary_score_rejects_out_of_range_values() -> None:
    with pytest.raises(ValueError):
        primary_score("FOMC", {"accuracy": 1.5})


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(ValueError):
        caculate_accuracy(["a"], ["a", "b"])


# ---------------------------------------------------------------------------
# OP and BWT, paper eqs. 2-3
# ---------------------------------------------------------------------------


def test_op_and_bwt_follow_the_paper_normalization() -> None:
    matrix = [
        [0.8, None, None],
        [0.6, 0.9, None],
        [0.5, 0.7, 0.4],
    ]
    metrics = trace_cl_metrics(matrix)
    assert metrics.overall_performance == pytest.approx((0.5 + 0.7 + 0.4) / 3)
    # Paper: (1/t) * sum_{i<t} (R[t,i] - R[i,i]), t = 3
    assert metrics.backward_transfer == pytest.approx(((0.5 - 0.8) + (0.7 - 0.9)) / 3)
    assert metrics.backward_transfer_old_mean == pytest.approx(((0.5 - 0.8) + (0.7 - 0.9)) / 2)
    assert metrics.per_task_final == pytest.approx((0.5, 0.7, 0.4))


def test_bwt_reads_the_last_row_not_the_second() -> None:
    """The Fase 3 bug: a reader that took matrix[1] reported stage-1 retention."""

    matrix = [
        [1.0, None, None, None],
        [1.0, 1.0, None, None],
        [0.5, 0.5, 1.0, None],
        [0.0, 0.0, 0.0, 1.0],
    ]
    metrics = trace_cl_metrics(matrix)
    assert metrics.backward_transfer == pytest.approx(-3.0 / 4)
    assert metrics.overall_performance == pytest.approx(0.25)


def test_trace_metrics_reject_missing_lower_triangle() -> None:
    with pytest.raises(ValueError):
        trace_cl_metrics([[0.5, None], [None, 0.5]])
