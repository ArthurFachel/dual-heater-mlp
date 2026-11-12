"""TRACE task metrics, ported from the official repository.

TRACE (Wang et al., 2023, arXiv 2310.06762) scores each of its eight tasks with
a different metric and folds one number per task into Overall Performance (OP)
and Backward Transfer (BWT). This module reproduces the OFFICIAL scoring code,
quirks included, because a number that will be compared against the paper must
be computed the way the paper computed it.

Sources ported (fetched 2026-10-05, not installed as dependencies):

* ``BeyonderXX/TRACE@master``: ``metrics.py`` and ``evaluations/eval_*.py``.
* ``pltrdy/rouge@1.0.1`` (``rouge==1.0.1`` pinned in TRACE's requirements):
  ``Rouge.get_scores`` and ``rouge_l_summary_level``.
* ``nltk@3.5`` (pinned): ``sentence_bleu`` with ``SmoothingFunction().method0``.
* ``seatgeek/fuzzywuzzy``: ``fuzz.ratio`` on the pure-Python ``difflib`` path.
* ``huggingface/datasets@2.14.0`` ``metrics/sari`` (identical core to
  ``evaluate``'s), with the ``sacrebleu`` ``13a`` tokenizer.

Official quirks reproduced on purpose
-------------------------------------
1. Empty-string predictions or targets are SKIPPED in the numerator but still
   COUNTED in the denominator, so they score as zero (accuracy, BLEU, ROUGE-L,
   fuzz).
2. ``caculate_rouge`` calls ``score_rouge(target, prediction)``: the reference
   is passed as the hypothesis. Summary-level ROUGE-L is not symmetric, so the
   swap is kept.
3. ``rouge`` 1.0.1 with ``exclusive=True`` counts UNIQUE words and builds the
   LCS union as a set of words, not of positions.
4. Accuracy is raw string equality, no stripping or case folding. A leading
   space in a generation is a miss.
5. ScienceQA answers are the FIRST CHARACTER of the generation, the reasoning is
   everything from index 2 onward.

Declared deviations (each pinned by a test)
-------------------------------------------
D1. A prediction that ``rouge`` would split into zero sentences (for example
    ``"..."``) raises ``ValueError`` in the official code and kills the whole
    evaluation. Here it scores 0.0 for that example.
D2. An empty ScienceQA generation raises ``IndexError`` in the official
    ``resolve``. Here its answer and reasoning are both ``""``, which the
    official skip rule then scores as zero.
D3. ``fuzz.ratio`` uses ``difflib`` (fuzzywuzzy's fallback). TRACE does not
    pin ``python-Levenshtein``; if it was installed upstream the edit
    similarity can differ slightly. Not validated against the authors' runs.
D4. ``caculate_sari`` has a trailing comma, so upstream it returns a 1-tuple.
    The value is identical; only the container differs.
D5. OP and BWT are reported in [0, 1]. The paper prints Py150 similarity and
    20Minuten SARI divided by 100 (Table 6 values are all in [0, 1]); the
    division is done explicitly in :func:`primary_score`.
D6. The ``rouge`` LCS reconstruction is recursive upstream and overflows the
    Python stack on long sentences. It is iterative here, with the same
    tie-breaking.
"""

from __future__ import annotations

import math
import re
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher

__all__ = [
    "TRACE_PRIMARY_METRIC",
    "TRACE_TASKS",
    "TraceCLMetrics",
    "bleu_score",
    "caculate_accuracy",
    "caculate_bleu",
    "caculate_fuzz",
    "caculate_rouge",
    "caculate_sari",
    "evaluate_task",
    "fuzz_ratio",
    "postprocess_py150",
    "primary_score",
    "resolve_scienceqa",
    "rouge_l_f",
    "sari_corpus",
    "trace_cl_metrics",
]

#: Official task order (``scripts/train_seq_naive.sh``, ``--dataset_name``).
TRACE_TASKS: tuple[str, ...] = (
    "C-STANCE",
    "FOMC",
    "MeetingBank",
    "Py150",
    "ScienceQA",
    "NumGLUE-cm",
    "NumGLUE-ds",
    "20Minuten",
)

#: The single metric each task contributes to OP/BWT (paper, Table 4), and the
#: key it has in the official evaluation dictionary.
TRACE_PRIMARY_METRIC: Mapping[str, str] = {
    "C-STANCE": "accuracy",
    "FOMC": "accuracy",
    "MeetingBank": "rouge-L",
    "Py150": "similarity",
    "ScienceQA": "accuracy",
    "NumGLUE-cm": "accuracy",
    "NumGLUE-ds": "accuracy",
    "20Minuten": "sari",
}

#: Divisor that maps each primary metric into [0, 1] (deviation D5).
_PRIMARY_SCALE: Mapping[str, float] = {
    "accuracy": 1.0,
    "rouge-L": 1.0,
    "similarity": 100.0,
    "sari": 100.0,
}


def _check_lengths(results: Sequence[str], data: Sequence[str]) -> None:
    if len(results) != len(data):
        raise ValueError(
            f"predições e referências com tamanhos diferentes: {len(results)} != {len(data)}"
        )
    if not results:
        raise ValueError("avaliação vazia: nenhuma predição")


# ---------------------------------------------------------------------------
# Accuracy (exact match, raw string)
# ---------------------------------------------------------------------------


def caculate_accuracy(results: Sequence[str], data: Sequence[str]) -> float:
    """Official ``caculate_accuracy``: raw ``==``, empties count as misses."""

    _check_lengths(results, data)
    scores = 0
    for prediction, target in zip(results, data):
        if prediction == "" or target == "":
            continue
        if prediction == target:
            scores += 1
    return scores / len(results)


# ---------------------------------------------------------------------------
# BLEU (nltk 3.5 sentence_bleu, method0 smoothing)
# ---------------------------------------------------------------------------


def _bleu_tokenize(text: str) -> list[str]:
    """Official TRACE tokenizer: split on whitespace OR period, drop empties."""

    return [token for token in re.split(r"\s|\.", text) if len(token) > 0]


def _ngram_counts(tokens: Sequence[str], n: int) -> Counter:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _sentence_bleu(reference: Sequence[str], hypothesis: Sequence[str], weights) -> float:
    """``nltk.translate.bleu_score.sentence_bleu`` (3.5) for ONE reference.

    method0 smoothing: a zero-count order contributes ``sys.float_info.min`` to
    the geometric mean, so any missing order drives the score to ~0.
    """

    numerators: list[int] = []
    denominators: list[int] = []
    reference_counts_cache: dict[int, Counter] = {}
    for order in range(1, len(weights) + 1):
        counts = _ngram_counts(hypothesis, order)
        reference_counts = reference_counts_cache.setdefault(
            order, _ngram_counts(reference, order)
        )
        clipped = sum(min(count, reference_counts[gram]) for gram, count in counts.items())
        numerators.append(clipped)
        denominators.append(max(1, sum(counts.values())))

    if numerators[0] == 0:
        return 0.0

    hyp_len = len(hypothesis)
    ref_len = len(reference)
    if hyp_len > ref_len:
        brevity = 1.0
    elif hyp_len == 0:
        brevity = 0.0
    else:
        brevity = math.exp(1 - ref_len / hyp_len)

    precisions = [
        numerator / denominator if numerator != 0 else sys.float_info.min
        for numerator, denominator in zip(numerators, denominators)
    ]
    log_sum = math.fsum(w * math.log(p) for w, p in zip(weights, precisions))
    return brevity * math.exp(log_sum)


_BLEU_WEIGHTS = {
    1: (1.0,),
    2: (1.0 / 2.0, 1.0 / 2.0),
    3: (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
    4: (1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0),
}


def bleu_score(reference: str, hypothesis: str, gram: int) -> float:
    """Official ``bleu_score``: TRACE tokenization, then nltk sentence BLEU."""

    if gram not in _BLEU_WEIGHTS:
        raise ValueError("gram deve ser 1, 2, 3 ou 4")
    return _sentence_bleu(
        _bleu_tokenize(reference), _bleu_tokenize(hypothesis), _BLEU_WEIGHTS[gram]
    )


def caculate_bleu(results: Sequence[str], data: Sequence[str], gram: int) -> float:
    _check_lengths(results, data)
    bleus = []
    for prediction, target in zip(results, data):
        if prediction == "" or target == "":
            continue
        bleus.append(bleu_score(target, prediction, gram))
    return sum(bleus) / len(results)


# ---------------------------------------------------------------------------
# ROUGE-L (pltrdy/rouge 1.0.1, summary level, exclusive=True)
# ---------------------------------------------------------------------------


def _split_sentences(text: str) -> list[str]:
    """``Rouge.get_scores`` preprocessing: split on '.', collapse whitespace."""

    return [" ".join(part.split()) for part in text.split(".") if len(part) > 0]


def _split_into_words(sentences: Sequence[str]) -> list[str]:
    return [word for sentence in sentences for word in sentence.split(" ")]


def _recon_lcs_words(x: Sequence[str], y: Sequence[str]) -> set[str]:
    """Words on the LCS path between ``x`` and ``y``, as a set.

    Iterative version of ``rouge_score._recon_lcs`` (deviation D6) with the
    same tie-breaking: match -> diagonal; else up if ``T[i-1][j] > T[i][j-1]``;
    else left.
    """

    n, m = len(x), len(y)
    table = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        row, previous = table[i], table[i - 1]
        xi = x[i - 1]
        for j in range(1, m + 1):
            if xi == y[j - 1]:
                row[j] = previous[j - 1] + 1
            else:
                row[j] = max(previous[j], row[j - 1])
    words: set[str] = set()
    i, j = n, m
    while i > 0 and j > 0:
        if x[i - 1] == y[j - 1]:
            words.add(x[i - 1])
            i -= 1
            j -= 1
        elif table[i - 1][j] > table[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return words


def _rouge_l_summary_level(
    evaluated_sentences: Sequence[str], reference_sentences: Sequence[str]
) -> float:
    if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
        raise ValueError("Collections must contain at least 1 sentence.")

    m = len(set(_split_into_words(reference_sentences)))
    n = len(set(_split_into_words(evaluated_sentences)))

    union: set[str] = set()
    llcs = 0
    for reference_sentence in reference_sentences:
        previous_count = len(union)
        reference_words = _split_into_words([reference_sentence])
        for evaluated_sentence in evaluated_sentences:
            evaluated_words = _split_into_words([evaluated_sentence])
            union = union | _recon_lcs_words(reference_words, evaluated_words)
        llcs += len(union) - previous_count

    r_lcs = llcs / m
    p_lcs = llcs / n
    return 2.0 * ((p_lcs * r_lcs) / (p_lcs + r_lcs + 1e-8))


def rouge_l_f(hypothesis: str, reference: str) -> float:
    """``Rouge(metrics=['rouge-l']).get_scores(hypothesis, reference)`` F value.

    Deviation D1: inputs that split into zero sentences score 0.0 instead of
    raising.
    """

    evaluated = _split_sentences(hypothesis)
    references = _split_sentences(reference)
    if not evaluated or not references:
        return 0.0
    return _rouge_l_summary_level(evaluated, references)


def caculate_rouge(results: Sequence[str], data: Sequence[str]) -> float:
    """Official ``caculate_rouge``, INCLUDING the argument swap (quirk 2)."""

    _check_lengths(results, data)
    rouges = []
    for prediction, target in zip(results, data):
        if prediction == "" or target == "":
            continue
        # Upstream: score_rouge(target, prediction) -> get_scores(target, prediction)
        rouges.append(rouge_l_f(target, prediction))
    return sum(rouges) / len(results)


# ---------------------------------------------------------------------------
# Edit similarity (fuzzywuzzy fuzz.ratio, difflib path)
# ---------------------------------------------------------------------------


def fuzz_ratio(s1: str | None, s2: str | None) -> int:
    """``fuzz.ratio`` with its three decorators, in upstream order."""

    if s1 is None or s2 is None:
        return 0
    if s1 == s2:
        return 100
    if len(s1) == 0 or len(s2) == 0:
        return 0
    # fuzzywuzzy utils.intr: int(round(n)). round() already returns int here.
    return round(100 * SequenceMatcher(None, s1, s2).ratio())


def caculate_fuzz(results: Sequence[str], data: Sequence[str]) -> float:
    _check_lengths(results, data)
    scores = 0
    for prediction, target in zip(results, data):
        if prediction == "" or target == "":
            continue
        scores += fuzz_ratio(prediction, target)
    return scores / len(results)


_PY150_LITERAL = re.compile(r"<(STR|NUM|CHAR)_LIT:(.*?)>", re.DOTALL)


def postprocess_py150(code: str) -> str:
    """Official ``eval_Py150.postprocess``: restore literal placeholders."""

    code = code.replace("<NUM_LIT>", "0").replace("<STR_LIT>", "").replace("<CHAR_LIT>", "")
    for kind, value in re.findall(_PY150_LITERAL, code):
        code = code.replace(f"<{kind}_LIT:{value}>", value)
    return code


# ---------------------------------------------------------------------------
# SARI (HF datasets/evaluate, sacrebleu 13a normalization)
# ---------------------------------------------------------------------------

_13A_RULES = (
    (re.compile(r"([\{-\~\[-\` -\&\(-\+\:-\@\/])"), r" \1 "),
    (re.compile(r"([^0-9])([\.,])"), r"\1 \2 "),
    (re.compile(r"([\.,])([^0-9])"), r" \1 \2"),
    (re.compile(r"([0-9])(-)"), r"\1 \2 "),
)


def _tokenize_13a(line: str) -> str:
    line = line.replace("<skipped>", "")
    line = line.replace("-\n", "")
    line = line.replace("\n", " ")
    if "&" in line:
        line = line.replace("&quot;", '"')
        line = line.replace("&amp;", "&")
        line = line.replace("&lt;", "<")
        line = line.replace("&gt;", ">")
    line = f" {line} "
    for pattern, replacement in _13A_RULES:
        line = pattern.sub(replacement, line)
    return " ".join(line.split())


def _sari_normalize(sentence: str) -> str:
    return _tokenize_13a(sentence.lower())


def _sari_ngram(sgrams, cgrams, rgramslist, numref):
    rgramsall = [rgram for rgrams in rgramslist for rgram in rgrams]
    rgramcounter = Counter(rgramsall)

    sgramcounter = Counter(sgrams)
    sgramcounter_rep = Counter()
    for sgram, scount in sgramcounter.items():
        sgramcounter_rep[sgram] = scount * numref

    cgramcounter = Counter(cgrams)
    cgramcounter_rep = Counter()
    for cgram, ccount in cgramcounter.items():
        cgramcounter_rep[cgram] = ccount * numref

    # KEEP
    keepgramcounter_rep = sgramcounter_rep & cgramcounter_rep
    keepgramcountergood_rep = keepgramcounter_rep & rgramcounter
    keepgramcounterall_rep = sgramcounter_rep & rgramcounter

    keeptmpscore1 = 0
    keeptmpscore2 = 0
    for keepgram in keepgramcountergood_rep:
        keeptmpscore1 += keepgramcountergood_rep[keepgram] / keepgramcounter_rep[keepgram]
        keeptmpscore2 += keepgramcountergood_rep[keepgram]
    keepscore_precision = 1
    keepscore_recall = 1
    if len(keepgramcounter_rep) > 0:
        keepscore_precision = keeptmpscore1 / len(keepgramcounter_rep)
    if len(keepgramcounterall_rep) > 0:
        keepscore_recall = keeptmpscore2 / sum(keepgramcounterall_rep.values())
    keepscore = 0
    if keepscore_precision > 0 or keepscore_recall > 0:
        keepscore = (
            2 * keepscore_precision * keepscore_recall
            / (keepscore_precision + keepscore_recall)
        )

    # DELETION
    delgramcounter_rep = sgramcounter_rep - cgramcounter_rep
    delgramcountergood_rep = delgramcounter_rep - rgramcounter
    deltmpscore1 = 0
    for delgram in delgramcountergood_rep:
        # Upstream also accumulates an unused deltmpscore2 from
        # delgramcounterall_rep; its denominator is provably > 0 here, so
        # dropping it changes nothing.
        deltmpscore1 += delgramcountergood_rep[delgram] / delgramcounter_rep[delgram]
    delscore_precision = 1
    if len(delgramcounter_rep) > 0:
        delscore_precision = deltmpscore1 / len(delgramcounter_rep)

    # ADDITION
    addgramcounter = set(cgramcounter) - set(sgramcounter)
    addgramcountergood = set(addgramcounter) & set(rgramcounter)
    addgramcounterall = set(rgramcounter) - set(sgramcounter)

    addtmpscore = len(addgramcountergood)
    addscore_precision = 1
    addscore_recall = 1
    if len(addgramcounter) > 0:
        addscore_precision = addtmpscore / len(addgramcounter)
    if len(addgramcounterall) > 0:
        addscore_recall = addtmpscore / len(addgramcounterall)
    addscore = 0
    if addscore_precision > 0 or addscore_recall > 0:
        addscore = (
            2 * addscore_precision * addscore_recall
            / (addscore_precision + addscore_recall)
        )

    return keepscore, delscore_precision, addscore


def _ngrams_upstream(grams1: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Upstream builds 2/3/4-grams with that exact (slightly odd) loop."""

    g2: list[str] = []
    g3: list[str] = []
    g4: list[str] = []
    for i in range(len(grams1) - 1):
        if i < len(grams1) - 1:
            g2.append(grams1[i] + " " + grams1[i + 1])
        if i < len(grams1) - 2:
            g3.append(grams1[i] + " " + grams1[i + 1] + " " + grams1[i + 2])
        if i < len(grams1) - 3:
            g4.append(
                grams1[i] + " " + grams1[i + 1] + " " + grams1[i + 2] + " " + grams1[i + 3]
            )
    return g2, g3, g4


def _sari_sentence(ssent: str, csent: str, rsents: Sequence[str]) -> float:
    numref = len(rsents)
    s1 = ssent.split(" ")
    c1 = csent.split(" ")
    s2, s3, s4 = _ngrams_upstream(s1)
    c2, c3, c4 = _ngrams_upstream(c1)
    r1l, r2l, r3l, r4l = [], [], [], []
    for rsent in rsents:
        r1 = rsent.split(" ")
        r2, r3, r4 = _ngrams_upstream(r1)
        r1l.append(r1)
        r2l.append(r2)
        r3l.append(r3)
        r4l.append(r4)

    k1, d1, a1 = _sari_ngram(s1, c1, r1l, numref)
    k2, d2, a2 = _sari_ngram(s2, c2, r2l, numref)
    k3, d3, a3 = _sari_ngram(s3, c3, r3l, numref)
    k4, d4, a4 = _sari_ngram(s4, c4, r4l, numref)
    keep = sum([k1, k2, k3, k4]) / 4
    delete = sum([d1, d2, d3, d4]) / 4
    add = sum([a1, a2, a3, a4]) / 4
    return (keep + delete + add) / 3


def sari_corpus(
    sources: Sequence[str],
    predictions: Sequence[str],
    references: Sequence[Sequence[str]],
) -> float:
    """``load_metric('sari').compute(...)['sari']``, in [0, 100]."""

    if not (len(sources) == len(predictions) == len(references)):
        raise ValueError("Sources length must match predictions and references lengths.")
    if not predictions:
        raise ValueError("avaliação vazia: nenhuma predição")
    total = 0.0
    for source, prediction, refs in zip(sources, predictions, references):
        total += _sari_sentence(
            _sari_normalize(source),
            _sari_normalize(prediction),
            [_sari_normalize(sentence) for sentence in refs],
        )
    return 100 * total / len(predictions)


def caculate_sari(
    inputs: Sequence[str], results: Sequence[str], data: Sequence[str]
) -> dict[str, float]:
    """Official ``caculate_sari`` (minus the trailing-comma tuple, D4)."""

    return {"sari": sari_corpus(inputs, results, [[label] for label in data])}


# ---------------------------------------------------------------------------
# Per-task evaluation, as dispatched by inference/infer_single.py
# ---------------------------------------------------------------------------


def resolve_scienceqa(dataset: Sequence[str]) -> dict[str, list[str]]:
    """Official ``eval_ScienceQA.resolve``; empty input handled per D2."""

    answers = [item[0] if item else "" for item in dataset]
    reasonings = [item[2:] for item in dataset]
    return {"answers": answers, "reasonings": reasonings}


def _eval_accuracy(_sources, predictions, truths) -> dict[str, float]:
    return {"accuracy": caculate_accuracy(predictions, truths)}


def _eval_meetingbank(_sources, predictions, truths) -> dict[str, float]:
    return {
        "bleu-1": caculate_bleu(predictions, truths, 1),
        "bleu-4": caculate_bleu(predictions, truths, 4),
        "rouge-L": caculate_rouge(predictions, truths),
    }


def _eval_py150(_sources, predictions, truths) -> dict[str, float]:
    outputs = [postprocess_py150(item) for item in predictions]
    targets = [postprocess_py150(item) for item in truths]
    return {"similarity": caculate_fuzz(outputs, targets)}


def _eval_scienceqa(_sources, predictions, truths) -> dict[str, float]:
    outputs = resolve_scienceqa(predictions)
    targets = resolve_scienceqa(truths)
    return {
        "bleu-1": caculate_bleu(outputs["reasonings"], targets["reasonings"], 1),
        "bleu-4": caculate_bleu(outputs["reasonings"], targets["reasonings"], 4),
        "rouge-L": caculate_rouge(outputs["reasonings"], targets["reasonings"]),
        "accuracy": caculate_accuracy(outputs["answers"], targets["answers"]),
    }


def _eval_20minuten(sources, predictions, truths) -> dict[str, float]:
    return {
        "bleu-1": caculate_bleu(predictions, truths, 1),
        "bleu-4": caculate_bleu(predictions, truths, 4),
        "rouge-L": caculate_rouge(predictions, truths),
        "sari": caculate_sari(sources, predictions, truths)["sari"],
    }


_EVALUATORS: Mapping[str, Callable[..., dict[str, float]]] = {
    "C-STANCE": _eval_accuracy,
    "FOMC": _eval_accuracy,
    "MeetingBank": _eval_meetingbank,
    "Py150": _eval_py150,
    "ScienceQA": _eval_scienceqa,
    "NumGLUE-cm": _eval_accuracy,
    "NumGLUE-ds": _eval_accuracy,
    "20Minuten": _eval_20minuten,
}


def evaluate_task(
    task: str,
    sources: Sequence[str],
    predictions: Sequence[str],
    ground_truths: Sequence[str],
) -> dict[str, float]:
    """The official evaluation dictionary for one TRACE task.

    ``sources`` are the prompts; only 20Minuten (SARI) reads them. Unknown task
    names raise instead of returning ``{}`` as upstream does: an empty dict
    would silently drop a task from OP.
    """

    if task not in _EVALUATORS:
        raise KeyError(f"tarefa TRACE desconhecida: {task!r}")
    if len(sources) != len(predictions):
        raise ValueError("sources e predictions com tamanhos diferentes")
    return _EVALUATORS[task](list(sources), list(predictions), list(ground_truths))


def primary_score(task: str, evaluation: Mapping[str, float]) -> float:
    """The [0, 1] number this task contributes to OP/BWT (D5)."""

    key = TRACE_PRIMARY_METRIC[task]
    if key not in evaluation:
        raise KeyError(f"{task}: métrica primária {key!r} ausente da avaliação")
    value = float(evaluation[key]) / _PRIMARY_SCALE[key]
    if not 0.0 <= value <= 1.0 + 1e-9:
        raise ValueError(f"{task}: escore primário fora de [0, 1]: {value}")
    return min(value, 1.0)


# ---------------------------------------------------------------------------
# OP and BWT (paper eqs. 2-3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TraceCLMetrics:
    """TRACE's sequence metrics after the LAST task.

    ``backward_transfer`` follows the PAPER's eq. 3, which divides by ``t``
    (all tasks), not ``t - 1`` (old tasks). ``backward_transfer_old_mean`` is
    the conventional ``t - 1`` mean used by :func:`dual_heater.metrics.compute_cl_metrics`.
    The two differ by the factor ``(t - 1) / t``; both are reported so neither
    convention is silently substituted for the other.
    """

    overall_performance: float
    backward_transfer: float
    backward_transfer_old_mean: float
    per_task_final: tuple[float, ...]


def trace_cl_metrics(matrix: Sequence[Sequence[float | None]]) -> TraceCLMetrics:
    """OP_t = (1/t) sum_{i<=t} R[t,i];  BWT_t = (1/t) sum_{i<t} (R[t,i] - R[i,i]).

    ``matrix[t][i]`` is the primary score on task ``i`` after training through
    task ``t``. Only the lower triangle and diagonal are read.
    """

    task_count = len(matrix)
    if task_count == 0:
        raise ValueError("matriz vazia")
    lower: list[list[float]] = []
    for stage, row in enumerate(matrix):
        if len(row) != task_count:
            raise ValueError("a matriz deve ser quadrada")
        values: list[float] = []
        for task in range(stage + 1):
            raw = row[task]
            if raw is None or not math.isfinite(float(raw)):
                raise ValueError(f"R[{stage}][{task}] ausente ou não finito")
            value = float(raw)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"R[{stage}][{task}] fora de [0, 1]")
            values.append(value)
        lower.append(values)

    last = task_count - 1
    final = tuple(lower[last])
    op = sum(final) / task_count
    drops = [lower[last][task] - lower[task][task] for task in range(last)]
    bwt = sum(drops) / task_count
    bwt_old = sum(drops) / last if last else 0.0
    return TraceCLMetrics(
        overall_performance=op,
        backward_transfer=bwt,
        backward_transfer_old_mean=bwt_old,
        per_task_final=final,
    )
