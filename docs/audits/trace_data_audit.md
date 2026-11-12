# TRACE data audit

Release audited: official TRACE archive, Google Drive id
`1S0SmU0WEw5okW_XvP2Ns0URflNzZq6sV` (linked from `BeyonderXX/TRACE` README),
downloaded 2026-10-05. `TRACE-Benchmark.zip`, sha256
`11152d50f8a093b1fc9c6c924ec207915d173b9e2d715396ec8f8a837e2668a8`, 76.4 MB.
Extracted under `data/TRACE_raw/` (gitignored, 334 MB). The archive holds four
variants (`_500`, `_1000`, `_5000`, `_Reasoning`) plus `Lima` (replay data).
`data/TRACE` points at `LLM-CL-Benchmark_5000`, the one the paper and
`scripts/train_*.sh` use.

Every number below comes from `docs/audits/trace_data_audit.json`, produced by
`scripts/audit_trace_data.py` with the Qwen2.5-0.5B-Instruct tokenizer and the
official limits (`max_prompt_len=1024`, `max_ans_len=512`). Per-file sha256 of
every split is in the JSON.

## Contents vs. what the paper says

The paper (Table 4, §4.1) states 5,000 train and 2,000 test examples per task.
The release does not match on either:

| task | train | unique train prompts | test | test prompts in train | eval = test | train truncated | test prompt truncated | official label mask wrong | split label mask wrong |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| C-STANCE | 5000 | 5000 | 2000 | 0 | no | 0.0% | 0.0% | 0.0% | 0.0% |
| FOMC | 5000 | 1915 | 496 | 16 | yes | 0.0% | 0.0% | 100.0% | 0.0% |
| MeetingBank | 5000 | 5000 | 692 | 1 | no | 45.6% | 55.2% | 57.1% | 0.0% |
| Py150 | 5000 | 4634 | 2000 | 213 | no | 12.4% | 19.8% | 24.6% | 0.0% |
| ScienceQA | 5000 | 5000 | 2000 | 0 | no | 0.0% | 0.0% | 100.0% | 0.0% |
| NumGLUE-cm | 5000 | 282 | 81 | 0 | no | 0.0% | 0.0% | 0.7% | 0.0% |
| NumGLUE-ds | 5000 | 1069 | 325 | 34 | no | 0.0% | 0.0% | 0.0% | 0.0% |
| 20Minuten | 5000 | 5000 | 200 | 0 | no | 2.1% | 15.0% | 18.3% | 0.0% |

What this means:

1. **"5,000 examples" is 5,000 rows, not 5,000 examples.** NumGLUE-cm has 282
   distinct prompts (each repeated up to 20x), NumGLUE-ds 1,069, FOMC 1,915.
   Five epochs on NumGLUE-cm is roughly 88 passes over each distinct item.
   Memorization on these tasks is not comparable to the others.
2. **Test leaks into train.** Py150: 213/2000 (10.7%) test prompts appear in
   train, 154 with the identical answer. NumGLUE-ds: 34/325 (10.5%), all 34
   identical. FOMC: 16/496. Scores on these tasks partly measure recall of
   training items. Not corrected here; reported so it is not read as
   generalization.
3. **Test sizes are 81-2000, not 2000.** NumGLUE-cm has 81 test items: one
   item is 1.2 points of accuracy. Per-task forgetting on it has a coarse grid.
4. **FOMC `eval.json` is a reshuffle of `test.json`.** Any model selection on
   "eval" is selection on test. The runner never reads `eval.json`.
5. **The 1024-token prompt limit cuts MeetingBank in half.** 55% of test
   prompts are truncated (from the left, as upstream, so the transcript's
   beginning is lost). Py150 20%, 20Minuten 15%.

## Label mask (deviation R10)

Upstream's collator supervises the last `len(tokens(answer)) + 1` tokens of
`tokens(prompt + answer)`. Qwen's BPE merges the prompt's final `":"` with a
leading letter (`"Stance:" + "C"` -> `":C"`), so on FOMC and ScienceQA
**every** training example supervises `":C"` after `"Stance"`, while at
inference the prompt already ends in `":"` and the model must emit `"C"`, a
transition it never trained on. The `split` tokenization (default in this
repo) tokenizes prompt and answer separately: 0.0% span mismatch and 0.0%
context mismatch on every task, measured through the real collator. The
`official` mode is kept and pinned by test.

This is a property of the Qwen tokenizer. Whether LLaMA-2's SentencePiece hits
the same merge in the authors' runs is not measured here.

## Metric floors

Official scorer, primary metric, test split. Any near-chance gate must be set
per task against these, not against zero:

| task | oracle | empty prediction | majority train answer | copy of prompt |
|---|---:|---:|---:|---:|
| C-STANCE | 1.000 | 0.000 | 0.348 | 0.000 |
| FOMC | 1.000 | 0.000 | 0.494 | 0.000 |
| MeetingBank | 1.000 | 0.000 | 0.110 | 0.111 |
| Py150 | 1.000 | 0.000 | 0.042 | 0.107 |
| ScienceQA | 1.000 | 0.000 | 0.440 | 0.094 |
| NumGLUE-cm | 1.000 | 0.000 | 0.025 | 0.000 |
| NumGLUE-ds | 1.000 | 0.000 | 0.132 | 0.000 |
| 20Minuten | 0.996 | **0.370** | 0.387 | 0.353 |

- **20Minuten rewards emitting nothing**: an empty prediction scores 0.370
  SARI/100, above copying the source (0.353). A collapsed arm still collects
  ~0.05 of OP from this task alone (0.370 / 8). The oracle is 0.996, not 1.0,
  because SARI also scores against the source.
- **ScienceQA's primary metric is the first character only**; predicting the
  most common letter gives 0.440.
- **FOMC majority class is 0.494** (C = neutral, 245/496), so "accuracy 0.5"
  on FOMC is the floor, not half of the ceiling.
- Mean of the majority floors: 0.247. A model that learned only label priors
  would report OP ~= 0.25.

## Cost

A full protocol evaluates tasks 0..t after every task t: 42,717 test
generations per arm, with `max_new_tokens=512`. Training is 86.3M tokens per
arm over all official epochs (`experiments.trace_cl --dry-run`). Neither has
been timed on GPU.
