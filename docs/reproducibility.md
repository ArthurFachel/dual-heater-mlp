# Synthetic continual-learning protocol

The canonical synthetic experiment is `experiments/synthetic_cl.py`. It is a
class-incremental benchmark with one shared output head. Evaluation receives no
task ID, but SlowHeat receives oracle task-boundary events for `consolidate()`.
The protocol is therefore not task-free.

The classifier has final capacity from initialization for paired parameter
comparisons, but logits beyond the classes seen at a stage are sliced out of
both cross-entropy and evaluation. Future classifier rows consequently receive
no gradient before their classes arrive. Scores for a future task use the
paired random model with the same stage-specific visible-class count.

## Fairness controls

- Every method receives byte-identical trainable parameter initialization.
- Dataset generation and train/test splits are deterministic for each seed.
- Minibatch index schedules are precomputed once and reused by every method.
- Every stage evaluates all tasks seen so far and stores the lower triangle of `A[t, k]`.
- Average accuracy, average forgetting, backward transfer and forward transfer are computed by `dual_heater.metrics`.
- FWT compares each task's score immediately before training with its score under the paired random initialization.
- SlowHeat protects the output head and registers factorized row/column masks.
- Results record realized protected and plastic fractions for every SlowHeat layer.
- `optimizer_state_policy` and `plasticity_budget` are frozen in `config.json`.

## Lightweight CPU smoke run

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
PYTHONPATH=src:. python -m experiments.synthetic_cl \
  --config configs/synthetic_smoke.json \
  --output-dir results/synthetic_smoke
```

The smoke configuration is intentionally too small for scientific claims. It validates wiring and artifacts only.

Each run writes:

- `config.json`: frozen configuration;
- `results.json`: accuracy matrices, losses, pretraining scores, metrics and timings;
- `summary.csv`: one-row-per-method summary.

The serial multi-seed wrapper writes one subdirectory per seed plus
`aggregate.json` and `multi_seed_config.json`:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
PYTHONPATH=src:. python -m experiments.multi_seed \
  --config configs/synthetic_ablation_pilot.json \
  --seeds 11 22 33 \
  --output-dir results/synthetic_ablation_pilot
```

The superseded three-seed pilot is preserved separately in
`artifacts/synthetic_ablation_pilot/`, including its environment manifest. Do
not confuse that archive with outputs from the current functional method.

## Interpretation constraints

Do not compare wall time across different machines. Do not report one-seed
results as evidence of superiority. The `reduced_lr` control is mandatory
because an apparent gain may come from generic update reduction rather than
selective consolidation. `slowheat_max_native_state`,
`slowheat_max_unidirectional` and `slowheat_max_unbudgeted` isolate the new
components. Scientific use requires multiple seeds, paired confidence
intervals and a validation-only capacity signal. Specialized baselines are
available in the separate Split-MNIST/visual engine, not in this synthetic
runner; results from the two protocols must not be combined as if they shared
a dataset or training budget.
