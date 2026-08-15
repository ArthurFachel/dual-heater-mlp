# Synthetic continual-learning protocol

The canonical synthetic experiment is `experiments/synthetic_cl.py`. It is a class-incremental benchmark with one shared output head. Evaluation receives no task ID, but SlowHeat receives oracle task-boundary events for `consolidate()`. The protocol is therefore not task-free.

## Fairness controls

- Every method receives byte-identical trainable parameter initialization.
- Dataset generation and train/test splits are deterministic for each seed.
- Minibatch index schedules are precomputed once and reused by every method.
- Every stage evaluates all tasks seen so far and stores the lower triangle of `A[t, k]`.
- Average accuracy, average forgetting, backward transfer and forward transfer are computed by `dual_heater.metrics`.
- FWT compares each task's score immediately before training with its score under the paired random initialization.

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

## Interpretation constraints

Do not compare wall time across different machines. Do not report one-seed results as evidence of superiority. The `reduced_lr` control is mandatory because an apparent gain may come from generic update reduction rather than selective consolidation. The next scientific stage requires multiple seeds, confidence intervals and method-specific ablations.
