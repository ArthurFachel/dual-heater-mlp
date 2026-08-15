# DualHeat / SlowHeat Research Prototype

Research code for neuron-level plasticity mechanisms in continual learning.

> Status: exploratory. The repository does not yet establish that SlowHeat outperforms established continual-learning methods. Historical claims such as “34% less forgetting” are not supported by the corrected protocol and have been removed.

## Current focus

The current research direction is **SlowHeat with task-boundary MAX consolidation** and explicit optimizer-aware update masking.

SlowHeat maintains one importance value per output unit:

1. During a task, it tracks pre-activation magnitude.
2. At a task boundary, it consolidates the task statistic into persistent importance.
3. Persistent importance produces a plasticity factor.
4. `SlowHeatAdamW` or `SlowHeatSGD` applies that factor to the optimizer’s final parameter update.

The mechanism is not EWC. It does not use Fisher information, old-parameter anchors or a quadratic restoring penalty.

## Installation

```bash
python -m pip install -e .
```

Development dependencies:

```bash
python -m pip install -e '.[dev]'
```

## Minimal usage

```python
from dual_heater import SlowHeatAdamW, SlowHeatLinear

layer = SlowHeatLinear(128, 64, slow_strength=3.0)
optimizer = SlowHeatAdamW(
    layer.parameters(),
    lr=1e-3,
    weight_decay=1e-2,
)
optimizer.register_slow_heat_module(layer)
```

After each task:

```python
layer.consolidate(strategy="max")
```

Available experimental consolidation strategies:

- `max`: elementwise maximum across task statistics;
- `mean`: running mean across task statistics;
- `sum`: unnormalized accumulated importance.

These strategies produce different importance scales. Comparing them with the same `slow_strength` does not guarantee matched regularization strength.

## Why optimizer-aware masking is necessary

The legacy implementation multiplied raw gradients by:

```text
1 / (1 + beta * slow_heat)
```

That directly scales an update only for simple SGD-like cases. Adam and AdamW normalize gradients through first and second moments, so a persistent multiplicative gradient factor can largely cancel. AdamW’s decoupled weight decay can also move a parameter independently of raw-gradient masking.

The corrected optimizers:

1. compute the native optimizer update;
2. measure the resulting parameter delta;
3. apply the plasticity mask to that final delta, including weight decay.

Mask `1` preserves the native update. Mask `0` blocks it.

Optimizer moments still evolve from unmasked gradients. This is an explicit experimental design choice, not a proven optimal rule. It requires ablation against state-masking alternatives.

See `docs/optimizer_semantics.md`.

## Metrics

`dual_heater.metrics.compute_cl_metrics` consumes a stage-by-task accuracy matrix `A[t,k]`.

It reports:

- final average accuracy;
- peak-based average forgetting over old tasks;
- diagonal-based backward transfer;
- forward transfer using pretraining scores minus random/chance baselines;
- forgetting per old task.

The API validates matrix shape, finitude and accuracy range `[0,1]`.

## Reproducible synthetic protocol

The canonical synthetic runner is class-incremental with one shared output head. Inference receives no task ID, but SlowHeat receives oracle task-boundary events to call `consolidate()`:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
PYTHONPATH=src:. python -m experiments.synthetic_cl \
  --config configs/synthetic_smoke.json \
  --output-dir results/synthetic_smoke
```

Methods receive:

- identical trainable parameter initialization;
- identical generated data and splits;
- identical minibatch schedules;
- the same evaluation points.

A serial multi-seed runner is available:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
PYTHONPATH=src:. python -m experiments.multi_seed \
  --config configs/synthetic_ablation_pilot.json \
  --seeds 11 22 33 \
  --output-dir results/synthetic_ablation_pilot
```

See `docs/reproducibility.md`.

## Diagnostic pilot

A tiny three-seed CPU pilot was used to validate wiring and expose confounds. It is not publication evidence.

Main observation:

- optimizer-aware MAX reduced measured forgetting;
- final average accuracy dropped substantially;
- the result is a stability-plasticity trade-off, not a general improvement;
- vanilla showed little forgetting and positive backward transfer, so the pilot has a forgetting floor and is not suitable for efficacy claims.

The SGD and reduced-learning-rate variants in this pilot are diagnostics, not fair tuned comparisons. The current SGD comparison lacks matched `vanilla_sgd` and `slowheat_none_sgd` controls. MAX, mean and sum also require separate calibration of effective protection strength.

See `docs/synthetic_ablation_pilot.md`.

## Known limitations

- No standard image or language continual-learning benchmark has been completed with the corrected optimizer.
- No specialized replay, EWC, SI, MAS, UCB, HAT, NAI or SLNID baseline is implemented in the corrected protocol.
- The shared classifier is currently unprotected in `SlowHeatMLP`.
- The pilot uses only three seeds and a simple Gaussian dataset.
- Raw pilot artifacts are generated under ignored `results/` directories and must be archived with environment and Git metadata before publication.
- LoRA output masking does not guarantee independent protection of every output because `lora_A` is shared across outputs.
- The forward inhibition mechanism is a train-only regularizer; evaluation uses the uninhibited function.
- Runtime and memory scalability have not been established. Importance state is per unit, but applying masks and cloning optimizer deltas touches protected parameters and requires temporary parameter-scale memory.

## Safe claims

Supported:

- raw-gradient scaling is not equivalent to final-update scaling under AdamW;
- the corrected wrappers mask the complete AdamW/SGD parameter delta;
- the synthetic runner pairs initialization and minibatches;
- a tiny diagnostic pilot exposed a stability-plasticity trade-off.

Not supported:

- a general reduction of forgetting by 34%;
- superiority to EWC or other continual-learning baselines;
- novelty of MAX consolidation by itself;
- guaranteed convergence, specialization or neuron recruitment;
- validated effectiveness on convolutional networks, transformers or real-world tasks.

## Project structure

```text
src/dual_heater/
  dual_heat.py       legacy DualHeat mechanism
  slow_heat.py       SlowHeat linear/conv layers and consolidation
  lora.py            experimental LoRA adaptation
  optim.py           optimizer-aware update masking
  metrics.py         continual-learning metrics

experiments/
  synthetic_cl.py    deterministic synthetic benchmark
  multi_seed.py      serial multi-seed aggregation

tests/               unit and integration tests
docs/                optimizer, protocol and pilot documentation
article/manuscript.md technical manuscript draft
```

## Verification

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=src:. pytest -q -p no:cacheprovider

ruff check .
```

## Manuscript

The current technical draft is available at:

`article/manuscript.md`

It is intentionally conservative and lists the experiments required before submission.
