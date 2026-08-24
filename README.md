# DualHeat / SlowHeat Research Prototype

Research code for neuron-level plasticity mechanisms in continual learning.

> Status: exploratory. The repository does not yet establish that SlowHeat outperforms established continual-learning methods. Historical claims such as “34% less forgetting” are not supported by the corrected protocol and have been removed.

## Current focus

The current research direction is **Functional SlowHeat**: scale-invariant
neuron utility, factorized path protection, capacity budgeting and explicit
optimizer-state semantics.

SlowHeat maintains one importance value per output unit:

1. During backward it tracks normalized first-order utility
   `|z * dL/dz|`, not activation magnitude alone.
2. At a task boundary it consolidates utility into persistent evidence.
3. A layer-wise rank budget guarantees a minimum fraction of plastic neurons.
4. Factorized masks protect both the incoming row and outgoing columns of an
   important neuron.
5. `SlowHeatAdamW` or `SlowHeatSGD` applies the mask to both the final parameter
   delta and, by default, tensor-valued optimizer-state deltas.

The mechanism is not EWC. It does not use Fisher information, old-parameter anchors or a quadratic restoring penalty.

## Installation

```bash
python -m pip install -e .
```

Development dependencies:

```bash
python -m pip install -e '.[dev]'
```

Research notebook dependencies:

```bash
python -m pip install -e '.[research]'
```

The ready-to-run Split-MNIST class-incremental notebook is
`notebooks/functional_slowheat_split_mnist.ipynb`. It compares the complete
method at `beta = 10, 30, 100` with vanilla AdamW, exact hard-freeze, replay,
distillation and combined variants. It reports class-incremental and task-aware
matrices, locates classifier interference, plots the stability-plasticity
trade-off and aggregates ten paired seeds. It also compares 1, 5 and 10
epochs for replay and hidden-only SlowHeat + replay variants with lower beta,
larger plasticity budgets and paired differences against replay.

The same ten-seed protocol can be launched without Jupyter:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
PYTHONPATH=src:. python -m experiments.run_split_mnist_10seeds --device cpu
```

Per-seed outputs and `aggregate.csv` are written to
`results/split_mnist_10seeds/` by default.

The frozen independent-confirmation protocol and expanded baseline suite are
in `notebooks/split_mnist_confirmatory_suite.ipynb`. That notebook exposes
DER++, ER-ACE, A-GEM, EWC, SI, calibrated LwF, balanced replay,
equal-epoch/equal-example comparisons, early stopping and compute accounting
through one interface. Read `docs/confirmatory_protocol.md` before running it;
the notebook is committed without executed cells, and no confirmatory result
artifacts are versioned in this repository.

The complete chronological record of Split-MNIST tests, statistical
conclusions and the meaning of structured method names such as
`slowheat_replay_hidden_beta_30_budget_0.25` is available in
`docs/split_mnist_experiment_log.md`.

## Minimal usage

```python
from dual_heater import SlowHeatAdamW, SlowHeatMLP

model = SlowHeatMLP(
    128, 64, 32, 10,
    slow_strength=3.0,
    plasticity_budget=0.25,
    protect_output=True,
)
optimizer = SlowHeatAdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-2,
    state_policy="follow_update",
)
optimizer.register_slow_heat_model(model)
```

For images, the same lifecycle is available through a true convolutional
network. Its adaptive pooling keeps the Conv→Linear channel mapping explicit:

```python
from dual_heater import SlowHeatAdamW, SlowHeatCNN

model = SlowHeatCNN(
    in_channels=3,
    num_classes=10,
    channels=(32, 64),
    pooled_size=(2, 2),
    slow_strength=3.0,
    plasticity_budget=0.25,
)
optimizer = SlowHeatAdamW(model.parameters(), lr=1e-3)
optimizer.register_slow_heat_model(model)
```

After each task:

```python
model.consolidate(strategy="max")
```

Available experimental consolidation strategies:

- `max`: elementwise maximum across task statistics;
- `mean`: running mean across task statistics;
- `sum`: unnormalized accumulated importance.

The persistent evidence is converted to a `[0, 1]` protection vector after
each consolidation, so at least `plasticity_budget` of every layer remains
unprotected. `adapt_capacity()` can update that budget from a separately held
out validation acquisition score. Test-set scores must not drive the controller.

See `docs/functional_slowheat.md` for the method contract.

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

With the default `state_policy="follow_update"`, tensor-valued moment deltas
are interpolated by the same mask as the parameter delta. The ablation
`state_policy="native"` retains the previous behavior in which moments evolve
from the native optimizer trajectory.

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

The canonical synthetic runner is class-incremental with one shared output
head. Inference receives no task ID, but SlowHeat receives oracle task-boundary
events to call `consolidate()`. Logits for future classes are masked during
training and evaluation, leaving their classifier rows untouched until their
classes arrive:

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

A tiny three-seed CPU pilot was used to validate the previous activation-based
implementation and expose confounds. It is historical diagnostic evidence and
must not be presented as a result of Functional SlowHeat.

Main observation:

- optimizer-aware MAX reduced measured forgetting;
- final average accuracy dropped substantially;
- the result is a stability-plasticity trade-off, not a general improvement;
- vanilla showed little forgetting and positive backward transfer, so the pilot has a forgetting floor and is not suitable for efficacy claims.

The SGD and reduced-learning-rate variants in this pilot are diagnostics, not fair tuned comparisons. The current SGD comparison lacks matched `vanilla_sgd` and `slowheat_none_sgd` controls. MAX, mean and sum also require separate calibration of effective protection strength.

See `docs/synthetic_ablation_pilot.md`.

## Known limitations

- No completed result on a harder visual or language continual-learning
  benchmark is versioned. Partial Split-CIFAR-10 per-seed artifacts are kept as
  execution diagnostics; there is no completed CIFAR aggregate or Split-CIFAR-100
  result.
- Replay, DER++, ER-ACE, A-GEM, EWC, SI and calibrated LwF are implemented in
  the shared Split-MNIST/visual runner, but they have not all received
  method-specific tuning or independent replication. MAS, UCB, HAT, NAI,
  SLNID and joint-training controls are not implemented.
- The output classifier is protected by default, but classifier expansion and
  alignment have not yet been evaluated on a standard benchmark.
- The pilot uses only three seeds and a simple Gaussian dataset.
- The superseded synthetic pilot is archived under
  `artifacts/synthetic_ablation_pilot/`. Existing exploratory outputs under
  `results/` remain versioned as historical evidence, while new generated
  results are ignored. Current runners write `environment.json` with package,
  Python, platform and Git provenance alongside new outputs.
- LoRA output masking does not guarantee independent protection of every output because `lora_A` is shared across outputs.
- The forward inhibition mechanism is a train-only regularizer; evaluation uses the uninhibited function.
- Runtime and memory scalability have not been established. Persistent
  importance state is per unit, but parameter and optimizer-state snapshots
  require temporary parameter-scale memory.
- The adaptive capacity API requires a held-out validation signal; the
  synthetic runner currently uses a fixed predeclared budget to avoid test
  leakage.
- Functional importance is first-order and local. It is scale-invariant under
  reciprocal homogeneous-neuron reparameterization, but it is not a causal or
  second-order importance oracle.

## Safe claims

Supported:

- raw-gradient scaling is not equivalent to final-update scaling under AdamW;
- the corrected wrappers mask the complete AdamW/SGD parameter delta and can
  make tensor-valued optimizer state follow that mask;
- `|z * dL/dz|` passes a reciprocal ReLU reparameterization test and assigns
  zero utility to a dead ReLU unit;
- factorized registration protects output rows and downstream input columns;
- capacity budgeting enforces a minimum realized plastic fraction;
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
  _layers.py         shared activation and MLP validation helpers

experiments/
  split_mnist.py                 shared benchmark and baseline engine
  split_mnist_suite.py           fairness, ablations and orchestration
  confirmatory_split_mnist.py    frozen confirmation entry point
  visual_generalization.py       Permuted-MNIST and Split-CIFAR adapters
  synthetic_cl.py                deterministic synthetic benchmark
  multi_seed.py                  serial synthetic aggregation
  provenance.py                  environment and Git metadata for new runs

tests/               unit and integration tests
docs/                method contracts, protocols and experiment records
notebooks/           interactive experiment entry points
article/manuscript.md technical manuscript draft
```

Documentation entry points:

- `docs/functional_slowheat.md`: method contract;
- `docs/optimizer_semantics.md`: masking and checkpoint semantics;
- `docs/confirmatory_protocol.md`: frozen confirmation and baseline suite;
- `docs/split_cifar.md`: exact Split-CIFAR-10/100 protocol;
- `docs/reproducibility.md`: synthetic protocol;
- `docs/split_mnist_experiment_log.md`: chronological experimental record;
- `docs/project_methods_and_results.md`: catalog and analysis of the external
  exploratory result export;
- `docs/synthetic_ablation_pilot.md`: superseded diagnostic pilot.

## Verification

The complete protocol from
`notebooks/split_mnist_confirmatory_suite.ipynb` can be executed without
Jupyter. Preview every section, method, seed and output path without training:

```bash
python run_all_tests.py --num-seeds 10 --dry-run
```

Run every compatible method on every dataset/stream in the project:

```bash
python run_all_tests.py --num-seeds 10 --all-datasets-all-methods --device cpu --no-download
```

This selects the synthetic benchmark plus Split-MNIST, Permuted-MNIST,
Split-CIFAR-10 and Split-CIFAR-100. It runs all 11 synthetic methods on the
synthetic stream and all 32 visual methods on each visual stream.
`--num-seeds` is required and generates that many distinct, reproducible
pseudorandom secondary seeds. The synthetic runner is CPU-only.

Run the complete confirmatory, baseline, fairness, ablation, DER++, Split-MNIST
generalization, Permuted-MNIST, Split-CIFAR-10 and Split-CIFAR-100 protocol.
This command does not run pytest:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_all_tests.py --num-seeds 10 --device cpu
```

Runs resume matching completed seeds by default. Use `--sections` to execute a
subset, for example the separate SlowHeat+DER++ comparison:

```bash
python run_all_tests.py --num-seeds 10 --sections slowheat-derpp
```

The CIFAR streams can also be run independently. By default, torchvision
downloads their data under `data/`:

```bash
python run_all_tests.py --num-seeds 10 --sections split-cifar10 split-cifar100
```

Split-CIFAR-10 uses five Class-IL tasks with two classes each;
Split-CIFAR-100 uses ten Class-IL tasks with ten classes each. Evaluation does
not receive a task ID. Images are normalized and flattened for the repository's
paired MLP engine, so these runs test harder visual streams but are not CNN
benchmarks. See [`docs/split_cifar.md`](docs/split_cifar.md) for the exact
protocol.

The small real-CNN pilot is a separate, opt-in section so it cannot alter the
historical flattened-MLP runs:

```bash
PYTHONPATH=src:. python run_all_tests.py \
  --num-seeds 3 \
  --sections split-cifar10-cnn \
  --device cuda
```

It runs five paired methods (`vanilla`, no-consolidation, destination-only,
factorized SlowHeat and hard freeze) for five epochs per task. Use `--dry-run`
to inspect the exact protocol without downloading or training.

Each visual all-methods section runs all 32 methods implemented or explicitly
configured by the project, including vanilla, the SlowHeat controls and beta
variants, Replay, distillation, DER++, ER-ACE, A-GEM, EWC, SI, calibrated LwF,
fairness controls and the executable ablations. Use `--dry-run` to inspect the
exact ordered list before starting this computationally expensive suite.
Partial Split-CIFAR-10 seed outputs are versioned, but the interrupted run has
no aggregate and must not be presented as a completed CIFAR result.

Unit/integration tests are optional and only run when explicitly requested
with `--run-unit-tests`.

Progress is recorded in
`results/split_mnist_protocol/benchmark_index.json`; the frozen primary result
is extracted to `results/split_mnist_protocol/primary_result.json`.

The complete default protocol is computationally expensive: it contains 20
frozen confirmatory seeds plus repeated ten-seed secondary analyses.

Individual verification commands remain available:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=src:. pytest -q -p no:cacheprovider

ruff check .
```

## Manuscript

The current technical draft is available at:

`article/manuscript.md`

It is intentionally conservative and lists the experiments required before submission.
