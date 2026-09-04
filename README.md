# DualHeat / SlowHeat Research Prototype

Research code for neuron-level plasticity mechanisms in continual learning.

> Status: exploratory. The repository does not yet establish that SlowHeat outperforms established continual-learning methods. Historical claims such as “34% less forgetting” are not supported by the corrected protocol and have been removed.

## Current focus

The repository now distinguishes **Functional SlowHeat**, the new
**Functional DualHeat** (activation FastHeat + Functional SlowHeat), and the
historical `DualHeatLinear`/`DualHeatMLP` API.

Ten-seed exploratory Split-CIFAR-10 benchmarks are now available for VGG11 and
ResNet18. Functional DualHeat did not produce a multiplicity-adjusted gain in
the primary final-average-accuracy contrasts. On VGG11, DualHeat minus SlowHeat
was `-0.00483`, while DualHeat+LPR minus SlowHeat+LPR was `+0.01430`; both had
Holm-adjusted `p = 0.0803`. All four ResNet18 contrasts were close to zero and
had Holm-adjusted `p = 1.0`. The VGG11 DualHeat comparison reduced average
forgetting by `0.02488` while reducing final accuracy, so the evidence supports
a stability-plasticity trade-off rather than superiority. These results are
exploratory, and all recorded paired fairness checks passed.

A runner audit on 2026-09-02 found that epoch-end evaluation entered eval mode
without restoring the learner's previous mode. SlowHeat sweeps without any
FastHeat method consequently accumulated functional importance only during the
first epoch of each stage. Any frozen Split-MNIST confirmation executed before
the correction is invalid and must not be reported. The correction preserves
the preregistered seeds, hyperparameters, endpoint and analysis; a valid
confirmation must use the corrected source and a new output directory. See
[`docs/confirmatory_protocol.md`](docs/confirmatory_protocol.md) and the
[experiment log](docs/split_mnist_experiment_log.md).

### Method versus method + DualHeat

The older dedicated MLP comparison is retained under its historical artifact
name, but it evaluates **Functional SlowHeat only** as an add-on to four
learners. It does not evaluate Functional DualHeat or the legacy
`DualHeatMLP` class.
Each learner is compared only with its own augmented counterpart, using the
same initialization, data, training/replay schedules and epoch budget.

```bash
python run_dualheat_pairs.py --datasets split_mnist --num-seeds 10 --device cpu
```

Run from the project root; no machine-specific paths or `PYTHONPATH` setup
are needed. This direct entry writes to `results/dualheat_pairs/split_mnist/`.
Add `--dry-run` to preview without training, or `--no-download` if the data is
already available. Reports store their source paths relative to the report
directory so the result tree can be moved to another machine. Multi-seed runs
also store `run_identity.json`; resume is accepted only when the requested
seeds, loader, configuration and source fingerprint still match. Older result
trees without this identity remain readable but are not resumed. Each seed also
stores `data_identity.json`, containing a hash of the exact task tensors used.

Alternatively, use the general protocol runner:

```bash
python run_all_tests.py --num-seeds 10 --sections dualheat-pairs --dry-run
python run_all_tests.py --num-seeds 10 --sections dualheat-pairs --device cpu --no-download
```

This opt-in section does not change the default suite or frozen confirmation.
It writes `pair_report.md`, `pair_report.json`, `pair_summary.csv` and
`pair_differences.csv` under `results/split_mnist_protocol/dualheat_pairs/`.
The report includes paired confidence intervals, accuracy p-values adjusted
for the four comparisons, negative results and observed runtime overhead.
See [the paired protocol](docs/dualheat_paired_protocol.md) for other MLP
datasets, reanalysis of saved results and interpretation limits.

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

Replay can also let each learner rank which training images enter its episodic
memory. The `first`, `loss`, `representative` and `hybrid` policies work with
both MLP and CNN backbones; no-memory controls remain available through
`vanilla` and hidden-only SlowHeat. See
[the replay-selection protocol](docs/replay_selection.md) for configuration,
the ten-seed visual sweep and task-boundary checkpoint behavior.

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
  fast_heat.py       normalized activation gate and online FastHeat state
  slow_heat.py       SlowHeat linear/conv layers and consolidation
  resnet.py          CIFAR ResNet18 controls and Functional DualHeat variant
  lora.py            experimental LoRA adaptation
  optim.py           optimizer-aware update masking
  metrics.py         continual-learning metrics
  _layers.py         shared activation and MLP validation helpers

experiments/
  functional_dualheat.py        pilot, frozen manifest and 13-method reports
  lpr.py                         LPR covariance preconditioner
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

It runs the five original CNN controls plus three normal/SlowHeat pairs:
`lpr`/`slowheat_lpr`,
`classifier_expander`/`slowheat_classifier_expander`, and
`scroll`/`slowheat_scroll`. Use `--dry-run` to inspect the exact protocol
without downloading or training.

The SCROLL paper assumes a suitably pre-trained representation. To keep this
small runner self-contained, both paired SCROLL variants use task 0 as an
explicit representation bootstrap, followed by accumulated ridge-regression
statistics and replay-only representation adaptation. This is a documented
benchmark adaptation, not a claim of exact reproduction of pretrained SCROLL.

The VGG11-CIFAR pilot uses the standard eight-convolution VGG11 feature pattern
without BatchNorm, five max-pooling stages, adaptive `1x1` pooling and one
linear classifier. Its first pilot runs the five core controls (`vanilla`,
`slowheat_none`, `slowheat_unidirectional`, `slowheat` and `hard_freeze`) in a
separate output tree:

```bash
PYTHONPATH=src:. python run_all_tests.py \
  --num-seeds 3 \
  --sections split-cifar10-vgg11 \
  --device cuda
```

This CIFAR-sized adaptation avoids the ImageNet VGG classifier and isolates
depth from BatchNorm running-statistics effects. Its artifacts are written to
`results/split_mnist_protocol/split_cifar10_vgg11/`.

The opt-in deep-CNN sweep runs the same eleven controls and paired auxiliary
methods on VGG11 and CIFAR ResNet-18. ResNet-18 uses a 3x3 stride-1 stem,
GroupNorm, stages `(64,128,256,512)` with `(2,2,2,2)` residual blocks and
global average pooling. Exact LPR is retained with an update interval of 300
optimizer steps to control the covariance-inversion cost:

```bash
PYTHONPATH=src:. python run_all_tests.py \
  --num-seeds 10 \
  --sections split-cifar10-vgg11-all-methods split-cifar10-resnet18-all-methods \
  --device cuda \
  --run-unit-tests
```

This is 220 learner runs (2 architectures x 11 methods x 10 paired seeds).
The new output directories end in `_all_methods`, so historical VGG and
ResNet artifacts cannot be mistaken for resumable runs of this protocol.

The actual Functional DualHeat protocol is separate. It first calibrates
FastHeat on validation only, freezes the selected configuration, and then runs
fresh 13-method VGG11 and ResNet18 benchmarks in new output directories:

```bash
python run_all_tests.py \
  --num-seeds 10 \
  --sections functional-dualheat-pilot \
    split-cifar10-vgg11-functional-dualheat \
    split-cifar10-resnet18-functional-dualheat \
  --device cuda --run-unit-tests
```

See [the Functional DualHeat protocol](docs/functional_dualheat.md) for the
gate equation, placement, fixed seeds, calibration rule and paired contrasts.

After the pilot, run the preselected CNN stability/plasticity sweep with ten
paired seeds:

```bash
PYTHONPATH=src:. python run_all_tests.py \
  --num-seeds 10 \
  --sections split-cifar10-cnn-sweep \
  --device cuda
```

This separate section crosses destination-only and hidden-only protection with
`beta={3,10}` and plasticity budget `{0.50,0.75}`. Its artifacts are written to
`results/split_mnist_protocol/split_cifar10_cnn_sweep/`.

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
