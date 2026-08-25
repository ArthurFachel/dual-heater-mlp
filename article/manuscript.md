# Functional SlowHeat: Scale-Invariant Neuron Utility, Factorized Protection and Capacity-Aware Plasticity

## Status

Technical manuscript draft. The current implementation is validated by unit
and integration tests. Exploratory Split-MNIST comparisons against Replay,
DER++, ER-ACE, A-GEM, EWC, SI and calibrated LwF are documented and their raw
per-seed artifacts are versioned, but the frozen independent confirmation has
no versioned execution artifacts. Split-CIFAR-10/100 support is implemented.
A local ten-seed real-CNN screening export contains paired differences but
not complete accuracy matrices or an environment manifest; the historical
flattened-MLP Split-CIFAR-10 run remains partial and non-aggregate. The
three-seed pilot in Section 5 predates the
functional-importance, factorized-protection, capacity-budget and
optimizer-state changes; it is retained only as historical motivation and
must not be reported as evidence for the current method.

## Abstract

Neuron-level importance masks are an appealing way to regulate the
stability-plasticity trade-off without replay, but activation magnitude is not
invariant to function-preserving neuron reparameterization and row-only masks do
not protect outgoing connectivity. Functional SlowHeat estimates normalized
first-order neuron utility with `|z dL/dz|`, consolidates persistent evidence,
derives protection under an explicit plastic-capacity budget and applies a
factorized mask to both incoming rows and downstream columns. Optimizer-aware
AdamW and SGD wrappers apply the mask to the final parameter delta and, under
the default policy, to tensor-valued optimizer-state deltas. The method is
implemented and covered by falsification tests. Exploratory Split-MNIST
comparisons and a ten-seed Split-CIFAR-10 CNN screening exist. A graph-aware
CIFAR ResNet-18 with GroupNorm is implemented but has no multi-seed result.
Independent confirmation and Split-CIFAR-100 CNN evaluation remain
outstanding. No general efficacy or state-of-the-art claim is made.

## 1. Motivation

Continual learners must preserve performance on previously learned tasks while retaining enough plasticity to acquire new ones. Parameter-importance methods typically protect parts of a model that matter for old tasks. Examples include Elastic Weight Consolidation (EWC), Synaptic Intelligence (SI), activation-based neuron importance and uncertainty-guided learning-rate adaptation.

SlowHeat explores a local mechanism with one persistent value per output
neuron. The original prototype tracked pre-activation magnitude. The current
method instead tracks normalized first-order functional utility during backward
and treats the original statistic as a required ablation.

The initial implementation and benchmark were not sufficient to support an efficacy claim. They mixed an imprecise forgetting metric, unmatched model initialization and raw-gradient masking under AdamW. This article is therefore framed around methodological correction and diagnostic evidence, not around a claim that SlowHeat outperforms established continual-learning methods.

## 2. Method

### 2.1 Within-task importance

For output unit `i` at optimization step `s`, let `z_i^(s)` be its
pre-activation. Functional SlowHeat tracks:

```text
u_i <- sum_samples |z_i * dL/dz_i|
u_normalized,i <- u_i / (mean_j(u_j) + epsilon)
h_task,i <- EMA(u_normalized,i)
```

For linear layers, the reduction covers every leading dimension and preserves
the last output dimension. Convolutional layers reduce batch and spatial
dimensions while preserving channels. Under reciprocal reparameterization of a
positively homogeneous unit, `z` scales by `c` and `dL/dz` by `1/c`, leaving
their product unchanged. A dead ReLU receives zero utility.

### 2.2 Task-boundary consolidation

The main SlowHeat rule is:

```text
h_slow,i <- max(h_slow,i, h_task,i)
```

The maximum is monotonic and preserves the strongest recorded evidence. It is
not intrinsically novel. After evidence consolidation, a rank-based budget
selects at most `floor((1-p)N)` protected units, where `p` is the minimum
plastic fraction. Selected evidence is normalized to `[0,1]`. A bounded
validation-driven controller may increase `p` when acquisition is below a
predeclared target; test accuracy may not drive this controller.

The implementation also supports mean and sum consolidation for controlled ablations. These alternatives are experimental controls, not recommended defaults.

### 2.3 Plasticity mask

Persistent importance is converted into a factor:

```text
m_i = 1 / (1 + beta * h_slow,i)
```

where `beta >= 0` controls protection strength. `m_i = 1` permits the native update. Values near zero strongly protect the corresponding output row and bias.

For `W_l[i,j]`, the final factor is the minimum of the destination-neuron and
source-neuron plasticity factors. This protects both a neuron's incoming row and
its outgoing columns without storing per-weight importance. The mechanism is
not EWC: it contains no Fisher information, old-parameter snapshot or quadratic
restoring force.

## 3. Why Raw Gradient Scaling Is Not AdamW Update Scaling

A raw-gradient hook applies:

```text
g_i <- m_i * g_i
```

For plain SGD without momentum or weight decay, this directly scales the parameter update. Under Adam-like normalization, a persistent positive factor affects the first moment approximately linearly and the second moment approximately quadratically:

```text
m_t proportional to c * g
v_t proportional to c^2 * g^2
m_t / sqrt(v_t) approximately cancels c
```

The cancellation is not exact in every transient regime, but it invalidates the interpretation of the hook as a guaranteed effective learning rate. AdamW also applies decoupled weight decay outside the raw gradient.

The corrected research optimizer first lets AdamW or SGD compute its complete native parameter delta and then applies the plasticity mask:

```text
Delta_native = theta_after_native_step - theta_before_step
Delta_applied = M * Delta_native
theta <- theta_before_step + Delta_applied
```

This contract is directly testable:

- mask `1` matches the native optimizer;
- mask `0` blocks both gradient and weight-decay movement;
- mask `0.1` produces one tenth of the final native update;
- optimizer state survives checkpoint round trips.

The default `follow_update` policy applies the same interpolation to
tensor-valued optimizer-state deltas. The `native` policy retains unmasked
moment evolution as an explicit ablation. Scalar AdamW step counters remain
global, a limitation of using the native PyTorch optimizer state layout.

## 4. Experimental Corrections

### 4.1 Scenario

The synthetic and Split-MNIST protocols are class-incremental: all tasks share
one output head and evaluation does not provide a task identifier. SlowHeat
nevertheless receives oracle task-boundary events to call `consolidate()`. It
is therefore boundary-aware rather than task-free, which limits comparison
with methods that do not receive boundary information. Permuted-MNIST is
domain-incremental. The general Split-CIFAR-10/100 adapters are Class-IL and
flatten normalized images for the same paired MLP engine. A separate opt-in
Split-CIFAR-10 section preserves NCHW tensors and uses a real two-convolution
`3→32→64` network with a shared ten-class head. A second opt-in section uses a
CIFAR-style ResNet-18 with a `3x3` stem, GroupNorm, graph-registered residual
branches and parameter-free post-add importance trackers.

### 4.2 Paired controls

In the synthetic protocol, every method receives:

- byte-identical trainable parameter initialization;
- the same generated dataset and train/test splits;
- the same ordered minibatch schedule;
- the same number of optimization steps;
- the same evaluation points.

The Split-MNIST/visual engine also pairs initialization, partitions, current
minibatch schedules and replay indices within each seed. Equal-epoch and
equal-example analyses are separate: `replay_more_epochs` and
`replay_early_stopping` intentionally do not have the same step count as the
ten-epoch comparison.

`slowheat_none`, which registers the corrected optimizer but never consolidates importance, exactly matches vanilla in the diagnostic pilot. This is a key wiring and fairness check.

### 4.3 Metrics

Let `A[t,k]` be accuracy on task `k` after training through task `t`.

Final average accuracy:

```text
ACC = mean_k A[T-1,k]
```

Forgetting for each old task:

```text
F_k = max_{l=k,...,T-1} A[l,k] - A[T-1,k]
```

Average forgetting excludes the final task because it has no subsequent task over which to forget.

Backward transfer compares final performance with performance immediately after learning each old task. Forward transfer uses performance immediately before training a future task minus a separately measured random-initialization baseline. Accuracy after training the new task is not forward transfer.

### 4.4 Exploratory Split-CIFAR-10 CNN screening

The local `results/paired_differences.csv` export was analyzed on August 25,
2026. It contains ten paired seeds for LPR, Classifier Expander and SCROLL,
each with and without SlowHeat. Differences below are
`method+SlowHeat - method`; negative forgetting and classifier gap are
favorable.

| Paired contrast | Final accuracy | Forgetting | BWT | Task-aware accuracy | Classifier gap |
|---|---:|---:|---:|---:|---:|
| SlowHeat+LPR − LPR | +0.78 pp | −4.66 pp | +4.66 pp | −1.29 pp | −2.07 pp |
| SlowHeat+Classifier Expander − Classifier Expander | −0.88 pp | −5.13 pp | +10.28 pp | −2.05 pp | −1.17 pp |
| SlowHeat+SCROLL − SCROLL | +5.78 pp | −1.25 pp | −4.07 pp | +3.83 pp | −1.95 pp |

SlowHeat+LPR improved mean final accuracy and reduced forgetting in all ten seeds.
SlowHeat+Classifier Expander reduced forgetting but did not improve mean final
accuracy. SlowHeat+SCROLL improved final and task-aware accuracy in all ten
seeds; its forgetting and BWT effects remain inconclusive. Approximate paired
Student-t 95% intervals for final-accuracy change are +0.19 to +1.38 pp for
LPR, −1.80 to +0.03 pp for Classifier Expander and +4.14 to +7.42 pp for
SCROLL.

These observations are screening evidence only. The intervals do not correct
for exploratory multiplicity, and the export omits absolute per-seed scores,
task-accuracy matrices and environment provenance. SCROLL also uses task 0 as
a shared representation bootstrap because the runner does not distribute an
external pretrained checkpoint; this is not an exact reproduction of the
original pretrained-representation protocol.

## 5. Historical Diagnostic Pilot (Superseded Method)

The CPU-only pilot used three seeds, three tasks, two classes per task and 20 optimizer steps per task. It was designed to verify the protocol and expose confounds, not to estimate benchmark performance.

| Method | Final average accuracy | Average forgetting | BWT | FWT |
|---|---:|---:|---:|---:|
| Vanilla AdamW | 0.7153 | 0.0312 | 0.2604 | -0.1094 |
| SlowHeat without consolidation | 0.7153 | 0.0312 | 0.2604 | -0.1094 |
| SlowHeat MAX, optimizer-aware AdamW | 0.4757 | 0.0052 | 0.1250 | -0.1094 |
| SlowHeat mean, optimizer-aware AdamW | 0.4861 | 0.0052 | 0.1302 | -0.1094 |
| SlowHeat sum, optimizer-aware AdamW | 0.4375 | 0.0052 | 0.0938 | -0.1094 |
| SlowHeat MAX, legacy raw-gradient AdamW | 0.6701 | 0.0000 | 0.2917 | -0.1042 |
| Reduced learning-rate control | 0.2396 | 0.0000 | 0.0156 | -0.0156 |
| SlowHeat MAX, optimizer-aware SGD | 0.1910 | 0.0208 | -0.0208 | -0.0156 |

### Interpretation

The corrected MAX mechanism was more stable according to observed forgetting, but it was also much less plastic. Its final average accuracy was lower than vanilla. Vanilla already had low forgetting and strongly positive backward transfer, indicating weak initial acquisition and a forgetting floor. The pilot therefore does not provide an informative catastrophic-forgetting regime and does not show a general benefit.

The legacy AdamW variant retained more final accuracy. A plausible explanation is that adaptive normalization weakened its raw-gradient protection. This is a hypothesis about optimizer semantics, not evidence that the legacy method is preferable.

MAX and mean were close in this short run. Sum was more restrictive. Three seeds cannot resolve these differences. The reduced-learning-rate and SGD controls were poorly matched to the short training horizon and require their own tuning before scientific comparison.

## 6. Related Work and Positioning

The closest conceptual precedents include:

- EWC: parameter protection using Fisher-weighted quadratic penalties ([Kirkpatrick et al., 2017](https://arxiv.org/abs/1612.00796)).
- SI: online synaptic contribution estimates ([Zenke et al., 2017](https://arxiv.org/abs/1703.04200)).
- Selfless Sequential Learning / SLNID: neuron-level importance and lateral inhibition ([Aljundi et al., 2019](https://arxiv.org/abs/1806.05421)).
- HAT: cumulative task masks using elementwise maximum ([Serrà et al., 2018](https://arxiv.org/abs/1801.01423)).
- Uncertainty-guided continual learning: importance-dependent learning-rate modulation ([Ebrahimi et al., 2020](https://arxiv.org/abs/1906.02425)).
- Neuron Activation Importance: activation-based neuron importance ([Jung et al., 2022](https://doi.org/10.1007/978-3-031-06427-2_26)).
- LPR: replay-conditioned layerwise proximal optimization ([Yoo et al., 2024](https://proceedings.mlr.press/v235/yoo24a.html)).
- Classifier Expander: two-stage inner-task and cross-task classifier training ([Liu et al., 2024](https://proceedings.mlr.press/v222/liu24b.html)).
- SCROLL: schedule-robust ridge classification over a pretrained representation followed by replay-only adaptation ([Wang et al., 2025](https://doi.org/10.1109/TPAMI.2025.3614868)).

No claim of being the first method to use neuron importance, MAX masks, lateral inhibition or importance-dependent plasticity is justified. Novelty, if established, must be argued at the level of the exact combination and its optimizer-aware formulation.

## 7. Required Experiments Before Submission

1. Tune learning rate and protection strength separately for every optimizer family using a declared validation protocol.
2. Use at least five seeds for screening and preferably ten for final tables.
3. Execute the implemented CIFAR ResNet-18 section with ten paired seeds,
   archive complete trajectories and environment provenance, then execute a
   real-CNN Split-CIFAR-100 protocol.
4. Independently repeat the implemented Replay, DER++, ER-ACE, A-GEM, EWC, SI
   and LwF comparisons with declared tuning, and add MAS, activation-based
   importance, UCB, HAT, SLNID and joint training.
5. Report final average accuracy, average forgetting, BWT, FWT, per-task trajectories, runtime and peak memory.
6. Ablate functional versus activation importance, factorized versus row-only
   masking, fixed/adaptive/no capacity budget and protected/unprotected output.
7. Ablate MAX, mean, sum, no consolidation, legacy gradient masking,
   optimizer-aware masking and global learning-rate reduction.
8. Compare `follow_update` and `native` optimizer-state policies.
9. Sweep `beta` and the capacity budget and plot the Pareto frontier between
   final accuracy and forgetting instead of optimizing either metric alone.
10. Repeat efficiency measurements after warm-up on the same hardware and
    software stack.

## 8. Safe Claims

Currently supported:

- Raw gradient scaling is not equivalent to explicit final-update scaling under AdamW.
- The corrected implementation masks the complete AdamW/SGD update, including
  weight decay, and can make tensor-valued state follow the applied update.
- Functional utility passes reciprocal ReLU reparameterization and dead-unit
  falsification tests.
- Factorized masks protect incoming rows and downstream columns, while the
  budget guarantees a minimum plastic fraction up to integer rounding.
- The benchmark now uses paired initialization, fixed batches and a complete accuracy matrix.
- In a historical pilot of the superseded method, stronger protection reduced
  measured forgetting while reducing final accuracy.
- In one ten-seed Split-CIFAR-10 CNN screening, SlowHeat+SCROLL increased
  final accuracy in all ten pairs and SlowHeat+LPR reduced forgetting in all
  ten pairs.

Not currently supported:

- SlowHeat outperforms established continual-learning baselines.
- MAX consolidation is novel by itself.
- The method reduces forgetting by 34 percent in general.
- The method is equivalent to EWC.
- The ten-seed small-CNN observations generalize to residual networks,
  transformers or real-world tasks.

## 9. Reproducibility Artifacts

- Core implementation: `src/dual_heater/`
- Optimizer contract: `docs/optimizer_semantics.md`
- Functional method contract: `docs/functional_slowheat.md`
- Synthetic protocol: `docs/reproducibility.md`
- Diagnostic pilot: `docs/synthetic_ablation_pilot.md`
- Frozen confirmation protocol: `docs/confirmatory_protocol.md`
- Split-CIFAR protocol: `docs/split_cifar.md`
- Local CNN paired-difference export: `results/paired_differences.csv`
- Residual implementation: `src/dual_heater/resnet.py`
- Split-MNIST experiment record: `docs/split_mnist_experiment_log.md`
- Smoke config: `configs/synthetic_smoke.json`
- Ablation pilot config: `configs/synthetic_ablation_pilot.json`
- Experiment runner: `experiments/synthetic_cl.py`
- Multi-seed aggregation: `experiments/multi_seed.py`
- Complete protocol runner: `run_all_tests.py`
