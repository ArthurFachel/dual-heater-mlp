# Functional SlowHeat: Scale-Invariant Neuron Utility, Factorized Protection and Capacity-Aware Plasticity

## Status

Technical manuscript draft. The current implementation is validated by unit
and integration tests. Exploratory Split-MNIST comparisons against Replay,
DER++, ER-ACE, A-GEM, EWC, SI and calibrated LwF are documented and their raw
per-seed artifacts are versioned. The frozen 20-seed confirmation of
SlowHeat+Replay against Replay has been executed twice independently; both
runs are versioned and agree on every scientific metric, though both record a
dirty Git tree. Ten-seed paired suites for Split-CIFAR-10 and Split-CIFAR-100
are versioned and analysed in Section 6: they show that the sign of the
SlowHeat effect depends on the base method it is attached to. Ten-seed
exploratory Split-CIFAR-10 benchmarks with VGG11 and ResNet18 do not establish
a multiplicity-adjusted final-accuracy advantage for Functional DualHeat.
Section 8 reports a five-target suite, frozen before execution and run from a
clean Git tree, that removes the confound between protection regime and
architecture: hard protection does not beat soft protection anywhere, and
loses where capacity is tight. The three-seed pilot in Section 5 predates the
functional-importance, factorized-protection, capacity-budget and
optimizer-state changes; it is retained only as historical motivation and must
not be reported as evidence for the current method.

Per-architecture evidence summaries, each stating its claim, citable numbers,
provenance and explicit non-claims, are in `docs/arch_mlp.md`,
`docs/arch_cnn.md`, `docs/arch_bert.md` and `docs/arch_qwen.md`. The
hard-versus-soft results are in `docs/hard_vs_soft_results.md`. The
provenance status of every versioned aggregate, including what must be
re-executed before submission, is in `docs/results_provenance_status.md`.

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
implemented and covered by falsification tests. A frozen 20-seed Split-MNIST
confirmation of SlowHeat+Replay against Replay gives a paired final-accuracy
difference of +0.87 percentage points (95% CI [+0.29, +1.45], t = 3.16,
p = 0.0052, 17/20 seeds favourable) at 1.8x to 1.9x the wall-clock time; the
result was reproduced by a second independent execution. Exploratory ten-seed
convolutional suites show that the sign of the effect depends on the base
method: attached to ER-ACE, SlowHeat improves final accuracy by +4.15 points on
Split-CIFAR-10 and +1.27 on Split-CIFAR-100, whereas attached to DER++ it
*reduces* accuracy by 1.13 and 0.53 points respectively, all four surviving
Holm correction within their dataset. Ten-seed Split-CIFAR-10 comparisons do
not show a multiplicity-adjusted final-accuracy advantage for Functional
DualHeat. A five-target suite frozen before execution removes the confound
between protection regime and architecture and finds that hard protection
never beats soft protection outside Transformers, losing by 1.41 points on
Split-CIFAR-100 (10/10 seeds): the hard-protection advantage measured on BERT
does not transfer. No general efficacy or state-of-the-art claim is made.

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
domain-incremental; the implemented Split-CIFAR-10/100 adapters are Class-IL
and flatten normalized images for the same paired MLP engine.

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

### 4.4 Evaluation-mode restoration correction

An audit on 2026-09-02 found that epoch-end accuracy evaluation called
`model.eval()` without restoring the learner's previous mode. Functional
importance hooks accumulate only while the model is in training mode. In a
SlowHeat sweep containing no FastHeat method, importance was therefore
accumulated only during the first epoch of each stage. Sweeps containing a
FastHeat method happened to restore all learners at the next epoch and exposed
an invalid dependence on sweep composition.

Evaluation now uses a side-effect-free context that restores the previous mode,
and class-incremental and task-aware accuracy share the same forward passes.
Regression tests verify mode restoration, independence from the presence of a
FastHeat method and exact agreement between vanilla and the no-consolidation
SlowHeat control. The frozen Split-MNIST confirmation contains only Replay and
SlowHeat+Replay, so any run made before this correction is invalid. Its frozen
seeds, hyperparameters, endpoint and analysis remain unchanged; a valid run
must use the corrected source and a new output directory.

The two versioned confirmation runs satisfy this requirement. The
side-effect-free evaluation context `experiments/evaluation.py` was introduced
in commit `d5b22ad` (2026-09-03), which is the commit recorded in the
environment manifest of the first run; the second uses `f2f7616`
(2026-09-04). Both wrote to fresh output directories.

### 4.5 Frozen Split-MNIST confirmation

The preregistered contrast is `slowheat_replay_hidden_beta_30_budget_0.25`
minus `replay` over 20 frozen seeds, with final average class-incremental
accuracy after the fifth task as the primary endpoint. The preregistration lock
carries `status = frozen_before_execution`, `preregistered_at = 2026-08-15` and
an identical `sha256` in both runs, so the preregistration was not edited
between them.

| Metric | Mean paired difference | 95% t CI (df=19) | t | two-sided p | Signs |
|---|---:|---:|---:|---:|---|
| **Final average accuracy** | **+0.00869** | [+0.00293, +0.01445] | 3.157 | **0.0052** | 17+ / 3- |
| Average forgetting | -0.01462 | [-0.02207, -0.00718] | -4.112 | 0.00059 | 2+ / 18- |
| Classifier gap | -0.00735 | [-0.01304, -0.00166] | -2.704 | 0.0141 | 4+ / 16- |
| Task-aware final accuracy | +0.00134 | [+0.00005, +0.00263] | 2.168 | 0.0431 | 14+ / 6- |

Marginal means: Replay reaches 0.77040 final accuracy with 0.27244 forgetting;
SlowHeat+Replay reaches 0.77909 with 0.25781. Only the primary endpoint was
preregistered; the remaining rows are secondary and descriptive. The
task-aware row should not be read as a positive result: the paired t test
gives p = 0.043, but the exact sign test on the same 20 pairs gives p = 0.115.

Wall-clock cost is the one dimension on which the two executions differ, since
it is machine-dependent. The first run records 3.003 s for Replay against
5.496 s for SlowHeat+Replay (+2.493 s, t = 35.95, 1.83x); the second records
3.036 s against 5.828 s (+2.792 s, t = 135.06, 1.92x). The overhead is
positive in 20 of 20 seeds in both runs.

A second independent execution reproduced the first: across the 20 per-seed
result files, 21 of 100 scalar fields differ and all of them are cost fields
(elapsed time, optimizer step time, peak memory, selection time). No
scientific metric differs.

The limitation is provenance: both runs record a dirty Git tree, and the
executed diff was not fingerprinted, so bit-for-bit reproduction cannot be
established from the manifests alone.

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

## 6. Convolutional Benchmarks: the Effect Depends on the Base Method

The paired suite `dualheat_pairs` runs ten seeds per dataset with paired
initialization, partitions, minibatch schedules and replay indices. For each
dataset we evaluate four contrasts, always
`slowheat_<base>_hidden_beta_30_budget_0.25` minus `<base>`, on the same
primary endpoint used throughout: final average class-incremental accuracy.
Holm correction is applied within each dataset over its four contrasts.

| Dataset | vs ER-ACE | vs Replay | vs DER++ | vs Vanilla |
|---|---:|---:|---:|---:|
| Split-MNIST | -0.08 (ns) | +0.14 (ns) | **+3.18** | **-0.12** |
| Permuted-MNIST | **+0.53** | **+0.42** | **+0.10** | **+7.67** |
| Split-CIFAR-10 | **+4.15** | +0.35 (ns) | **-1.13** | -0.00 (ns) |
| Split-CIFAR-100 | **+1.27** | **+0.62** | **-0.53** | +0.05 (ns) |

Differences are percentage points. Bold entries survive Holm at 5% within
their dataset; `ns` marks contrasts that do not. The Split-CIFAR-10 Replay
contrast lands at Holm-adjusted p = 0.0509 and is therefore reported as
non-significant.

Two Holm-surviving entries must not be read as efficacy results. The
Split-MNIST vanilla contrast (-0.12 points) compares 19.64% against 19.52%,
both at the 20% chance floor of a five-task two-class Class-IL stream: the
difference is statistically detectable and practically meaningless. The
Permuted-MNIST vanilla contrast (+7.67 points) is large, but the reference is
unregularized sequential fine-tuning, which is the weakest possible baseline.

The convolutional results are not a uniform confirmation or a uniform failure.
On both CIFAR benchmarks the ER-ACE contrast is strongly positive and unanimous
across seeds (10+/0- on CIFAR-10, 10+/0- on CIFAR-100), while the DER++
contrast is negative and nearly unanimous (0+/10- and 1+/9-). Both directions
survive multiplicity correction. The same inversion does not appear on the MLP
benchmarks, where the DER++ contrast is the strongest positive result.

We report the inversion as an observation and not as a mechanism. A natural
reading is that SlowHeat and DER++ compete for the same stability budget while
ER-ACE and SlowHeat are complementary, but no ablation in this work tests that
hypothesis. A competing explanation cannot be excluded from the present data:
every aggregate uses a fixed `lr = 1e-3` for all methods, so DER++ and ER-ACE
may be unequally tuned, and an interaction with learning rate would produce a
similar pattern. Resolving this requires the per-method tuning listed in
Section 10, item 1.

Two further limitations bound these numbers. Absolute accuracy on
Split-CIFAR-100 is low in every arm, between 5.4% and 14.7%, so the
differences describe a weak-performance regime. And while the four contrasts
within each dataset are Holm-corrected, the four datasets are not corrected
against each other; a global correction over all sixteen contrasts would
weaken the marginal entries.

A separate ten-seed study compares Functional DualHeat against SlowHeat on
Split-CIFAR-10 with VGG11 and ResNet18 backbones. No contrast survives Holm at
5% in either architecture. The largest mean gain, VGG11 with LPR at +1.43
points, has an adjusted p of 0.0803; all four ResNet18 contrasts fall between
-0.50 and -0.01 points with adjusted p of 1.000. Adding FastHeat to SlowHeat
therefore has no robust effect in this protocol.

Full tables, artifacts and provenance are in `docs/arch_cnn.md`.

## 7. BERT/CLINC150 Mechanism Diagnostics

We implemented Functional SlowHeat in Hugging Face BERT without replacing its
native attention computation. Training-time trackers estimate normalized
`|z dL/dz|` utility for post-GELU feed-forward units and for attention heads
using Q, K, V and merged head outputs. The resulting factors mask producer rows
and consumer columns through an optimizer-aware AdamW update. These runs use
validation only and the first two domains of a 150-way Class-IL CLINC150 stream;
they are mechanism diagnostics rather than a full benchmark.

Across ten paired seeds, every evaluated SlowHeat configuration improved mean
final accuracy over standard sequential BERT. Vanilla reached 48.47%, compared
with 51.75%, 59.65% and 66.05% for soft beta 3, 10 and 30, respectively, and
71.38% for learned hard protection. The learned-hard gain over vanilla was
22.92 percentage points and was positive in all ten seeds; T1 forgetting fell
from 90.90 to 41.03 points. This establishes that the Transformer integration
is functional and beneficial relative to sequential fine-tuning in this
two-task protocol.

The mechanism controls narrow the claim. Learned hard protection improved final
average accuracy by 3.65 points over matched random hard protection, but the
sign reversed in three seeds. T1 retention improved by 9.37 points in all ten
seeds, while T2 acquisition decreased by 2.07 points on average. Protected
parameter drift was exactly zero, yet the learned-hard condition still forgot
41.03 points of T1. The ranking therefore contains retention signal, while the
current protected topology and plasticity allocation do not eliminate the
stability-plasticity trade-off.

A predeclared replay interaction test rejected scaling the mechanism. With 20
stored examples per class, hard SlowHeat+replay trailed replay by 0.83 points of
final average accuracy in all three seeds and required approximately 2.21 times
the elapsed time. Under the primary one-example-per-class budget,
hard SlowHeat+replay improved final average accuracy by 2.17 points in all three
seeds, but reduced T2 acquisition by 3.67 points. This exceeded the predeclared
maximum acceptable acquisition loss of two points. The secondary five- and
ten-example budgets did not produce a consistent benefit.

We therefore conclude that SlowHeat outperformed standard sequential BERT in
the completed two-task diagnostic, but we do not advance this SlowHeat+replay
configuration to the ten-task sequence. The evidence supports a functional
Transformer implementation and a stability mechanism, not superiority over
replay. Complete artifacts and provenance limitations are recorded in
`docs/bert_slowheat_diagnostic_results.md`.

## 8. Hard versus Soft Protection: the Regime Does Not Transfer

Section 7 reports the Transformer result under *hard* protection: consolidated
units frozen outright. Every MLP and convolutional result in Sections 5 and 6
uses *soft* protection, `1 / (1 + 30 h)`. That left the protection regime
perfectly confounded with the architecture, so the 22.92-point BERT gain could
not be attributed to either.

To remove the confound we ran both regimes on the non-Transformer hosts under
one paired protocol, frozen to `hard_vs_soft_protocol.json` before the first
training step and executed from a clean Git tree at commit `6f4d12d` — the only
suite in the project other than the Split-MNIST confirmation to satisfy both.
The two arms share the capacity budget (0.25), so they protect the same units
and differ only in how hard those units are held, and the soft comparator is
the output-protecting variant, because hard freezing protects the output layer
too. Ten paired seeds per target; Holm within each target's four contrasts.

The expectation was declared in the protocol file before execution: if the
regime explains the Transformer result, hard should win here as well, and a
null or reversed result would be reported as informative.

| Target | Hard − Soft (pp) | 95% CI | p (Holm) | seeds |
|---|---:|---|---:|---|
| Split-MNIST / MLP | +0.50 | [−1.76, +2.75] | 1.00 | 5/10 |
| Permuted-MNIST / MLP | +0.88 | [−0.14, +1.89] | 0.17 | 7/10 |
| Split-CIFAR-10 / MLP | +1.11 | [−0.18, +2.41] | 0.17 | 7/10 |
| **Split-CIFAR-100 / MLP** | **−1.41** | [−1.82, −1.00] | **2.9e−05** | **10/10** |
| Split-CIFAR-10 / CNN | −1.16 | [−2.53, +0.22] | 0.089 | 6/10 |

Hard protection does not beat soft protection on any of the five targets. The
only Holm-surviving primary contrast is Split-CIFAR-100, where hard **loses**,
with all ten seeds agreeing. The BERT advantage is therefore specific to the
architecture or to its capacity regime; it is not a property of binary freezing
that transfers to smaller hosts.

The BERT replay finding is likewise not general. Hard+replay minus replay
replicates the negative result on two targets (Split-CIFAR-100/MLP −3.19 pp and
Split-CIFAR-10/CNN −6.22 pp, both 10/10 seeds under Holm), reverses on one
(Split-CIFAR-10/MLP, +2.55 pp, 9/10) and is null on the remaining two.

The sign tracks capacity pressure rather than architecture. Split-CIFAR-100
runs on the same MLP [1024, 512] as Split-CIFAR-10 and flips the sign, with
twice the boundaries, five times the classes per task and a tenth of the
examples per class. This is consistent with the plasticity floor of
Section 2.3: freezing `P` of `N` units costs plasticity in proportion, and the
cost is only absorbable while `N − P` still suffices for the new task. We note
that capacity, boundary count and data volume vary together across these
targets, so the ordering is suggestive rather than an isolated capacity
manipulation.

A reading that cites forgetting alone would invert this conclusion. On the CNN,
hard protection cuts forgetting by 8.34 points against vanilla and still loses
2.97 points of final accuracy; Split-CIFAR-100 shows the same pattern (−11.35
forgetting, −1.83 accuracy). Less forgetting did not help because acquisition
fell with it.

Full results, per-target secondary contrasts and provenance are in
`docs/hard_vs_soft_results.md`; the frozen design is in
`docs/hard_vs_soft_protection.md`.

## 9. Related Work and Positioning

The closest conceptual precedents include:

- EWC: parameter protection using Fisher-weighted quadratic penalties ([Kirkpatrick et al., 2017](https://arxiv.org/abs/1612.00796)).
- SI: online synaptic contribution estimates ([Zenke et al., 2017](https://arxiv.org/abs/1703.04200)).
- Selfless Sequential Learning / SLNID: neuron-level importance and lateral inhibition ([Aljundi et al., 2019](https://arxiv.org/abs/1806.05421)).
- HAT: cumulative task masks using elementwise maximum ([Serrà et al., 2018](https://arxiv.org/abs/1801.01423)).
- Uncertainty-guided continual learning: importance-dependent learning-rate modulation ([Ebrahimi et al., 2020](https://arxiv.org/abs/1906.02425)).
- Neuron Activation Importance: activation-based neuron importance ([Jung et al., 2022](https://doi.org/10.1007/978-3-031-06427-2_26)).

No claim of being the first method to use neuron importance, MAX masks, lateral inhibition or importance-dependent plasticity is justified. Novelty, if established, must be argued at the level of the exact combination and its optimizer-aware formulation.

## 10. Required Experiments Before Submission

1. Tune learning rate and protection strength separately for every optimizer family using a declared validation protocol.
2. Use at least five seeds for screening and preferably ten for final tables.
3. Use Split MNIST as a debugging benchmark. The implemented Split-CIFAR-10 and
   Split-CIFAR-100 streams have been executed for ten paired seeds and are
   analysed in Section 6; what remains is a preregistered confirmation of the
   ER-ACE interaction rather than a first execution.
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

## 11. Safe Claims

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
- In a two-task BERT/CLINC150 diagnostic, SlowHeat improved final average
  accuracy over sequential BERT in all ten learned-hard paired seeds. Learned
  rankings improved old-task retention over matched random masks, while replay
  comparisons exposed an acquisition-retention trade-off and failed the
  predeclared scaling gate.
- In the frozen 20-seed Split-MNIST confirmation, SlowHeat+Replay improved
  final average accuracy over Replay by +0.87 percentage points
  (p = 0.0052, 17/20 seeds) and reduced average forgetting by 1.46 points
  (p = 0.00059, 18/20 seeds), at 1.8x to 1.9x the wall-clock time. The result
  was reproduced by a second independent execution.
- In ten-seed convolutional suites, the sign of the SlowHeat effect depends on
  the base method. Attached to ER-ACE it improved final average accuracy by
  +4.15 points on Split-CIFAR-10 (10/10 seeds) and +1.27 on Split-CIFAR-100
  (10/10 seeds); attached to DER++ it reduced accuracy by 1.13 and 0.53 points
  respectively (0/10 and 1/10 seeds favourable). All four survive Holm
  correction within their dataset.
- Adding FastHeat to SlowHeat produced no Holm-surviving final-accuracy change
  on Split-CIFAR-10 with either VGG11 or ResNet18.
- Hard protection does not beat soft protection on any of five non-Transformer
  targets under a protocol frozen before execution with a clean Git tree. On
  Split-CIFAR-100/MLP hard is worse by 1.41 points (Holm p = 2.9e-05, 10/10
  seeds). The BERT hard-protection advantage is therefore specific to that
  architecture or capacity regime, not a transferable property of the regime.

Not currently supported:

- That the protection regime explains the Transformer result. This was the
  declared expectation of the hard-versus-soft suite and it was not observed.
- Any isolated causal account of *why* hard loses under capacity pressure.
  Across the five targets capacity, boundary count and data volume vary
  together; a width sweep with everything else fixed has not been run.

- SlowHeat outperforms established continual-learning baselines. The confirmed
  contrast is against plain Replay on Split-MNIST only; DER++ and ER-ACE were
  not part of the frozen confirmation.
- Any causal account of why the convolutional effect inverts between ER-ACE and
  DER++. The pattern is measured; the competing-stability-budget reading is an
  untested hypothesis, and unequal per-method tuning is not excluded.
- Any task-aware accuracy gain in the frozen confirmation. The paired t test
  gives p = 0.043, but the exact sign test on the same pairs gives p = 0.115.
- The SlowHeat+DER++ combination, which is the strongest exploratory result on
  the MLP benchmarks, has any preregistered confirmation of its own.
- MAX consolidation is novel by itself.
- The method reduces forgetting by 34 percent in general.
- The method is equivalent to EWC.
- Results generalize to other Transformer architectures, datasets, real-world
  tasks or a full CLINC150 task sequence.

## 12. Reproducibility Artifacts

Per-architecture evidence summaries, each with citable numbers, artifact
paths, provenance and explicit non-claims:

- MLP evidence and limits: `docs/arch_mlp.md`
- Convolutional evidence and limits: `docs/arch_cnn.md`
- BERT evidence and limits: `docs/arch_bert.md`
- Qwen2 status (no citable evidence yet): `docs/arch_qwen.md`
- Hard versus soft protection, design and results:
  `docs/hard_vs_soft_protection.md` and `docs/hard_vs_soft_results.md`
- Provenance status and what must be re-executed: `docs/results_provenance_status.md`
- Index of every versioned result directory: `docs/results_index.md`

Implementation and protocol:

- Core implementation: `src/dual_heater/`
- Optimizer contract: `docs/optimizer_semantics.md`
- Functional method contract: `docs/functional_slowheat.md`
- Synthetic protocol: `docs/reproducibility.md`
- Diagnostic pilot: `docs/synthetic_ablation_pilot.md`
- Frozen confirmation protocol: `docs/confirmatory_protocol.md`
- Split-CIFAR protocol: `docs/split_cifar.md`
- Split-MNIST experiment record: `docs/split_mnist_experiment_log.md`
- BERT/CLINC150 diagnostic decision: `docs/bert_slowheat_diagnostic_results.md`
- Smoke config: `configs/synthetic_smoke.json`
- Ablation pilot config: `configs/synthetic_ablation_pilot.json`
- Experiment runner: `experiments/synthetic_cl.py`
- Multi-seed aggregation: `experiments/multi_seed.py`
- Complete protocol runner: `run_all_tests.py`

### Provenance caveat

Of the 93 versioned aggregates, 89 were produced with a dirty Git tree, and
only the two confirmation runs carry a frozen preregistration lock. The
recorded commit therefore does not fully describe the executed code for most
results, and the executed diff was not fingerprinted. The frozen Split-MNIST
confirmation is the result for which this matters most; its mitigation is the
independent second execution, which reproduced every scientific metric from a
different commit. `docs/results_provenance_status.md` lists, in priority
order, what should be re-executed with a clean tree before submission.

The hard-versus-soft suite of Section 8 is the exception and the template: it
froze its protocol before the first training step, refused to start from a
dirty tree, and recorded the commit (`6f4d12d`) in its queue log beforehand.
Its CNN target carries one additional note — the ten seeds trained on 23/09
but the analysis step aborted, and the report was regenerated on 24/09 from
the trained artifacts alone, with `analysis_provenance` in `pair_report.json`
distinguishing the analysis fingerprint from the training identity.
