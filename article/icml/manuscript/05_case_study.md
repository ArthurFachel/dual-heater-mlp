# 5 Case study: two ways a pre-registered pipeline was wrong

Both episodes below happened inside the protocol of Section 3.2. Neither was
caught by a gate. We report them because the controls in this paper exist to
prevent the first, and the second limits how far Table 1 can be read.

## 5.1 A confirmed result that described conditions that did not exist

**The claim.** On Qwen-LoRA (10 tasks), our masked arm `exact` against a
learning-rate control was pre-registered with the arms matched at E = 0.85. A
pre-run check (H1) required E = 0.8500 in every seed. It passed. The primary
contrast confirmed: forgetting -0.0921, 9 of 10 seeds, exact sign test
p = 0.02148.

**What was wrong.** The function computing E averaged the mask only over
parameters that carried a SlowHeat binding. `exact` also freezes the LoRA
down-projection A, 2,555,904 parameters with an update factor of exactly zero,
and none of them entered the mean. Measured over the reference trainable
surface (Section 2.2) the contrast compared E = 0.532 against E = 0.850:
32 points of plasticity removed, in the direction that favors the treated arm.
The check verified the metric, and the metric was wrong.

**Re-run, matched on the full surface** (protocol d9b57fc, new seed band,
10 seeds). The treatment is now the frozen-A arm alone (LoRA-FA without its
closed-form gradient correction); the control scales the learning rate by
0.62246, the fraction of trainable parameters frozen-A keeps. Both arms remove
the same plasticity from the same surface, one by freezing and one by step size.

| arm | forgetting | FAA | last-task acquisition | first-task retention |
|---|---:|---:|---:|---:|
| vanilla | 0.4125 | 0.5879 | 0.923 | 0.049 |
| lr control (E = 0.62246) | 0.4741 | 0.5275 | 0.918 | 0.043 |
| frozen A (E = 0.62246) | **0.3480** | **0.6267** | 0.895 | **0.093** |

frozen_A - lr control: forgetting -0.1260 and FAA +9.92 pp, 10/10 seeds on
both, p_Holm = 0.0039 (the minimum for n = 10). The verdict survived. Two
descriptive contrasts changed its meaning:

- lr control - vanilla: forgetting **+0.0616**, 9/10, p = 0.0215. The
  comparator got worse than doing nothing.
- frozen A - vanilla: forgetting -0.0644, 8/10, p = 0.109. Not significant.

About half of the 12.6-point gap is the treatment improving on vanilla and half
is the control degrading below it. The original result was right in sign and
wrong in cause.

**Then the mask itself.** A second pre-registered run (f5d77ce) split `exact`
into its two operations: frozen_A - vanilla = -0.1118 forgetting (9/10,
p_Holm = 0.043) and exact - frozen_A = -0.0092 (5+/5-, p = 1.000). The masking
layer we had built adds nothing measurable to LoRA-FA [CITE LoRA-FA], published
in 2023, and costs 1.8x the run time. The frozen-A contrast against vanilla
reproduces the original confirmation down to the p value (0.02148, 9 negative
signs). The effect we had attributed to SlowHeat was, number for number, the
effect of freezing A.

The control that would have found this on day one was a one-line
`requires_grad_(False)` arm. It was not in the design because nobody had
separated the two things `exact` does.

## 5.2 A correct endpoint with no variance

**The setup.** To extend the companion paper's BERT result from 2 to 5 tasks,
we pre-registered a 5-task sequence with task-1 retention as the primary
endpoint and a gate requiring FAA well above chance. One calibration seed,
outside the confirmatory band, ran first to measure cost.

**What happened.** The gate passed (FAA 14x chance). Task-1 retention was
0.0000 in all four arms. At T = 2 the spread of the same endpoint between arms
was 0.354; at T = 5 every arm had forgotten task 1 completely. A family of
contrasts on that endpoint returns p = 1.00 by construction.

**What we did.** We did not run the 10 confirmatory seeds, which would have
spent the band on a result that cannot answer anything, and we did not switch
to a secondary endpoint that did order the arms (average forgetting), because
promoting it after seeing it order the arms is endpoint fishing. The phase is
frozen; it can be resumed with a re-registered endpoint.

**What the gates missed.** Our gates checked the regime (accuracy far from
chance), the ranking of arms, and mask overlap. None checked whether the primary
endpoint had dispersion. In Section 5.1 a metric was computed wrongly; here it
was computed correctly and was uninformative. We now require a calibration seed
to show a nonzero between-arm spread on the primary endpoint before a band is
spent. [TODO: decide whether to state this as a rule or as a recommendation.]

This matters for Table 1. Qwen-FT retention is measured close to a floor:
first-task retention of the best arm is 9.0% (F14), 7.9% (F15) and 4.8% (F16),
and in F16 six of ten seeds have the hard arm at or below 3%. The refuted F16
contrast may be partly a floor effect. We do not use that possibility to
reinterpret it.

## 5.3 Precedent

An earlier audit found that our accuracy evaluation called `model.eval()`
without restoring the previous mode, so importance hooks, which accumulate only
in training mode, collected statistics during the first epoch of each stage and
not afterwards, depending on which other methods shared the sweep. All runs
before the fix were discarded, not reinterpreted, and the confirmation was rerun
under the same frozen seeds and analysis in a new output directory.
[TODO: keep as one paragraph or move to the appendix.]
