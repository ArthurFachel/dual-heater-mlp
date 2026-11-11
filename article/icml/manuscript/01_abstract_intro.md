# Abstract

Methods that protect parameters during continual learning (CL) are usually
evaluated against an unconstrained learner. Every such method also removes
plasticity, and a learner that changes less forgets less. We introduce
SlowHeat, a masking method that ranks hidden units by importance, consolidates
the ranking at task boundaries under a per-layer capacity budget, and masks the
optimizer's native update rather than the gradient. We then ask how much of its
gain is learning less. We split the gain over the unconstrained learner into
three paired terms, each measured against a control with the same effective
plasticity: *plasticity* (a uniform update scale with the method's mean factor),
*shape* (the same mask on randomly chosen units) and *selection* (the chosen
units against random ones). We test four model families in seven settings: an
MLP on Split- and Permuted-MNIST, a 4-layer BERT on CLINC150, a 410M concept
language model under continued pre-training, and Qwen2.5-0.5B under full
fine-tuning and under LoRA, on a classification stream and on a generative
instruction stream. Primary contrasts were pre-registered and run on fresh
seeds. Three findings hold. First, the method beats a plasticity-matched
uniform control in all 11 pre-registered contrasts. Second, inside the
criterion, activation magnitude alone matches the first-order utility
(-0.17 pp) and both beat a same-size random mask by +12.23 pp. Third, most of
the gain is not selection. The plasticity term ranges from +30.4 to -6.0 pp
across settings, a random mask is worse than a uniform scale in one host and
better in others, and the selection term is large in BERT, small and once
refuted in Qwen, and zero under LoRA on both streams (within 1 pp of AP on the
generative one), where freezing the LoRA down-projection explains the gain.
Comparisons against the unconstrained learner, or against a random mask alone,
cannot tell these cases apart. We release the controls and the full
pre-registration trail, including the refuted results.

[NOTE for the LaTeX version: the method name is a macro (\method). The
submitted version uses a neutral name for double-blind review; "SlowHeat" only
in the camera-ready.]

# 1 Introduction

LoRA learns less and forgets less [CITE Biderman et al. 2024]. The observation
generalizes. A CL method that protects some parameters while training on a new
task does two things at once: it decides *which* parameters keep their values,
and it reduces *how much* the network can change. The second effect alone
reduces forgetting. Reporting a method's gain over the unconstrained learner
mixes both effects, and the usual second control, a random mask of the same
size, mixes in a third one, the structure of the mask.

We ran into this while developing our own method. SlowHeat (Section 2) keeps
a per-unit importance score, protects the top units at each task boundary under
a capacity budget, and applies the protection to the optimizer's update. Its
early results looked strong against vanilla and against random masks. Then a
confirmatory, pre-registered result that had passed Holm correction and five
integrity gates turned out to compare an arm with effective plasticity 0.532
against a control at 0.850, because our plasticity metric averaged only over
parameters that carried a mask (Section 6). The verdict survived once the arms
were matched, but the magnitude was the sum of a real gain and a degradation of
the comparator. If a matched claim can be wrong under pre-registration, an
unmatched one says little. This paper reports the method together with the
controls that tell what its gain is made of.

**The decomposition.** For a masking method M with mean update factor E over
the trainable surface, define three controls on the same host, seeds and data:
the vanilla learner V; a uniform scale U that multiplies every update by E; and
a random mask R with the same count, layout and hardness as M on units chosen
without the importance criterion. Then

    M - V = (U - V)  +  (R - U)  +  (M - R)
            plasticity   shape      selection

Each term is a paired contrast with a fixed meaning. Only the last one is
evidence that the importance criterion matters.

**What we find.**

1. *The method beats matched plasticity.* M - U is positive in all 11
   pre-registered primary contrasts: 4 replay-family baselines on Split-MNIST,
   5 domain pairs in BERT, continued pre-training of a 410M language model, and
   a 10-task sequence in Qwen2.5-0.5B (Table 1).
2. *A cheap criterion is enough.* Ranking units by activation magnitude |z|
   matches the first-order utility |z * dL/dz| (-0.17 pp, 5/10 seeds), and both
   beat a random mask of the same capacity by +12.23 pp of retention (10/10).
3. *Most of the gain is not selection.* The plasticity term ranges from
   +30.4 pp of retention to -6.0 pp of accuracy. The shape term changes sign
   across hosts. The selection term is +9.8 pp in BERT, present in the language
   model, +4.10 pp in Qwen in one pre-registered band and refuted at +1.50 pp
   against a fixed random set in the next, and zero under LoRA on both streams,
   where freezing the LoRA down-projection, a published method (LoRA-FA),
   explains the gain.
4. *Rankings move.* Among five LoRA variants, an arm that removed 94% of the
   plasticity ranks second unmatched, above vanilla, and below vanilla once
   matched (Section 5.6, exploratory).

**Contributions.**

- A masking method for CL with a capacity budget and an optimizer contract that
  masks the update and its state, unit-tested to reproduce the native optimizer
  at M = 1 and to block all motion, including weight decay, at M = 0.
- A three-term, plasticity-matched decomposition for masking methods, with two
  controls that are cheap to add to existing code (a uniform update scale and a
  same-shape random mask).
- Pre-registered evidence in seven settings, including the parts that go
  against our method: a refuted selection contrast, a null against LoRA-FA on
  two streams, and a null for the gradient term of our own criterion.
- A case study of a confirmed, pre-registered result whose plasticity matching
  was wrong, and of a correctly computed endpoint that had no variance.

We do not claim a new state of the art. We claim that the method's gain over
matched plasticity is real, and we show how much of it is selection.

[TODO: Figure 1. Stacked bars per setting: plasticity / shape / selection,
C/E tags, only rows whose three terms come from the same seeds. Data in Table 1.]
