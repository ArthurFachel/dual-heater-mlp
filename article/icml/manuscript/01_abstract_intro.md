# Abstract

Continual learning (CL) methods that protect parameters are usually evaluated
against an unconstrained baseline. Any such method also removes plasticity, so
part of its reported gain may come from simply learning less. We make this
explicit by splitting the gain of a masking method over the vanilla learner into
three paired terms, each measured against a control with the same effective
plasticity: *plasticity* (a uniform update scale with the method's mean factor
versus vanilla), *shape* (the same mask structure on randomly chosen units versus
the uniform scale), and *selection* (the method's chosen units versus random
ones). We apply the decomposition to our own method, SlowHeat, across five
hosts: an MLP on Split- and Permuted-MNIST, a 4-layer BERT on CLINC150, a 410M
concept language model under continued pre-training, and Qwen2.5-0.5B under
full fine-tuning and under LoRA. All primary contrasts were pre-registered and
run on fresh seeds. Three findings hold. First, the method beats a
plasticity-matched uniform control in every host where we tested it (11 of 11
pre-registered contrasts). Second, none of the three terms has a fixed sign: the
plasticity term alone is worth +30.4 pp of retention in BERT and costs
-6.0 pp of accuracy in Qwen-LoRA, and a random mask is *worse* than a uniform
scale in BERT but better in the language model (these sign results come from
exploratory runs and from descriptive contrasts of confirmatory runs; Table 1
tags each cell). Third, the selection term, the
part that justifies an importance criterion, is the smallest and least stable:
large in BERT, present in the language model, small and not robust in Qwen
(+4.10 pp, then refuted at +1.50 pp against a fixed random set), and zero under
LoRA, where freezing the down-projection (LoRA-FA) already accounts for the
whole gain. Comparisons against vanilla, or against a random mask alone, cannot
tell these cases apart. We release the controls and the pre-registration trail,
including the refuted results.

# 1 Introduction

A CL method that protects some parameters while training on a new task does two
things at once. It decides *which* parameters keep their values, and it reduces
*how much* the network can change. The second effect alone reduces forgetting:
a learner with a smaller step forgets less and also learns less. Reporting a
method's gain over the unconstrained learner mixes both effects, and the usual
second control, a random mask of the same size, mixes a third one: the mask's
structure.

We ran into this with our own method. A confirmatory result that passed a
pre-registered analysis with Holm correction and five integrity gates compared
an arm with effective plasticity 0.532 against a control at 0.850, because the
plasticity metric averaged only over the parameters that carried a mask
(Section 5). The verdict survived once the arms were matched, but the
magnitude was the sum of a real gain and a degradation of the comparator. That
episode is the reason for this paper: if a matched-plasticity claim can be wrong
under pre-registration, an unmatched one says little.

**The decomposition.** For a masking method M with mean update factor E over
the trainable surface, define three controls on the same host, seeds and data:
the vanilla learner V; a uniform scale U that multiplies every update by E; and
a random mask R with the same count, layout and hardness as M on units chosen
without the importance criterion. Then

    M - V = (U - V)  +  (R - U)  +  (M - R)
            plasticity   shape      selection

Each term is a paired contrast with a fixed meaning. Only the last one is
evidence that the importance criterion matters.

**What we find.** Applied to SlowHeat on five hosts:

1. *Method versus matched plasticity holds.* M - U is positive in all 11
   pre-registered contrasts: 4 replay-family baselines on Split-MNIST, 5 domain
   pairs in BERT, one continued pre-training run of a 410M language model, and
   one 10-task sequence in Qwen2.5-0.5B (Table 1).
2. *No term has a fixed sign.* The plasticity term ranges from +30.4 pp of
   retention (BERT, hard mask) to -6.0 pp of accuracy (Qwen-LoRA). The shape term
   is negative in BERT (a random hard mask is 5.9 pp worse than a uniform scale)
   and positive in the language model. A method ranked against vanilla or
   against a random mask inherits whichever sign the host gives.
3. *Selection is the smallest and least stable term.* It is +9.8 pp in BERT
   (exploratory), present in the language model (exploratory), +4.10 pp in Qwen
   in one pre-registered band of seeds and refuted at +1.50 pp [-0.07, +3.07]
   against a fixed random set in the next one, and zero under LoRA, where
   freezing the LoRA down-projection, a published method (LoRA-FA), explains the
   entire gain we had attributed to our mask.
4. *Rankings move.* In a 5-arm LoRA comparison, an arm that had removed 94% of
   plasticity ranks second unmatched, above vanilla, and third matched, below
   vanilla (Section 4.5, exploratory).

**Contributions.**

- A three-term, plasticity-matched decomposition for masking CL methods, with
  controls that are cheap to add to existing code (a uniform update scale and a
  same-shape random mask).
- Pre-registered evidence on five hosts, including the parts that go against
  our own method: a refuted selection contrast (F16), a null against LoRA-FA
  (R-C), and a null for the gradient term of our importance criterion
  (companion paper).
- A case study of a confirmed, pre-registered result whose plasticity matching
  was wrong, and of a correctly computed endpoint that had no variance in its
  regime. Both passed every gate we had.

We do not claim a new state of the art. The method is a case study for the
controls.

[TODO: Figure 1. Stacked bars per host: plasticity / shape / selection, with
C/E tags. Data in Table 1.]
