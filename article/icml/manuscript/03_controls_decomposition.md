# 3 Effective plasticity, controls and the three-term decomposition

Section 2 defines what the method does. This section defines what it is
compared against. Every control removes the same amount of plasticity as the
method, measured on the same surface.

## 3.1 Effective plasticity

For a method that multiplies updates by a factor field M over a reference
trainable surface S (all parameters the vanilla learner would train), we define
the surface plasticity

    E_sup = (1 / |S|) * sum_{theta in S} M_theta

with frozen parameters contributing M = 0. For methods that do not act through
a multiplicative mask (penalties, regularizers) we use the ratio of update norms
against a shadow vanilla step taken from the same state,

    E = || Delta_method || / || Delta_vanilla ||,

measured at every step. [TODO: state the lemma E_sup <= E_bind from
goals/protocol_plasticity_generalization.md and its condition.]

The definition over the *whole* reference surface is not a detail. Averaging M
only over parameters that carry a mask ("binding" parameters) drops every
parameter the method froze outright. Section 6 shows a pre-registered result in
which this omission turned E = 0.532 into a reported E = 0.850.

## 3.2 Controls

All controls run in the same process, on the same seed, data order and device
as the method arm.

- **V (vanilla).** Same model and optimizer, no protection.
- **U (uniform scale).** Every parameter of the surface covered by M receives
  the same factor, equal to the mean of M over that surface, recomputed at
  every consolidation. Same E_sup as M, no selection, nothing frozen. Under a
  mask applied to the full trainable surface this is a learning-rate reduction
  (the "reduced_lr" arm in Qwen).
- **R (random units, same shape).** Same protected count per layer, same mask
  hardness or soft values, units drawn without the importance score. We use two
  variants: *re-drawn* at every boundary (values of h permuted across units of a
  layer after each consolidation) and *fixed* (one random order per layer drawn
  once per run; the first k units are protected throughout). R matches M in E,
  count and shape.
- **M (method).**

## 3.3 The decomposition

On paired seeds,

    M - V  =  (U - V)  +  (R - U)  +  (M - R)
              plasticity  shape      selection

*Plasticity* is what a method would gain by learning less, with no choice of
where. *Shape* is what the mask's structure (binary versus graded, concentrated
versus spread) adds or costs when units are chosen at random. *Selection* is
what the importance criterion adds over random units of the same shape. M - U,
the contrast we pre-registered most often, is shape plus selection.

Two remarks. First, the decomposition is an identity on means; it is
informative because each term is a paired contrast with its own interval, not
because it sums. Second, it is not unique: ordering R before U gives a
different split of the same total. We fix the order above because U is the
control any method can implement without knowing its structure.

**Structural variants.** Some methods remove plasticity by a structural choice
rather than a mask. In LoRA, freezing the down-projection A (LoRA-FA
[CITE LoRA-FA arXiv 2308.03303]) removes 37.75% of the trainable parameters of
our classification configuration (4,214,016 of 6,769,920 remain; r = 16). We
treat this as a shape term with its own uniform control: a learning-rate scale
equal to the remaining fraction, 4,214,016 / 6,769,920 = 0.62246, computed
without reading any accuracy. Our masked LoRA arm ("exact") adds a row mask on B
on top of the frozen A. On the classification stream we split it as
exact - frozen_A. On the generative stream (r = 8, where frozen A keeps
2,039,808 of 3,317,760 parameters, 0.61481) we also ran the proper R arm: the
same multiset of B-row factors per module assigned to a fixed random order of
rows (random_fixed), matched to exact in E within 0.0006 in every seed. There
exact - random_fixed is the selection term and random_fixed - U is shape.
