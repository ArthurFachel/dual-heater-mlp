# 2 Effective plasticity and the three-term decomposition

## 2.1 The method under study

SlowHeat keeps, for every hidden unit i of a protected layer, an importance
score accumulated during training on a task (an EMA of |z_i| or of the
first-order utility |z_i * dL/dz_i|, normalized within the layer). At each task
boundary the scores are consolidated into a protection value h_i in [0, 1]
subject to a per-layer capacity budget p (the fraction of units left free):

    protected_count <= floor((1 - p) * unit_count)

and h is turned into a per-unit update factor m_i:

    hard mask:  m_i = 0 for the top (1 - p) units by importance, 1 otherwise
    soft mask:  m_i = 1 / (1 + beta * h_i)          (beta = 30 unless stated)

For a weight W[i, j] the factor is min(m_out_i, m_in_j), so an important unit
protects its incoming row and its outgoing column.

**Masking the update, not the gradient.** Under AdamW, scaling the raw gradient
by a constant c scales the first moment by about c and the second by about c^2,
so m_t / sqrt(v_t) largely cancels c; decoupled weight decay is applied outside
the gradient altogether. A gradient hook is therefore not a learning-rate
control. We let the optimizer compute its native step and mask the result:

    Delta_applied = M * (theta_after_native_step - theta_before_step)

with the same interpolation applied to tensor-valued optimizer state. This
contract is unit-tested: M = 1 reproduces the native optimizer, M = 0 blocks
both gradient and weight-decay motion, M = 0.1 yields a tenth of the native
update. [TODO: 1-sentence pointer to the follow_update vs native state ablation
if we keep it.]

## 2.2 Effective plasticity

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
parameter the method froze outright. Section 5 shows a pre-registered result in
which this omission turned E = 0.532 into a reported E = 0.850.

## 2.3 Controls

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

## 2.4 The decomposition

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
our configuration (4,214,016 of 6,769,920 remain). We treat this as a shape term
with its own uniform control: a learning-rate scale equal to the remaining
fraction, 4,214,016 / 6,769,920 = 0.62246, computed without reading any accuracy.
Our masked LoRA arm ("exact") adds unit selection on top of the frozen A, so
exact - frozen_A is the selection term in that host.
