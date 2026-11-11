# 2 The method

SlowHeat protects hidden units of a network trained on a sequence of tasks. It
has three parts: an importance score per unit, a consolidation step at each task
boundary that turns scores into protection under a capacity budget, and an
optimizer contract that applies the protection to the update.

## 2.1 Importance

For a hidden unit i with pre-activation z_i we consider two criteria, both
normalized within the layer:

    magnitude:   u_i = sum_samples |z_i|
    functional:  u_i = sum_samples |z_i * dL/dz_i|
    u_norm,i = u_i / (mean_j u_j + eps)

The functional score is a first-order estimate of the loss change from removing
the unit. It is invariant to the rescaling that leaves a positively homogeneous
unit's function unchanged (incoming weights times c, outgoing times 1/c), and a
dead ReLU receives zero utility. It needs an activation-gradient hook; the
magnitude score needs only a forward hook. During a task the layer keeps an
online mean followed by an EMA of u_norm. Section 5.5 compares the two criteria
under matched capacity and finds them equivalent, so the cheaper one is a valid
default. [Per-setting criterion in Table 2 / Appendix C.]

## 2.2 Consolidation under a capacity budget

At a task boundary the task score is merged into a persistent
`importance_memory` (element-wise max over tasks by default; mean and sum are
supported) and the task EMA is reset. A protection vector h in [0, 1] is derived
from the memory, and the highest-scoring units are protected subject to a
per-layer budget p, the fraction of units that must stay free:

    protected_count <= floor((1 - p) * unit_count)

This guarantees at least the requested free capacity up to integer rounding.
Because the max is re-ranked at every boundary, the protected set can change
between tasks; in Qwen, 5 to 15% of the set turns over per boundary (Section
5.4, exploratory measurement).

## 2.3 From protection to an update factor

Protection becomes a per-unit factor m_i in [0, 1]:

    hard:  m_i = 0 for protected units, 1 otherwise
    soft:  m_i = 1 / (1 + beta * h_i)

For a weight W_l[i, j] the factor combines the output unit of layer l with the
input unit from layer l-1,

    M_l[i, j] = min(m_l[i], m_{l-1}[j]),

so an important unit protects both its incoming row and its outgoing column.
In a transformer FFN one unit is a row of the up-projection (and its bias) plus
the matching column of the down-projection. Attention, embeddings and
normalization layers are left unprotected unless stated (Table 2). Under LoRA,
our "exact" variant freezes the down-projection A and applies the factor to the
rows of B; Section 5.4 shows that the freeze, not the row mask, carries the
effect.

## 2.4 Masking the update, not the gradient

Under AdamW, scaling the raw gradient by a constant c scales the first moment by
about c and the second by about c^2, so m_t / sqrt(v_t) largely cancels c, and
decoupled weight decay is applied outside the gradient altogether. A gradient
hook is therefore not a learning-rate control. We let the optimizer compute its
native step and mask the result:

    theta_new = theta_old + M * (theta_native - theta_old)

with the same interpolation applied to every tensor-valued optimizer state of
parameter shape (AdamW moments, AMSGrad state, SGD momentum). Scalar step
counters stay global. The contract is unit-tested: M = 1 reproduces the native
optimizer, M = 0 blocks both gradient and weight-decay motion, and M = 0.1
yields a tenth of the native update. Interpolating the state ("follow_update")
versus leaving it native is a declared ablation (Appendix I).

## 2.5 Cost

The magnitude criterion adds one forward hook per protected layer and a buffer
per unit. The functional criterion adds a backward hook. Under LoRA the masked
arm took 1.8x the wall time of the frozen-A arm (611 s vs 336 s per seed,
Section 5.4).
