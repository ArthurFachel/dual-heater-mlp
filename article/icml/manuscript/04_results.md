# 4 Results

Signs are oriented so that **positive = better for the first arm** in every
cell. For the language model the endpoint is a loss, so we report the loss
reduction (U - M in nats, etc.). C = confirmatory primary contrast;
c = descriptive contrast from a confirmatory run (outside the tested family);
E = exploratory. Intervals are 95% paired t intervals; "k/n" counts seeds in
favor of the first arm.

## 4.1 Table 1: the decomposition per host

| host (endpoint) | plasticity U - V | shape R - U | selection M - R | M - U (shape + selection) |
|---|---|---|---|---|
| MLP-S + DER++ (FAA, pp) | +1.51 c / +1.40 [+0.95, +1.86] 19/20 E | not run | not run | **+1.91 [+1.39, +2.43] 19/20 C** |
| MLP-S + replay calibrated | +1.05 c / +0.84 [+0.49, +1.19] 18/20 E | not run | not run | **+1.49 [+1.21, +1.78] 20/20 C** |
| MLP-S + replay | +0.58 c | not run | not run | **+0.62 [+0.25, +0.98] 17/20 C** |
| MLP-S + ER-ACE | -0.65 c | not run | not run | **+0.39 [+0.15, +0.64] 15/20 C** |
| MLP-P, hard (FAA, pp) | +3.02 [+1.41, +4.63] 9/10 E | not run | not run | +13.47 [+11.57, +15.36] 10/10 E |
| BERT, hard (T1 retention, pp) | +30.44 E | **-5.93**, 0/10 E | +9.83, 10/10 E | +3.90, 10/10 E |
| BERT, soft (T1 retention, pp), 5 pairs | +5.48 to +12.10 c | not run | not run | **+2.93 to +5.20, 19-20/20 per pair, C (5/5)** |
| LM-410M, soft (MiniPile loss, nats) | not run | +0.00436, 6/6 E | +0.00339, 6/6 E | **+0.00764 [+0.00743, +0.00785] 10/10 C** |
| Qwen-FT, hard (T1 retention, pp) | +1.60 [+0.29, +2.91] 8/10 c | +1.97 [+0.23, +3.71] 5/10 c | **+4.10 [+0.61, +7.59] 9/10 C** (re-drawn R); **+1.50 [-0.07, +3.07] 8/10 C, REFUTED** (fixed R) | **+5.67 [+3.00, +8.33] 9/10 C** |
| Qwen-LoRA (FAA, pp) | -6.04 c | **+9.92, 10/10 C** (frozen A vs matched lr) | +0.69, 5/5 C, null (exact vs frozen A) | n/a |

"c" values in the MLP-S plasticity column are differences of means from F11
(FAA of B+U minus FAA of B); the paired versions come from the exploratory F7
seeds. The LM-410M shape and selection terms come from F3 with 6 seeds; the
plasticity term was not measured there. Every cell is traced to a file in
Appendix A.

Reading across rows gives the three findings of the paper.

## 4.2 Method versus matched plasticity holds everywhere it was tested

M - U was pre-registered 11 times on four hosts (4 baselines in MLP-S, 5 domain
pairs in BERT, LM-410M, Qwen-FT) and confirmed 11 times. Effect sizes are
modest. In MLP-S the method adds +0.39 to +1.91 pp of FAA over a baseline that
received the same mean plasticity. In BERT it adds +2.93 to +5.20 pp of task-1
retention, with FAA also in favor in all five pairs (+0.84 to +2.40 pp,
descriptive) and acquisition of task 2 lower by 0.4 to 1.4 pp. In LM-410M it
removes 10.7% of the forgetting of the uniform arm. In Qwen-FT the gain is
+5.67 pp of first-task retention with no visible acquisition cost
(-0.10 pp [-1.99, +1.79] on the last task, descriptive).

## 4.3 No term has a fixed sign

**Plasticity.** Reducing plasticity uniformly helps a lot in BERT with a hard
budget (+30.44 pp of retention, E), a little in MLP-P and Qwen-FT, and hurts in
two places: ER-ACE on Split-MNIST, where U is 0.65 pp below the baseline (c), and
Qwen-LoRA, where a learning-rate scale of 0.62246 lowers FAA by 6.04 pp and
raises forgetting by 0.0616 (9/10 seeds, p = 0.0215, c). In the LoRA host a
uniform scale of 0.85 had no detectable effect (p = 0.754) in an earlier run;
the sign depends on the size of the cut as well as on the host (two points only,
not a curve).

The ER-ACE row shows why this matters for headline numbers. SlowHeat + ER-ACE
beats ER-ACE + U by +0.39 pp (C), yet its FAA is 0.26 pp *below* ER-ACE alone
(0.7083 vs 0.7109). The selection gain is real and smaller than the plasticity
loss it rides on. Against the baseline alone, the method looks useless; against
the matched control, it looks like it works. Both are true.

**Shape.** In BERT with a hard budget a random mask of the same count is
5.93 pp worse in retention than spreading the same plasticity uniformly (10/10
seeds, E): freezing 75% of randomly chosen units is worse than shrinking every
step. In LM-410M with a soft mask, the same-shape random arm is better than
uniform (+0.00436 nats, 6/6, E), and in Qwen-FT with a hard mask the re-drawn
random arm is better than reduced lr (+1.97 pp, c, but only 5/10 seeds). In
Qwen-LoRA the structural term (freezing A) is the largest gain in the table:
+9.92 pp of FAA and -0.1260 forgetting against a learning-rate control with
identical surface plasticity, 10/10 seeds on both endpoints, p_Holm = 0.0039,
which is the floor for n = 10.

A random mask is therefore not a neutral control. Depending on the host it
inflates (BERT) or deflates (LM-410M) the apparent value of selection. Reporting
M - R alone, as is common, measures selection plus an unknown shape term
measured against the wrong reference.

## 4.4 Selection is the smallest and least stable term

Where we could measure it:

- BERT, hard: +9.83 pp of retention over a random mask, 10/10 (E). This
  replicates the companion paper's +12.23 pp (C) in a new seed band. Against U,
  only +3.90 pp survives, and FAA ties (+0.65 pp, 6/10).
- LM-410M, soft: +0.00339 nats over the random arm, 6/6 (E): about 44% of
  M - U.
- Qwen-FT, hard: +4.10 pp over a re-drawn random set in seeds 300-309 (C,
  p = 0.026, effect concentrated in three seeds). In seeds 400-409, +1.50 pp
  [-0.07, +3.07] over a fixed random set: refuted under the frozen rule
  (p = 0.060). In the same run fixed and re-drawn random were equivalent
  (+0.30 pp [-1.06, +1.66]), so the weaker F15 control does not explain the
  drop; the selection effect itself shrank (+1.80 pp over re-drawn in the
  F16 band). We read this as a positive but small and unstable term, not as
  2/3 of the gain, which is what F15 alone suggested.
- Qwen-LoRA: exact - frozen_A = -0.0092 forgetting, 5+/5-, p = 1.000 (C).
  Selection adds nothing to LoRA-FA. The masked arm costs 1.8x the wall time
  (611 s vs 336 s per seed).

One plausible reason for the LoRA null is geometric: A is shared by all output
rows through the rank-r bottleneck, so protecting rows of B leaks. We have not
tested this.

## 4.5 Rankings change when plasticity is matched (E)

Five LoRA variants on Qwen-LoRA, 10 seeds:

| arm | E unmatched | rank unmatched (FAA) | rank at E = 0.85 |
|---|---:|---:|---:|
| exact | 0.9234 | 1 (0.6365) | 1 (0.6494) |
| slice | **0.0625** | 2 (0.6203) | 3 (0.5875) |
| vanilla | 1.000 | 3 (0.6016) | 2 (0.6016) |
| rank | 0.6454 | 4 (0.5983) | 5 (0.5808) |
| leak | 0.6586 | 5 (0.5958) | 4 (0.5865) |

`slice` ranks second only because it removed 94% of the plasticity. Matched at
E = 0.85 it falls below vanilla. (The project compendium says "falls to 4th";
counting the five arms in this table it is 3rd. The claim that matters, above
vanilla unmatched and below it matched, holds either way.) [TODO: lr_control at E = 0.85 appears in the
matched table (FAA 0.5684) but its E column says 1.000 in the compendium; check
before printing.]

## 4.6 Penalty methods: E depends on the optimizer

[Short; may move to the appendix.] For EWC, SI and MAS with published strengths
[CITE Hsu et al. 2018], E measured as an update-norm ratio is above 1 under AdamW
in all 12 seeds on Split-CIFAR-100 (EWC 1.00128, SI 1.01238, MAS 1.01939): the
penalty *amplifies* the step, so a "learning-rate matched" control would need a
learning rate above the baseline. For MAS the update can flip direction (cosine
with the vanilla step down to -0.876). Under plain SGD the three methods give
E < 1 in 12/12 seeds (E = 0.955, 0.976, 0.993 at lr 3e-3), so E > 1 is an
interaction with the adaptive optimizer, not a property of the penalties. With
a matched SGD control on Split-MNIST, MAS at 30x beats the control in forgetting
(-0.0400, 12/12, p = 0.00049), but all three arms end near chance (final
accuracy 0.149 / 0.198 / 0.160 against chance 0.10), so the contrast cannot
separate consolidation from changing less. We report it with that caveat.
