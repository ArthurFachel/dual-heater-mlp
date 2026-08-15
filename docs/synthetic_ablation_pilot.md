# Synthetic ablation pilot (3 seeds, superseded implementation)

> Diagnostic CPU-only pilot. This is not publication evidence.

> These artifacts predate functional `|z*dL/dz|` importance, factorized
> row/column masks, capacity budgeting, output-head protection, future-logit
> masking and `follow_update` optimizer-state semantics. They must not be used
> as results for the current implementation.

The raw matrices, losses, configs and per-seed summaries are archived in
`artifacts/synthetic_ablation_pilot/`. The run predates a commit identifier and
the manifest therefore records an uncommitted source tree.

Seeds: 11, 22, 33. Three class-incremental tasks, two classes per task, 20 optimizer steps per task.

| Method | Final avg. accuracy | Avg. forgetting | BWT | FWT |
|---|---:|---:|---:|---:|
| `slowheat_none` | 0.7153 | 0.0312 | 0.2604 | -0.1094 |
| `vanilla` | 0.7153 | 0.0312 | 0.2604 | -0.1094 |
| `slowheat_max_legacy_adamw` | 0.6701 | 0.0000 | 0.2917 | -0.1042 |
| `slowheat_mean` | 0.4861 | 0.0052 | 0.1302 | -0.1094 |
| `slowheat_max` | 0.4757 | 0.0052 | 0.1250 | -0.1094 |
| `slowheat_sum` | 0.4375 | 0.0052 | 0.0938 | -0.1094 |
| `reduced_lr` | 0.2396 | 0.0000 | 0.0156 | -0.0156 |
| `slowheat_max_sgd` | 0.1910 | 0.0208 | -0.0208 | -0.0156 |

## What this pilot establishes

1. `slowheat_none` exactly matches vanilla on aggregate metrics. This is a useful fairness control: paired initialization, batches and the corrected optimizer with an all-one mask preserve native AdamW behavior.
2. Corrected `slowheat_max` reduced mean forgetting relative to vanilla in this tiny pilot, but its final average accuracy was substantially lower. The result is a stability-plasticity trade-off, not a win.
3. MAX and mean were close; sum was more restrictive. Three seeds are insufficient to distinguish them.
4. `slowheat_max_legacy_adamw` retained much more final accuracy than optimizer-aware MAX here. This does not validate the legacy implementation; it is consistent with AdamW cancelling much of raw gradient scaling, so the effective protection is weaker.
5. The fixed learning-rate reduction and SGD controls underfit at this short training horizon. Learning-rate tuning must be method-specific before a fair scientific comparison.

## Numerical contrast

- Vanilla: accuracy 0.7153, forgetting 0.0312.
- SlowHeat MAX, optimizer-aware: accuracy 0.4757, forgetting 0.0052.
- SlowHeat MAX, legacy raw-gradient AdamW: accuracy 0.6701, forgetting 0.0000.
- SlowHeat without consolidation: accuracy 0.7153, forgetting 0.0312.

## Limits

- Vanilla has strongly positive BWT and very low mean forgetting, so this is
  not an informative catastrophic-forgetting regime. The apparent reduction
  in forgetting is close to a floor effect.
- Three seeds, a tiny synthetic dataset and only 20 steps per task.
- Confidence intervals use a normal approximation and are unstable at n=3.
- The exact sign test cannot reach conventional significance with only three non-tied pairs.
- No specialized continual-learning baselines are included yet.
- No hyperparameter tuning was performed.
- The SGD comparison has no matched vanilla-SGD/no-consolidation-SGD controls.
- MAX, mean and sum use the same beta despite different heat scales.
- Runtime values are diagnostic only and should not be compared across machines.

## Decision

Do not write an efficacy claim from this pilot. The next experiment must tune the stability-plasticity strength, compare MAX against mean with at least five seeds, and include specialized baselines. The article can already use this pilot as evidence for the optimizer-semantics analysis and for why raw forgetting alone is misleading.
