# Optimizer-aware plasticity masking

SlowHeat estimates an importance value per output unit and converts it to a plasticity factor:

```text
m_i = 1 / (1 + beta * slow_heat_i)
```

## Why raw gradient hooks are insufficient

For plain SGD, multiplying a gradient by `m_i` scales the update directly. Adam and AdamW normalize gradients using first and second moments, so a persistent multiplicative factor can largely cancel. Decoupled weight decay can also move a supposedly protected parameter independently of its gradient.

`SlowHeatAdamW` and `SlowHeatSGD` therefore:

1. let the native optimizer compute its complete update;
2. measure the resulting parameter delta;
3. multiply that final delta by the registered plasticity mask;
4. apply the masked delta;
5. optionally apply the same mask to tensor-valued optimizer-state deltas.

The mask affects gradients, momentum/preconditioning and weight decay together. Mask `1` preserves the native update. Mask `0` blocks it.

## Usage

```python
from dual_heater import SlowHeatAdamW, SlowHeatMLP

model = SlowHeatMLP(128, 64, 32, 10, plasticity_budget=0.25)
optimizer = SlowHeatAdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-2,
    state_policy="follow_update",
)
optimizer.register_slow_heat_model(model)
```

Registration disables legacy raw-gradient hooks, avoiding double masking. The
model-level registration validates the sequential topology and creates
factorized masks: output rows are protected by the current layer and input
columns by the preceding layer.

## State policies

`follow_update` is the default. For each tensor-valued state with the same
shape as its parameter, it applies:

```text
s_applied = s_before + mask * (s_native - s_before)
```

This covers AdamW first/second moments, AMSGrad maximum moments and SGD momentum
buffers. A zero mask blocks both movement and local moment accumulation. Scalar
step counters remain global.

`native` retains native optimizer states while masking only parameter movement.
It exists as a required scientific ablation, not as the recommended default.

## Checkpoints

Save both model and optimizer state dictionaries. Importance buffers belong to
the model; moments belong to the optimizer. Checkpoints store the state policy,
parameter-group positions, registration kinds and factorized source positions.
`step()` fails closed until the same masks are registered. Loading a checkpoint
with a different state policy is rejected. Legacy optimizer checkpoints without
metadata remain loadable but cannot supply a fail-closed mask expectation.

## Trade-offs

The implementation clones each registered parameter once per optimizer step to
recover the native delta. Under `follow_update`, it also snapshots tensor-valued
state. Peak temporary memory is therefore proportional to protected parameter
and state size. This is intentional for semantic clarity and must be measured
before claiming scalability.

The update rule is explicit but is not yet a fused optimizer kernel. It can
introduce graph breaks under `torch.compile`, and distributed/fused optimizer
compatibility has not been established.
