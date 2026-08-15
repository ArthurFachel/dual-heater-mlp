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
4. apply the masked delta.

The mask affects gradients, momentum/preconditioning and weight decay together. Mask `1` preserves the native update. Mask `0` blocks it.

## Usage

```python
from dual_heater import SlowHeatAdamW, SlowHeatLinear

layer = SlowHeatLinear(128, 64, slow_strength=3.0)
optimizer = SlowHeatAdamW(layer.parameters(), lr=1e-3, weight_decay=1e-2)
optimizer.register_slow_heat_module(layer)
```

Registration disables the layer's legacy raw-gradient hook, avoiding double masking. Register every protected SlowHeat layer after constructing the optimizer.

## Checkpoints

Save both model and optimizer state dictionaries. Heat buffers belong to the model. AdamW moments belong to the optimizer. The optimizer checkpoint also stores the expected parameter-group positions and registration kinds of protected masks. You may register the recreated module before loading, or load first and then call `register_slow_heat_module()`. `step()` fails closed until protected metadata and current SlowHeat weight/bias registrations match. Saving an awaiting-restoration checkpoint preserves its expected metadata. Legacy checkpoints without metadata remain compatible; loading one preserves any masks already registered, but the legacy file itself cannot supply a fail-closed expectation.

## Trade-offs

The implementation clones each registered parameter once per optimizer step to recover the native delta. Peak temporary memory is therefore proportional to the protected parameter count. This is intentional for semantic clarity in the research implementation, but it must be measured before claiming scalability.

The optimizer moments evolve from raw gradients even when the final parameter update is heavily masked. This is an explicit design choice and should be included in ablations against alternative state-masking rules.
