#!/usr/bin/env python3
"""EXPLORATORY: how much does the hard arm's protected set change between boundaries?

Runs the real F14/F15/F16 configuration (Qwen2.5-0.5B, CLINC150, 10 domains, 120 steps, E*=0.75) for
the hard arm only, and records WHICH units are protected at every boundary by wrapping
`protected_digest`, which the runner calls once per boundary after the mask is final.

Output JSON: per boundary, per layer, the protected index set sizes and overlaps:
  consecutive: |S_b & S_{b-1}| / |S_b|            (fraction kept from the previous boundary)
  vs_first:    |S_b & S_0| / |S_0|                (fraction of the task-1 set still protected)
Not a protocol; seed 900 is outside every confirmatory band.
"""
import json
import sys

import torch

import experiments.qwen_iso_plasticity as qip

SETS: list[list[set[int]]] = []
_orig = qip.protected_digest


def recording_digest(model):
    SETS.append([
        set(torch.nonzero(t.slow_heat > 0.0).flatten().cpu().tolist()) for t in model.get_slow_states()
    ])
    return _orig(model)


qip.protected_digest = recording_digest
out = sys.argv[1]
sys.argv = [sys.argv[0]] + sys.argv[2:]
qip.main()

layers = len(SETS[0])
rows = []
for b, sets in enumerate(SETS):
    cons = [len(s & p) / max(len(s), 1) for s, p in zip(sets, SETS[b - 1])] if b else None
    first = [len(s & f) / max(len(f), 1) for s, f in zip(sets, SETS[0])]
    rows.append({
        "boundary": b,
        "sizes": [len(s) for s in sets],
        "consecutive_kept_mean": sum(cons) / layers if cons else None,
        "consecutive_kept_min": min(cons) if cons else None,
        "vs_first_kept_mean": sum(first) / layers,
        "vs_first_kept_min": min(first),
    })
json.dump({"layers": layers, "boundaries": rows}, open(out, "w"), indent=1)
for r in rows:
    print(r["boundary"], r["sizes"][0], "consec", r["consecutive_kept_mean"], "vs_first", round(r["vs_first_kept_mean"], 4))
