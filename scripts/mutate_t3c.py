"""Directed mutations for T3c. Run from the worktree root on the Lab."""
import os
import subprocess
from pathlib import Path

TARGET = Path("experiments/split_mnist.py")
GOOD = TARGET.read_text()
PY = "/mnt/B-SSD/fachel/dual-heater-mlp/.venv/bin/python"
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4", PYTHONPATH=".:src")

MUTATIONS = {
    "M1 runner never reassigns": (
        "                if method in RANDOM_FIXED_METHODS:\n                    random_fixed_history.append(",
        "                if False:\n                    random_fixed_history.append(",
    ),
    "M2 order redrawn at every boundary (seed + boundary count)": (
        'orders = getattr(model, "_random_fixed_orders", None)\n    if orders is None:',
        'orders = None\n    model._rf_calls = getattr(model, "_rf_calls", 0) + 1\n    seed = seed + 7919 * model._rf_calls\n    if orders is None:',
    ),
    "M3 order ignores the seed": (
        "(seed + RANDOM_FIXED_ORDER_SEED_OFFSET) * 1009 + index",
        "1009 + index",
    ),
    "M4 values not moved": (
        "        moved[order.to(values.device)] = values\n",
        "        moved = flat.clone()\n",
    ),
    "M5 random hard not hard": (
        '    "uniform_hard_freeze",\n    "random_fixed_hard_freeze",\n}',
        '    "uniform_hard_freeze",\n}',
    ),
    "M6 er_ace spec drops er_ace": (
        '    "random_fixed_er_ace_hidden_beta_30_budget_0.25": MethodSpec(\n        slowheat=True,\n        replay=True,\n        er_ace=True,',
        '    "random_fixed_er_ace_hidden_beta_30_budget_0.25": MethodSpec(\n        slowheat=True,\n        replay=True,',
    ),
    "M7 unsorted values (no fixed prefix)": (
        "values = torch.sort(flat, descending=True, stable=True).values",
        "values = flat.clone()",
    ),
    "M8 history not recorded in results": (
        '            "random_fixed_history": random_fixed_history,\n',
        '            "random_fixed_history": [],\n',
    ),
}

try:
    for name, (old, new) in MUTATIONS.items():
        assert GOOD.count(old) == 1, (name, GOOD.count(old))
        TARGET.write_text(GOOD.replace(old, new))
        r = subprocess.run(
            [PY, "-m", "pytest", "-q", "-x", "tests/test_random_fixed_mlp.py"],
            env=ENV, capture_output=True, text=True,
        )
        print(f"{name}: {'MORTA' if r.returncode != 0 else 'SOBREVIVEU'}", flush=True)
finally:
    TARGET.write_text(GOOD)
    print("restored", TARGET.read_text() == GOOD)
