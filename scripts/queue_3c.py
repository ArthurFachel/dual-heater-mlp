#!/usr/bin/env python3
"""Job queue for Fase 3c (goals/protocol_3c_random_fixed.md): one random_fixed run per seed.

One worker per GPU: `python scripts/queue_3c.py --gpu 0`. Workers share the queue through atomic mkdir
claims under ROOT/claims. A failed job is retried once (protocol V5). --dry replaces the runner with a fake
that writes a minimal manifest, to test the queue on CPU. Arguments are the 3b ones (scripts/confirm_3b_queue.py
@ deea8d2) with --arms random_fixed; the runner, venv and environment are the same.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path("/mnt/B-SSD/fachel/dh-qwen-f14")
PY = "/mnt/B-SSD/fachel/dual-heater-mlp/.venv/bin/python"
SEEDS = list(range(1000, 1010))
COMMON = [
    "--order", "1", "--epochs-per-task", "1", "--batch-size", "2", "--eval-batch-size", "8",
    "--precision", "fp32", "--device", "cuda:0",
    "--acquisition-margin", "0.05", "--min-vanilla-forgetting", "0.05",
]
MAX_ATTEMPTS = 2


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def command(root: Path, seed: int, dry: bool) -> list[str]:
    out = root / f"s{seed}"
    if dry:
        return [sys.executable, __file__, "--fake-run", str(out)]
    return [PY, "experiments/superni_cl.py", "--output", str(out), "--seed", str(seed), *COMMON,
            "--arms", "random_fixed"]


def fake_run(out: str) -> int:
    time.sleep(1)
    fail = os.environ.get("FAKE_FAIL")
    if fail and fail in out:
        return 1
    o = Path(out)
    o.mkdir(parents=True, exist_ok=True)
    (o / "manifest.json").write_text(json.dumps({"results": [{"method": "random_fixed"}]}))
    return 0


def worker(root: Path, gpu: str, dry: bool) -> None:
    for d in ("claims", "status", "logs"):
        (root / d).mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=gpu,
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", PYTHONPATH=".:src",
               HF_HOME="/mnt/B-SSD/fachel/dual-heater-mlp/.hf-cache", HF_HUB_OFFLINE="1",
               HF_DATASETS_OFFLINE="1", OMP_NUM_THREADS="4")
    for seed in SEEDS:
        try:
            os.mkdir(root / "claims" / f"s{seed}")
        except FileExistsError:
            continue
        code = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            cmd = command(root, seed, dry)
            log = root / "logs" / f"s{seed}.try{attempt}.log"
            print(f"=== s{seed} gpu={gpu} try={attempt} start {now()}", flush=True)
            with open(log, "w") as fh:
                code = subprocess.run(cmd, cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
            print(f"=== s{seed} gpu={gpu} try={attempt} exit={code} {now()}", flush=True)
            if code == 0:
                break
        (root / "status" / f"s{seed}").write_text(f"{code}\n")
    print(f"=== worker gpu={gpu} no jobs left {now()}", flush=True)


def main() -> None:
    if len(sys.argv) > 2 and sys.argv[1] == "--fake-run":
        raise SystemExit(fake_run(sys.argv[2]))
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", required=True)
    p.add_argument("--root", default=str(REPO / "results" / "3c_random_fixed"))
    p.add_argument("--dry", action="store_true")
    a = p.parse_args()
    worker(Path(a.root), a.gpu, a.dry)


if __name__ == "__main__":
    main()
