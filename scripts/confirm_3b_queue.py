#!/usr/bin/env python3
"""Job queue for the Phase 3b confirmatory run (goals/protocol_3b_confirmatory.md).

One worker per GPU: `python scripts/confirm_3b_queue.py --gpu 0`. Workers share the queue through atomic
mkdir claims under ROOT/claims. Jobs per seed: main (vanilla exact frozen_a_control), lrc_lorafa (fixed
structural scale), lrc_exact (scale = surface_plasticity of exact read from that seed's main manifest, so it
waits for main). A failed job is retried once (protocol V4). --dry replaces the runner with a fake that
writes a minimal manifest, to test the queue logic on CPU.
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
LORAFA_SCALE = 2039808 / 3317760  # structural E of frozen_a_control (0.6148148148148148)
COMMON = [
    "--order", "1", "--epochs-per-task", "1", "--batch-size", "2", "--eval-batch-size", "8",
    "--precision", "fp32", "--device", "cuda:0",
    "--acquisition-margin", "0.05", "--min-vanilla-forgetting", "0.05",
]
MAX_ATTEMPTS = 2


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def jobs() -> list[tuple[int, str]]:
    return [(seed, kind) for seed in SEEDS for kind in ("main", "lrc_lorafa", "lrc_exact")]


def name(seed: int, kind: str) -> str:
    return f"s{seed}_{kind}"


def status(root: Path, seed: int, kind: str) -> str | None:
    p = root / "status" / name(seed, kind)
    return p.read_text().strip() if p.exists() else None


def exact_scale(root: Path, seed: int) -> float:
    m = json.loads((root / f"s{seed}" / "main" / "manifest.json").read_text())
    rows = [r for r in m["results"] if r["method"] == "exact"]
    if len(rows) != 1 or rows[0].get("surface_plasticity") is None:
        raise ValueError(f"seed {seed}: exact surface_plasticity missing in main manifest")
    e = float(rows[0]["surface_plasticity"])
    if not 0.0 < e <= 1.0:
        raise ValueError(f"seed {seed}: exact surface_plasticity out of (0,1]: {e}")
    return e


def command(root: Path, seed: int, kind: str, dry: bool) -> list[str]:
    out = root / f"s{seed}" / kind
    if kind == "main":
        arms = ["--arms", "vanilla", "exact", "frozen_a_control"]
    elif kind == "lrc_lorafa":
        arms = ["--arms", "lr_control", "--learning-rate-scale", repr(LORAFA_SCALE)]
    else:
        arms = ["--arms", "lr_control", "--learning-rate-scale", repr(exact_scale(root, seed))]
    if dry:
        return [sys.executable, __file__, "--fake-run", str(out), *arms]
    return [PY, "experiments/superni_cl.py", "--output", str(out), "--seed", str(seed), *COMMON, *arms]


def fake_run(out: str, rest: list[str]) -> int:
    time.sleep(1)
    fail = os.environ.get("FAKE_FAIL")
    if fail and fail in out:
        return 1
    o = Path(out)
    o.mkdir(parents=True, exist_ok=True)
    results = [{"method": "exact", "surface_plasticity": 0.55}] if "exact" in rest else []
    (o / "manifest.json").write_text(json.dumps({"results": results, "argv": rest}))
    return 0


def try_claim(root: Path, seed: int, kind: str) -> bool:
    try:
        os.mkdir(root / "claims" / name(seed, kind))
        return True
    except FileExistsError:
        return False


def eligible(root: Path, seed: int, kind: str) -> str:
    """'ready', 'wait' (dependency still running) or 'blocked' (dependency failed)."""
    if kind != "lrc_exact":
        return "ready"
    dep = status(root, seed, "main")
    if dep is None:
        return "wait"
    return "ready" if dep == "0" else "blocked"


def worker(root: Path, gpu: str, dry: bool) -> None:
    for d in ("claims", "status", "logs"):
        (root / d).mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=gpu,
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", PYTHONPATH=".:src",
               HF_HOME="/mnt/B-SSD/fachel/dual-heater-mlp/.hf-cache", HF_HUB_OFFLINE="1",
               HF_DATASETS_OFFLINE="1", OMP_NUM_THREADS="4")
    priority = {"lrc_exact": 0, "main": 1, "lrc_lorafa": 2}
    while True:
        pending = [(s, k) for s, k in jobs() if not (root / "claims" / name(s, k)).exists()]
        if not pending:
            print(f"=== worker gpu={gpu} no jobs left {now()}", flush=True)
            return
        pending.sort(key=lambda sk: (priority[sk[1]], sk[0]))
        picked = None
        for s, k in pending:
            state = eligible(root, s, k)
            if state == "blocked":
                if try_claim(root, s, k):
                    (root / "status" / name(s, k)).write_text("skipped_dep\n")
                    print(f"=== {name(s, k)} skipped: main failed {now()}", flush=True)
                continue
            if state == "ready" and try_claim(root, s, k):
                picked = (s, k)
                break
        if picked is None:
            time.sleep(2 if dry else 60)
            continue
        s, k = picked
        code = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                cmd = command(root, s, k, dry)
            except Exception as exc:  # dependency manifest unreadable
                print(f"=== {name(s, k)} gpu={gpu} cannot build command: {exc} {now()}", flush=True)
                code = 97
                break
            log = root / "logs" / f"{name(s, k)}.try{attempt}.log"
            print(f"=== {name(s, k)} gpu={gpu} try={attempt} start {now()} :: {' '.join(cmd[-4:])}", flush=True)
            with open(log, "w") as fh:
                code = subprocess.run(cmd, cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
            print(f"=== {name(s, k)} gpu={gpu} try={attempt} exit={code} {now()}", flush=True)
            if code == 0:
                break
        (root / "status" / name(s, k)).write_text(f"{code}\n")


def main() -> None:
    if len(sys.argv) > 2 and sys.argv[1] == "--fake-run":
        raise SystemExit(fake_run(sys.argv[2], sys.argv[3:]))
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", required=True)
    p.add_argument("--root", default=str(REPO / "results" / "3b_confirm"))
    p.add_argument("--dry", action="store_true")
    a = p.parse_args()
    worker(Path(a.root), a.gpu, a.dry)


if __name__ == "__main__":
    main()
