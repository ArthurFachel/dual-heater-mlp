"""Portable environment and Git provenance for generated research artifacts."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

TRACKED_PACKAGES = (
    "dual-heater",
    "numpy",
    "torch",
    "torchvision",
    "scipy",
    "pandas",
)


def _git_output(project_root: Path, *args: str) -> str | None:
    completed = subprocess.run(
        ("git", *args),
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def environment_manifest(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    packages: dict[str, str | None] = {}
    for package in TRACKED_PACKAGES:
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    status = _git_output(root, "status", "--short")
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "git": {
            "commit": _git_output(root, "rev-parse", "HEAD"),
            "branch": _git_output(root, "branch", "--show-current"),
            "dirty": bool(status),
        },
        "command": [Path(sys.executable).name, *sys.argv],
    }


def write_environment_manifest(
    output_dir: str | Path,
    *,
    project_root: str | Path,
) -> Path:
    destination = Path(output_dir) / "environment.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(
            environment_manifest(project_root),
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    return destination
