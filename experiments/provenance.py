"""Portable environment and Git provenance for generated research artifacts."""

from __future__ import annotations

import json
import os
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


def relative_path(path: str | Path, *, base: str | Path) -> str:
    """Serialize a path relative to its declared base, including siblings."""
    try:
        relative = os.path.relpath(Path(path).resolve(), Path(base).resolve())
    except ValueError as error:
        raise ValueError(
            "caminhos relativos requerem diretórios no mesmo volume"
        ) from error
    return Path(relative).as_posix()


def _portable_command(project_root: Path) -> list[str]:
    """Avoid saving machine-specific script or CLI argument locations."""
    arguments = []
    for argument in sys.argv:
        option, separator, value = argument.partition("=")
        if separator and option.startswith("--") and Path(value).is_absolute():
            argument = f"{option}={relative_path(value, base=project_root)}"
        elif Path(argument).is_absolute():
            argument = relative_path(argument, base=project_root)
        arguments.append(argument)
    return [Path(sys.executable).name, *arguments]


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
        "command": _portable_command(root),
        "command_path_base": "project_root",
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
