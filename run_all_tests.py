#!/usr/bin/env python3
"""Run the complete project test suite from any working directory."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> int:
    env = os.environ.copy()
    project_paths = [str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)]
    if existing_pythonpath := env.get("PYTHONPATH"):
        project_paths.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(project_paths)

    command = [
        sys.executable,
        "-m",
        "pytest",
        str(PROJECT_ROOT / "tests"),
        "-q",
        "-p",
        "no:cacheprovider",
        *sys.argv[1:],
    ]
    return subprocess.call(command, cwd=PROJECT_ROOT, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
