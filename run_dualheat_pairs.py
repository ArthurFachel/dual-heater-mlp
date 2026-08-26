"""Run the paired MLP comparison from the project root, without PYTHONPATH."""

from __future__ import annotations

import sys
from pathlib import Path

# Resolve only at runtime: no machine-specific directory is embedded in the file.
PROJECT_ROOT = Path(__file__).resolve().parent
for import_path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from experiments.dualheat_pairs import main

if __name__ == "__main__":
    raise SystemExit(main())
