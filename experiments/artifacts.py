"""Versioned, atomic persistence helpers for research artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TextIO

import torch

RUN_IDENTITY_SCHEMA_VERSION = 1


@contextmanager
def atomic_text_writer(
    path: str | Path,
    *,
    newline: str | None = None,
) -> Iterator[TextIO]:
    """Write a text artifact completely before replacing its destination."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline=newline,
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def write_json_atomic(path: str | Path, payload: Any) -> None:
    with atomic_text_writer(path) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_torch_atomic(path: str | Path, payload: Any) -> None:
    """Atomically persist a tensor-only checkpoint suitable for safe loading."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def read_torch_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load an internal checkpoint without permitting arbitrary pickle globals."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError(f"checkpoint deve conter um objeto: {path}")
    return payload


def read_json_object(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"artefato JSON deve conter um objeto: {path}")
    return payload


def source_fingerprint(project_root: str | Path) -> str:
    """Hash code and protocol configuration that can affect benchmark results."""

    root = Path(project_root).resolve()
    candidates: list[Path] = []
    for directory, pattern in (
        (root / "src", "*.py"),
        (root / "experiments", "*.py"),
        (root / "configs", "*.json"),
    ):
        if directory.is_dir():
            candidates.extend(directory.rglob(pattern))
    candidates.extend(
        item
        for item in (
            root / "pyproject.toml",
            root / "run_all_tests.py",
            root / "run_dualheat_pairs.py",
        )
        if item.is_file()
    )

    digest = hashlib.sha256()
    for source in sorted(candidates):
        relative = source.relative_to(root).as_posix().encode("utf-8")
        content = source.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def task_data_fingerprint(tasks: Iterable[Any]) -> str:
    """Hash the exact materialized tensors consumed by one seeded run."""

    digest = hashlib.sha256()
    for task_index, task in enumerate(tasks):
        classes = json.dumps(list(task.classes), separators=(",", ":")).encode()
        digest.update(task_index.to_bytes(8, "big"))
        digest.update(classes)
        for name in (
            "train_x",
            "train_y",
            "validation_x",
            "validation_y",
            "test_x",
            "test_y",
        ):
            tensor = getattr(task, name).detach().cpu().contiguous()
            metadata = json.dumps(
                {
                    "name": name,
                    "dtype": str(tensor.dtype),
                    "shape": list(tensor.shape),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            content = memoryview(tensor.numpy()).cast("B")
            digest.update(len(metadata).to_bytes(8, "big"))
            digest.update(metadata)
            digest.update(len(content).to_bytes(8, "big"))
            digest.update(content)
    return digest.hexdigest()


def build_run_identity(
    config: Mapping[str, Any],
    *,
    project_root: str | Path,
    task_loader: str,
) -> dict[str, Any]:
    return {
        "schema_version": RUN_IDENTITY_SCHEMA_VERSION,
        "python_major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
        "source_sha256": source_fingerprint(project_root),
        "task_loader": task_loader,
        "config": json.loads(json.dumps(dict(config), allow_nan=False)),
    }


def ensure_run_identity(
    output_dir: str | Path,
    identity: Mapping[str, Any],
    *,
    resume: bool,
) -> Path:
    """Fail closed when completed seeds cannot be tied to the current code."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output / "run_identity.json"
    completed_results = tuple(output.glob("seed_*/results.json"))
    if identity_path.is_file():
        saved = read_json_object(identity_path)
        if saved != dict(identity):
            raise RuntimeError(
                "identidade da execução difere do código, loader ou configuração atual"
            )
        if completed_results and not resume:
            raise FileExistsError(
                "resultados existentes requerem resume=True ou um novo output_dir"
            )
        return identity_path
    if completed_results:
        raise RuntimeError(
            "resultados existentes não possuem run_identity.json; "
            "não é seguro retomá-los"
        )
    write_json_atomic(identity_path, dict(identity))
    return identity_path
