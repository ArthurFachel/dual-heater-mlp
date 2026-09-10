"""Append-only, read-only telemetry for continual-learning experiments."""

from __future__ import annotations

import hashlib
import json
import math
import re
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Self

import torch
from torch import Tensor, nn

from experiments.artifacts import read_json_object, write_json_atomic

TELEMETRY_SCHEMA_VERSION = 1
_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def identity_sha256(identity: dict[str, Any]) -> str:
    encoded = json.dumps(
        identity,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_events(
    path: str | Path,
    *,
    after: int = 0,
    limit: int = 10_000,
) -> list[dict[str, Any]]:
    """Read valid events while tolerating a truncated final JSONL line."""

    if after < 0:
        raise ValueError("after deve ser >= 0")
    if limit < 1:
        raise ValueError("limit deve ser positivo")
    source = Path(path)
    if not source.is_file():
        return []
    events: list[dict[str, Any]] = []
    with source.open(encoding="utf-8") as handle:
        for raw_line in handle:
            try:
                candidate = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if not isinstance(candidate, dict):
                continue
            sequence = candidate.get("sequence")
            if not isinstance(sequence, int) or sequence <= after:
                continue
            events.append(candidate)
            if len(events) >= limit:
                break
    return events


def _last_sequence(path: Path) -> int:
    sequence = 0
    for event in read_events(path, limit=2**31 - 1):
        sequence = max(sequence, int(event["sequence"]))
    return sequence


def _needs_jsonl_separator(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    with path.open("rb") as handle:
        handle.seek(-1, 2)
        return handle.read(1) != b"\n"


def _tensor_values(tensor: Tensor) -> list[float]:
    values = tensor.detach().to(device="cpu", dtype=torch.float32).flatten()
    return [float(value) for value in values]


def _tracker_snapshot(tracker: nn.Module, layer: int) -> dict[str, Any]:
    task_importance = tracker.task_ema
    consolidated = tracker.importance_memory
    heat = tracker.slow_heat
    scales = tracker.get_lr_scales()
    protected = heat.detach() > 0.0
    return {
        "layer": layer,
        "units": int(heat.numel()),
        "task_importance": _tensor_values(task_importance),
        "consolidated_importance": _tensor_values(consolidated),
        "heat": _tensor_values(heat),
        "plasticity": _tensor_values(scales),
        "protected_fraction": float(protected.float().mean().cpu()),
        "plastic_fraction": float((~protected).float().mean().cpu()),
    }


def build_heat_snapshot(
    model: nn.Module | None,
    *,
    context: dict[str, Any],
    run_id: str,
    session_id: str,
    sequence: int,
) -> dict[str, Any]:
    """Copy complete unit-level SlowHeat state without retaining the graph."""

    snapshot: dict[str, Any] = {
        "schema_version": TELEMETRY_SCHEMA_VERSION,
        "run_id": run_id,
        "session_id": session_id,
        "sequence": sequence,
        "timestamp": _utc_now(),
        "context": dict(context),
        "available": model is not None,
        "attention": [],
        "ffn": [],
    }
    if model is None:
        return snapshot
    attention_getter = getattr(model, "get_attention_trackers", None)
    ffn_getter = getattr(model, "get_ffn_trackers", None)
    if callable(attention_getter):
        snapshot["attention"] = [
            _tracker_snapshot(tracker, layer)
            for layer, tracker in enumerate(attention_getter())
        ]
    if callable(ffn_getter):
        snapshot["ffn"] = [
            _tracker_snapshot(tracker, layer)
            for layer, tracker in enumerate(ffn_getter())
        ]
    fast_getter = getattr(model, "get_fast_states", None)
    if callable(fast_getter):
        gates = list(fast_getter())
        for index, entry in enumerate(snapshot["ffn"]):
            if index < len(gates):
                entry["fast_heat"] = _tensor_values(gates[index].fast_heat)
    snapshot["available"] = bool(snapshot["attention"] or snapshot["ffn"])
    return snapshot


class TelemetryWriter:
    """Publish telemetry that can be consumed by a separate dashboard process."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        identity: dict[str, Any],
        every: int = 10,
        resumed: bool = False,
    ) -> None:
        if not isinstance(every, int) or isinstance(every, bool) or every < 1:
            raise ValueError("telemetry_every deve ser um inteiro positivo")
        self.run_dir = Path(run_dir).resolve()
        self.telemetry_dir = self.run_dir / "telemetry"
        self.telemetry_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.telemetry_dir / "events.jsonl"
        self.heat_latest_path = self.telemetry_dir / "heat-latest.json"
        self.heat_history_dir = self.telemetry_dir / "heat"
        self.every = every
        self.run_id = identity_sha256(identity)
        self.session_id = f"{time.time_ns()}-{secrets.token_hex(4)}"
        self._started = time.perf_counter()
        self._write_seconds = 0.0
        self._closed = False
        self.sequence = _last_sequence(self.events_path)
        manifest_path = self.telemetry_dir / "manifest.json"
        if manifest_path.is_file():
            manifest = read_json_object(manifest_path)
            if manifest.get("run_id") != self.run_id:
                raise RuntimeError(
                    "telemetria existente pertence a um protocolo incompatível"
                )
        else:
            write_json_atomic(
                manifest_path,
                {
                    "schema_version": TELEMETRY_SCHEMA_VERSION,
                    "run_id": self.run_id,
                    "created_at": _utc_now(),
                    "identity": identity,
                },
            )
        needs_separator = _needs_jsonl_separator(self.events_path)
        self._handle = self.events_path.open("a", encoding="utf-8", buffering=1)
        if needs_separator:
            self._handle.write("\n")
            self._handle.flush()
        self.emit(
            "session_start",
            resumed=resumed,
            telemetry_every=every,
        )

    def emit(self, event_type: str, **payload: Any) -> int:
        if self._closed:
            raise RuntimeError("telemetria já foi encerrada")
        if not event_type:
            raise ValueError("event_type não pode ser vazio")
        reserved = {
            "schema_version",
            "run_id",
            "session_id",
            "sequence",
            "timestamp",
            "session_elapsed_seconds",
            "event",
        }
        overlap = reserved.intersection(payload)
        if overlap:
            raise ValueError(f"campos reservados na telemetria: {sorted(overlap)}")
        write_started = time.perf_counter()
        self.sequence += 1
        event = {
            "schema_version": TELEMETRY_SCHEMA_VERSION,
            "run_id": self.run_id,
            "session_id": self.session_id,
            "sequence": self.sequence,
            "timestamp": _utc_now(),
            "session_elapsed_seconds": time.perf_counter() - self._started,
            "event": event_type,
            **payload,
        }
        line = json.dumps(
            event,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        self._handle.write(line + "\n")
        self._handle.flush()
        self._write_seconds += time.perf_counter() - write_started
        return self.sequence

    def publish_heat(
        self,
        model: nn.Module | None,
        *,
        context: dict[str, Any],
        stage_snapshot: bool = False,
    ) -> dict[str, Any]:
        write_started = time.perf_counter()
        snapshot = build_heat_snapshot(
            model,
            context=context,
            run_id=self.run_id,
            session_id=self.session_id,
            sequence=self.sequence,
        )
        write_json_atomic(self.heat_latest_path, snapshot)
        if stage_snapshot and model is not None:
            method = _SAFE_COMPONENT.sub("_", str(context.get("method", "method")))
            stage = int(context.get("stage", 0)) + 1
            write_json_atomic(
                self.heat_history_dir
                / f"{method}-stage-{stage:02d}-seq-{self.sequence:08d}.json",
                snapshot,
            )
        self._write_seconds += time.perf_counter() - write_started
        return snapshot

    def should_publish_batch(self, method_step: int) -> bool:
        return method_step > 0 and method_step % self.every == 0

    @property
    def is_closed(self) -> bool:
        return self._closed

    @property
    def overhead_seconds(self) -> float:
        return self._write_seconds

    def close(
        self, *, status: str = "complete", error: BaseException | None = None
    ) -> None:
        if self._closed:
            return
        if error is None:
            self.emit("session_end", status=status)
        else:
            self.emit(
                "session_error",
                status="failed",
                error_type=type(error).__name__,
                error_message=str(error)[:2_000],
            )
        self._closed = True
        self._handle.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close(error=exc)


def cuda_memory_payload(device: str) -> dict[str, int | None]:
    selected = torch.device(device)
    if selected.type != "cuda":
        return {
            "gpu_allocated_bytes": None,
            "gpu_reserved_bytes": None,
            "gpu_peak_bytes": None,
        }
    return {
        "gpu_allocated_bytes": int(torch.cuda.memory_allocated(selected)),
        "gpu_reserved_bytes": int(torch.cuda.memory_reserved(selected)),
        "gpu_peak_bytes": int(torch.cuda.max_memory_allocated(selected)),
    }


def finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None
