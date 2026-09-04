"""Backend-explicit process and PyTorch peak-memory measurements."""

from __future__ import annotations

import os
import threading
import weakref
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch


MemoryResult = dict[str, int | float | str | bool | None]


def process_rss_bytes() -> int:
    """Return this Linux process's current resident set size in bytes."""

    try:
        resident_pages = int(
            Path("/proc/self/statm").read_text(encoding="ascii").split()[1]
        )
    except (OSError, IndexError, ValueError) as error:
        raise RuntimeError("process RSS is unavailable from /proc/self/statm") from error
    return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))


def _stop_sampler(stop: threading.Event, thread: threading.Thread) -> None:
    stop.set()
    if thread.is_alive() and thread is not threading.current_thread():
        thread.join(timeout=1.0)


def _sample_rss_loop(
    reader: Callable[[], int],
    interval_seconds: float,
    stop: threading.Event,
    peak_holder: list[int],
    lock: threading.Lock,
) -> None:
    while not stop.wait(interval_seconds):
        observed = int(reader())
        with lock:
            peak_holder[0] = max(peak_holder[0], observed)


class PeakMemoryTracker:
    """Measure one method run without equating CPU RSS and CUDA allocator use."""

    def __init__(
        self,
        device: str,
        *,
        interval_seconds: float = 0.01,
        rss_reader: Callable[[], int] = process_rss_bytes,
    ) -> None:
        if interval_seconds <= 0.0:
            raise ValueError("interval_seconds must be positive")
        self.device = torch.device(device)
        self.interval_seconds = float(interval_seconds)
        self.rss_reader = rss_reader
        self._started = False
        self._stopped = False
        self._baseline: int | None = None
        self._peak: int | None = None
        self._reserved: int | None = None
        self._stop: threading.Event | None = None
        self._thread: threading.Thread | None = None
        self._peak_holder: list[int] | None = None
        self._lock = threading.Lock()
        self._finalizer: weakref.finalize | None = None

    def start(self) -> "PeakMemoryTracker":
        if self._started:
            raise RuntimeError("peak-memory tracker already started")
        self._started = True
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            self._baseline = int(torch.cuda.memory_allocated(self.device))
            torch.cuda.reset_peak_memory_stats(self.device)
            return self
        if self.device.type != "cpu":
            raise ValueError(f"unsupported memory backend for device {self.device}")

        self._baseline = int(self.rss_reader())
        self._peak_holder = [self._baseline]
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=_sample_rss_loop,
            args=(
                self.rss_reader,
                self.interval_seconds,
                self._stop,
                self._peak_holder,
                self._lock,
            ),
            name="peak-rss-sampler",
            daemon=True,
        )
        self._thread.start()
        self._finalizer = weakref.finalize(
            self,
            _stop_sampler,
            self._stop,
            self._thread,
        )
        return self

    def snapshot(self) -> MemoryResult:
        if not self._started:
            raise RuntimeError("peak-memory tracker has not started")
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            self._peak = int(torch.cuda.max_memory_allocated(self.device))
            self._reserved = int(torch.cuda.max_memory_reserved(self.device))
        else:
            observed = int(self.rss_reader())
            assert self._peak_holder is not None
            with self._lock:
                self._peak_holder[0] = max(self._peak_holder[0], observed)
                self._peak = self._peak_holder[0]
        return self._build_result()

    def stop(self) -> MemoryResult:
        if not self._started:
            raise RuntimeError("peak-memory tracker has not started")
        if not self._stopped:
            if self._stop is not None and self._thread is not None:
                _stop_sampler(self._stop, self._thread)
            self.snapshot()
            if self._finalizer is not None:
                self._finalizer.detach()
            self._stopped = True
        return self._build_result()

    def result(self) -> MemoryResult:
        return self.snapshot() if not self._stopped else self._build_result()

    def _build_result(self) -> MemoryResult:
        if self._baseline is None or self._peak is None:
            raise RuntimeError("peak-memory measurement is incomplete")
        backend = (
            "cuda_allocator_allocated"
            if self.device.type == "cuda"
            else "process_rss_sampled"
        )
        return {
            "peak_memory_available": True,
            "peak_memory_backend": backend,
            "peak_memory_bytes": self._peak,
            "peak_memory_baseline_bytes": self._baseline,
            "peak_memory_delta_bytes": max(0, self._peak - self._baseline),
            "peak_cuda_reserved_bytes": self._reserved,
            "peak_memory_sampling_interval_seconds": (
                self.interval_seconds if self.device.type == "cpu" else None
            ),
        }

    def __enter__(self) -> "PeakMemoryTracker":
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.stop()
