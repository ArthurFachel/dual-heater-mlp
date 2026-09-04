from __future__ import annotations

import pytest
import torch

from experiments.peak_memory import PeakMemoryTracker
from experiments.split_mnist import _merge_peak_memory, _unavailable_peak_memory


def test_cpu_tracker_reports_sampled_rss_peak_and_stops_thread():
    readings = iter((100, 140, 125))
    tracker = PeakMemoryTracker(
        "cpu",
        interval_seconds=3600.0,
        rss_reader=lambda: next(readings),
    ).start()

    assert tracker.snapshot() == {
        "peak_memory_available": True,
        "peak_memory_backend": "process_rss_sampled",
        "peak_memory_bytes": 140,
        "peak_memory_baseline_bytes": 100,
        "peak_memory_delta_bytes": 40,
        "peak_cuda_reserved_bytes": None,
        "peak_memory_sampling_interval_seconds": 3600.0,
    }
    result = tracker.stop()

    assert result["peak_memory_bytes"] == 140
    assert tracker._thread is not None
    assert not tracker._thread.is_alive()


def test_cpu_tracker_stops_sampler_when_context_raises():
    tracker = PeakMemoryTracker(
        "cpu",
        interval_seconds=3600.0,
        rss_reader=lambda: 100,
    )

    with pytest.raises(RuntimeError, match="boom"):
        with tracker:
            raise RuntimeError("boom")

    assert tracker._thread is not None
    assert not tracker._thread.is_alive()


def test_resume_merge_keeps_largest_peak_and_rejects_backend_change():
    cost = _unavailable_peak_memory()
    first = {
        "peak_memory_available": True,
        "peak_memory_backend": "process_rss_sampled",
        "peak_memory_bytes": 300,
        "peak_memory_baseline_bytes": 100,
        "peak_memory_delta_bytes": 200,
        "peak_cuda_reserved_bytes": None,
        "peak_memory_sampling_interval_seconds": 0.01,
    }
    second = {**first, "peak_memory_bytes": 250, "peak_memory_delta_bytes": 150}

    _merge_peak_memory(cost, first)
    _merge_peak_memory(cost, second)

    assert cost["peak_memory_bytes"] == 300
    assert cost["peak_memory_delta_bytes"] == 200
    with pytest.raises(RuntimeError, match="incompatible peak-memory backends"):
        _merge_peak_memory(
            cost,
            {
                **first,
                "peak_memory_backend": "cuda_allocator_allocated",
                "peak_cuda_reserved_bytes": 500,
            },
        )


def test_cuda_tracker_synchronizes_and_keeps_allocated_and_reserved_separate(
    monkeypatch,
):
    calls = []
    device = torch.device("cuda:1")
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda observed: calls.append(("synchronize", observed)),
    )
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda observed: 100)
    monkeypatch.setattr(
        torch.cuda,
        "reset_peak_memory_stats",
        lambda observed: calls.append(("reset", observed)),
    )
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda observed: 300)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda observed: 500)

    tracker = PeakMemoryTracker(str(device)).start()
    result = tracker.stop()

    assert calls[:2] == [("synchronize", device), ("reset", device)]
    assert calls[-1] == ("synchronize", device)
    assert result == {
        "peak_memory_available": True,
        "peak_memory_backend": "cuda_allocator_allocated",
        "peak_memory_bytes": 300,
        "peak_memory_baseline_bytes": 100,
        "peak_memory_delta_bytes": 200,
        "peak_cuda_reserved_bytes": 500,
        "peak_memory_sampling_interval_seconds": None,
    }
