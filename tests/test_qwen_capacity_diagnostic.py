"""CPU tests for the Qwen capacity diagnostic runner.

Only the pieces that do not need a checkpoint or a dataset are exercised here:
task-order resolution and the per-layer report assembly. The measurement itself
is a script, verified by running it; what is pinned here is the contract that
declares the protocol (which domains, in which order).
"""

import pytest
import torch

pytest.importorskip("transformers")

from experiments.qwen_capacity_diagnostic import _resolve_domains
from experiments.split_clinc150 import CLINC150_DOMAINS


def test_tasks_count_takes_the_declared_dictionary_prefix():
    assert _resolve_domains(None, 2) == ["banking", "credit_cards"]
    assert _resolve_domains(None, 3) == [
        "banking",
        "credit_cards",
        "kitchen_and_dining",
    ]


def test_tasks_count_is_bounded_by_the_available_domains():
    with pytest.raises(ValueError, match="entre 2 e 10"):
        _resolve_domains(None, 1)
    with pytest.raises(ValueError, match="entre 2 e 10"):
        _resolve_domains(None, len(CLINC150_DOMAINS) + 1)


def test_explicit_domains_preserve_the_given_order():
    """Order is the point: the reversed sequence must not be re-sorted."""

    assert _resolve_domains(["credit_cards", "banking"], 2) == [
        "credit_cards",
        "banking",
    ]


def test_explicit_domains_override_the_task_count():
    resolved = _resolve_domains(["travel", "work", "home"], 2)

    assert resolved == ["travel", "work", "home"]


def test_unknown_domain_is_rejected_with_the_valid_names():
    with pytest.raises(ValueError, match="domínio desconhecido: bankingg"):
        _resolve_domains(["bankingg", "banking"], 2)


def test_repeated_domain_is_rejected():
    """A repeated domain would silently train the same task twice."""

    with pytest.raises(ValueError, match="não pode repetir"):
        _resolve_domains(["banking", "banking"], 2)


def test_a_single_domain_is_rejected():
    with pytest.raises(ValueError, match="ao menos dois"):
        _resolve_domains(["banking"], 2)


def test_layer_report_carries_the_raw_profile_per_layer():
    """The anomaly reading needs the pre-budget shape, not only the heat."""

    from experiments.qwen_capacity_diagnostic import _layer_report

    class _Tracker:
        def __init__(self, memory):
            self.importance_memory = memory

    class _Model:
        def __init__(self, memories):
            self._trackers = [_Tracker(memory) for memory in memories]

        def get_ffn_trackers(self):
            return self._trackers

    concentrated = torch.full((100,), 1e-6)
    concentrated[0] = 1.0
    flat = torch.linspace(1.0, 2.0, 100)

    report = _layer_report(_Model([concentrated, flat]), 0.25, "local")

    assert [entry["layer"] for entry in report] == [0, 1]
    assert report[0]["raw"]["top_1_mass"] > 0.99
    assert report[1]["raw"]["top_1_mass"] < 0.05
    # The heat-side concentration ratio still separates them too.
    assert report[0]["concentration_ratio"] < report[1]["concentration_ratio"]
