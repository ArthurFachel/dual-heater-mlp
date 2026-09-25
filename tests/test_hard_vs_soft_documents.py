"""The hard-versus-soft documents must quote the artifacts, not memory.

This repository's largest documented debt is documents whose numbers no
aggregate on disk reproduces (`docs/audits/results_provenance_status.md`). This test
re-derives every primary-endpoint number quoted in the hard-versus-soft
documents straight from `pair_report.json` and fails if a document drifts.

It is skipped when the results are absent, so a fresh clone without the
artifacts still passes the suite.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "hard_vs_soft"
TARGETS = (
    "split_mnist_mlp",
    "permuted_mnist_mlp",
    "split_cifar10_mlp",
    "split_cifar100_mlp",
    "split_cifar10_cnn",
)
PRIMARY = "final_average_accuracy"


def _reports() -> dict[str, dict]:
    reports = {}
    for target in TARGETS:
        path = RESULTS / target / "pair_report.json"
        if not path.is_file():
            pytest.skip(f"artefato ausente: {path.relative_to(ROOT)}")
        reports[target] = json.loads(path.read_text(encoding="utf-8"))
    return reports


def _primary_deltas(reports: dict[str, dict]) -> dict[tuple[str, str], float]:
    """(target, contrast label) -> primary-endpoint difference in points."""
    return {
        (target, pair["label"]): pair["metrics"][PRIMARY]["mean_difference"] * 100
        for target, report in reports.items()
        for pair in report["pairs"]
    }


def _quoted_numbers(text: str) -> set[float]:
    """Every signed decimal in the text, normalized to two decimals.

    Documents quote in both locales (`-1.407` and `−1,41`), so the scan
    normalizes the Unicode minus and the decimal comma before rounding.
    """
    normalized = text.replace("\u2212", "-")
    found = set()
    for match in re.finditer(r"[-+]?\d+[.,]\d+", normalized):
        found.add(round(float(match.group().replace(",", ".")), 2))
    return found


def test_hard_never_beats_soft_on_the_primary_endpoint() -> None:
    """The documents' central claim, re-derived rather than trusted.

    Every document says hard protection wins nowhere. If a rerun ever made a
    hard arm win with Holm significance, the prose would silently become false.
    """
    reports = _reports()
    winners = [
        (target, pair["label"])
        for target, report in reports.items()
        for pair in report["pairs"]
        if pair["label"] == "Hard vs Soft"
        and pair["metrics"][PRIMARY]["mean_difference"] > 0
        and pair["metrics"][PRIMARY]["holm_adjusted_p"] < 0.05
    ]
    assert not winners, (
        "hard venceu soft com significância em "
        f"{winners}; a prosa de docs/results/hard_vs_soft_results.md está obsoleta"
    )


def test_cifar100_is_the_only_holm_surviving_primary_contrast() -> None:
    """The one result the documents report as significant, and only that one."""
    reports = _reports()
    survivors = {
        target
        for target, report in reports.items()
        for pair in report["pairs"]
        if pair["label"] == "Hard vs Soft"
        and pair["metrics"][PRIMARY]["holm_adjusted_p"] < 0.05
    }
    assert survivors == {"split_cifar100_mlp"}, survivors


def test_results_document_quotes_every_primary_delta() -> None:
    """Each of the 5 primary deltas must appear verbatim in the results doc."""
    deltas = _primary_deltas(_reports())
    text = (ROOT / "docs" / "results" / "hard_vs_soft_results.md").read_text(encoding="utf-8")
    quoted = _quoted_numbers(text)
    missing = [
        (target, round(value, 2))
        for (target, label), value in deltas.items()
        if label == "Hard vs Soft" and round(value, 2) not in quoted
    ]
    assert not missing, f"deltas primários ausentes ou divergentes no doc: {missing}"


@pytest.mark.parametrize(
    ("document", "target"),
    [
        # Each document must quote the target it actually covers. arch_cnn.md
        # covers the CNN, so demanding the MLP number there would be wrong.
        ("docs/results/hard_vs_soft_results.md", "split_cifar100_mlp"),
        ("docs/results/hard_vs_soft_results.md", "split_cifar10_cnn"),
        ("docs/architectures/arch_mlp.md", "split_cifar100_mlp"),
        ("docs/architectures/arch_cnn.md", "split_cifar10_cnn"),
        ("article/manuscript.md", "split_cifar100_mlp"),
    ],
)
def test_each_document_quotes_the_target_it_covers(document: str, target: str) -> None:
    """The headline delta each document leans on must match its artifact."""
    reports = _reports()
    pairs = {pair["label"]: pair for pair in reports[target]["pairs"]}
    delta = pairs["Hard vs Soft"]["metrics"][PRIMARY]["mean_difference"] * 100
    quoted = _quoted_numbers((ROOT / document).read_text(encoding="utf-8"))
    assert round(delta, 2) in quoted, (
        f"{document} não cita {delta:.2f}, o delta real de {target}"
    )


def test_every_target_reports_ten_paired_seeds() -> None:
    reports = _reports()
    for target, report in reports.items():
        assert len(report["seeds"]) == 10, target
        for pair in report["pairs"]:
            assert pair["metrics"][PRIMARY]["n_pairs"] == 10, (target, pair["label"])


def test_the_cnn_report_records_its_rebuild_provenance() -> None:
    """The CNN report was regenerated after its analysis step aborted.

    That fact must stay attached to the artifact, not only to the prose, or a
    later reader cannot tell the analysis code from the training identity.
    """
    reports = _reports()
    provenance = reports["split_cifar10_cnn"].get("analysis_provenance")
    assert provenance and provenance["rebuilt_from_existing_seeds"] is True
    assert len(provenance["analysis_source_sha256"]) == 64
    assert (RESULTS / "split_cifar10_cnn" / "run_identity.json").is_file()


def test_the_cnn_report_names_the_cnn_backbone() -> None:
    """A CNN run labelled 'MLP' would misreport where the numbers came from."""
    text = (RESULTS / "split_cifar10_cnn" / "pair_report.md").read_text(
        encoding="utf-8"
    )
    assert "CNN" in text
    assert "MLP [1]" not in text
