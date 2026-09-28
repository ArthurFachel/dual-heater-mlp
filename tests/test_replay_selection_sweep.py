

def test_replay_selector_confirmatory_seeds_are_registered_and_disjoint() -> None:
    """The IJCNN selector-inversion confirmation must own a disjoint seed band.

    Pre-registration: goals/protocol_replay_selector_confirmation.md (R4).
    A markdown freeze does not stop anyone from running the wrong seeds; the
    band has to exist in code so the guard-rail can reject overlaps.
    """

    import json
    from pathlib import Path

    from experiments.confirmatory_split_mnist import (
        CONFIRMATORY_SEEDS,
        DECLARED_EXPLORATORY_SEEDS,
        DERPP_CONFIRMATORY_SEEDS,
    )
    from experiments.replay_selection_sweep import (
        REPLAY_SELECTOR_CONFIRMATORY_SEEDS,
    )

    seeds = REPLAY_SELECTOR_CONFIRMATORY_SEEDS
    assert len(seeds) == 20, "R4 freezes 20 seeds"
    assert len(set(seeds)) == 20, "seeds must be unique"

    for name, reserved in (
        ("split_mnist_confirmatory", CONFIRMATORY_SEEDS),
        ("declared_exploratory", DECLARED_EXPLORATORY_SEEDS),
        ("derpp_confirmatory", DERPP_CONFIRMATORY_SEEDS),
    ):
        overlap = set(seeds) & set(reserved)
        assert not overlap, f"selector band collides with {name}: {sorted(overlap)}"

    # The 50 exploratory sweep seeds are the band this confirmation must not
    # reuse: reusing them would re-analyse the data that generated the
    # hypothesis, which is the exact failure the pre-registration exists to
    # prevent. They are stored in the artefact, not in code.
    report = Path("results/cache_derpp_10seeds/replay_selection_sweep/sweep_report.json")
    if report.exists():
        exploratory = set(json.loads(report.read_text())["seeds"])
        overlap = set(seeds) & exploratory
        assert not overlap, f"selector band reuses exploratory sweep seeds: {sorted(overlap)}"
