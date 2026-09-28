

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


def test_resume_survives_json_tuple_to_list_roundtrip(tmp_path) -> None:
    """Resume must not break because JSON turns tuples into lists.

    The sweep index stores `configs` via config_payload(), which contains tuples
    (class_order, hidden_dims, methods). json.dump writes them as lists, so the
    reloaded index never compares equal to a freshly built one and resume dies
    with "indice incompativel" even when nothing changed. A killed run could
    therefore never be resumed -- the failure mode that matters, since these
    sweeps run for hours.
    """

    import json

    from experiments.replay_selection_sweep import (
        _index_identity_matches,
        config_payload,
        replay_selection_configs,
    )

    payload = {"split_cifar100": config_payload(replay_selection_configs("cuda")["split_cifar100"])}
    identity = {"seeds": [1, 2, 3], "configs": payload}

    # What lands on disk after a json round-trip.
    saved = json.loads(json.dumps(identity))

    assert _index_identity_matches(saved, identity), (
        "resume must treat a JSON round-trip of the same identity as equal"
    )


def test_index_identity_still_rejects_a_real_change() -> None:
    """The tuple/list fix must not turn the guard-rail into a no-op."""

    from experiments.replay_selection_sweep import _index_identity_matches

    base = {"seeds": [1, 2, 3], "configs": {"a": {"device": "cuda"}}}
    changed = {"seeds": [1, 2, 4], "configs": {"a": {"device": "cuda"}}}
    device_changed = {"seeds": [1, 2, 3], "configs": {"a": {"device": "cpu"}}}

    assert not _index_identity_matches(base, changed), "different seeds must be rejected"
    assert not _index_identity_matches(base, device_changed), "different device must be rejected"
