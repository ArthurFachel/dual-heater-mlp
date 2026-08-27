import json
import sys
from pathlib import Path

from experiments.multi_seed import exact_two_sided_sign_test, run_multi_seed
from experiments.artifacts import build_run_identity, ensure_run_identity
from experiments.provenance import environment_manifest
from experiments.synthetic_cl import SyntheticConfig


def test_run_identity_rejects_results_after_source_changes(tmp_path):
    project_root = tmp_path / "project"
    source = project_root / "src" / "package.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    first = build_run_identity(
        {"seeds": [1]},
        project_root=project_root,
        task_loader="tests.loader",
    )
    output = tmp_path / "results"
    ensure_run_identity(output, first, resume=True)
    (output / "seed_1").mkdir()
    (output / "seed_1" / "results.json").write_text("{}", encoding="utf-8")

    source.write_text("VALUE = 2\n", encoding="utf-8")
    second = build_run_identity(
        {"seeds": [1]},
        project_root=project_root,
        task_loader="tests.loader",
    )
    try:
        ensure_run_identity(output, second, resume=True)
    except RuntimeError:
        pass
    else:
        raise AssertionError("retomada aceitou resultados produzidos por outro código")


def test_environment_command_records_relative_script_and_path_arguments(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    monkeypatch.setattr(sys, "argv", [
        str(root / "run_dualheat_pairs.py"),
        f"--data-dir={tmp_path / 'data'}",
        "--output-dir", str(root / "results/paired"),
        "--device", "cpu",
    ])
    manifest = environment_manifest(root)
    assert manifest["command"] == [
        Path(sys.executable).name, "run_dualheat_pairs.py", "--data-dir=../data",
        "--output-dir", "results/paired", "--device", "cpu",
    ]
    assert manifest["command_path_base"] == "project_root"
    assert str(tmp_path) not in json.dumps(manifest)


def test_exact_sign_test_handles_ties_and_one_sided_outcomes():
    assert exact_two_sided_sign_test([0.0, 1.0, 2.0, 3.0]) == 2 * (1 / 2**3)
    assert exact_two_sided_sign_test([0.0, 0.0]) is None


def test_multi_seed_runner_is_serial_and_writes_aggregate(tmp_path):
    config = SyntheticConfig(
        n_features=3,
        classes_per_task=2,
        task_count=1,
        train_per_class=3,
        test_per_class=2,
        hidden_dims=(4,),
        batch_size=2,
        steps_per_task=1,
        methods=("vanilla", "slowheat_max"),
    )

    aggregate = run_multi_seed(config, seeds=[2, 5], output_dir=tmp_path)

    assert aggregate["seeds"] == [2, 5]
    assert aggregate["methods"].keys() == {"vanilla", "slowheat_max"}
    assert (tmp_path / "seed_2" / "results.json").is_file()
    assert (tmp_path / "seed_5" / "results.json").is_file()
    saved = json.loads((tmp_path / "aggregate.json").read_text())
    assert saved["seeds"] == [2, 5]
