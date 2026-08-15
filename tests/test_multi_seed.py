import json

from experiments.multi_seed import exact_two_sided_sign_test, run_multi_seed
from experiments.synthetic_cl import SyntheticConfig


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
