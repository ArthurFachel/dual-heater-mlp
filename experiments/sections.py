"""Single registry for sections exposed by the benchmark CLI."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SectionSpec:
    name: str
    output_dir: str
    default: bool = True
    all_dataset_method: bool = False


SECTION_SPECS = (
    SectionSpec("dualheat-pairs", "dualheat_pairs", default=False),
    SectionSpec(
        "replay-selection-sweep",
        "replay_selection_sweep",
        default=False,
    ),
    SectionSpec(
        "synthetic-all-methods",
        "synthetic_all_methods",
        default=False,
        all_dataset_method=True,
    ),
    SectionSpec(
        "split-mnist-all-methods",
        "split_mnist_all_methods",
        default=False,
        all_dataset_method=True,
    ),
    SectionSpec("confirmation", "confirmation"),
    SectionSpec("all-baselines", "all_baselines_equal_epochs"),
    SectionSpec("equal-examples", "all_baselines_equal_examples"),
    SectionSpec("ablations", "ablations"),
    SectionSpec("slowheat-derpp", "slowheat_derpp_exploratory"),
    SectionSpec("split-mnist-generalization", "split_mnist_generalization"),
    SectionSpec(
        "permuted-mnist",
        "permuted_mnist",
        all_dataset_method=True,
    ),
    SectionSpec(
        "split-cifar10",
        "split_cifar10",
        all_dataset_method=True,
    ),
    SectionSpec("split-cifar10-cnn", "split_cifar10_cnn", default=False),
    SectionSpec(
        "split-cifar10-cnn-sweep",
        "split_cifar10_cnn_sweep",
        default=False,
    ),
    SectionSpec("split-cifar10-vgg11", "split_cifar10_vgg11", default=False),
    SectionSpec(
        "split-cifar10-vgg11-all-methods",
        "split_cifar10_vgg11_all_methods",
        default=False,
    ),
    SectionSpec(
        "split-cifar10-resnet18-all-methods",
        "split_cifar10_resnet18_all_methods",
        default=False,
    ),
    SectionSpec(
        "functional-dualheat-pilot",
        "functional_dualheat_pilot",
        default=False,
    ),
    SectionSpec(
        "split-cifar10-vgg11-functional-dualheat",
        "split_cifar10_vgg11_functional_dualheat",
        default=False,
    ),
    SectionSpec(
        "split-cifar10-resnet18-functional-dualheat",
        "split_cifar10_resnet18_functional_dualheat",
        default=False,
    ),
    SectionSpec(
        "split-cifar100",
        "split_cifar100",
        all_dataset_method=True,
    ),
)

SECTION_NAMES = tuple(spec.name for spec in SECTION_SPECS)
DEFAULT_SECTION_NAMES = tuple(spec.name for spec in SECTION_SPECS if spec.default)
ALL_DATASET_METHOD_SECTIONS = tuple(
    spec.name for spec in SECTION_SPECS if spec.all_dataset_method
)
SECTION_OUTPUT_DIRS = {spec.name: spec.output_dir for spec in SECTION_SPECS}
