from experiments.split_mnist_suite import SLOWHEAT_DERPP_METHODS
from experiments.visual_generalization import generalization_configs


def test_generalization_configs_preserve_scenario_semantics():
    configs = generalization_configs()

    assert configs["permuted_mnist"].scenario == "domain_incremental"
    assert configs["permuted_mnist"].task_count == 5
    assert configs["split_cifar100"].scenario == "class_incremental"
    assert len(configs["split_cifar100"].class_order) == 100
    tiny_imagenet = configs["tiny_imagenet"]
    assert tiny_imagenet.input_dim == 3 * 64 * 64
    assert tiny_imagenet.scenario == "class_incremental"
    assert tiny_imagenet.task_count == 10
    assert tiny_imagenet.classes_per_task == 20
    for config in configs.values():
        assert config.methods == SLOWHEAT_DERPP_METHODS
        config.validate()
