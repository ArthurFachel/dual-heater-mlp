from experiments.visual_generalization import generalization_configs


def test_generalization_configs_preserve_scenario_semantics():
    configs = generalization_configs()

    assert configs["permuted_mnist"].scenario == "domain_incremental"
    assert configs["permuted_mnist"].task_count == 5
    assert configs["split_cifar100"].scenario == "class_incremental"
    assert len(configs["split_cifar100"].class_order) == 100
    assert configs["tiny_imagenet"].input_dim == 3 * 64 * 64
    for config in configs.values():
        config.validate()
