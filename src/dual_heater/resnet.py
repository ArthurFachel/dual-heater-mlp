"""CIFAR-style ResNet-18 backbones with graph-aware SlowHeat instrumentation."""

from __future__ import annotations

from collections.abc import Callable

from torch import Tensor, nn

from .slow_heat import SlowHeatChannelTracker, SlowHeatConv2d, SlowHeatLinear


def _group_count(channels: int, maximum: int = 8) -> int:
    """Return the largest valid GroupNorm group count up to ``maximum``."""

    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class _CIFARBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int,
        convolution: Callable[..., nn.Module],
        normalization: Callable[[int], nn.Module],
        tracker: Callable[[int], nn.Module],
    ) -> None:
        super().__init__()
        self.conv1 = convolution(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.norm1 = normalization(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = convolution(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm2 = normalization(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.downsample_conv: nn.Module | None = convolution(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
            )
            self.downsample_norm: nn.Module | None = normalization(out_channels)
        else:
            self.downsample_conv = None
            self.downsample_norm = None
        self.output_tracker = tracker(out_channels)

    def forward(self, inputs: Tensor) -> Tensor:
        identity = inputs
        output = self.relu(self.norm1(self.conv1(inputs)))
        output = self.norm2(self.conv2(output))
        if self.downsample_conv is not None:
            assert self.downsample_norm is not None
            identity = self.downsample_norm(self.downsample_conv(identity))
        return self.output_tracker(self.relu(output + identity))


class _CIFARResNet18(nn.Module):
    """ResNet-18 with a 3x3 CIFAR stem, GroupNorm and global pooling."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        stage_channels: tuple[int, int, int, int] = (64, 128, 256, 512),
        blocks_per_stage: tuple[int, int, int, int] = (2, 2, 2, 2),
        slowheat: bool,
        slow_strength: float = 3.0,
        plasticity_budget: float = 0.25,
        importance_decay: float = 0.99,
        importance_eps: float = 1e-8,
        protect_output: bool = True,
        output_slow_strength: float | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(in_channels, int) or isinstance(in_channels, bool) or in_channels < 1:
            raise ValueError("in_channels deve ser um inteiro positivo")
        if not isinstance(num_classes, int) or isinstance(num_classes, bool) or num_classes < 1:
            raise ValueError("num_classes deve ser um inteiro positivo")
        if len(stage_channels) != 4 or any(
            not isinstance(width, int) or isinstance(width, bool) or width < 1
            for width in stage_channels
        ):
            raise ValueError("stage_channels deve conter quatro inteiros positivos")
        if len(blocks_per_stage) != 4 or any(
            not isinstance(count, int) or isinstance(count, bool) or count < 1
            for count in blocks_per_stage
        ):
            raise ValueError(
                "blocks_per_stage deve conter quatro inteiros positivos"
            )

        common = {
            "slow_strength": slow_strength,
            "plasticity_budget": plasticity_budget,
            "importance_decay": importance_decay,
            "importance_eps": importance_eps,
        }

        def convolution(*args, **kwargs):
            return (
                SlowHeatConv2d(*args, **kwargs, **common)
                if slowheat
                else nn.Conv2d(*args, **kwargs)
            )

        def normalization(channels: int) -> nn.Module:
            return nn.GroupNorm(_group_count(channels), channels, affine=True)

        def tracker(channels: int) -> nn.Module:
            return SlowHeatChannelTracker(channels, **common) if slowheat else nn.Identity()

        self.stem = convolution(
            in_channels,
            stage_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.stem_norm = normalization(stage_channels[0])
        self.relu = nn.ReLU()
        self.stem_tracker = tracker(stage_channels[0])

        stages: list[nn.ModuleList] = []
        current_channels = stage_channels[0]
        for stage_index, (width, block_count) in enumerate(
            zip(stage_channels, blocks_per_stage, strict=True)
        ):
            blocks: list[nn.Module] = []
            for block_index in range(block_count):
                stride = 2 if stage_index > 0 and block_index == 0 else 1
                blocks.append(
                    _CIFARBasicBlock(
                        current_channels,
                        width,
                        stride=stride,
                        convolution=convolution,
                        normalization=normalization,
                        tracker=tracker,
                    )
                )
                current_channels = width
            stages.append(nn.ModuleList(blocks))
        self.stages = nn.ModuleList(stages)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        if slowheat and protect_output:
            classifier_common = dict(common)
            if output_slow_strength is not None:
                classifier_common["slow_strength"] = output_slow_strength
            self.classifier: nn.Module = SlowHeatLinear(
                stage_channels[-1], num_classes, **classifier_common
            )
        else:
            self.classifier = nn.Linear(stage_channels[-1], num_classes)
        self.slowheat = slowheat

    def forward_features(self, inputs: Tensor) -> Tensor:
        output = self.stem_tracker(self.relu(self.stem_norm(self.stem(inputs))))
        for stage in self.stages:
            for block in stage:
                output = block(output)
        return self.flatten(self.global_pool(output))

    def forward(self, inputs: Tensor) -> Tensor:
        return self.classifier(self.forward_features(inputs))

    def _slow_blocks(self) -> list[_CIFARBasicBlock]:
        if not self.slowheat:
            return []
        return [block for stage in self.stages for block in stage]

    def get_slow_layers(self) -> list[SlowHeatConv2d | SlowHeatLinear]:
        if not self.slowheat:
            return []
        layers: list[SlowHeatConv2d | SlowHeatLinear] = [self.stem]
        for block in self._slow_blocks():
            assert isinstance(block.conv1, SlowHeatConv2d)
            assert isinstance(block.conv2, SlowHeatConv2d)
            layers.extend((block.conv1, block.conv2))
            if block.downsample_conv is not None:
                assert isinstance(block.downsample_conv, SlowHeatConv2d)
                layers.append(block.downsample_conv)
        if isinstance(self.classifier, SlowHeatLinear):
            layers.append(self.classifier)
        return layers

    def get_slow_states(
        self,
    ) -> list[SlowHeatConv2d | SlowHeatLinear | SlowHeatChannelTracker]:
        if not self.slowheat:
            return []
        states: list[
            SlowHeatConv2d | SlowHeatLinear | SlowHeatChannelTracker
        ] = list(self.get_slow_layers())
        assert isinstance(self.stem_tracker, SlowHeatChannelTracker)
        states.append(self.stem_tracker)
        for block in self._slow_blocks():
            assert isinstance(block.output_tracker, SlowHeatChannelTracker)
            states.append(block.output_tracker)
        return states

    def get_slow_connections(
        self,
    ) -> list[tuple[nn.Module, nn.Module | None, int, str | None]]:
        """Return destination/source edges for graph-aware optimizer masks."""

        if not self.slowheat:
            return []
        connections: list[tuple[nn.Module, nn.Module | None, int, str | None]] = [
            (self.stem, None, 1, None)
        ]
        source: nn.Module = self.stem_tracker
        source_key = "stem_output"
        for stage_index, stage in enumerate(self.stages):
            for block_index, block in enumerate(stage):
                connections.append((block.conv1, source, 1, source_key))
                connections.append((block.conv2, block.conv1, 1, None))
                if block.downsample_conv is not None:
                    connections.append((block.downsample_conv, source, 1, source_key))
                source = block.output_tracker
                source_key = f"stage_{stage_index}_block_{block_index}_output"
        if isinstance(self.classifier, SlowHeatLinear):
            connections.append((self.classifier, source, 1, source_key))
        return connections

    def get_slow_channel_modules(
        self,
    ) -> list[tuple[nn.Module, nn.Module, str]]:
        """Return affine normalizers and the channel state protecting them."""

        if not self.slowheat:
            return []
        modules: list[tuple[nn.Module, nn.Module, str]] = [
            (self.stem_norm, self.stem, "stem")
        ]
        for stage_index, stage in enumerate(self.stages):
            for block_index, block in enumerate(stage):
                prefix = f"stage_{stage_index}_block_{block_index}"
                modules.extend(
                    (
                        (block.norm1, block.conv1, f"{prefix}_conv1"),
                        (block.norm2, block.conv2, f"{prefix}_conv2"),
                    )
                )
                if block.downsample_conv is not None:
                    assert block.downsample_norm is not None
                    modules.append(
                        (
                            block.downsample_norm,
                            block.downsample_conv,
                            f"{prefix}_downsample",
                        )
                    )
        return modules

    def consolidate(self, strategy: str = "max") -> None:
        for state in self.get_slow_states():
            state.consolidate(strategy=strategy)

    def get_lr_scales(self) -> list[Tensor]:
        return [state.get_lr_scales() for state in self.get_slow_states()]

    def adapt_capacity(
        self,
        *,
        acquisition_score: float,
        target_score: float,
        adaptation_rate: float = 0.1,
        minimum: float = 0.05,
        maximum: float = 0.95,
    ) -> list[float]:
        return [
            state.adapt_capacity(
                acquisition_score=acquisition_score,
                target_score=target_score,
                adaptation_rate=adaptation_rate,
                minimum=minimum,
                maximum=maximum,
            )
            for state in self.get_slow_states()
        ]


class CIFARResNet18(_CIFARResNet18):
    """Native CIFAR ResNet-18 control with GroupNorm."""

    def __init__(self, in_channels: int, num_classes: int, **kwargs) -> None:
        super().__init__(in_channels, num_classes, slowheat=False, **kwargs)


class SlowHeatResNet18(_CIFARResNet18):
    """CIFAR ResNet-18 instrumented with graph-aware Functional SlowHeat."""

    def __init__(self, in_channels: int, num_classes: int, **kwargs) -> None:
        super().__init__(in_channels, num_classes, slowheat=True, **kwargs)
