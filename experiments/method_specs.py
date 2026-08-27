"""Declarative capabilities shared by benchmark method registries."""

from __future__ import annotations

import re
from dataclasses import dataclass

STRUCTURED_METHOD = re.compile(
    r"slowheat"
    r"(?:_(?P<auxiliary>replay|distillation|unidirectional))?"
    r"(?:_(?P<scope>hidden))?"
    r"_beta_(?P<beta>\d+(?:\.\d+)?)"
    r"(?:_budget_(?P<budget>\d+(?:\.\d+)?))?$"
)


@dataclass(frozen=True)
class MethodSpec:
    """Resolved capabilities for one benchmark method."""

    slowheat: bool = False
    replay: bool = False
    distillation: bool = False
    derpp: bool = False
    er_ace: bool = False
    lpr: bool = False
    classifier_expander: bool = False
    scroll: bool = False
    strength: float | None = None
    budget: float | None = None
    protect_output: bool = True
    epoch_budget_policy: str = "default"
    disable_capacity_budget: bool = False
    partial_output_protection: bool = False


def structured_method_match(method: str) -> re.Match[str] | None:
    return STRUCTURED_METHOD.fullmatch(method)


def method_epoch_budget(
    spec: MethodSpec,
    *,
    stage: int,
    default: int,
    replay_more: int,
    early_stopping: int,
) -> int:
    if spec.epoch_budget_policy == "scroll" and stage > 0:
        return 0
    if spec.epoch_budget_policy == "replay_more":
        return replay_more
    if spec.epoch_budget_policy == "early_stopping":
        return early_stopping
    if spec.epoch_budget_policy != "default":
        raise ValueError(f"política de épocas desconhecida: {spec.epoch_budget_policy}")
    return default
