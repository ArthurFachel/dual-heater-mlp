"""Model-ranked episodic replay memory for continual-learning experiments."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

ReplaySelectionStrategy = Literal["first", "loss", "representative", "hybrid"]
REPLAY_SELECTION_STRATEGIES: tuple[ReplaySelectionStrategy, ...] = (
    "first",
    "loss",
    "representative",
    "hybrid",
)
HYBRID_WEIGHTS = (0.50, 0.30, 0.20)
REPLAY_BUFFER_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ReplaySelection:
    """One task-boundary selection, already materialized on CPU."""

    inputs: Tensor
    targets: Tensor
    source_tasks: Tensor
    source_indices: Tensor
    scores: Tensor
    score_components: Tensor
    logits: Tensor | None
    selector_forward_examples: int
    selector_distance_flops: int
    history: dict[str, Any]


def _empty_tensor(dtype: torch.dtype, *shape: int) -> Tensor:
    return torch.empty(shape, dtype=dtype)


class ReplayBuffer:
    """Append-only, task-balanced replay memory with safe tensor state."""

    def __init__(self) -> None:
        self.inputs: Tensor | None = None
        self.targets = _empty_tensor(torch.long, 0)
        self.source_tasks = _empty_tensor(torch.long, 0)
        self.source_indices = _empty_tensor(torch.long, 0)
        self.scores = _empty_tensor(torch.float32, 0)
        self.score_components = _empty_tensor(torch.float32, 0, 3)
        self.logits: Tensor | None = None
        self.selection_history: list[dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self.targets)

    def as_memory(self) -> tuple[Tensor, Tensor] | None:
        if self.inputs is None or len(self) == 0:
            return None
        return self.inputs, self.targets

    def append(self, selection: ReplaySelection) -> None:
        count = len(selection.targets)
        fields = (
            selection.inputs,
            selection.source_tasks,
            selection.source_indices,
            selection.scores,
            selection.score_components,
        )
        if any(len(field) != count for field in fields):
            raise ValueError("seleção de replay contém tensores desalinhados")
        if selection.inputs.device.type != "cpu":
            raise ValueError("ReplayBuffer aceita somente tensores em CPU")
        if selection.inputs.dtype != torch.float32:
            raise ValueError("imagens de replay devem usar float32")
        if selection.targets.dtype != torch.long:
            raise ValueError("rótulos de replay devem usar int64")
        if selection.score_components.shape != (count, 3):
            raise ValueError("score_components deve ter forma [N, 3]")
        if selection.logits is not None and len(selection.logits) != count:
            raise ValueError("logits de replay estão desalinhados")
        if self.logits is not None and selection.logits is None:
            raise ValueError("uma memória com logits requer logits em todas as seleções")
        if self.logits is None and len(self) > 0 and selection.logits is not None:
            raise ValueError("não é possível adicionar logits a uma memória sem logits")

        self.inputs = (
            selection.inputs.detach().cpu().contiguous()
            if self.inputs is None
            else torch.cat((self.inputs, selection.inputs.detach().cpu()))
        )
        self.targets = torch.cat((self.targets, selection.targets.detach().cpu()))
        self.source_tasks = torch.cat(
            (self.source_tasks, selection.source_tasks.detach().cpu())
        )
        self.source_indices = torch.cat(
            (self.source_indices, selection.source_indices.detach().cpu())
        )
        self.scores = torch.cat((self.scores, selection.scores.detach().cpu()))
        self.score_components = torch.cat(
            (self.score_components, selection.score_components.detach().cpu())
        )
        if selection.logits is not None:
            selected_logits = selection.logits.detach().cpu().contiguous()
            self.logits = (
                selected_logits
                if self.logits is None
                else torch.cat((self.logits, selected_logits))
            )
        self.selection_history.append(selection.history)

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REPLAY_BUFFER_SCHEMA_VERSION,
            "inputs": self.inputs,
            "targets": self.targets,
            "source_tasks": self.source_tasks,
            "source_indices": self.source_indices,
            "scores": self.scores,
            "score_components": self.score_components,
            "logits": self.logits,
            "selection_history": self.selection_history,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("schema_version") != REPLAY_BUFFER_SCHEMA_VERSION:
            raise ValueError("versão incompatível do ReplayBuffer")
        required = {
            "inputs",
            "targets",
            "source_tasks",
            "source_indices",
            "scores",
            "score_components",
            "logits",
            "selection_history",
        }
        if not required.issubset(state):
            raise ValueError("estado do ReplayBuffer está incompleto")
        inputs = state["inputs"]
        targets = state["targets"]
        if inputs is not None and not isinstance(inputs, Tensor):
            raise TypeError("inputs inválido no ReplayBuffer")
        if not isinstance(targets, Tensor):
            raise TypeError("targets inválido no ReplayBuffer")
        count = len(targets)
        tensor_fields = {
            name: state[name]
            for name in (
                "source_tasks",
                "source_indices",
                "scores",
                "score_components",
            )
        }
        if any(not isinstance(value, Tensor) for value in tensor_fields.values()):
            raise TypeError("metadados tensoriais inválidos no ReplayBuffer")
        if inputs is not None and len(inputs) != count:
            raise ValueError("inputs e targets desalinhados no ReplayBuffer")
        if any(len(value) != count for value in tensor_fields.values()):
            raise ValueError("metadados desalinhados no ReplayBuffer")
        logits = state["logits"]
        if logits is not None and (not isinstance(logits, Tensor) or len(logits) != count):
            raise ValueError("logits desalinhados no ReplayBuffer")
        history = state["selection_history"]
        if not isinstance(history, list) or any(not isinstance(item, dict) for item in history):
            raise TypeError("selection_history inválido no ReplayBuffer")

        self.inputs = None if inputs is None else inputs.detach().cpu().contiguous()
        self.targets = targets.detach().cpu().to(torch.long).contiguous()
        self.source_tasks = state["source_tasks"].detach().cpu().to(torch.long)
        self.source_indices = state["source_indices"].detach().cpu().to(torch.long)
        self.scores = state["scores"].detach().cpu().to(torch.float32)
        self.score_components = (
            state["score_components"].detach().cpu().to(torch.float32)
        )
        self.logits = None if logits is None else logits.detach().cpu().contiguous()
        self.selection_history = list(history)

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> ReplayBuffer:
        buffer = cls()
        buffer.load_state_dict(state)
        return buffer

    @property
    def replay_memory_bytes(self) -> int:
        if self.inputs is None:
            return 0
        return self.inputs.numel() * self.inputs.element_size() + (
            self.targets.numel() * self.targets.element_size()
        )

    @property
    def stored_logits_bytes(self) -> int:
        if self.logits is None:
            return 0
        return self.logits.numel() * self.logits.element_size()

    @property
    def metadata_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                self.source_tasks,
                self.source_indices,
                self.scores,
                self.score_components,
            )
        )


def _features_and_logits(model: nn.Module, inputs: Tensor) -> tuple[Tensor, Tensor]:
    extractor = getattr(model, "forward_features", None)
    if callable(extractor):
        features = extractor(inputs)
        classifier = getattr(model, "classifier", None)
        if isinstance(classifier, nn.Module):
            return features.flatten(1), classifier(features)
        if isinstance(model, nn.Sequential) and len(model) > 0:
            return features.flatten(1), model[-1](features)
        raise TypeError("modelo com forward_features não expõe classificador")
    if isinstance(model, nn.Sequential) and len(model) > 0:
        features = inputs
        for module in list(model.children())[:-1]:
            features = module(features)
        return features.flatten(1), model[-1](features)
    raise TypeError("modelo deve expor forward_features ou ser nn.Sequential")


def _percentile_rank(values: Tensor) -> Tensor:
    """Average percentile ranks with deterministic treatment of ties."""

    if values.ndim != 1 or len(values) == 0:
        raise ValueError("values deve ser um vetor não vazio")
    if len(values) == 1:
        return torch.ones_like(values, dtype=torch.float32)
    unique, inverse, counts = torch.unique(
        values, sorted=True, return_inverse=True, return_counts=True
    )
    del unique
    starts = torch.cumsum(counts, dim=0) - counts
    average_positions = starts.to(torch.float32) + (counts.to(torch.float32) - 1) / 2
    return average_positions[inverse] / (len(values) - 1)


def _herding(features: Tensor, count: int) -> tuple[Tensor, Tensor, int]:
    normalized = F.normalize(features, dim=1)
    centroid = normalized.mean(dim=0)
    selected: list[int] = []
    utilities: list[float] = []
    selected_sum = torch.zeros_like(centroid)
    flops = 0
    for step in range(count):
        candidate_means = (normalized + selected_sum) / (step + 1)
        distances = (candidate_means - centroid).square().sum(dim=1)
        if selected:
            distances[selected] = torch.inf
        index = int(torch.argmin(distances).item())
        finite = distances[torch.isfinite(distances)]
        maximum = float(finite.max().item()) if len(finite) else 0.0
        utility = 1.0 if maximum == 0.0 else 1.0 - float(distances[index]) / maximum
        selected.append(index)
        utilities.append(utility)
        selected_sum = selected_sum + normalized[index]
        flops += len(features) * features.shape[1] * 3
    return torch.tensor(selected, dtype=torch.long), torch.tensor(utilities), flops


def _hybrid_selection(
    features: Tensor,
    loss_ranks: Tensor,
    entropy_ranks: Tensor,
    count: int,
) -> tuple[Tensor, Tensor, Tensor, int]:
    normalized = F.normalize(features, dim=1)
    centroid = F.normalize(normalized.mean(dim=0), dim=0)
    centrality = 1.0 - _percentile_rank(1.0 - normalized @ centroid)
    selected: list[int] = []
    utilities: list[float] = []
    coverages: list[float] = []
    flops = len(features) * features.shape[1] * 2
    difficulty_weight, uncertainty_weight, coverage_weight = HYBRID_WEIGHTS
    for _ in range(count):
        if not selected:
            coverage = centrality
        else:
            similarities = normalized @ normalized[selected].T
            distances = (1.0 - similarities).amin(dim=1)
            coverage = _percentile_rank(distances)
            flops += len(features) * len(selected) * features.shape[1] * 2
        utility = (
            difficulty_weight * loss_ranks
            + uncertainty_weight * entropy_ranks
            + coverage_weight * coverage
        )
        if selected:
            utility[selected] = -torch.inf
        index = int(torch.argmax(utility).item())
        selected.append(index)
        utilities.append(float(utility[index].item()))
        coverages.append(float(coverage[index].item()))
    return (
        torch.tensor(selected, dtype=torch.long),
        torch.tensor(utilities, dtype=torch.float32),
        torch.tensor(coverages, dtype=torch.float32),
        flops,
    )


def _selection_digest(inputs: Tensor, targets: Tensor, indices: Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in (inputs, targets, indices):
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(str(contiguous.dtype).encode())
        digest.update(str(tuple(contiguous.shape)).encode())
        digest.update(memoryview(contiguous.numpy()).cast("B"))
    return digest.hexdigest()


@torch.no_grad()
def select_task_exemplars(
    model: nn.Module,
    task: Any,
    *,
    task_index: int,
    seen_classes: tuple[int, ...],
    samples_per_class: int,
    strategy: ReplaySelectionStrategy,
    device: str,
    batch_size: int = 1_024,
    store_logits: bool = False,
) -> ReplaySelection:
    """Rank only training examples and return a deterministic per-class cache."""

    if strategy not in REPLAY_SELECTION_STRATEGIES:
        raise ValueError(f"estratégia de replay inválida: {strategy}")
    if samples_per_class < 1 or batch_size < 1:
        raise ValueError("samples_per_class e batch_size devem ser positivos")
    inputs = task.train_x.detach().cpu().to(torch.float32)
    targets = task.train_y.detach().cpu().to(torch.long)
    if len(inputs) != len(targets) or len(inputs) == 0:
        raise ValueError("tarefa de replay deve conter treino não vazio e alinhado")

    needs_ranking = strategy != "first"
    needs_forward = needs_ranking
    features_parts: list[Tensor] = []
    logits_parts: list[Tensor] = []
    was_training = model.training
    if needs_forward:
        model.eval()
        for start in range(0, len(inputs), batch_size):
            batch = inputs[start : start + batch_size].to(device)
            features, logits = _features_and_logits(model, batch)
            features_parts.append(features.detach().cpu().to(torch.float32))
            logits_parts.append(logits.detach().cpu().to(torch.float32))
        model.train(was_training)
    features = torch.cat(features_parts) if features_parts else None
    logits = torch.cat(logits_parts) if logits_parts else None

    if needs_ranking:
        assert logits is not None
        class_indices = torch.tensor(seen_classes, dtype=torch.long)
        seen_logits = logits.index_select(1, class_indices)
        label_to_local = torch.full((logits.shape[1],), -1, dtype=torch.long)
        label_to_local[class_indices] = torch.arange(len(class_indices))
        local_targets = label_to_local[targets]
        if (local_targets < 0).any():
            raise ValueError("treino contém classe fora de seen_classes")
        losses = F.cross_entropy(seen_logits, local_targets, reduction="none")
        probabilities = F.softmax(seen_logits, dim=1)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(1)
        if len(seen_classes) > 1:
            entropy = entropy / math.log(len(seen_classes))
    else:
        losses = torch.zeros(len(inputs), dtype=torch.float32)
        entropy = torch.zeros(len(inputs), dtype=torch.float32)

    chosen_parts: list[Tensor] = []
    score_parts: list[Tensor] = []
    component_parts: list[Tensor] = []
    distance_flops = 0
    class_history: list[dict[str, Any]] = []
    for label in task.classes:
        candidates = torch.nonzero(targets == label, as_tuple=False).flatten()
        count = min(samples_per_class, len(candidates))
        class_losses = losses[candidates]
        class_entropy = entropy[candidates]
        loss_ranks = _percentile_rank(class_losses)
        entropy_ranks = _percentile_rank(class_entropy)
        coverage = torch.zeros(count, dtype=torch.float32)
        if strategy == "first":
            local = torch.arange(count)
            scores = torch.zeros(count, dtype=torch.float32)
        elif strategy == "loss":
            local = torch.argsort(class_losses, descending=True, stable=True)[:count]
            scores = loss_ranks[local]
        elif strategy == "representative":
            assert features is not None
            local, scores, flops = _herding(features[candidates], count)
            coverage = scores.clone()
            distance_flops += flops
        else:
            assert features is not None
            local, scores, coverage, flops = _hybrid_selection(
                features[candidates], loss_ranks, entropy_ranks, count
            )
            distance_flops += flops
        chosen = candidates[local]
        chosen_parts.append(chosen)
        score_parts.append(scores)
        component_parts.append(
            torch.stack(
                (loss_ranks[local], entropy_ranks[local], coverage), dim=1
            )
        )
        class_history.append(
            {
                "label": int(label),
                "candidate_count": len(candidates),
                "selected_count": count,
                "selected_source_indices": chosen.tolist(),
                "selected_scores": [float(value) for value in scores],
                "selected_components": [
                    [float(component) for component in row]
                    for row in component_parts[-1]
                ],
            }
        )

    chosen = torch.cat(chosen_parts)
    selected_inputs = inputs[chosen].contiguous()
    selected_targets = targets[chosen].contiguous()
    selected_scores = torch.cat(score_parts).to(torch.float32)
    selected_components = torch.cat(component_parts).to(torch.float32)
    selected_logits = None if logits is None or not store_logits else logits[chosen]
    stored_logit_examples = 0
    if store_logits and selected_logits is None:
        stored_parts: list[Tensor] = []
        model.eval()
        for start in range(0, len(selected_inputs), batch_size):
            batch = selected_inputs[start : start + batch_size].to(device)
            stored_parts.append(model(batch).detach().cpu().to(torch.float32))
        model.train(was_training)
        selected_logits = torch.cat(stored_parts)
        stored_logit_examples = len(selected_inputs)
    history = {
        "task_index": task_index,
        "strategy": strategy,
        "candidate_count": len(inputs),
        "selected_count": len(chosen),
        "selected_sha256": _selection_digest(
            selected_inputs, selected_targets, chosen
        ),
        "classes": class_history,
        "score_min": float(selected_scores.min().item()),
        "score_mean": float(selected_scores.mean().item()),
        "score_max": float(selected_scores.max().item()),
    }
    return ReplaySelection(
        inputs=selected_inputs,
        targets=selected_targets,
        source_tasks=torch.full((len(chosen),), task_index, dtype=torch.long),
        source_indices=chosen.to(torch.long),
        scores=selected_scores,
        score_components=selected_components,
        logits=selected_logits,
        selector_forward_examples=(
            len(inputs) if needs_forward else stored_logit_examples
        ),
        selector_distance_flops=distance_flops,
        history=history,
    )
