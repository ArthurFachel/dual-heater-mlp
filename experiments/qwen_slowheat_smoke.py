"""CPU smoke check for SlowHeat on the real Qwen2.5-0.5B checkpoint.

Verifies, on the actual pretrained weights, that: instrumentation attaches to
all 24 layers, the trainable envelope matches the FFN-only design, masks are
well formed, one optimizer step leaves hard-protected units exactly unmoved,
and consolidation respects the plasticity budget.

Run:
    CUDA_VISIBLE_DEVICES= HF_HOME=.hf-cache .venv/bin/python \\
        experiments/qwen_slowheat_smoke.py
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoTokenizer

from dual_heater.optim import SlowHeatAdamW
from dual_heater.qwen import (
    QwenSlowHeatConfig,
    SlowHeatQwen2ForSequenceClassification,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--labels", type=int, default=150)
    parser.add_argument("--budget", type=float, default=0.25)
    parser.add_argument("--slow-strength", type=float, default=3.0)
    args = parser.parse_args()

    torch.manual_seed(0)
    slowheat = QwenSlowHeatConfig(
        slow_strength=args.slow_strength,
        ffn_plasticity_budget=args.budget,
        freeze_unbound_parameters=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = SlowHeatQwen2ForSequenceClassification.from_pretrained(
        args.model,
        num_labels=args.labels,
        slowheat_config=slowheat,
        dtype=torch.float32,
    )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    trackers = model.get_ffn_trackers()
    print(f"layers instrumented: {len(trackers)} / {model.config.num_hidden_layers}")
    print(f"units per tracker:   {trackers[0].units} (intermediate_size)")

    summary = model.mask_coverage_summary()
    total = sum(p.numel() for p in model.parameters())
    print(f"total parameters:    {total:,}")
    print(f"trainable:           {summary['trainable_parameter_count']:,}")
    print(f"masked:              {summary['masked_parameter_count']:,}")
    print(f"exempt (score head): {summary['exempt_parameter_count']:,}")
    print(f"bindings:            {summary['binding_count']}")
    print(f"uncovered trainable: {model.uncovered_trainable_parameters()}")

    for binding in model.mask_bindings():
        mask = binding.mask()
        assert (
            torch.broadcast_shapes(mask.shape, binding.parameter.shape)
            == binding.parameter.shape
        ), binding.kind
        assert torch.isfinite(mask).all(), binding.kind
    print("all masks broadcast and finite: ok")

    model.train()
    batch = tokenizer(
        ["what is my account balance", "transfer money to savings please"],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=32,
    )
    labels = torch.tensor([0, 1])

    optimizer = SlowHeatAdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-5
    )
    model.register_plasticity_masks(optimizer, hard=True)

    # Task 1: learn, then consolidate.
    optimizer.zero_grad()
    output = model(**batch, labels=labels)
    output.loss.backward()
    optimizer.step()
    print(f"task 1 loss:         {output.loss.item():.4f}")
    print(f"task_step recorded:  {trackers[0].task_step.item()}")

    model.consolidate()
    protected = [int((t.slow_heat > 0.0).sum().item()) for t in trackers]
    cap = int((1.0 - args.budget) * trackers[0].units)
    print(f"protected per layer: min={min(protected)} max={max(protected)} cap={cap}")
    assert max(protected) <= cap

    # Task 2: hard-protected rows must not move at all.
    gate = model.model.layers[0].mlp.gate_proj.weight
    down = model.model.layers[0].mlp.down_proj.weight
    heat = trackers[0].slow_heat
    frozen = heat > 0.0
    gate_before = gate.detach().clone()
    down_before = down.detach().clone()

    optimizer.zero_grad()
    output = model(**batch, labels=torch.tensor([2, 3]))
    output.loss.backward()
    optimizer.step()
    print(f"task 2 loss:         {output.loss.item():.4f}")

    gate_drift = (gate[frozen] - gate_before[frozen]).abs().max().item()
    down_drift = (down[:, frozen] - down_before[:, frozen]).abs().max().item()
    plastic_drift = (gate[~frozen] - gate_before[~frozen]).abs().max().item()
    print(f"protected gate drift: {gate_drift:.3e} (must be 0)")
    print(f"protected down drift: {down_drift:.3e} (must be 0)")
    print(f"plastic gate drift:   {plastic_drift:.3e} (must be > 0)")
    assert gate_drift == 0.0 and down_drift == 0.0
    assert plastic_drift > 0.0

    model.consolidate()
    assert all(t.consolidated_tasks.item() == 2 for t in trackers)
    print("two-task sequence completed: ok")


if __name__ == "__main__":
    main()
