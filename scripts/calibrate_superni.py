#!/usr/bin/env python3
"""Calibração do SuperNI em Qwen2.5-0.5B — Tarefa 3b.2 do roadmap.

**Isto NÃO é um experimento confirmatório.** Não há pré-registro, não há seeds
confirmatórias, não há endpoint. Mede três coisas, nessa ordem de importância:

1. **Dispersão.** O ROUGE-L separa os braços, ou todos colapsam no mesmo valor?
   Esta é a pergunta que decide se o SuperNI é usável. A Fase 3.5 morreu com
   `task1_retention` = 0,0000 nos quatro braços: métrica calculada certa, regime
   sem dispersão, e nenhum gate vigiava isso. Aqui a checagem vem antes de
   qualquer leitura de efeito.
2. **Memória.** Cabe em 11 GB com o comprimento de sequência real?
3. **Custo.** Quanto tempo por tarefa, para dimensionar (ou descartar) a fase.

Roda UM braço (vanilla) sobre uma sequência curta. Comparar braços exigiria
pré-registro — e o ponto aqui é descobrir se vale a pena escrevê-lo.

Uso:
    PYTHONPATH=. .venv/bin/python scripts/calibrate_superni.py --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import time
from pathlib import Path
from typing import Any

import torch

from dual_heater.superni_data import SuperNITask, group_examples_by_task
from dual_heater.superni_metrics import exact_match, rouge_l

#: Seed de calibração. FORA de qualquer banda confirmatória, pelo mesmo motivo
#: que a seed 11_900_003 ficou fora na Fase 3: uma seed gasta em calibração não
#: pode reaparecer num experimento, ou o resultado herda o viés de ter sido
#: olhado antes.
CALIBRATION_SEED = 7_900_019

#: Tarefas escolhidas por COBERTURA DE FORMATO, não por desempenho esperado:
#: duas de classificação (saída curta, Exact Match no paper, piso de chance
#: definível) e duas de geração (saída livre, onde o ROUGE-L é a única opção).
#: Se a dispersão morrer, importa saber em qual dos dois regimes ela morreu.
#:
#: Os nomes vêm do split `test/` do repositório, que é o track inglês oficial
#: (119 tarefas). **Verificados contra a listagem do repo**, não copiados do
#: paper: a primeira tentativa usou quatro nomes plausíveis e nenhum existia.
DEFAULT_TASKS = (
    "task1344_glue_entailment_classification",  # classificação
    "task1529_scitail1.1_classification",  # classificação
    "task039_qasc_find_overlapping_words",  # geração curta
    "task102_commongen_sentence_generation",  # geração livre
)


def load_tasks(
    task_names: tuple[str, ...],
    *,
    train_per_task: int,
    eval_per_task: int,
    max_prompt_characters: int,
) -> list[SuperNITask]:
    """Baixa só os arquivos das tarefas pedidas, do split `test/` oficial.

    Streaming sequencial do dataset inteiro **não serve**: 400 mil linhas
    cobrem apenas 47 das 1.616 tarefas, então uma tarefa específica pode nunca
    aparecer. O repositório expõe um JSONL por tarefa, e baixar quatro arquivos
    é mais barato e determinístico que varrer 5M de instâncias.
    """

    from huggingface_hub import hf_hub_download

    flat: list[dict[str, Any]] = []
    needed = train_per_task + eval_per_task
    for name in task_names:
        path = hf_hub_download(
            repo_id="Muennighoff/natural-instructions",
            filename=f"test/{name}_test.jsonl",
            repo_type="dataset",
        )
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if len(flat) and sum(1 for r in flat if r["task_name"] == name) >= needed:
                    break
                row = json.loads(line)
                row.setdefault("task_name", name)
                flat.append(row)

    return group_examples_by_task(
        flat,
        task_names=list(task_names),
        train_per_task=train_per_task,
        eval_per_task=eval_per_task,
        max_prompt_characters=max_prompt_characters,
    )


@torch.no_grad()
def generate_and_score(
    model,
    tokenizer,
    task: SuperNITask,
    *,
    device: str,
    max_prompt_tokens: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Gera e pontua, guardando os valores POR EXEMPLO.

    Os valores individuais são o ponto: a média sozinha não distingue "todos os
    exemplos em 0,3" de "metade em 0 e metade em 0,6", e essa diferença é
    exatamente a dispersão que decide a viabilidade.
    """

    model.eval()
    rouge_scores: list[float] = []
    exact_scores: list[float] = []
    samples: list[dict[str, str]] = []
    empty_generations = 0

    for example in task.evaluation:
        encoded = tokenizer(
            example.prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_prompt_tokens,
        ).to(device)
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        completion = tokenizer.decode(
            generated[0][encoded["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()
        if not completion:
            empty_generations += 1
        rouge_scores.append(rouge_l(completion, example.target))
        exact_scores.append(exact_match(completion, example.target))
        if len(samples) < 3:
            samples.append(
                {
                    "prompt_tail": example.prompt[-180:],
                    "generated": completion[:180],
                    "target": example.target[:180],
                }
            )

    return {
        "task": task.name,
        "n_eval": len(rouge_scores),
        "rouge_l_mean": st.mean(rouge_scores) if rouge_scores else 0.0,
        "rouge_l_stdev": st.stdev(rouge_scores) if len(rouge_scores) > 1 else 0.0,
        "rouge_l_min": min(rouge_scores) if rouge_scores else 0.0,
        "rouge_l_max": max(rouge_scores) if rouge_scores else 0.0,
        "rouge_l_zeros": sum(1 for score in rouge_scores if score == 0.0),
        "exact_match_mean": st.mean(exact_scores) if exact_scores else 0.0,
        "empty_generations": empty_generations,
        "per_example_rouge": rouge_scores,
        "samples": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--train-per-task", type=int, default=32)
    parser.add_argument("--eval-per-task", type=int, default=24)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-prompt-characters", type=int, default=6000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--output", type=Path, default=Path("results/superni_calibration"))
    args = parser.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(CALIBRATION_SEED)
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    print(f"carregando tarefas: {args.tasks}", flush=True)
    tasks = load_tasks(
        tuple(args.tasks),
        train_per_task=args.train_per_task,
        eval_per_task=args.eval_per_task,
        max_prompt_characters=args.max_prompt_characters,
    )
    for task in tasks:
        print(
            f"  {task.name}: {len(task.train)} treino, {len(task.evaluation)} eval",
            flush=True,
        )
    missing = [task.name for task in tasks if not task.train or not task.evaluation]
    if missing:
        print(f"ABORTADO: tarefas sem exemplos suficientes: {missing}")
        return 1
    load_seconds = time.perf_counter() - started

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # fp16: Pascal (CC 6.x) não tem bf16 nativo.
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16)
    lora = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora).to(args.device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"treináveis: {trainable:,}", flush=True)

    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.learning_rate
    )

    # Avaliação ANTES de treinar: é o piso do modelo base. Sem ela não dá para
    # dizer se o ROUGE-L observado depois veio do treino ou já estava lá.
    print("\n=== zero-shot (antes de treinar) ===", flush=True)
    zero_shot = [
        generate_and_score(
            model,
            tokenizer,
            task,
            device=args.device,
            max_prompt_tokens=args.max_prompt_tokens,
            max_new_tokens=args.max_new_tokens,
        )
        for task in tasks
    ]
    for row in zero_shot:
        print(
            f"  {row['task'][:46]:46s} ROUGE-L {row['rouge_l_mean']:.4f} "
            f"±{row['rouge_l_stdev']:.4f} [{row['rouge_l_min']:.2f},{row['rouge_l_max']:.2f}] "
            f"zeros {row['rouge_l_zeros']}/{row['n_eval']} vazias {row['empty_generations']}",
            flush=True,
        )

    print("\n=== treino sequencial ===", flush=True)
    train_seconds: list[float] = []
    for index, task in enumerate(tasks, start=1):
        task_started = time.perf_counter()
        model.train()
        for _ in range(args.epochs):
            for example in task.train:
                text = f"{example.prompt} {example.target}{tokenizer.eos_token}"
                encoded = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=args.max_prompt_tokens + args.max_new_tokens,
                ).to(args.device)
                outputs = model(**encoded, labels=encoded["input_ids"])
                outputs.loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        elapsed = time.perf_counter() - task_started
        train_seconds.append(elapsed)
        print(f"  [{index}/{len(tasks)}] {task.name[:46]:46s} {elapsed:6.1f} s", flush=True)

    print("\n=== após o treino sequencial ===", flush=True)
    after = [
        generate_and_score(
            model,
            tokenizer,
            task,
            device=args.device,
            max_prompt_tokens=args.max_prompt_tokens,
            max_new_tokens=args.max_new_tokens,
        )
        for task in tasks
    ]
    for row in after:
        print(
            f"  {row['task'][:46]:46s} ROUGE-L {row['rouge_l_mean']:.4f} "
            f"±{row['rouge_l_stdev']:.4f} [{row['rouge_l_min']:.2f},{row['rouge_l_max']:.2f}] "
            f"zeros {row['rouge_l_zeros']}/{row['n_eval']} vazias {row['empty_generations']}",
            flush=True,
        )

    peak_mib = (
        torch.cuda.max_memory_allocated(args.device) / 1024**2
        if args.device.startswith("cuda")
        else 0.0
    )
    total_minutes = (time.perf_counter() - started) / 60

    # O veredito de dispersão, declarado ANTES de olhar os números: se todos os
    # ROUGE-L por exemplo forem iguais dentro de uma tarefa, nenhum contraste
    # entre braços pode existir ali. É o teste que faltou na Fase 3.5.
    dispersion = {
        row["task"]: {
            "stdev": row["rouge_l_stdev"],
            "distinct_values": len(set(row["per_example_rouge"])),
            "all_zero": row["rouge_l_zeros"] == row["n_eval"],
        }
        for row in after
    }
    degenerate = [name for name, info in dispersion.items() if info["distinct_values"] <= 1]

    report = {
        "kind": "calibration_not_confirmatory",
        "seed": CALIBRATION_SEED,
        "model": args.model,
        "device": args.device,
        "trainable_parameters": trainable,
        "config": vars(args) | {"output": str(args.output)},
        "zero_shot": zero_shot,
        "after_sequential": after,
        "dispersion": dispersion,
        "degenerate_tasks": degenerate,
        "peak_memory_mib": peak_mib,
        "load_seconds": load_seconds,
        "train_seconds_per_task": train_seconds,
        "total_minutes": total_minutes,
    }
    destination = args.output / "calibration.json"
    destination.write_text(json.dumps(report, indent=2))

    print(f"\npico de memória : {peak_mib:,.0f} MiB")
    print(f"tempo total     : {total_minutes:.1f} min")
    print(f"tarefas degeneradas (ROUGE-L sem variação): {degenerate or 'nenhuma'}")
    print(f"relatório       : {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
