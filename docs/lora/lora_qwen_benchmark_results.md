# Aplicação 4 — Benchmark de LoRA em Qwen2.5-0.5B

**Fonte:** `experiments/qwen_lora_sweep.py`, `experiments/qwen_lora_slowheat.py`
**Resultados:** `results/qwen_lora_sweep_10seed/sweep.json`
**Estado:** run principal concluída. **Confundida por capacidade** — ver §5.

## 1. Protocolo

| item | valor |
|---|---|
| modelo | Qwen2.5-0.5B (`Qwen2ForSequenceClassification`) |
| adaptador | LoRA `r=16`, α=32, alvos `gate_proj`/`up_proj`/`down_proj` |
| benchmark | Split-CLINC150 class-incremental, 10 domínios oficiais |
| seeds | 10 (11, 22, 33, 44, 55, 66, 77, 88, 99, 110) |
| dados | 750 treino / 300 validação por tarefa |
| épocas | 3 por tarefa |
| precisão | fp32 (Pascal não tem bf16 nativo) |
| hardware | 2× GTX 1080 Ti + 1× Titan Xp, uma seed inteira por GPU |

Seeds exploratórias declaradas (múltiplos de 11). As seeds confirmatórias
(104729+) permanecem **reservadas** e não foram tocadas.

Todos os braços compartilham stream de dados, tokenização, otimizador, learning
rate e código de avaliação. **Tokens idênticos entre braços (205.570 ± 568)**
confirmam o pareamento por seed — a contagem de tokens aqui é variável de
controle, não de resultado.

## 2. Resultados agregados

| arm | FAA (média±sd) | forgetting | BWT | min/seed | pico MiB | E_eff |
|---|---|---|---|---|---|---|
| vanilla | 0,6016±0,0297 | +0,3967±0,0323 | −0,3967 | 10,6 | 2.759,8 | 1,000 |
| **exact** | **0,6365±0,0349** | **+0,3338±0,0341** | −0,3338 | 17,7 | 2.827,7 | 0,923 |
| slice | 0,6203±0,0360 | +0,3725±0,0399 | −0,3725 | 13,1 | 2.759,7 | 0,062 |
| rank | 0,5983±0,0447 | +0,4012±0,0445 | −0,4012 | 19,3 | 2.760,8 | 0,645 |
| leak | 0,5958±0,0404 | +0,4001±0,0439 | −0,4001 | 19,2 | 3.008,1 | 0,659 |

Custo total: **5h08min de wall-clock**, 13,3 GPU-horas, 50 runs, todas com
sucesso.

## 3. Contrastes pareados vs vanilla

Teste de sinal exato bicaudal sobre as 10 diferenças por seed:

| arm | métrica | dif. média | sd | p | vitórias |
|---|---|---|---|---|---|
| **exact** | **forgetting** | **−0,0629** | 0,0366 | **0,0020** | **10/10** |
| exact | FAA | +0,0349 | 0,0326 | 0,1094 | 8/10 |
| slice | FAA | +0,0188 | 0,0356 | 0,3438 | 7/10 |
| slice | forgetting | −0,0243 | 0,0397 | 0,3438 | 7/10 |
| leak | FAA | −0,0058 | 0,0314 | 0,3438 | 3/10 |
| leak | forgetting | +0,0033 | 0,0327 | 0,7539 | 6/10 |
| rank | FAA | −0,0032 | 0,0236 | 1,0000 | 5/10 |
| rank | forgetting | +0,0044 | 0,0239 | 1,0000 | 5/10 |

**Correção de Holm sobre as 8 comparações (α=0,05):** apenas
`exact / forgetting` sobrevive (p=0,0020 < 0,00625). A segunda menor p já falha
(0,1094 > 0,00714), e Holm para aí.

## 4. Leitura honesta

**Os três mecanismos novos falharam.** `rank` é indistinguível do vanilla — 5
vitórias em 10, p=1,0 nas duas métricas, ao custo de 1,82× o tempo. `leak` teve
o pior FAA médio e o maior pico de memória. `slice` tem o sinal correto e 7/10
vitórias, mas não chega perto da significância.

**O único braço que funciona é o que já existia.** `exact` é a Solução A de
`build_exact_slowheat_lora`, usada como referência do experimento, não como
contribuição.

## 5. A confusão que invalida a comparação causal

`E_eff` varia de 0,062 (`slice`) a 0,923 (`exact`) entre os braços, e **a
ordenação do FAA segue `E_eff` quase monotonicamente**:

```text
exact  E=0,923  FAA=0,637   <- mais plasticidade retida, melhor FAA
slice  E=0,062  FAA=0,620
rank   E=0,645  FAA=0,598
leak   E=0,659  FAA=0,596
```

Este sweep mede principalmente **quanta** plasticidade cada braço removeu, não
se o mecanismo de **seleção** funciona. É exatamente a confusão entre quantidade
e distribuição que o protocolo iso-plasticidade existe para evitar (F5 em
`goals/opcoes_novidade_e_proximos_passos.md`).

Consequência direta: **nem o resultado do `exact` é atribuível ao mecanismo**.
Ele retém mais plasticidade que os outros e, simultaneamente, paga custo de
aquisição (acurácia 10 pontos menor ao fim da primeira tarefa, por ter metade
dos parâmetros treináveis). A leitura alternativa não descartada é
*ele esquece menos porque aprendeu menos*.

## 6. Run iso-plasticidade (corretiva)

Em execução na data desta revisão, saída em `results/qwen_lora_iso_10seed/`.
Duas mudanças:

1. **Pareamento em E = 0,85.** Todos os braços mascarados resolvem seu próprio
   botão por bisseção em cada fronteira, até a plasticidade **medida** igualar
   0,85. Verificado no probe: `exact`, `rank`, `leak` e `slice` atingiram
   0,8500 exato nos três estágios.
2. **Braço `lr_control`.** Remove a mesma plasticidade via `lr · 0,85`, sem
   máscara. Qualquer mecanismo que não o supere não está fazendo nada que um
   escalar não faça.

Essa é a run que permite (ou não) uma afirmação causal. A atual não permite.

## 7. Custo computacional

| arm | min/seed | vs vanilla | pico MiB |
|---|---|---|---|
| vanilla | 10,6 ± 2,4 | 1,00× | 2.759,8 |
| slice | 13,1 ± 2,5 | 1,24× | 2.759,7 |
| exact | 17,7 ± 4,1 | 1,67× | 2.827,7 |
| leak | 19,2 ± 3,7 | 1,81× | 3.008,1 |
| rank | 19,3 ± 3,4 | 1,82× | 2.760,8 |

O sobrecusto é dominado pelo passo pontual mascarado do `SlowHeatAdamW`, que
percorre parâmetro a parâmetro em Python sem `foreach` — requisito pendente
listado na §13 de `functional_slowheat_transformers.md`. `slice` é o mais barato
porque sua máscara é um vetor binário de `[r]` sem tracker nenhum; `rank` e
`leak` pagam dois bindings por módulo.

## 8. Pendência que bloqueia publicação

O-LoRA, InfLoRA e a família de LoRA para continual learning **não foram
levantados**. Vários desses trabalhos particionam ou ortogonalizam subespaço por
tarefa, o que toca `slice` diretamente e `rank` em parte. Nenhuma reivindicação
de novidade pode ser feita antes dessa checagem.

## Referências

- [Índice das aplicações de LoRA](lora_applications.md)
- [Os três mecanismos](lora_slowheat_mechanisms.md)
- [Aplicação 2, o braço que funcionou](lora_exact_producer_only.md)
- [Protocolo confirmatório](../protocols/confirmatory_protocol.md)
