# Aplicação 4 — Benchmark de LoRA em Qwen2.5-0.5B

**Fonte:** `experiments/qwen_lora_sweep.py`, `experiments/qwen_lora_slowheat.py`
**Resultados:** `results/qwen_lora_iso_10seed/` (principal, iso-plasticidade) e
`results/qwen_lora_sweep_10seed/` (primeira run, confundida por capacidade)
**Estado:** duas runs de 10 seeds concluídas. A segunda pareia os braços em
plasticidade e inclui o controle de learning rate.

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
controle, não de resultado. O agregador pareia **por seed, nunca por posição na
lista**, propriedade coberta por `tests/test_lora_sweep_aggregate.py`.

## 2. Resultados — run iso-plasticidade (principal)

Todos os braços mascarados em `E = 0,85`, verificado exato nas 10 seeds.
`lr_control` remove a mesma plasticidade via `lr × 0,85`, sem máscara.

| arm | FAA (média±sd) | forgetting | min/seed | pico MiB | E_eff |
|---|---|---|---|---|---|
| **exact** | **0,6494±0,0297** | **+0,3160±0,0281** | 10,4 | 2.827,3 | 0,850 |
| vanilla | 0,6016±0,0297 | +0,3967±0,0323 | 5,9 | 2.759,8 | 1,000 |
| slice | 0,5875±0,0422 | +0,4126±0,0426 | 8,9 | 2.759,7 | 0,850 |
| leak | 0,5865±0,0304 | +0,4136±0,0319 | 12,6 | 3.007,0 | 0,850 |
| rank | 0,5808±0,0580 | +0,4191±0,0612 | 12,2 | 2.760,8 | 0,850 |
| lr_control | 0,5684±0,0542 | +0,4350±0,0602 | 5,9 | 2.759,7 | 1,000 |

Custo: 3h48min de wall-clock, 60 runs, todas com sucesso.

## 3. O contraste que isola o mecanismo

`mecanismo − lr_control`: mesma plasticidade removida, a única diferença é
**seletiva (máscara) contra uniforme (learning rate)**.

| arm | métrica | dif. média | p | a favor |
|---|---|---|---|---|
| **exact** | **forgetting** | **−0,1190** | **0,0020** | **10/10** |
| exact | FAA | +0,0810 | 0,0215 | 9/10 |
| slice | FAA | +0,0191 | 0,7539 | 6/10 |
| leak | FAA | +0,0181 | 0,7539 | 6/10 |
| rank | FAA | +0,0124 | 0,1094 | 8/10 |

Sob Holm (8 comparações), apenas `exact / forgetting` sobrevive
(0,0020 < 0,00625).

**Esse é o resultado central do documento.** Ele diz que a proteção seletiva do
braço `exact` faz algo que um escalar no learning rate não faz — pergunta que a
primeira run não conseguia responder.

## 4. O que o pareamento mudou

Na primeira run os braços tinham `E_eff` entre 0,062 e 0,923, e a ordenação do
FAA seguia `E_eff` quase monotonicamente: o sweep media **quanta** plasticidade
cada braço removeu, não se a **seleção** funcionava. É a confusão entre
quantidade e distribuição que o protocolo iso-plasticidade existe para evitar
(F5 em `goals/opcoes_novidade_e_proximos_passos.md`).

Com todos em 0,85 a ordenação deixa de seguir a plasticidade, porque ela é
constante. Efeitos:

| contraste | run 1 (sem pareamento) | run 2 (E=0,85) |
|---|---|---|
| exact FAA vs vanilla | +0,0349 (p=0,11) | **+0,0478 (p=0,02)** |
| exact forgetting vs vanilla | −0,0629 (p=0,002) | **−0,0807 (p=0,002)** |
| exact forgetting vs lr_control | — | **−0,1190 (p=0,002)** |

O `exact` ficou **mais** forte sob o controle mais rigoroso, não mais fraco.

## 5. Os três mecanismos novos falharam, agora sem defesa disponível

`rank`, `leak` e `slice` batem o `lr_control` por +0,012 a +0,019 de FAA,
nenhuma diferença significativa (p ≥ 0,11), e **todos os três ficam abaixo do
vanilla**.

O `slice` tinha a defesa de que `E_eff = 0,062` o penalizava por remoção de
capacidade na primeira run. Com `E = 0,85` ele continua perdendo (0,5875 contra
0,6016 do vanilla). **Essa defesa caiu.**

O único braço que funciona é o que já existia: `exact` é a Solução A de
`build_exact_slowheat_lora`, usada como referência do experimento, não como
contribuição.

## 6. O `lr_control` é pior que o vanilla

> **Correção de 29/09/2026 — o título desta seção não replicou.** Ela descreve a
> run exploratória. Nas 10 seeds confirmatórias (banda 700001+) o sinal inverte
> nos dois endpoints e nenhum é significativo: FAA `lr_control − vanilla` =
> **+0,0084** (6+/4−, p = 0,754); forgetting **−0,0096** (3+/7−, p = 0,344).
> **O resultado correto é nulo:** reduzir o lr uniformemente em 15% não muda
> nada detectável em n=10. O conteúdo abaixo é mantido como registro
> exploratório e **não deve ser citado como achado**.

FAA 0,5684 contra 0,6016; forgetting +0,4350 contra +0,3967, o pior de todos os
braços. Reduzir o learning rate uniformemente em 15% não ajuda em continual
learning — atrapalha.

Isso valida o falsificador como teste não-trivial: ele não é um alvo fácil
posicionado abaixo de todo mundo, é um braço que **piora** o resultado, e ainda
assim três dos quatro mecanismos mal o superam.

## 7. Custo computacional

| arm | min/seed | vs vanilla | pico MiB |
|---|---|---|---|
| vanilla | 5,9 | 1,00× | 2.759,8 |
| lr_control | 5,9 | 1,00× | 2.759,7 |
| slice | 8,9 | 1,51× | 2.759,7 |
| exact | 10,4 | 1,76× | 2.827,3 |
| rank | 12,2 | 2,06× | 2.760,8 |
| leak | 12,6 | 2,13× | 3.007,0 |

O sobrecusto é dominado pelo passo pontual mascarado do `SlowHeatAdamW`, que
percorre parâmetro a parâmetro em Python sem `foreach` — requisito pendente
listado na §13 de
[functional_slowheat_transformers.md](../mechanisms/functional_slowheat_transformers.md).
`slice` é o mais barato dos mascarados porque sua máscara é um vetor binário de
`[r]` sem tracker; `rank` e `leak` pagam dois bindings por módulo.

## 8. O que estes números ainda não sustentam

- **Exploratório, não confirmatório.** Seeds múltiplas de 11; as confirmatórias
  seguem reservadas. Um resultado exploratório com p=0,002 justifica
  pré-registro, não publicação.
- **Um único alvo.** 10 domínios do CLINC150, um modelo, um tamanho de rank. A
  generalização para outros `r`, outros hosts e outros benchmarks é desconhecida.
- **Um único ponto de plasticidade.** `E = 0,85` foi escolhido por ser
  alcançável por todos os braços. O comportamento em E baixo (onde `slice` vivia
  na run 1) não foi medido sob pareamento.
- **O custo de aquisição do `exact` permanece.** Ele treina metade dos
  parâmetros do adaptador e chega mais baixo ao fim da primeira tarefa. Parte do
  seu forgetting menor ainda pode ser "aprendeu menos", e o pareamento em E não
  corrige isso — E mede plasticidade retida da máscara, não capacidade do
  adaptador.
- **Prioridade não verificada.** O-LoRA, InfLoRA e a família de LoRA para
  continual learning não foram levantados. Vários desses trabalhos particionam
  ou ortogonalizam subespaço por tarefa, o que toca `slice` diretamente e `rank`
  em parte.

## Referências

- [Índice das aplicações de LoRA](lora_applications.md)
- [Os três mecanismos](lora_slowheat_mechanisms.md)
- [Aplicação 2, o braço que funcionou](lora_exact_producer_only.md)
- [Protocolo confirmatório](../protocols/confirmatory_protocol.md)
