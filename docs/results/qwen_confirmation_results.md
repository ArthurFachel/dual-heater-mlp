# Qwen2 — confirmação de 10 seeds (120 passos)

Execução completa das 10 seeds pré-registradas (10–19) do protocolo
`goals/protocol_iso_plasticity.md`. Este documento traz os resultados
**completos**, incluindo os valores por seed.

- Artefatos: `results/qwen_iso_plasticity/confirm120_seed{10..19}/manifest.json`
- Agregador: `experiments/analyze_confirmation.py` (commit `6f4d12d`)
- Tarefas: 10 domínios do CLINC150, 15 intents cada (**piso do acaso ≈ 6,7%**)

## 1. Integridade

| Campo | Valor |
|---|---|
| Seeds | 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 |
| `protocol_hash` único | sim — `2cd785d8635661708b574249985da49e93e34088f10b8afbdd240b6045eb30ea` |
| `task_fingerprint` único | sim |
| Passos por tarefa | 120 |
| Tarefas | 10 |
| Precisão | fp32 |
| Escopo de capacidade | local |

As 10 seeds vêm de um único protocolo declarado. Sem isso, nada abaixo é
agregável.

## 2. Endpoints médios por braço

### E* = 0,75

| Braço | FAA | Retenção t0 | Forgetting | Aquisição t_últ |
|---|---:|---:|---:|---:|
| iso_b0.05 | 0,4271 ± 0,0498 | 0,0417 ± 0,0346 | 0,4989 ± 0,0534 | 0,8307 ± 0,0313 |
| iso_b0.1 | 0,4342 ± 0,0516 | 0,0393 ± 0,0303 | 0,4919 ± 0,0522 | 0,8370 ± 0,0327 |
| iso_b0.25 | 0,4330 ± 0,0465 | 0,0457 ± 0,0389 | 0,4920 ± 0,0485 | 0,8270 ± 0,0374 |
| iso_b0.5 | 0,4205 ± 0,0403 | 0,0513 ± 0,0462 | 0,5068 ± 0,0431 | 0,8340 ± 0,0262 |
| permuted_b0.25 | 0,4062 ± 0,0364 | 0,0337 ± 0,0353 | 0,5207 ± 0,0405 | 0,8230 ± 0,0344 |
| **hard_b0.75** | **0,4584 ± 0,0403** | **0,0953 ± 0,0487** | **0,4612 ± 0,0408** | 0,8210 ± 0,0336 |
| reduced_lr | 0,4243 ± 0,0438 | 0,0323 ± 0,0362 | 0,4944 ± 0,0563 | 0,8150 ± 0,0446 |
| vanilla | 0,4074 ± 0,0259 | 0,0203 ± 0,0252 | 0,5226 ± 0,0287 | 0,8197 ± 0,0304 |

### E* = 0,50

| Braço | FAA | Retenção t0 | Forgetting | Aquisição t_últ |
|---|---:|---:|---:|---:|
| iso_b0.05 | 0,3770 ± 0,0479 | 0,0340 ± 0,0226 | 0,5442 ± 0,0471 | 0,8093 ± 0,0339 |
| iso_b0.1 | 0,3710 ± 0,0501 | 0,0357 ± 0,0245 | 0,5517 ± 0,0509 | 0,8203 ± 0,0407 |
| iso_b0.25 | 0,3889 ± 0,0474 | 0,0487 ± 0,0292 | 0,5311 ± 0,0494 | 0,8183 ± 0,0290 |
| iso_b0.5 | — descartado em todas as seeds — | | | |
| permuted_b0.25 | 0,3813 ± 0,0294 | 0,0310 ± 0,0223 | 0,5445 ± 0,0274 | 0,8190 ± 0,0317 |
| **hard_b0.5** | **0,4474 ± 0,0393** | **0,1140 ± 0,0508** | **0,4650 ± 0,0357** | 0,8143 ± 0,0342 |
| reduced_lr | 0,3944 ± 0,0405 | 0,0277 ± 0,0255 | 0,5104 ± 0,0391 | 0,8083 ± 0,0314 |
| vanilla | 0,4074 ± 0,0259 | 0,0203 ± 0,0252 | 0,5226 ± 0,0287 | 0,8197 ± 0,0304 |

**Descarte de `iso_b0.5`**: razão registrada no manifesto —
*"budget 0.5 não alcança E\*=0.5 na fronteira 0→1"*. É uma condição declarada
do protocolo, não falha de execução: o braço não consegue atingir a
plasticidade-alvo com aquele orçamento. Deve ser reportado como restrição do
desenho.

## 3. Valores por seed

### E* = 0,75 — FAA (%)

| Seed | vanilla | permutado | iso_b0.25 | hard_b0.75 |
|---:|---:|---:|---:|---:|
| 10 | 42,53 | 40,00 | 49,50 | 48,10 |
| 11 | 40,83 | 42,80 | 44,20 | 47,63 |
| 12 | 44,07 | 42,97 | 46,17 | 42,47 |
| 13 | 39,93 | 42,33 | 43,67 | 46,30 |
| 14 | 45,70 | 46,67 | 49,47 | 53,13 |
| 15 | 38,60 | 40,40 | 38,77 | 41,73 |
| 16 | 38,50 | 38,30 | 45,93 | 48,23 |
| 17 | 39,87 | 35,93 | 39,57 | 46,90 |
| 18 | 39,63 | 42,37 | 39,97 | 45,07 |
| 19 | 37,73 | 34,40 | 35,80 | 38,87 |

### E* = 0,75 — retenção da primeira tarefa (%)

| Seed | vanilla | permutado | iso_b0.25 | hard_b0.75 |
|---:|---:|---:|---:|---:|
| 10 | 0,33 | 1,00 | 1,00 | 3,00 |
| 11 | 0,00 | 2,67 | 2,67 | 15,00 |
| 12 | 0,00 | 0,00 | 7,67 | 7,67 |
| 13 | 0,00 | 4,33 | 3,67 | 7,33 |
| 14 | 8,00 | 12,33 | 13,33 | 17,33 |
| 15 | 3,67 | 4,33 | 6,33 | 9,67 |
| 16 | 3,33 | 3,33 | 5,33 | 8,00 |
| 17 | 2,67 | 0,33 | 1,33 | 8,33 |
| 18 | 1,33 | 3,67 | 4,00 | 15,33 |
| 19 | 1,00 | 1,67 | 0,33 | 3,67 |

### E* = 0,50 — FAA (%)

| Seed | vanilla | permutado | iso_b0.25 | hard_b0.5 |
|---:|---:|---:|---:|---:|
| 10 | 42,53 | 42,83 | 39,60 | 46,07 |
| 11 | 40,83 | 37,93 | 43,70 | 49,00 |
| 12 | 44,07 | 37,13 | 43,67 | 40,33 |
| 13 | 39,93 | 34,63 | 33,80 | 42,47 |
| 14 | 45,70 | 37,80 | 42,30 | 44,67 |
| 15 | 38,60 | 41,33 | 42,40 | 50,53 |
| 16 | 38,50 | 40,17 | 42,77 | 48,73 |
| 17 | 39,87 | 38,67 | 32,57 | 44,70 |
| 18 | 39,63 | 37,87 | 35,47 | 42,53 |
| 19 | 37,73 | 32,90 | 32,60 | 38,40 |

### E* = 0,50 — retenção da primeira tarefa (%)

| Seed | vanilla | permutado | iso_b0.25 | hard_b0.5 |
|---:|---:|---:|---:|---:|
| 10 | 0,33 | 0,67 | 0,67 | 6,67 |
| 11 | 0,00 | 3,00 | 7,00 | 18,00 |
| 12 | 0,00 | 0,00 | 5,33 | 5,33 |
| 13 | 0,00 | 3,67 | 4,33 | 7,33 |
| 14 | 8,00 | 6,33 | 10,00 | 15,00 |
| 15 | 3,67 | 5,33 | 6,33 | 16,00 |
| 16 | 3,33 | 3,00 | 2,67 | 8,33 |
| 17 | 2,67 | 1,33 | 2,00 | 10,00 |
| 18 | 1,33 | 6,00 | 7,67 | 19,00 |
| 19 | 1,00 | 1,67 | 2,67 | 8,33 |

O `hard` vence o `vanilla` em retenção nas **10 de 10 seeds** nas duas famílias
— a única regularidade perfeita do conjunto.

## 4. Diferenças pareadas (sign-flip exato, n=10)

Holm sobre as 24 comparações da família. `*` = p_holm < 0,05.

| E* | Comparação | Endpoint | Diferença | Seeds | p | p_holm |
|---|---|---|---:|---:|---:|---:|
| 0,75 | iso − permutado | FAA | +0,0269 ± 0,0368 | 8/10 | 0,0371 | 0,3164 |
| 0,75 | iso − permutado | retenção t0 | +0,0120 ± 0,0251 | 6/10 | 0,1250 | 0,6250 |
| 0,75 | iso − permutado | forgetting | −0,0287 ± 0,0373 | 2/10 | 0,0254 | 0,2578 |
| 0,75 | hard − iso | FAA | +0,0254 ± 0,0310 | 8/10 | 0,0352 | 0,3164 |
| 0,75 | hard − iso | retenção t0 | +0,0497 ± 0,0402 | 9/10 | 0,0039 | 0,0859 |
| 0,75 | hard − iso | forgetting | −0,0308 ± 0,0422 | 2/10 | 0,0566 | 0,3398 |
| 0,75 | hard − vanilla | FAA | +0,0510 ± 0,0333 | 9/10 | 0,0059 | 0,1113 |
| **0,75** | **hard − vanilla** | **retenção t0** | **+0,0750 ± 0,0425** | **10/10** | **0,0020** | **0,0469 \*** |
| 0,75 | hard − vanilla | forgetting | −0,0614 ± 0,0369 | 1/10 | 0,0039 | 0,0859 |
| 0,75 | iso − vanilla | FAA | +0,0256 ± 0,0309 | 8/10 | 0,0234 | 0,2578 |
| 0,75 | iso − vanilla | retenção t0 | +0,0253 ± 0,0268 | 8/10 | 0,0156 | 0,1914 |
| 0,75 | iso − vanilla | forgetting | −0,0307 ± 0,0318 | 2/10 | 0,0137 | 0,1914 |
| 0,50 | iso − permutado | FAA | +0,0076 ± 0,0412 | 5/10 | 0,5605 | 1,0000 |
| 0,50 | iso − permutado | retenção t0 | +0,0177 ± 0,0190 | 8/10 | 0,0078 | 0,1250 |
| 0,50 | iso − permutado | forgetting | −0,0134 ± 0,0457 | 4/10 | 0,3750 | 1,0000 |
| 0,50 | hard − iso | FAA | +0,0586 ± 0,0410 | 9/10 | 0,0059 | 0,1113 |
| 0,50 | hard − iso | retenção t0 | +0,0653 ± 0,0356 | 9/10 | 0,0039 | 0,0859 |
| 0,50 | hard − iso | forgetting | −0,0662 ± 0,0494 | 1/10 | 0,0059 | 0,1113 |
| 0,50 | hard − vanilla | FAA | +0,0400 ± 0,0494 | 8/10 | 0,0352 | 0,3164 |
| **0,50** | **hard − vanilla** | **retenção t0** | **+0,0937 ± 0,0488** | **10/10** | **0,0020** | **0,0469 \*** |
| 0,50 | hard − vanilla | forgetting | −0,0577 ± 0,0536 | 2/10 | 0,0078 | 0,1250 |
| 0,50 | iso − vanilla | FAA | −0,0185 ± 0,0424 | 3/10 | 0,1973 | 0,7891 |
| 0,50 | iso − vanilla | retenção t0 | +0,0283 ± 0,0281 | 8/10 | 0,0137 | 0,1914 |
| 0,50 | iso − vanilla | forgetting | +0,0085 ± 0,0472 | 6/10 | 0,5742 | 1,0000 |

**Sobrevivem a Holm: 2 de 24.** Ambos são retenção da primeira tarefa contra
vanilla, ambos com 10/10 seeds.

## 5. Curva por tarefa

Acurácia final por tarefa (%), média de 10 seeds:

### E* = 0,75

| | t0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vanilla | 2,0 | 9,9 | 20,4 | 11,6 | 45,0 | 46,3 | 51,5 | 74,1 | 64,7 | 82,0 |
| permutado | 3,4 | 12,1 | 20,0 | 10,0 | 44,9 | 43,6 | 50,7 | 75,6 | 63,6 | 82,3 |
| iso_b0.25 | 4,6 | 14,0 | 24,1 | 12,7 | 45,6 | 50,3 | 54,9 | 77,5 | 66,7 | 82,7 |
| **hard_b0.75** | **9,5** | **18,9** | **26,6** | **16,4** | **48,9** | **52,9** | **57,3** | 77,8 | 68,0 | 82,1 |

### E* = 0,50

| | t0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vanilla | 2,0 | 9,9 | 20,4 | 11,6 | 45,0 | 46,3 | 51,5 | 74,1 | 64,7 | 82,0 |
| permutado | 3,1 | 7,0 | 15,1 | 7,4 | 42,9 | 44,0 | 50,0 | 70,7 | 59,0 | 81,9 |
| iso_b0.25 | 4,9 | 11,1 | 17,7 | 8,3 | 42,0 | 45,1 | 52,0 | 70,4 | 55,7 | 81,8 |
| **hard_b0.5** | **11,4** | **19,7** | **26,1** | 14,4 | 46,9 | 48,9 | 56,8 | 76,4 | 65,5 | 81,4 |

O ganho se concentra nas tarefas **antigas** (t0–t3) e desaparece nas recentes
(t7–t9), onde ainda não houve esquecimento a evitar. A média de 10 tarefas
dilui o efeito com sete tarefas onde havia pouco a ganhar.

## 6. Custo de plasticidade — o `hard` NÃO paga

| E* | Contraste | Aquisição t_últ | Δ | Seeds | p |
|---|---|---:|---:|---:|---:|
| 0,75 | hard − vanilla | 82,10 vs 81,97 | +0,13 pp | 3+/7− | 0,8926 |
| 0,75 | hard − iso | 82,10 vs 82,70 | −0,60 pp | 5+/5− | 0,6328 |
| 0,50 | hard − vanilla | 81,43 vs 81,97 | −0,53 pp | 5+/4− | 0,6406 |
| 0,50 | hard − iso | 81,43 vs 81,83 | −0,40 pp | 5+/4− | 0,7109 |

Nenhum contraste se distingue de zero. **A proteção hard melhora retenção sem
custo detectável em aquisição da última tarefa.** Isto contrasta com o MLP
(Split-MNIST), onde hard − vanilla dá **−6,42 pp, p = 0,0020** na aquisição —
sugerindo que o custo do congelamento depende da capacidade disponível.

## 7. Custo computacional (E* = 0,75)

| Braço | Parede | Pico de memória | Razão vs vanilla |
|---|---:|---:|---:|
| vanilla | 8,06 min | 6,60 GiB | 1,00× |
| reduced_lr | 8,04 min | 6,60 GiB | 1,00× |
| iso_b0.25 | 10,06 min | 5,52 GiB | 1,25× |
| hard_b0.75 | 10,03 min | 5,52 GiB | 1,24× |

72 máscaras registradas nos braços protegidos. O pico de memória é **menor**
nos braços com máscara porque o congelamento dispensa estado de otimizador
para os parâmetros protegidos.

## 8. O que se pode e o que não se pode afirmar

**Pode-se afirmar:**

- O `hard` melhora retenção da primeira tarefa sobre o `vanilla` em **10/10
  seeds**, nas duas famílias, sobrevivendo a Holm sobre 24 comparações
  (+7,50 pp em E\*=0,75; +9,37 pp em E\*=0,50).
- Não há custo detectável em aquisição da última tarefa.
- O custo computacional é de ~1,25× em tempo, com pico de memória menor.

**NÃO se pode afirmar:**

- **Nada sobre FAA.** Nenhum contraste de FAA sobrevive a Holm.
- **Nada sobre forgetting como resultado confirmatório.** A direção é
  consistente (8–10 de 10 seeds em todos os contrastes, reduções de 11–15%),
  mas o melhor p ajustado é 0,0859.
- **Que o ranking aprendido supera o aleatório.** `hard − permutado` não foi
  testado diretamente nesta família; `iso − permutado` não sobrevive a Holm.
- **Que a tarefa 0 foi recuperada.** Retenção de 9,5–11,4% contra piso do
  acaso de 6,7%: o ganho é relativo, não utilizável.

**Ameaças à validade:**

- O braço `hard` entrou pelo commit `095b1c8`, **ausente da §E** do protocolo
  congelado; registrado a posteriori na tabela K. Por isso é **exploratório,
  nunca confirmatório** — e é justamente o braço com os dois únicos resultados
  que sobrevivem a Holm.
- `iso_b0.5` descartado em todas as seeds de E\*=0,50 reduz a família de
  comparações planejada.
- 120 passos por tarefa é regime de baixo orçamento; nada aqui se estende a
  treinamento até convergência.

## 9. Reprodução

```bash
PYTHONPATH=.:src .venv/bin/python experiments/analyze_confirmation.py
```
