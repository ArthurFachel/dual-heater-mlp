# Protocolo congelado — ablação iso-plasticidade no Qwen2.5-0.5B

> Pré-registro. Este arquivo é escrito **antes** de qualquer run com acurácia e
> não é editado depois de ver acurácia. Se um valor mudar, o commit que o muda
> tem de vir antes da run correspondente, e a mudança fica registrada abaixo.

**Estado: RASCUNHO — NÃO CONGELADO.** Congelar exige fechar os itens do bloco A.
Enquanto houver item em aberto, este documento não autoriza run com acurácia.

Última atualização: 22 de setembro de 2026.
Gate 0: fechado — `concentration_ratio` 0,512 a 0,551, conjunto protegido é
funcional (`goals/qwen_heat_roadmap.md`, Meta 0).

---

## A. Decisões pendentes (bloqueiam o congelamento)

| # | Decisão | Valor no rascunho | Status |
|---|---|---|---|
| A1 | `E*` declarados | 0,75 (primário) e 0,50 (secundário) | a confirmar |
| A2 | `beta` fixo ou re-resolvido por fronteira | re-resolvido por fronteira | a confirmar |
| A3 | Seeds de calibração / confirmação | {0, 1, 2} / 10 seeds a declarar | a confirmar |
| A4 | `minimum_effective_plasticity` do critério declarado | 0,60, avaliado em `b=0,25` | a confirmar |

Regra de A4: o critério de `docs/qwen_capacity_calibration.md` lê **somente
mecanismo** e falha em vez de relaxar o piso. Nesta run ele nunca foi aplicado
(`selection=None`). Se o piso 0,60 não for atingível, o resultado é "critério
falhou" e isso é registrado, não contornado.

---

## B. Pergunta

Dada a mesma quantidade de plasticidade efetiva removida, importa **como** ela
foi distribuída entre as unidades?

O diagnóstico de BERT (`docs/bert_slowheat_diagnostic_results.md`) só conseguiu
concluir "mais proteção troca aquisição por retenção" — afirmação sobre
quantidade. Contra o controle de LR reduzido, que aquele diagnóstico não tinha.

## C. Hipótese declarada antes da run

*Se a distribuição da proteção não importar, os braços iso-`E` ficam dentro do
ruído entre seeds entre si, **e** todos ficam dentro do ruído do braço de LR
reduzido na mesma plasticidade efetiva.*

Resultado negativo é publicável (`docs/qwen_iso_plasticity_ablation.md`, §7). Não
há tentativa de salvar o endpoint do BERT.

## D. Cenário e modelo

| Item | Valor |
|---|---|
| Modelo | `Qwen/Qwen2.5-0.5B`, `num_labels=150`, fp32 em memória, `fp16=True` no Trainer |
| Dataset | `clinc/clinc_oos:plus`, Class-IL por domínio |
| Tarefas na calibração | 2 (`banking -> credit_cards`) |
| Tarefas na confirmação | 10 (sequência completa dos domínios) |
| `max_length` | 64 |
| `batch_size` | 2 |
| Passos por tarefa | 30 |
| `learning_rate` | 1e-5 |
| Escopo de capacidade | `local` (default de produção, obrigatório) |
| `freeze_unbound_parameters` | `True` |
| Cabeça de classificação | treinável e sem máscara; contagem reportada separada |

**Escopo de capacidade é parte do mecanismo.** Toda aritmética de heat usa as
variantes `*_scoped`. Concatenar camadas e normalizar por máximo global erra por
~2 ordens de magnitude — foi o bug de 22/09.

**Hardware.** GPUs heterogêneas (2x GTX 1080 Ti + 1x Titan Xp, Pascal CC 6.x):
fp16 obrigatório, `CUDA_VISIBLE_DEVICES` fixado em **uma** placa, DDP
desabilitado. Run em GPU exige autorização do Fachel.

## E. Braços

Todos com o mesmo `E*` por família. `beta` resolvido por
`solve_strength_for_plasticity_scoped` (bisseção, tolerância 1e-9), **nunca por
busca em grade com acurácia no loop**.

### E.1 Família primária, `E* = 0,75`

| Braço | budget | protegidas | `beta` | papel |
|---|---:|---:|---:|---|
| iso-E muitas-fracas | 0,05 | 110.880 (95%) | 8,139 | muitas unidades, proteção fraca |
| iso-E intermediário | 0,25 | 87.552 (75%) | 10,754 | intermediário |
| iso-E poucas-fortes | 0,50 | 58.368 (50%) | 20,998 | poucas unidades, proteção forte |

Spread de contagem 1,90x; spread de `beta` 2,58x. `b=0,10` (105.048, beta 8,614)
fica disponível como quarto braço e não é primário.

### E.2 Família secundária, `E* = 0,50`

| Braço | budget | protegidas | `beta` |
|---|---:|---:|---:|
| iso-E muitas-fracas | 0,05 | 110.880 (95%) | 31,168 |
| iso-E intermediário | 0,10 | 105.048 (90%) | 35,002 |
| iso-E poucas-fortes | 0,25 | 87.552 (75%) | 58,248 |

`b=0,50` **não é alcançável** em `E*=0,50`: o piso `(N-P)/N` já é 0,50, então
`beta` divergiria. `solve_*` retorna `None` e o braço é descartado, não saturado.
Consequência: nesta família o contraste de contagem é de apenas 1,27x. A família
de `E*=0,75` é a que carrega o contraste; a de 0,50 é replicação de direção.

### E.3 Controles, todos com `E*` pareado

| Braço | Definição | Papel |
|---|---|---|
| iso-E aleatório | **permutação do vetor de heat** que preserva a distribuição | o ranking informa? |
| LR reduzido | `lr * E*`, **sem máscara nenhuma** | o mecanismo é redutível a LR menor? |
| vanilla | `E = 1`, sem consolidação, sem máscara | âncora de forgetting |

Controle de LR reduzido é **obrigatório, não opcional**. É o controle que o
diagnóstico de BERT não tinha.

**O controle aleatório correto não é `randomize_slowheat_protection`.**
`experiments/split_clinc150.py:820` seta `slow_heat = 1,0` nas identidades
sorteadas, o que é proteção hard e **não** é um controle iso-E. O controle correto
permuta o vetor preservando a distribuição:

```python
permutation = torch.randperm(state.slow_heat.numel(), generator=generator)
state.slow_heat.copy_(state.slow_heat[permutation])
```

Escrever função nova e testar que distribuição de heat e `E` ficam idênticas após
a permutação, com `permutation != identity`.

## F. Resolução de `beta` e política temporal

`beta` é resolvido a partir da importância medida na **primeira** consolidação
(importância é acumulada durante a tarefa 1 e consolidada na fronteira 1->2).

**Política declarada (A2): re-resolvido a cada fronteira.** Em cada fronteira
`t -> t+1`, para cada braço, resolve-se `beta_t` tal que
`E(beta_t) = E*` sobre a importância consolidada naquele momento. O valor
saturado é sempre o alvo declarado, nunca um valor escolhido.

Consequências que precisam ser reportadas junto: `beta_t` por braço e por
fronteira; `E` atingido por braço e por fronteira (deve ser `E*` a 1e-6); e o
`beta` da primeira fronteira para comparabilidade com o diagnóstico de Gate 0.

Não misturar: ou `beta` fixo após a primeira consolidação, ou re-resolvido. Se o
braço de LR reduzido vencer, a leitura é que o mecanismo é uma redução de LR
disfarçada — em qualquer das duas políticas.

## G. Endpoints

Reportados por braço, por seed, com diferenças **pareadas por seed** e contagem
de sinais:

| Endpoint | Definição |
|---|---|
| FAA | acurácia média final, `mean_k A[T-1,k]` |
| Retenção T1 | `A[1,0]` |
| Aquisição T2 | `A[1,1]` |
| Forgetting médio | `mean_{k< T-1} (max_l A[l,k] - A[T-1,k])` — **exclui a última tarefa** |
| BWT | retenção versus desempenho logo após aprender cada tarefa antiga |
| Drift protegido | parâmetros mascarados, `capture_parameter_drift_reference` + `summarize_parameter_drift` |
| Drift plástico | mesmos instrumentos, parâmetros fora da máscara |
| Custo | pico de memória alocado e reservado, tempo de parede, tokens/s |

Não reportar retenção da tarefa 0 como forgetting global.

## H. Verificação em CPU antes de qualquer GPU

Modelo Qwen2 minúsculo, sem download, como em `tests/test_slow_heat_qwen.py`.
Asserções obrigatórias:

1. todo braço atinge o `E*` declarado dentro de 1e-6;
2. o braço aleatório tem **distribuição de heat idêntica** ao aprendido
   (mesmo histograma ordenado) e `E` idêntico;
3. o braço de LR reduzido **não registra máscara nenhuma**;
4. drift protegido é exatamente zero sob máscara hard;
5. o manifesto contém todos os braços, `beta` resolvidos por fronteira, `E`
   atingido, escopo de capacidade e o protocolo.

Mutações a injetar, cada uma tem de quebrar algum teste:

| Mutação | Teste que deve quebrar |
|---|---|
| inverter o eixo da máscara | 4 |
| trocar permutação por proteção hard | 2 |
| ignorar `capacity_scope` | 1 |
| aplicar `E*` errado num braço | 1 |
| saturar `beta` em vez de descartar budget inalcançável | 1 |
| esquecer `seen_tasks` na avaliação | 5 |

Teste novo verde de primeira é suspeito: injetar as mutações e confirmar.

## I. Gate 1 (fecha a calibração)

- todos os braços atingem `E*` com erro `< 1e-6`;
- drift protegido exatamente zero nos braços com máscara;
- `E` medido do braço de LR reduzido igual a `E*` por construção;
- diferenças pareadas por seed reportadas para cada endpoint, com contagem de sinais;
- nenhuma decisão de hiperparâmetro tomada após ver acurácia;
- suíte completa verde, com escopo e mutação testadas, não só "os testes passaram".

Gate 1 reprovado é resultado registrado, não obstáculo a contornar. Gate 1
fechado não autoriza benchmark completo.

## J. Gate 3 (precede a sequência de 10 tarefas)

Declarado antes da run de 10 domínios, conjunto:

- **mecanismo:** drift protegido exato, ranking útil (iso-E aprendido vs
  aleatório), aquisição adequada;
- **retenção:** o braço vencedor dentro do ruído de replay/EWC declarados.

Gate de ranking pode passar com o de retenção falhando; isso não autoriza
benchmark completo.

## K. Registro de alterações

| Data | Alteração | Antes da run? |
|---|---|---|
| 22/09 | criação do rascunho; Gate 0 registrado | sim |
| | **congelamento pendente dos itens A1-A4** | |
