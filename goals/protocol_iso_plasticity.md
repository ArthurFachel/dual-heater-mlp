# Protocolo congelado — ablação iso-plasticidade no Qwen2.5-0.5B

> Pré-registro. Este arquivo é escrito **antes** de qualquer run com acurácia e
> não é editado depois de ver acurácia. Se um valor mudar, o commit que o muda
> tem de vir antes da run correspondente, e a mudança fica registrada abaixo.

**Estado: CONGELADO em 22/09/2026.** As decisões A1 a A4 estão fechadas na
seção A. A partir deste commit o arquivo não é editado: qualquer mudança exige
um commit anterior à run correspondente, registrado na tabela K.

Última atualização: 22 de setembro de 2026.
Gate 0: fechado — `concentration_ratio` 0,512 a 0,551, conjunto protegido é
funcional (`goals/qwen_heat_roadmap.md`, Meta 0).

---

## A. Decisões congeladas

| # | Decisão | Valor congelado | Justificativa |
|---|---|---|---|
| A1 | `E*` declarados | 0,75 primário, 0,50 secundário | `E*=0,75` tem 4 budgets alcançáveis e contraste de 1,90x em contagem protegida; `E*=0,50` tem 3 e contraste de 1,27x, então replica a direção sem carregar o contraste |
| A2 | `beta` fixo ou re-resolvido por fronteira | re-resolvido a cada fronteira | mantém `E` no alvo declarado quando a importância consolidada muda; `beta` fixo deixaria `E` derivar e o braço deixaria de ser iso-`E` |
| A3 | Seeds de calibração / confirmação | calibração {0, 1, 2}; confirmação {10, 11, 12, 13, 14, 15, 16, 17, 18, 19} | as 10 de confirmação são declaradas **agora**, antes de qualquer acurácia; escolhê-las depois invalidaria o pré-registro |
| A4 | `minimum_effective_plasticity` do critério declarado | 0,60, avaliado em `b=0,25` | o regime medido é informativo (`heavy_tailed_warning=False`, `PR/N` agregado 11,5% contra guard-rail de 5%); o piso seleciona `beta=10` (`E=0,760`) na grade |

Regra de A4: o critério do Anexo A de `docs/protocols/qwen_iso_plasticity_ablation.md` lê **somente
mecanismo** e falha em vez de relaxar o piso. Se o piso 0,60 não for atingível
numa run futura, o resultado é "critério falhou" e isso é registrado, não
contornado.

Consequência de A2 que precisa ser reportada: `beta` só existe a partir da
fronteira 1->2, porque é resolvido sobre a importância consolidada. O `beta` da
primeira fronteira é reportado separado, para comparabilidade com Gate 0.

---

## B. Pergunta

Dada a mesma quantidade de plasticidade efetiva removida, importa **como** ela
foi distribuída entre as unidades?

O diagnóstico de BERT (`docs/results/bert_slowheat_diagnostic_results.md`) só conseguiu
concluir "mais proteção troca aquisição por retenção" — afirmação sobre
quantidade. Contra o controle de LR reduzido, que aquele diagnóstico não tinha.

## C. Hipótese declarada antes da run

*Se a distribuição da proteção não importar, os braços iso-`E` ficam dentro do
ruído entre seeds entre si, **e** todos ficam dentro do ruído do braço de LR
reduzido na mesma plasticidade efetiva.*

Resultado negativo é publicável (`docs/protocols/qwen_iso_plasticity_ablation.md`, §7). Não
há tentativa de salvar o endpoint do BERT.

## D. Cenário e modelo

| Item | Valor |
|---|---|
| Modelo | `Qwen/Qwen2.5-0.5B`, `num_labels=150`, fp32 em memória e **fp32 no treino** |
| Dataset | `clinc/clinc_oos:plus`, Class-IL por domínio |
| Tarefas na calibração | 2, na ordem `banking -> credit_cards` |
| Tarefas na confirmação | 10, na ordem de `CLINC150_DOMAINS` |
| `max_length` | 64 |
| `batch_size` | 2 |
| Passos por tarefa | 120 (era 30; ver tabela K, 22/09) |
| `learning_rate` | 1e-5 |
| Escopo de capacidade | `local` (default de produção, obrigatório) |
| `freeze_unbound_parameters` | `True` |
| Cabeça de classificação | treinável e sem máscara; contagem reportada separada |
| Seeds de calibração | 0, 1, 2 |
| Seeds de confirmação | 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 |

**A ordem das tarefas é parte do protocolo.** Declarada acima e passada
explicitamente por `--domains`, nunca inferida de uma contagem.

**Escopo de capacidade é parte do mecanismo.** Toda aritmética de heat usa as
variantes `*_scoped`. Concatenar camadas e normalizar por máximo global erra por
~2 ordens de magnitude — foi o bug de 22/09.

**Hardware.** GPUs heterogêneas (2x GTX 1080 Ti + 1x Titan Xp, Pascal CC 6.x):
`CUDA_VISIBLE_DEVICES` fixado em **uma** placa, DDP desabilitado. Run em GPU
exige autorização do Fachel.

**Precisão é fp32, não fp16.** Medido em 22/09 numa 1080 Ti: sob autocast fp16
com `GradScaler`, as camadas 0 a 18 registram importância **exatamente zero**;
só as 5 últimas gravam sinal. O gradiente que chega à entrada de `down_proj`
nas camadas iniciais faz underflow em fp16, e o estimador `|z dL/dz|` zera com
ele. O `GradScaler` não resolve: ele reescala a loss, mas o produto já foi
arredondado a zero no forward-backward antes do unscale.

Consequência se fosse mantido: `positive_fraction` cai de 1,00 para 0,2083
(5/24), o piso em `b=0,25` sobe de 0,25 para 0,84, e **nenhum braço iso-E é
alcançável**. O mecanismo seria medido em 5 camadas e reportado como se fossem
24.

Custo de usar fp32, medido na mesma placa com o braço mais protegido
(`b=0,50`): pico alocado 5,48 GiB contra 5,45 GiB em fp16, e 6,8s contra 6,3s
para 30 passos. Cabe com folga em 11 GB. A economia de fp16 é de 0,03 GiB e
~8% de tempo, contra perder 19 das 24 camadas.

## E. Braços

Todos com o mesmo `E*` por família. `beta` resolvido por
`solve_strength_for_plasticity_scoped` (bisseção, tolerância 1e-9), **nunca por
busca em grade com acurácia no loop**.

### E.1 Família primária, `E* = 0,75`

| Braço | budget | protegidas | `beta` | papel |
|---|---:|---:|---:|---|
| iso-E muitas-fracas | 0,05 | 110.880 (95%) | 8,139 | muitas unidades, proteção fraca |
| iso-E muitas-fracas-2 | 0,10 | 105.048 (90%) | 8,614 | ponto intermediário do lado fraco |
| iso-E intermediário | 0,25 | 87.552 (75%) | 10,754 | intermediário |
| iso-E poucas-fortes | 0,50 | 58.368 (50%) | 20,998 | poucas unidades, proteção forte |

Spread de contagem 1,90x; spread de `beta` 2,58x, medidos entre `b=0,05` e
`b=0,50`. Os quatro budgets são braços declarados: a grade de budgets é
`{0,05; 0,10; 0,25; 0,50}` e todo budget alcançável vira braço, sem seleção
posterior. Os valores de `beta` na tabela são os de Gate 0 e servem de
referência; na run cada `beta` é re-resolvido por fronteira (A2).

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

Contagem de braços: 4 iso-E + 3 controles = **7** em `E*=0,75`; 3 iso-E + 3
controles = **6** em `E*=0,50`. Total 13 por seed.

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
   (mesmo histograma ordenado) e `E` idêntico, com `permutation != identity`
   verificado;
3. o braço de LR reduzido **não registra máscara nenhuma** (`mask_bindings()`
   vazio);
4. drift protegido é exatamente zero sob máscara hard;
5. o manifesto contém todos os braços, `beta` resolvidos por fronteira, `E`
   atingido, escopo de capacidade, o protocolo e **a lista de budgets
   descartados por inalcançáveis**.

Mutações a injetar, cada uma tem de quebrar algum teste:

| Mutação | Teste que deve quebrar |
|---|---|
| inverter o eixo da máscara | 4 |
| trocar permutação por proteção hard | 2 |
| ignorar `capacity_scope` | 1 |
| aplicar `E*` errado num braço | 1 |
| saturar `beta` em vez de descartar budget inalcançável | 5 |
| registrar máscara no braço de LR reduzido | 3 |
| esquecer `seen_classes` na avaliação | teste de avaliação Class-IL |

Teste novo verde de primeira é suspeito: injetar as mutações e confirmar.
Mutação sobrevivente é lacuna de teste ou equivalência provada; investigar e
documentar qual das duas, sem deixar a sobrevivente sem explicação.

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
| 22/09 | **congelamento.** A1 = 0,75 primário e 0,50 secundário; A2 = `beta` re-resolvido por fronteira; A3 = calibração {0,1,2} e confirmação {10..19}; A4 = piso 0,60 em `b=0,25`. Ordem das tarefas declarada em D. `b=0,10` promovido a braço declarado em E.1 (4 braços iso-E, 7 no total em `E*=0,75`). Tabela H corrigida: `seen_classes` (não `seen_tasks`), saturação de `beta` mapeada para a asserção 5, mutação de máscara no braço de LR reduzido adicionada. | sim — nenhuma acurácia observada |
| 22/09 | **fp16 -> fp32 no treino.** Medição (sem acurácia) mostrou que sob autocast fp16 as camadas 0 a 18 registram importância exatamente zero: `positive_fraction` 0,2083 em vez de 1,00, piso em `b=0,25` de 0,84 em vez de 0,25, e nenhum braço iso-E alcançável. Custo medido de fp32: +0,03 GiB de pico e +8% de tempo. Ver seção D. | sim — a run descartada por este motivo não teve nenhum endpoint lido |
| 22/09 | **30 -> 120 passos por tarefa.** Medição de aquisição pura (uma tarefa isolada, sem mecanismo, sem comparação entre braços): com 30 passos e `batch_size=2` são 4 exemplos por classe e a acurácia fica em 0,31, contra teto prático de ~0,94. O modelo era interrompido no início da curva, então não havia conhecimento consolidado para nenhum mecanismo preservar — e no piloto de 10 tarefas os 4 braços iso ficaram indistinguíveis no piso (FAA 0,066 a 0,084, retenção ~0,015 = acaso). Com 120 passos (16 ex/classe) a acurácia é 0,78-0,81, o joelho da curva: 240 compra +0,06 por 60% mais tempo, e 960 já mostra overfitting em `credit_cards` (0,943 -> 0,870). Ver `results/run_logs/steps_sweep.log`. | sim — só aquisição de tarefa única foi lida; nenhum endpoint comparativo entre braços foi observado, o pré-registro segue intacto |

A partir daqui, qualquer alteração exige commit anterior à run correspondente e
uma linha nova nesta tabela.

### Desvio registrado a posteriori (22/09, revisão documental)

O commit `095b1c8` acrescentou os braços `hard_b0.75` e `hard_b0.5` ao runner,
e os manifestos `results/qwen_iso_plasticity/hard_seed*` foram executados com
**8 braços por família**, enquanto a seção E declara 7 braços em `E* = 0,75` e
6 em `E* = 0,50`. Essa alteração **não recebeu linha nesta tabela antes da
run**, violando a regra do parágrafo acima.

Consequência: o braço `hard` não é um braço pré-registrado. Ele pode ser
reportado como exploratório, nunca como parte do contraste congelado. As runs
afetadas usam 30 passos e já estão obsoletas por outro motivo, o que limita o
dano a este desvio.
