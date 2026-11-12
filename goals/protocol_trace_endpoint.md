# Protocolo congelado — endpoint de acurácia no host TRACE

> Pré-registro. Escrito **antes** de qualquer seed confirmatória deste host.
> Emendas só valem antes da primeira seed e devem ser commitadas antes do disparo.

**Estado:** congelado em 08/10/2026. Nenhuma seed confirmatória executada.

Cumpre a tarefa **3b.3** do `roadmap_icml_ijcnn.md` ("endpoint novo,
pré-registrado, com gate de regime"), que estava aberta. **Não** é emenda de
`protocol_exact_decomposition.md`: aquele protocolo foi executado e fechado em
29/09/2026 com 10 seeds e desfecho (a), e não aceita mais alteração.

---

## T0. Por que este documento existe

Uma seed-piloto de capacidade (seed 1234, 08/10/2026,
`results/trace_full/`, registro em
[`docs/results/trace_pilot_seed1234.md`](../docs/results/trace_pilot_seed1234.md))
mediu o custo do host e, ao fazê-lo, encontrou um **defeito de instrumento**: a
métrica oficial do TRACE reportou `0.0000` em células onde o modelo respondia
corretamente em até 56% dos exemplos.

O defeito é real e é do TRACE, não deste repositório. `caculate_accuracy`
compara strings com `==` cru, sem `strip` e sem case fold — quirk oficial 4,
reproduzido de propósito em `src/dual_heater/trace_metrics.py` para que OP/BWT
sejam comparáveis à Table 6 do paper. Uma geração `' A'` com o alvo `'A'`
conta como erro.

Isto exige pré-registro próprio **antes** de qualquer seed, porque a decisão
de como medir não pode ser tomada depois de ver contrastes.

## T1. O defeito, quantificado

Diagnóstico em `scripts/diagnose_accuracy_zeros.py` (CPU, sem modelo), sobre as
gerações salvas dos três braços da seed-piloto. 22 células com discrepância
material; as piores:

| braço | tarefa | estágio | oficial | recuperável |
|---|---|---|---:|---:|
| `exact` | ScienceQA | 7 | 0,0745 | **0,5610** |
| `exact` | FOMC | 2 | 0,0000 | 0,5202 |
| `frozen_a_control` | FOMC | 2 | 0,0000 | 0,5121 |
| `vanilla` | NumGLUE-ds | 7 | 0,0000 | 0,3754 |
| `vanilla` | C-STANCE | 3 | 0,0020 | 0,3660 |

Formas observadas: `'A '` (99% das gerações de C-STANCE no estágio 3), `' A'`,
`' A\nReasons:...'` em ScienceQA, e número seguido de prosa em 42-62% das
gerações de NumGLUE.

### T1.1 Por que isto é viés diferencial e não ruído

A degeneração **não é simétrica entre braços**. Os braços com `A` congelada
degeneram de forma homogênea — quase toda geração ganha o mesmo prefixo de
espaço, e a célula vai a zero exato. O `vanilla` degenera de forma
heterogênea, então uma fração das gerações continua sem prefixo e marca ponto.

Consequência: a métrica oficial mede, em parte, **a propensão do braço a
emitir espaço em branco**, e essa propensão correlaciona com o tratamento sob
teste. Isso confunde instrumento com efeito. Não é um erro que se cancela na
média nem que se dilui com mais seeds.

## T2. Endpoint primário (congelado)

**Acurácia normalizada**, definida por tarefa, idêntica para todos os braços:

| tarefa | regra |
|---|---|
| C-STANCE, FOMC | `pred.strip()[:1].upper() == alvo[:1].upper()` |
| ScienceQA | idem (o oficial já lê `item[0]`; a normalização só remove o prefixo) |
| NumGLUE-cm, NumGLUE-ds | primeiro número de `pred` == primeiro número do alvo, por `-?\d+(?:\.\d+)?` |
| MeetingBank, Py150, 20Minuten | **inalteradas** (ROUGE-L, fuzz, SARI já são tolerantes a prefixo) |

A regra é **declarada antes de qualquer seed**, aplicada igualmente aos três
braços, e deve estar implementada em `src/dual_heater/trace_metrics.py` com
teste que falha se ela divergir desta tabela, **antes** do disparo.

Endpoint: **forgetting médio** = `mean_{k<T} (max_l A[l,k] − A[T-1,k])` sobre a
matriz normalizada. Mesmo endpoint primário da linhagem Qwen, para
comparabilidade.

## T3. Métrica oficial: reportada, nunca descartada

A métrica oficial do TRACE continua computada e **reportada em paralelo em toda
tabela**, com a discrepância explícita. Razões:

1. é a única âncora externa ao projeto (Table 6 do paper);
2. a normalização é nossa decisão e precisa ficar auditável contra o original;
3. se as duas versões discordarem em ordenação de braços, isso é achado e vai
   no texto, não escondido.

Declarado como **desvio R11** na lista de desvios de `experiments/trace_cl.py`,
junto dos R1-R10 já existentes.

## T4. Gate de regime (§G.1 equivalente, exigido por 3b.3)

A tarefa 3b.3 do roadmap exige redefinir o gate de quase-chance para geração.
Definido aqui, antes de ver qualquer seed confirmatória:

Uma tarefa **não entra** no endpoint primário se, na condição normalizada, a
acurácia do `vanilla` no estágio em que a tarefa foi treinada (`A[k,k]`) ficar
**abaixo do baseline de maioria** daquela tarefa:

| tarefa | baseline de maioria |
|---|---:|
| C-STANCE | 0,3480 |
| FOMC | 0,4940 |
| ScienceQA | 0,0160 |
| NumGLUE-cm | 0,0864 |
| NumGLUE-ds | 0,1385 |

Justificativa: sem aprendizado acima do trivial, "esquecimento" não é
interpretável — foi o que a Fase 3.5 custou ao projeto. O gate usa **apenas o
`vanilla`** e **apenas a diagonal**, nunca um contraste, para não poder ser
usado para escolher tarefas por resultado.

A lista de tarefas que passam o gate é computada e **commitada antes do
disparo**, a partir da seed-piloto, e não é recalculada depois.

### T4.1 Gate avaliado na seed-piloto (commitado aqui)

| tarefa | maioria | `A[k,k]` normalizado (vanilla) | passa |
|---|---:|---:|---|
| C-STANCE | 0,3480 | 0,5090 | sim |
| FOMC | 0,4940 | 0,6290 | sim |
| ScienceQA | 0,0160 | 0,5575 | sim |
| NumGLUE-cm | 0,0864 | 0,3580 | sim |
| NumGLUE-ds | 0,1385 | 0,4892 | sim |

**As cinco passam.** O gate não exclui nenhuma tarefa, o que é o melhor
desfecho possível para ele: fica registrado que foi aplicado e não serviu de
pretexto para cortar nada.

## T5. O que este protocolo proíbe explicitamente

- **Reduzir o conjunto de tarefas.** Varredura exaustiva dos 120 subconjuntos
  (k≥2) da seed-piloto: o contraste `exact − frozen_a_control` varia de −0,2159
  a +0,0698, amplitude 0,2857 = **4,1× o contraste completo**, com troca de
  sinal disponível em k=4. Escolher subconjunto escolhe o resultado. Se o custo
  exigir corte, o lever é **reduzir seeds** com perda de poder declarada, nunca
  escolher tarefas. (Pergunta levantada e respondida em 08/10/2026, com os
  dados da piloto à vista; registrada aqui para não ser reaberta sem o número.)
- Trocar o endpoint primário depois de ver resultado.
- Recalcular Holm sobre família reduzida.
- Apagar resultado negativo.
- Usar a seed 1234 como seed confirmatória: é piloto de capacidade, foi
  lançada com árvore suja, e seus contrastes já foram vistos.

## T6. Seeds

A definir **antes** do disparo, em banda disjunta de todas as seeds já
reservadas nos artefatos de `results/`, verificada mecanicamente por teste.
A seed 1234 está queimada por T5.

## T7. Custo medido (não estimado)

Da seed-piloto, 8 tarefas, Qwen2.5-0.5B, fp32, 6254 passos:

| braço | parede | pico |
|---|---:|---:|
| `vanilla` | 23,8 h | 10,6 GiB |
| `frozen_a_control` | 24,4 h | 10,3 GiB |
| `exact` | 26,9 h | 10,5 GiB |

Distribuição: MeetingBank 24,3% e 20Minuten 30,4% do custo total — 55% em duas
tarefas. Eval é 40% do total e cresce com o triângulo de tarefas vistas.

Campanha de 10 seeds × 3 braços em 3 GPUs: **~250 h ≈ 10,4 dias** de relógio
contínuo. Requer autorização explícita.

## T8. Pré-condição de disparo

Nenhuma seed confirmatória antes de, nesta ordem:

1. normalização da §T2 implementada em `trace_metrics.py`, com teste que morre
   se a regra divergir da tabela, e mutação verificada;
2. gate da §T4 computado da piloto e commitado;
3. seeds da §T6 fixadas em código e verificadas por teste de disjunção;
4. desvio R11 escrito na docstring de `experiments/trace_cl.py`;
5. árvore Git limpa no disparo — a piloto não estabelece reprodução
   bit-a-bit e isso não pode repetir num run reportado;
6. autorização explícita de GPU.

## T9. O que este protocolo NÃO decide

Não decide se o SlowHeat funciona. **R-C já fechou essa questão em LoRA**: 10
seeds, `exact − frozen_a_control` 5+/5, p = 1,000, desfecho (a) — a máscara não
adiciona nada sobre LoRA-FA em Split-CLINC150. Este host mede outra coisa: se a
**ordenação dos braços muda em instruction tuning** quando a plasticidade é
pareada (tarefa 3b.4), com LoRA-FA presente como baseline publicada obrigatória
(3b.5).

Um resultado aqui que contradiga R-C não revoga R-C: são hosts e benchmarks
diferentes, e a leitura seria "o efeito depende do host", com os dois
reportados.

### T9.1 R-C auditado contra o defeito da §T1 — não afetado

Verificado em 08/10/2026, na dúvida de que o defeito de formato pudesse
contaminar o R-C e, por cascata, o arquivamento de
`protocol_slow_fast_coupling.md`.

**Não contamina.** O runner do R-C (`experiments/qwen_lora_slowheat.py`) usa
`Qwen2ForSequenceClassification` e mede acurácia por
`logits.masked_fill(...).argmax(-1) == labels` (`_evaluate`, linhas 170-197).
Nenhuma string participa do caminho da métrica: busca por `.generate(`,
`batch_decode`, `skip_special_tokens` e comparação com target textual retorna
zero ocorrências no arquivo. Os 10 manifests de
`results/exact_decomposition/seed_5*` não contêm campo de predição ou texto.

O defeito exige modelo gerando texto livre **e** métrica comparando strings.
Um prefixo de espaço não é representável num `argmax` sobre classes.

Escopo do que foi verificado: **apenas** esta classe de defeito. O R-C não foi
reauditado por outras causas (pareamento de dados, contagem de parâmetros,
instrumentação de plasticidade).
