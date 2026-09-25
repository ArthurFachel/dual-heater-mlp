# Qwen2 — evidência, protocolo e limites

Documento-guia do host Qwen2. **Esta arquitetura ainda não produz evidência
citável em artigo.** O documento existe para registrar o que está implementado,
o que está medido e o que falta antes que qualquer número possa ser reportado.

Escopo: `SlowHeatQwen2ForSequenceClassification` sobre `Qwen/Qwen2.5-0.5B`,
FFN gated (SwiGLU) apenas, CLINC150 class-incremental.

---

## Claim

**Nenhum claim de eficácia.** O que existe hoje:

1. O mecanismo está implementado e verificado para MLP gated (SwiGLU), com a
   máscara cobrindo corretamente linhas produtoras (`gate_proj`, `up_proj`) e
   colunas consumidoras (`down_proj`).
2. O custo real foi medido e é **35% menor que o estimado**: 5,447 GiB de pico
   alocado contra ~8,3 GB previstos.
3. A distribuição de importância do Qwen real foi medida e revelou **duas
   camadas anômalas** (L3 e L21) que concentram importância em 1–2% das
   unidades, enquanto as outras 22 usam 63–84%. **Sem explicação até hoje.**

Nenhum endpoint de continual learning foi produzido.

---

## Estado da evidência

### Q1 — Mecanismo verificado (CPU)

37 testes em `tests/test_slow_heat_qwen.py` com modelos Qwen2 minúsculos e
aleatórios. A Etapa E do contrato de Transformers (SwiGLU) está concluída:
`forward_pre_hook` em `down_proj` observa o produto gated
(`src/dual_heater/qwen.py:289-304`), e o agrupamento é fixado pelo teste
`test_producer_masks_are_rows_and_consumer_mask_is_columns`.

Mais 8 testes do diagnóstico de capacidade e 42 do runner de iso-plasticidade.

### Q2 — Custo medido (GPU)

`results/run_logs/results_gpu_memory_smoke.log`, GTX 1080 Ti, `--budget 0.50`,
batch 2:

| Grandeza | Valor |
|---|---:|
| Pico alocado | 5,447 GiB |
| Pico reservado | 6,033 GiB |
| Unidades protegidas | 58.368 |
| 10 passos | 3,17 s |

A estimativa analítica de 8,3 GB era pessimista. Isso é reportável como nota de
viabilidade, não como resultado.

### Q3 — Anomalia estrutural medida, não explicada

`results/qwen_layer_anomaly/`, 7 manifestos, 2 ordens × 3 seeds.

- **duas camadas em regime de cauda extrema**: L3 (PR = 104,8 de 4864 unidades,
  ou 2,2%) e L21 (PR = 42,4, ou 0,9%);
- **gradiente de profundidade nas demais**: as camadas iniciais e médias usam
  62–85% das unidades (pico em L8, 84,7%), mas o uso cai monotonicamente a
  partir de L19 — 47,5% (L19), 45,3% (L20), 27,7% (L22), 10,8% (L23). Ou seja,
  L3 e L21 são anomalias pontuais, enquanto o fim da rede exibe uma queda
  progressiva que é fenômeno distinto;
- o PR do sinal ao fim da run varia **11.567 a 27.955 entre as 6 runs**
  (razão 2,42×); considerando também o PR por estágio, o intervalo vai de
  11.567 a 43.373 (razão 3,75×), o que sugere forte dependência de seed;
- Jaccard do conjunto protegido após uma única fronteira: 0,870;
- turnover do pool livre: 0,345.

**A análise nunca foi escrita.** `docs/qwen_layer_anomaly.md`, prometido como
entregável de P6 e base do Gate 2, não existe. Os dados estão em disco.

---

## Por que nada disto é citável hoje

| Bloqueio | Detalhe |
|---|---|
| **Passos revogados** | Todas as 6 runs de calibração usam 30 passos/tarefa. A tabela K do protocolo congelado revogou esse valor em favor de 120, porque com 30 passos a acurácia fica em 0,31 contra teto prático de ~0,94 — o modelo era interrompido no início da curva |
| **Sem agregação** | Não há `aggregate.json`, nem diferenças pareadas, nem contagem de sinais |
| **Critério nunca aplicado** | `minimum_effective_plasticity: null`, `declared_before_run: false`, `selection: null` |
| **Gates não registrados** | Gate 1, Gate 2 e Gate 3 do roadmap não têm registro |
| **Confirmação incompleta** | 10 seeds declaradas; nenhum manifesto `confirm120_seed*` produzido |
| **Desvio de pré-registro** | Braço `hard` presente nos manifestos sem constar da seção E nem ter linha na tabela K antes da run |

---

## O que falta, em ordem

1. **Registrar Gate 1** conforme a seção I do protocolo congelado.
2. **Escrever `docs/qwen_layer_anomaly.md`** a partir dos dados já em disco, e
   fechar o Gate 2. A hipótese a testar: a queda de PR no fim da rede é efeito
   de profundidade (magnitude de `|z·dL/dz|` cai perto da cabeça porque a loss é
   dominada pelos logits), e L3/L21 são fenômeno distinto.
3. **Registrar Gate 3** antes da sequência de 10 tarefas — o protocolo exige
   isso e a confirmação foi iniciada sem ele.
4. **Concluir a confirmação** de 10 seeds × 10 domínios × 120 passos.
5. **Copiar manifestos para `artifacts/`** (o repositório ignora `results/`).
6. Rodar uma seed com `--capacity-scope hierarchical` (P6.3, nunca executado).

---

## Ameaças à validade, quando houver resultado

1. **Escopo restrito à FFN.** Atenção não é protegida: o Qwen2.5-0.5B tem 14
   heads de query e 2 de key/value, e o tracker de atenção não porta para GQA.
   Isso deve ser declarado como ameaça, não omitido.
2. **Dependência de seed na estrutura de importância** (PR variando 2,4×).
3. **Cabeça `score` aleatória.** Precisa ficar plástica; é contada separada dos
   parâmetros mascarados.
4. **Duas camadas em regime de cauda extrema** tornam o critério de plasticidade
   efetiva cego nelas — exatamente o risco previsto no Anexo A da ablação.
5. **Herança de endpoint negativo.** O BERT mostrou que SlowHeat perde para
   replay; o Qwen precisa declarar antes se sua pergunta é distinta ou
   continuação.

---

## Protocolo e implementação

Contrato do host, escopo da FFN gated e verificação por mutação em
[functional_slowheat_qwen.md](functional_slowheat_qwen.md).
Desenho da ablação iso-plasticidade e calibração de capacidade (Anexo A) em
[qwen_iso_plasticity_ablation.md](../protocols/qwen_iso_plasticity_ablation.md).
Protocolo **congelado** em `goals/protocol_iso_plasticity.md` — não editar;
alterações exigem commit anterior à run e linha na tabela K.
Roadmap, gates e plano de execução em `goals/qwen_heat_roadmap.md`.

Runners: `experiments/qwen_slowheat_smoke.py`,
`experiments/qwen_capacity_diagnostic.py`, `experiments/qwen_iso_plasticity.py`.
Driver da confirmação: `run_confirmation.sh`.