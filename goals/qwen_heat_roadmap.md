# Metas: SlowHeat no Qwen2

> Documento de controle. Cada meta tem um gate explícito. Não avançar para a
> meta seguinte sem o gate anterior fechado, e registrar no próprio arquivo se
> um gate falhar — um gate reprovado é resultado, não obstáculo a contornar.

Última atualização: 22 de setembro de 2026.

## Regras que valem para todas as metas

1. **Nenhum hiperparâmetro é escolhido olhando acurácia.** Proteção (`beta`) e
   budget saem de critério de mecanismo declarado antes da run, ou de dataset
   disjunto. Ver `docs/protocols/qwen_iso_plasticity_ablation.md`, Anexo A.
2. **Escopo de capacidade faz parte do mecanismo.** Qualquer aritmética de heat
   usa as variantes `*_scoped` com o escopo explícito. Concatenar camadas e
   normalizar por máximo global dá resposta errada por ~2 ordens de magnitude.
3. **Suíte verde toda vez que passa por um gate** com escopo e mutação testadas,
   não só "os testes passaram".
4. **Teste novo verde de primeira é suspeito.** Injetar mutações no código de
   produção e confirmar que cada uma quebra algum teste. Mutação sobrevivente é
   lacuna de teste ou equivalência provada — investigar e documentar qual.
5. **Isto não é continuação confirmatória do BERT.**
   `docs/results/bert_slowheat_diagnostic_results.md` fechou aquele endpoint. A pergunta
   aqui é outra: *a proteção seletiva tem efeito estrutural além da redução de
   capacidade que ela causa?* Resultado negativo é publicável.

---

## Estado atual

### Concluído

| Item | Onde | Verificação |
|---|---|---|
| Host Qwen2 com SlowHeat na FFN gated | `src/dual_heater/qwen.py` | 40 testes CPU; 10 mutações, 9 detectadas, 1 equivalência provada |
| Helpers de capacidade compartilhados BERT/Qwen | `src/dual_heater/transformer.py` | 44 testes BERT intactos |
| Aritmética de calibração + escopo | `experiments/capacity_calibration.py` | 72 testes; 5 mutações de escopo, 5 detectadas |
| Smoke no checkpoint real | `experiments/qwen_slowheat_smoke.py` | 313.920.768 treináveis, 72 bindings, drift protegido exatamente 0 |
| Diagnóstico de mecanismo | `experiments/qwen_capacity_diagnostic.py` | executado, 2 tarefas, seed 0 |
| Etapa 0.1 — concentração da proteção | `results/qwen_capacity_diagnostic/manifest.json` | executado 22/09 13:48, 2 tarefas, seed 0; **Gate 0 = funcional** |

Suíte completa: 498 testes.

### Medido no Qwen2.5-0.5B real (CLINC150, `banking -> credit_cards`, seed 0)

| Quantidade | Valor |
|---|---|
| unidades FFN (24 x 4864) | 116.736 |
| densidade de sinal | 1,000 em todas as camadas |
| PR agregado | 13.430 (11,5%) |
| PR por camada, típico | 3.000 a 4.100 (63 a 84%) |
| protegidas por camada, `b=0,25` | 3.648 (saturado) |
| heat médio por unidade protegida | 0,059 |
| `max_heat` por camada | 1,0 em todas as 24 |
| beta para `E*=0,75`, `b=0,25` | 10,75 |
| loss estágio 0 / 1 | 16,19 -> 2,06 / 11,67 -> 0,55 |
| Jaccard do conjunto protegido | 0,870 |
| turnover do pool livre | 0,345 |
| `concentration_ratio`, `b=0,25` | 0,519 |
| `effective_protected_units`, `b=0,25` | 45.460 de 87.552 |
| heat mediano da unidade protegida, `b=0,25` | 0,0466 |
| unidades com `h >= 0,5` (todos os budgets 0,10-0,50) | 142 |
| unidades com `h >= 0,1` (budgets 0,10-0,50) | 12.693 |

### Resultado analítico já estabelecido

`E(beta,b) >= (N-P)/N` para todo `beta`, com limite justo em `beta -> infinito`.
O budget define piso rígido de plasticidade; `beta` só move `E` dentro de
`[piso, 1]`. Independe de dados. Verificado em 3000 casos aleatórios.

### Achados sem explicação

- **Camadas 3 e 21** têm PR de 95 e 40 de 4864, contra 63-84% nas outras 22.
  Concentram importância em 1-2% das unidades. `mean_heat` de 0,0022 e 0,0013, e
  **`concentration_ratio` de 0,018 e 0,009**: nessas duas camadas as 3.648
  unidades marcadas são nominais — só ~65 e ~33 carregam a proteção de fato.
  A subproteção ali é real, não só baixa magnitude.
- **A queda de PR não são só duas camadas.** L19 (2.493), L20 (2.269),
  L23 (628) e L22 (1.212) também ficam abaixo da faixa 3.000-4.105, com
  `concentration_ratio` de 0,534 / 0,495 / 0,148 / 0,270. O padrão é de fim de
  rede, não de pontos isolados. Ver Meta 2.
- **Proteção rasa e espalhada, agora quantificada.** `max_heat=1,0` com
  `mean_protected_heat=0,059` e heat mediano de 0,0466: metade das protegidas
  retém **>= 67%** do learning rate em `beta=10,75`. Ameaça direta à Meta 1 —
  mas é ameaça de *magnitude*, não de nominalidade: o conjunto protegido como um
  todo é funcional (Gate 0 abaixo).

### Bugs encontrados e corrigidos

| Bug | Impacto | Correção |
|---|---|---|
| Aritmética agregava camadas ignorando `capacity_scope` | betas errados por ~2 ordens (266 vs 8,14) | variantes `*_scoped`, escopo no manifesto |
| Diagnóstico não passava `capacity_scope` ao `QwenSlowHeatConfig` | config do treino divergia da análise | `--capacity-scope` propagado |

### Hipótese minha que estava errada

Afirmei que hookar `act_fn` mediria o tensor errado na MLP gated. Falso: para
`z = a*u` vale `dL/da = dL/dz*u`, logo `|a*dL/da| = |z*dL/dz|`. O estimador é
invariante ao longo da cadeia (verificado a 2,4e-7). A posição do hook em
`down_proj` está certa por outro motivo — casa com a máscara de colunas — mas
não corrige bug de ranking nenhum.

---

## Meta 0 — Fechar a viabilidade da ablação

**Pergunta:** o conjunto protegido é funcional ou apenas nominal?

**Por que existe:** `mean_heat=0,059` com `max_heat=1,0` sugere que a maioria das
3.648 "protegidas" por camada mantém quase toda a plasticidade. Se for o caso, os
braços iso-E variam a contagem protegida (110.880 vs 58.368) numa faixa onde o
heat é irrelevante, e o contraste prometido — "muitas fracas vs poucas fortes" —
degenera em "muitas quase-nulas vs menos quase-nulas".

> **Refutado em 22/09.** O conjunto protegido é funcional; ver Gate 0. O que
> sobra da preocupação não é nominalidade, é a **magnitude média baixa**.

### Etapa 0.1 — Medir concentração

**Executada em 22/09 13:48.** `heat_concentration` em
`experiments/capacity_calibration.py:570`, integrada ao runner em
`qwen_capacity_diagnostic.py:108,306,403`. Comando usado:

```sh
cd /mnt/B-SSD/fachel/dual-heater-mlp
CUDA_VISIBLE_DEVICES= HF_HOME=.hf-cache PYTHONPATH=. \
  .venv/bin/python experiments/qwen_capacity_diagnostic.py \
    --tasks 2 --capacity-scope local \
    --iso-plasticity 0.75 --iso-plasticity 0.50
```

Manifesto: `results/qwen_capacity_diagnostic/manifest.json` (a versão anterior,
sem a chave `concentration`, ficou em `manifest.pre-concentration.json`). As
quantidades por camada e o sinal agregado reproduziram o manifesto antigo
exatamente, então a run é reprodutível e só a instrumentação era nova.

`concentration_ratio` = `effective_protected_units / protected_units`, onde
`effective` é a razão de participação do próprio vetor de heat.

| budget | protegidas | efetivas | ratio | `h>=0,1` | q50 de `h` |
|---|---:|---:|---:|---:|---:|
| 0,10 | 105.048 | 53.751 | 0,512 | 12.693 | 0,043 |
| 0,25 | 87.552 | 45.460 | 0,519 | 12.693 | 0,047 |
| 0,50 | 58.368 | 30.746 | 0,527 | 12.693 | 0,054 |
| 0,75 | 29.184 | 15.586 | 0,534 | 10.104 | 0,070 |
| 0,90 | 11.664 | 6.428 | 0,551 | 5.733 | 0,098 |

Observação que vale registrar: **142 unidades têm `h>=0,5` em todos os budgets
de 0,10 a 0,50**, e 12.693 têm `h>=0,1` nesses mesmos três budgets. O conjunto de
unidades de magnitude alta é estável; o que os braços acrescentam ao descer o
budget é a cauda `0,01 < h < 0,1` (92.606 unidades em `b=0,10` contra 52.233 em
`b=0,50`). É nessa banda que o contraste muitas-fracas / poucas-fortes acontece.

### Gate 0 (decide o desenho, não passa/falha)

| `concentration_ratio` | Leitura | Ação |
|---|---|---|
| `>= 0,30` | conjunto protegido é funcional | seguir para Meta 1 sem mudanças |
| `0,10` a `0,30` | zona cinzenta | seguir, mas incluir braço com heat renormalizado por percentil como quarto arm |
| `< 0,10` | eixo "contagem protegida" é vazio | **parar Meta 1.** Ir para Meta 0.2 |

**Veredito: `>= 0,30` em todos os budgets (0,512 a 0,551).** Seguir para a Meta 1
sem mudanças. O eixo "quantas unidades" existe: com `E*=0,75` fixo, o `beta`
exigido vai de 8,14 (`b=0,05`) a 21,00 (`b=0,50`), spread de 2,6x, enquanto a
contagem protegida varia 1,90x. `beta` e `budget` movem a plasticidade efetiva
por caminhos diferentes, e é essa separação que a ablação explora.

**Ressalva que sobrevive:** a camada 3 (`ratio` 0,018) e a camada 21 (0,009) são
nominais de fato, e as camadas 19, 20, 22 e 23 ficam entre 0,15 e 0,53. O Gate 0
é agregado; por camada, seis das 24 estão em zona ruim. Isso entra na Meta 2.

### Etapa 0.2 — **Descartada** (era: só se `ratio < 0,10`)

O Gate 0 reprovou a premissa (`ratio` medido 0,512-0,551), então a renormalização
por percentil deixa de ser correção obrigatória.

Ela continua sendo um **eixo de ablação legítimo**: se a normalização por quantil
redistribui a proteção sem mudar `E`, ela separa forma de quantidade, que é
exatamente a pergunta da Meta 1. Tratar como braço adicional
(`goals/opcoes_novidade_e_proximos_passos.md`, mecanismo M2), não como
pré-requisito. Se for implementada, troca `h = m / max(selecionado)` por
`h = clamp(m / quantile(selecionado, q), max=1)` com `q` declarado (sugestão
0,90), e exige:

- novo parâmetro em `_SlowHeatImportanceMixin._apply_capacity_budget`, com o
  comportamento atual como default para não quebrar o BERT;
- espelho em `protected_heat` e `protected_heat_scoped`;
- teste comparando os dois normalizadores em importância com outliers;
- rerodar Etapa 0.1 e reavaliar o Gate 0.

**`concentration_ratio` registrado: 0,519 em `b=0,25` (faixa 0,512-0,551).**

Pendência antes de fechar a Meta 1: o critério declarado de
O critério declarado (`docs/protocols/qwen_iso_plasticity_ablation.md`, Anexo A) **não foi aplicado** nesta run — o manifesto
grava `declared_before_run=False`, `minimum_effective_plasticity=None` e
`selection=None`. Isso precisa entrar no protocolo da Etapa 1.1.

---

## Meta 1 — Ablação iso-plasticidade

**Pergunta:** dada a mesma quantidade de plasticidade removida, importa como ela
foi distribuída?

**Estado:** desenho e aritmética prontos
(`docs/protocols/qwen_iso_plasticity_ablation.md`). Runner **implementado e executado**:
`experiments/qwen_iso_plasticity.py` (33.958 bytes, 42 testes em
`tests/test_qwen_iso_plasticity.py`), com 6 runs de calibração em
`results/qwen_iso_plasticity/`. O protocolo foi congelado em
`goals/protocol_iso_plasticity.md` (etapa 1.1 concluída, commit `6461827`).
Pendente: as runs existentes usam 30 passos, revogados em favor de 120 pela
tabela K; Gate 1 não foi registrado; nada foi copiado para `artifacts/`.

### Etapa 1.1 — Congelar o protocolo

Escrever `goals/protocol_iso_plasticity.md` **antes** de qualquer run com
acurácia, contendo:

- `E*` declarados (sugestão 0,75 e 0,50 — decisão do Fachel, registrar);
- budgets dos braços (sugestão 0,05 / 0,25 / 0,50);
- número de tarefas (sugestão 2 para calibração, sequência completa depois);
- seeds, separando calibração e confirmação;
- endpoints: FAA, retenção T1, aquisição T2, drift protegido/plástico;
- hipótese explícita: *se a distribuição não importa, os braços iso-E ficam
  dentro do ruído entre seeds e todos ficam dentro do ruído do LR reduzido.*

Congelado significa: commitado antes da primeira run, e não editado depois de
ver acurácia.

### Etapa 1.2 — Braços e controles

Todos com o mesmo `E*`:

| Braço | Descrição | Papel |
|---|---|---|
| iso-E `b=0,05` | 95% protegidas, beta menor | muitas-fracas |
| iso-E `b=0,25` | 75% protegidas, beta médio | intermediário |
| iso-E `b=0,50` | 50% protegidas, beta maior | poucas-fortes |
| iso-E aleatório | mesma **distribuição** de heat, identidades permutadas | ranking informa? |
| LR reduzido | `lr * E*`, sem máscara | **obrigatório** |
| vanilla | `E=1` | âncora de forgetting |

**Cuidado com o controle aleatório.** `randomize_slowheat_protection` em
`experiments/split_clinc150.py:820` seta `slow_heat = 1,0` nas identidades
sorteadas, ou seja, proteção hard. Isso **não** é um controle iso-E. O controle
correto permuta o vetor de heat preservando a distribuição:

```python
permutation = torch.randperm(state.slow_heat.numel(), generator=generator)
state.slow_heat.copy_(state.slow_heat[permutation])
```

Escrever função nova, não reaproveitar a do BERT, e testar que a distribuição de
heat e `E` ficam idênticas após a permutação.

**O controle de LR reduzido é o mais importante.** Se nenhum braço mascarado
superar redução uniforme de learning rate na mesma plasticidade efetiva, o
mecanismo não faz nada estrutural. É o controle que o diagnóstico de BERT não
tinha.

### Etapa 1.3 — Escrever o runner

Criar `experiments/qwen_iso_plasticity.py`. Reaproveitar de
`experiments/split_clinc150.py`:

- `build_clinc150_tasks`, `load_clinc150_tasks`, `text_task_fingerprint`
- `_evaluate` (linha 659) para a matriz de acurácia
- `capture_parameter_drift_reference` / `summarize_parameter_drift` (848/869)
- `select_replay_examples` (604) — só se replay entrar depois

Escrever novo:

- resolução de `beta` por braço via `iso_plasticity_family_scoped`, **após** a
  consolidação da primeira tarefa (o beta depende da importância medida);
- permutação de heat para o controle aleatório;
- braço de LR reduzido sem máscara;
- manifesto com protocolo, escopo, braços, betas resolvidos e `E` atingido.

Ponto delicado: `beta` é resolvido a partir da importância da tarefa 1, então ele
só existe a partir da fronteira 1->2. Decidir e registrar: `beta` fixo após a
primeira consolidação, ou re-resolvido a cada fronteira. As duas opções são
defensáveis, misturar não.

### Etapa 1.4 — Verificação em CPU

Modelo Qwen2 minúsculo, sem download, como em
`tests/test_slow_heat_qwen.py`. Testar que:

- todos os braços atingem o `E*` declarado dentro de 1e-6;
- o braço aleatório tem distribuição de heat idêntica ao aprendido;
- o braço de LR reduzido não registra máscara nenhuma;
- drift protegido é exatamente zero sob máscara hard;
- o manifesto contém todos os braços e betas.

Depois, mutações: inverter o eixo da máscara, trocar permutação por hard,
ignorar o escopo, aplicar `E*` errado num braço.

### Etapa 1.5 — Run de calibração

2 tarefas, seeds de calibração declaradas na Etapa 1.1. Em GPU: fp16 obrigatório
(Pascal, CC 6.x), batch 2, `CUDA_VISIBLE_DEVICES` numa única placa — as GPUs são
heterogêneas e DDP quebra.

**Pedir autorização ao Fachel antes de qualquer run em GPU.**

### Gate 1

- todos os braços atingem `E*` com erro `< 1e-6`;
- drift protegido exatamente zero nos braços hard;
- `E` medido do braço de LR reduzido igual a `E*` por construção;
- diferenças pareadas por seed reportadas para cada endpoint, com contagem de
  sinais;
- nenhuma decisão de hiperparâmetro tomada após ver acurácia.

---

## Meta 2 — Explicar as camadas anômalas

**Pergunta:** por que as camadas 3 e 21 concentram importância em 1-2% das
unidades quando as outras 22 usam 63-84%?

**Por que importa:** essas camadas recebem `mean_heat` de 0,0013-0,0023, ou seja,
proteção praticamente nula sob escopo local. Se forem funcionalmente relevantes,
há um vazamento de esquecimento que a contagem de protegidas esconde por
completo. Independente da Meta 1.

### Etapas

1. Confirmar que a anomalia não é artefato de seed nem da ordem das tarefas:
   rodar o diagnóstico com 3 seeds e ordem invertida. Só medição, sem acurácia.
2. Inspecionar a distribuição de importância crua dessas camadas: histograma,
   quantis, e se as unidades dominantes são as mesmas entre tarefas.
3. Verificar se coincidem com particularidades conhecidas do Qwen2.5-0.5B
   (camadas de transição, escala de RMSNorm, padrão de atenção).
4. Testar se escopo `hierarchical` corrige a subproteção — ele redistribui cota
   entre camadas e existe exatamente para esse caso.

### Gate 2

Explicação sustentada por medição, ou registro explícito de que continua sem
explicação. Não inventar mecanismo para fechar a lacuna.

---

## Meta 3 — Sequência completa e baselines

**Só depois dos Gates 1 e 2.**

**Pergunta:** o braço vencedor da Meta 1 se sustenta em 10 tarefas e contra
baselines padrão?

### Etapas

1. Sequência completa de 10 domínios do CLINC150, seeds de confirmação
   declaradas antes.
2. Baselines: vanilla, replay (o CLINC150 já tem `select_replay_examples`), EWC.
3. Métricas completas: matriz 10x10, FAA, forgetting médio, BWT. Não reportar
   retenção da tarefa 0 como forgetting global.
4. Custo: pico de memória absoluto (alocado e reservado), tempo, tokens/s.
5. Churn ao longo das 10 fronteiras — o Jaccard de 0,870 após **uma** fronteira
   sugere que o pool livre fossiliza; com 10 tarefas isso deve ficar visível.

### Gate 3

Gate conjunto declarado antes da run: mecanismo (drift exato, ranking útil,
aquisição adequada) **e** retenção. Gate de ranking pode passar com o de
retenção falhando; isso não autoriza benchmark completo.

---

## Meta 4 — Atenção GQA (opcional)

Fora de escopo hoje e provavelmente fora do escopo do trabalho.

O Qwen2.5-0.5B tem 14 heads de query e 2 de key/value, então Q, K e V não
compartilham hidden size e o `SlowHeatAttentionTracker` não porta. Exigiria um
tracker ciente de GQA mapeando cada grupo KV para seus query heads.

Só faz sentido se a Meta 1 der positivo e a limitação a FFN for apontada como
ameaça à validade.

---

## Riscos

| Risco | Probabilidade | Mitigação |
|---|---|---|
| Braços iso-E indistinguíveis por proteção rasa | **refutado para nominalidade** — Gate 0 = 0,512-0,551 | resta a magnitude média baixa (heat mediano 0,047); é o que a Meta 1 mede |
| Mecanismo não supera LR reduzido | média | é resultado negativo publicável, não falha |
| Escolher hiperparâmetro olhando acurácia | média — pressão real | protocolo congelado e commitado antes; o critério declarado ainda não foi aplicado (manifesto: `selection=None`) |
| Cauda pesada deixa `beta` sem sentido físico | baixa — PR por camada é 63-84% | WARNING automático se PR/N < 5% |
| Memória em GPU insuficiente | baixa | ~8,3 GB estimado, ainda não medido; smoke em GPU antes |
| Reaproveitar `randomize_slowheat_protection` por engano | média | documentado na Etapa 1.2 |

## Perguntas abertas para o Fachel

1. `E*` a declarar (sugestão 0,75 e 0,50).
2. `beta` fixo após a primeira consolidação ou re-resolvido a cada fronteira.
3. Seeds de calibração e de confirmação.
4. Meta 2 antes ou depois da Meta 1 — as camadas anômalas podem invalidar a
   leitura da ablação, mas investigar primeiro atrasa o resultado principal.
5. Qual é o núcleo da novidade: teorema do piso, protocolo metodológico, método
   novo sobre F1/F2/F5, ou combinação. Ver
   `goals/opcoes_novidade_e_proximos_passos.md`, seção 5.

## Perguntas já respondidas

- Gate 0: `concentration_ratio` de 0,512 a 0,551 (todos os budgets) -> conjunto
  protegido é funcional, seguir para a Meta 1 sem mudanças.

---

# Anexo B — Plano de execução P6/P1/P2/P3/P4

Este anexo era `goals/plano_execucao_p6_p1_p2_p3.md`. O plano foi em boa parte
aplicado, então as instruções passo a passo viraram histórico; o texto verbatim
está no histórico Git (commit `99e0190` e anteriores). O que permanece vivo está
consolidado abaixo: **estado**, **pendentes**, **pitfalls** e **riscos**.

## B.1. Estado verificado em 22 de setembro de 2026 (HEAD `4265b1c`)

| Item | Estado |
|---|---|
| P6.1 flag `--domains` | implementado (`qwen_capacity_diagnostic.py:88`) |
| P6.2 6 runs de medição | executado (`results/qwen_layer_anomaly/`) |
| P6.3 `dominant_unit_overlap` + perfil de importância | implementado (`capacity_calibration.py:185,137`) |
| P6.3 run com `--capacity-scope hierarchical` | **não executado** (todos os manifestos gravam `local`) |
| P6.4 `docs/qwen_layer_anomaly.md` + Gate 2 | **não entregue** |
| P1 congelar protocolo | executado (commit `6461827`) |
| P2 runner `qwen_iso_plasticity.py` | implementado e executado |
| P3 testes + mutações | 42 testes existem; relatório de mutação não escrito |
| P4 run de calibração | executado com **30 passos**, valor depois revogado para 120 |
| P4.3 copiar manifestos para `artifacts/` | **não executado**; nada versionado |
| P4.5 Gate 1 | **não registrado** |

Desvio registrado: os manifestos `hard_seed*` contêm um braço `hard` que não
consta da seção E do protocolo congelado e cuja adição (commit `095b1c8`) não
teve linha na tabela K antes da run. Ver a seção "Desvio registrado a
posteriori" em `goals/protocol_iso_plasticity.md`.

## B.2. Pendentes, com o detalhe necessário para executá-los

**P6.3 — escopo hierárquico.** Rodar uma seed com
`--capacity-scope hierarchical` e comparar `concentration_ratio` por camada com
o escopo local, para decidir se a redistribuição de cota corrige a subproteção
das camadas anômalas.

**P6.4 — `docs/qwen_layer_anomaly.md`.** Formato de
[`bert_slowheat_diagnostic_results.md`](../docs/results/bert_slowheat_diagnostic_results.md):
pergunta, protocolo, tabela por seed e por ordem, leitura e **o que continua sem
explicação**. Os dados já estão em `results/qwen_layer_anomaly/` (7 manifestos,
2 ordens x 3 seeds). Fecha o Gate 2.

Hipótese a confirmar ou descartar: a queda de PR no fim da rede é efeito de
profundidade (a magnitude de `|z dL/dz|` cai perto da cabeça porque a loss é
dominada pelos logits), e L3/L21 são outra coisa, sem relação com profundidade.
Se for isso, o tratamento separado vale só para L3/L21.

**P3 — relatório de mutação.** As sete mutações previstas e o teste que cada uma
deve quebrar:

| Mutação | Teste que deve quebrar |
|---|---|
| inverter o eixo da máscara | drift protegido exatamente zero |
| trocar permutação por proteção hard | distribuição de heat idêntica |
| ignorar `capacity_scope` | braço atinge `E*` dentro de 1e-6 |
| aplicar `E*` errado num braço | braço atinge `E*` dentro de 1e-6 |
| saturar `beta` em vez de descartar budget inalcançável | conteúdo do manifesto |
| esquecer `seen_classes` na avaliação | avaliação |
| registrar máscara no braço de LR reduzido | `mask_bindings()` vazio |

Mutação sobrevivente é lacuna de teste ou equivalência provada. Investigar e
documentar qual, como já foi feito para as duas equivalências de
`test_capacity_calibration.py`.

Cuidado conhecido: atenção causal mais pooling do último token não-pad já zeram
o gradiente de padding, então um teste de invariância a padding passa mesmo com
a máscara de validade removida. Testar a máscara no tracker diretamente.

**P4.3 e Gate 1.** Copiar os manifestos para `artifacts/` (o repo ignora
`results/`) e registrar o Gate 1 conforme a seção I do protocolo. Reprovado é
resultado registrado, não obstáculo.

## B.3. Pitfalls que o runner tem de respeitar

Referência viva, vale para qualquer run futura:

- carregar em fp32 e passar `fp16=True`, **ou** carregar fp16 com `fp16=False`.
  Nunca os dois: o scaler espera pesos mestres fp32. (O protocolo depois migrou
  o treino para fp32 por outro motivo — ver tabela K.)
- `gradient_checkpointing` conflita com forward hooks. Desligado;
- nenhum `device_map="auto"`: uma GPU só, DDP desabilitado (GPUs heterogêneas);
- `freeze_unbound_parameters=True` congela o envelope sem binding. A cabeça
  `score` é aleatória e **precisa** ficar plástica; ela é exposta por
  `exempt_parameter_names()` e contada separada de mascarados;
- um binding por parâmetro. Quando produtor e consumidor protegem o mesmo peso,
  combinar por mínimo, não registrar duas vezes;
- alocação de estado ativo: só para máscaras habilitadas;
- `evaluate_all`/`_evaluate`: `seen_tasks` é o terceiro posicional, `output_dir`
  o quarto. Não usar keyword `tasks=`;
- `softmax` de Class-IL: `_mask_unseen_logits` com `seen_classes` por tarefa;
- **não** reaproveitar `_find_slowheat_model` de `split_clinc150.py`: é tipada
  para `SlowHeatBertForSequenceClassification`.

O controle iso-E correto é permutar o vetor de heat inteiro, zeros incluídos —
isso preserva a contagem de protegidas e o multiconjunto de valores, logo `E`
fica idêntico por construção. Proteger identidades sorteadas com `hard`
(`split_clinc150.py:820`) **não** é iso-E e não serve como controle.

Ordem de treino por braço:

```text
para cada tarefa t:
  model.train()
  para cada step: forward, backward, optimizer.step()
  model.eval(); _evaluate em todas as tarefas vistas
  se t < T-1:
     model.consolidate(strategy="max")
     se policy == "per_boundary": re-resolver beta do braço em E(beta) = E*
     model.register_plasticity_masks(optimizer, hard=False)
     capturar drift de referência para a próxima tarefa
```

Detalhe que decide a leitura: a resolução de `beta` depende da importância da
primeira tarefa, então `beta` só existe a partir da fronteira 1->2. Com
`policy="first"` resolve uma vez e congela. Reportar os dois ou declarar um;
não misturar.

## B.4. Riscos de execução

| Risco | Onde aparece | Mitigação |
|---|---|---|
| `_evaluate` mudar de assinatura e quebrar o runner | P2 | confirmar antes; o teste de P3 cobre |
| Anomalia de L3/L21 ser artefato de seed | P6 | é o que P6 mede; os dados existem e a leitura falta |
| Ordem das tarefas mudar a leitura da Meta 1 | P6, P4 | ordem declarada no protocolo; P6 mede o efeito |
| Estimativa de memória otimista | P4 | **resolvido**: smoke em GPU mediu 5,447 GiB de pico contra ~8,3 GB estimados |
| Braços indistinguíveis do LR reduzido | P4, Gate 1 | resultado negativo publicável; o controle é obrigatório |
| Nome `SlowHeat` herdar o endpoint negativo do BERT | P9 | decidir o nome antes de escrever a Meta 3 |
| Manipulação de `slow_heat` pós-consolidação divergir do caminho de produção | P2, P3 | comparar `mask_bindings()` com e sem o hook de produção |

