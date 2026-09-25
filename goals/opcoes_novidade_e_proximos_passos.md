# Opções de novidade e próximos passos

> Documento de trabalho. Responde a uma pergunta que o `qwen_heat_roadmap.md` não
> faz: **onde exatamente está o método novo?** O roadmap é um plano de
> falsificação de um mecanismo existente. Nenhuma das Metas 0-4 produz um método.
> Este arquivo lista as opções de núcleo de novidade, os mecanismos candidatos e
> a ordem de execução.

Última atualização: 22 de setembro de 2026.

---

## 1. Diagnóstico: o roadmap atual não leva a um método novo

`qwen_heat_roadmap.md` responde "a proteção seletiva tem efeito estrutural além
da redução de capacidade que ela causa?". É uma pergunta de mecanismo, com
resposta possível sim ou não. As Metas 0 a 4 medem, ablam e comparam — não
introduzem algoritmo.

O próprio manuscrito já fechou as portas fáceis. Seção 7:

> No claim of being the first method to use neuron importance, MAX masks,
> lateral inhibition or importance-dependent plasticity is justified.

E seção 9 lista como **não sustentado**: "SlowHeat outperforms established
continual-learning baselines". Ou seja, hoje o artigo não tem nem prioridade
nem eficácia. Novidade tem que ser construída, não herdada — e há três ativos
reais no repositório que podem sustentá-la.

## 2. Os três ativos que já existem (e o que cada um vale)

| Ativo | Onde | Força | Limite |
|---|---|---|---|
| **A1 — Teorema `E(beta,b) >= (N-P)/N`** | `docs/protocols/qwen_iso_plasticity_ablation.md` §3 | Analítico, independente de dados, limite justo em `beta -> infinito`, verificado em 3000 casos | É um resultado sobre *quantidade*. Sozinho não é método. |
| **A2 — Construção iso-plasticidade** | `experiments/capacity_calibration.py` (`*_scoped`, `solve_strength_for_plasticity_scoped`) | Constrói braços de custo pareado em plasticidade **efetiva**, por bisseção, sem busca em grade | Precisa de uma run para virar afirmação |
| **A3 — Contrato de otimizador** | `src/dual_heater/optim.py`, `docs/mechanisms/optimizer_semantics.md` | Prova que escalar gradiente bruto não é escalar o passo sob Adam (`m/sqrt(v)` cancela `c`); mascara o delta nativo incluindo weight decay | É correção de bug conceitual, não mecanismo |

Nenhum deles é "o método novo" isoladamente. **A decisão que trava todo o resto é
em qual deles o artigo se ancora** (seção 6, item 5).

## 3. Insumo: as falhas já medidas

Um método novo precisa de um problema. Estes estão medidos no Qwen2.5-0.5B real
(`results/qwen_capacity_diagnostic/manifest.json`, 2 tarefas, seed 0) e
verificados nesta revisão:

| ID | Falha | Número |
|---|---|---|
| **F1** | Fossilização do conjunto protegido. `max` é monótono, então o pool livre converge para "unidades que nunca foram úteis". | Jaccard 0,870/0,890 após **uma** fronteira; turnover do pool livre 0,296-0,345 |
| **F2** | Magnitude de importância varia 72x entre camadas, mas cada camada normaliza pelo próprio máximo. Satura 22 camadas e subprotege as outras. | `mean_heat` por camada de 0,0013 (L21) a 0,0926 (L14) |
| **F3** | A anomalia não são 2 camadas. O PR cai sistematicamente no fim da rede. | PR: L19 2493, L20 2269, L22 1212, L23 628, L3 95, L21 40 — contra 3000-4105 nas demais |
| **F4** | A proteção é **rasa**, não nominal. | heat médio por unidade protegida 0,0591 e **mediano 0,0466** (`b=0,25`) -> metade das protegidas retém >= 67% do LR em `beta=10,75` |
| **F5** | `beta` e `budget` não são o mesmo botão, e `beta` é invertível. | `E*=0,75` -> beta 8,14 / 8,61 / 10,75 / 21,00 em `b`=0,05 / 0,10 / 0,25 / 0,50 (escopo local) |
| **F6** | A magnitude alta mora sempre nas mesmas unidades. | 142 unidades têm `h>=0,5` em **todos** os budgets de 0,10 a 0,50; 12.693 têm `h>=0,1` nesses três |

**F4 derruba a premissa da Meta 0, não o contrário.** Ver seção 4.

### 4. Gate 0: medido, e a resposta é que não há problema

A Etapa 0.1 foi executada em 22/09 13:48 (`results/qwen_capacity_diagnostic/`).
O manifesto anterior não tinha a chave `concentration`, porque a instrumentação
(`capacity_calibration.py:570`, ligada em `qwen_capacity_diagnostic.py:108,306,403`)
foi escrita depois dele. A run nova reproduziu perfeitamente as quantidades por
camada e o sinal agregado do manifesto antigo, então a única diferença é a
medição que faltava.

```text
budget | protegidas | efetivas | concentration_ratio | h>=0,1 | q50 de h
0,10   |   105.048  |  53.751  |       0,512         | 12.693 |  0,043
0,25   |    87.552  |  45.460  |       0,519         | 12.693 |  0,047
0,50   |    58.368  |  30.746  |       0,527         | 12.693 |  0,054
0,75   |    29.184  |  15.586  |       0,534         | 10.104 |  0,070
0,90   |    11.664  |   6.428  |       0,551         |  5.733 |  0,098
```

**Gate 0 = "conjunto protegido é funcional" em todos os budgets.** O corte em
0,30 não é aproximado: o medido é ~1,7x o corte. Seguir para a Meta 1 sem
mudanças; a Etapa 0.2 (renormalização por percentil) está descartada como
pré-requisito e passa a ser braço opcional (M2).

O que isso muda na leitura:

- A hipótese de "3.648 protegidas por camada são nominais" está **refutada** no
  agregado. O que sobra é **F4**: a proteção existe, mas é de magnitude baixa.
- O eixo "quantas unidades" tem contraste real: com `E*` fixo, o `beta` exigido
  varia 2,6x entre `b=0,05` e `b=0,50`, contra 1,90x de contagem.
- **O Gate 0 é agregado; por camada ele não vale.** L3 (ratio 0,018) e L21
  (0,009) são nominais de fato, e L19/L20/L22/L23 ficam entre 0,15 e 0,53. Seis
  das 24 camadas estão em zona ruim. Isso é matéria-prima para a Meta 2 e para
  M2, não para a Meta 1.
- F6 diz que o contraste entre braços não vem das unidades de magnitude alta
  (sempre as mesmas 142), vem da cauda `0,01 < h < 0,1`: 92.606 unidades em
  `b=0,10` contra 52.233 em `b=0,50`.

Retificação do que este documento dizia antes de P0: a reconstrução via
identidade de razão de participação deu `concentration_ratio >= 0,455` como
limite inferior. O medido é 0,512-0,551. O método estava certo e conservador.

## 5. Onde a novidade pode estar: quatro opções

### Opção A — Artigo metodológico/teórico

**Reivindicação:** um protocolo de avaliação controlado por capacidade para
métodos de proteção seletiva, com piso analítico. Contribuição = A1 + A2 + a
obrigatoriedade do controle de LR reduzido. Aplica-se a EWC, SI, MAS, SLNID,
não só ao SlowHeat.

- **Precisa:** Gate 0 fechado, protocolo congelado, uma run de calibração e uma
  demonstração de que o protocolo muda a conclusão de pelo menos um método
  publicado (o diagnóstico de BERT é o exemplo pronto: ele confundiu quantidade
  com distribuição).
- **Risco:** baixo. Não depende de ganhar nada. Publica com resultado negativo.
- **Encaixe:** conferência menor, ou workshop de avaliação/reprodutibilidade.

### Opção B — Método novo que corrige F1/F2/F5

**Reivindicação:** uma regra de consolidação e alocação cujo hiperparâmetro é
plasticidade retida medida, não força de proteção arbitrária. Mecanismos
candidatos na seção 6.

- **Precisa:** os três mecanismos implementados, a ablação iso-E intacta, a run
  de 2 tarefas e depois 10 domínios contra vanilla/replay/EWC.
- **Risco:** médio-alto. Pode perder para replay — como o diagnóstico de BERT já
  perdeu. Mas aí volta para A como artigo negativo.
- **Encaixe:** conferência de CL (Continual Learning workshop, CLVision) ou
  venue principal se a Meta 3 for limpa.

### Opção C — Novidade de combinação (a posição atual do manuscrito)

**Reivindicação:** primeiro método de proteção em nível de neurônio
*optimizer-aware*, com budget de capacidade explícito, máscara fatorizada nos
dois endpoints e piso de plasticidade provado.

- **Precisa:** exatamente o roadmap atual, mais baselines na sequência completa.
- **Risco:** alto de ser lido como incremental. "Combinação exata" é a
  formulação mais fraca das três e a seção 7 já admite isso.
- **Encaixe:** venue menor.

### Opção D — Extensão GQA/atenção

Fora de escopo. Só como resposta a "ameaça à validade: só FFN" depois de B dar
positivo.

## 6. Mecanismos candidatos (o insumo da Opção B)

Três regras, cada uma atacando uma falha medida. Nomes provisórios.

### M1 — Consolidação com decaimento (leaky-max)

```text
h_i <- max(gamma^d * h_i, h_task,i)      # d = distância em fronteiras, gamma <= 1
```

- **Ataca:** F1. `max` puro é monótono (HAT/MAX assumem que isso é estritamente
  bom). Se o pool livre fossiliza — e o Jaccard de 0,87 após *uma* fronteira diz
  que fossiliza — a monotonicidade é um defeito, não uma garantia.
- **Novo porque:** contradiz a premissa monotônica da linha HAT/MAX com medição
  de churn, não com argumento.
- **Teste mínimo (CPU):** com `gamma=1` a máscara é idêntica à produção
  (equivalência exata); com `gamma<1` o Jaccard entre fronteiras cai
  monotonicamente em `gamma`.
- **Falsificador:** Jaccard fica em ~0,87 para todo `gamma<1`: a fossilização
  não vem da recência, vem do ranking. Resultado negativo útil.

### M2 — Normalização por quantil, com equivalência de `E`

```text
h_i = clamp(m_i / quantile_q(m_selecionado), max=1)      # q declarado, sugestão 0,90
```

- **Ataca:** F2 e F4. Hoje `h = m / max` põe quase toda a massa de heat em
  poucas unidades de cada camada (L3 PR 95 de 4864). Renormalizar por quantil
  redistribui a proteção **sem mudar `E`**, se `beta` for re-resolvido.
- **Novo porque:** separa "quantas unidades" de "quão forte" no nível da
  normalização, e a equivalência de `E` permite provar que qualquer diferença
  observada não é efeito de capacidade. Sem isso, o resultado é sempre
  atribuível a "protegeu mais".
- **Teste mínimo:** os dois normalizadores sobre a mesma importância com
  outliers: `E` idêntico a 1e-9 após re-resolver `beta`; `concentration_ratio`
  mais alto com quantil; `beta` necessário menor.
- **Falsificador:** com `E` pareado, os dois braços ficam indistinguíveis — então
  a forma da distribuição não importa e a Opção B perde o eixo principal.

### M3 — Controlador de plasticidade declarada

```text
beta_t = solve_strength_for_plasticity_scoped(  E(beta) = E*(t)  )
E*(t): agendado (linear ou geométrico) ao longo das fronteiras, declarado antes
```

- **Ataca:** F5 e a ausência de interpretação de `beta`. Hoje `slow_strength` é
  um número sem unidade física. Com M3 o hiperparâmetro do método passa a ser
  "fração de plasticidade retida", que é medível antes de olhar acurácia — e
  portanto legal sob a regra 1 do roadmap.
- **Novo porque:** é proteção calibrada por uma propriedade medida da geometria
  do update, não por tuning.
- **Teste mínimo:** para uma grade de `E*`, `effective_plasticity` do modelo
  treinado bate o alvo em 1e-6 após a consolidação; monotonicidade de `beta` em
  `E*`.
- **Falsificador:** o controlador é indistinguível de um agendamento escalonado
  de LR (`lr_t = lr * E*(t)` sem máscara). Se for, o braço de LR reduzido do
  roadmap já mata o mecanismo e a Opção B vira Opção A.

### M4 — (reaproveitamento) FastHeat no lado plástico

`FastHeat` já existe (`src/dual_heater/fast_heat.py`) e faz competição divisiva
nas ativações. F# o par plástico natural de M1-M3. Não é novidade isolada, mas
fecha a história "um mecanismo para estabilidade, outro para plasticidade".

**Recomendação:** M1 + M3 como o método; M2 como eixo de ablação (é o que mais
aparece na literatura vizinha); M4 só se sobrar espaço.

## 7. Próximos passos, em ordem

Gates conforme `qwen_heat_roadmap.md`. Nada com acurácia antes de P1.

| # | Passo | Onde | Custo | Depende |
|---|---|---|---|---|
| **P0** | ~~Re-rodar Etapa 0.1~~ **Concluído 22/09 13:48.** `concentration_ratio` 0,512-0,551; Gate 0 = funcional. Registrado em `qwen_heat_roadmap.md`. | CPU | ~5 min | feito |
| **P1** | Escrever `goals/protocol_iso_plasticity.md`: `E*`, budgets, seeds, endpoints, hipótese. **Commitar antes da primeira run com acurácia.** | CPU | texto | P0 |
| **P2** | Escrever `experiments/qwen_iso_plasticity.py`. Controle de permutação de heat **novo** (não `randomize_slowheat_protection`, que é hard e não iso-E), braço de LR reduzido sem máscara, manifesto com protocolo/escopo/betas/`E` atingido. | CPU | código | P1 |
| **P3** | Testes + mutações. As quatro do roadmap (eixo da máscara, permutação->hard, escopo ignorado, `E*` errado) **mais** as dos mecanismos novos: `gamma=1` == produção, normalizador quantil->max, agendamento de `E*` congelado. | CPU | código | P2 |
| **P4** | Run de calibração, 2 tarefas, seeds de calibração. fp16, batch 2, uma GPU só (heterogêneas, DDP quebra). **Autorização do Fachel.** | GPU | horas | P3 |
| **P5** | Registrar Gate 1, congelar conclusão, atualizar `qwen_heat_roadmap.md` com os números. | CPU | texto | P4 |
| **P6** | Meta 2 reduzida: 3 seeds, ordem de tarefas invertida, PR por profundidade (F3). Barata, só medição, e F3 muda como a Meta 1 é lida. | CPU | ~1 h | P0 |
| **P7** | Meta 1 completa + M1/M2/M3 como braços adicionais. | GPU | dias | P5 |
| **P8** | Meta 3: 10 domínios, baselines (vanilla, replay, EWC), matriz 10x10, FAA, forgetting, BWT, churn por fronteira, pico de memória. | GPU | dias | P7 |
| **P9** | Pivotar o manuscrito: reescrever §1, §7, §9 para a reivindicação escolhida na seção 6, item 5 abaixo. | CPU | texto | P7 |

## 8. Decisões que só o Fachel fecha

1. **Qual é o núcleo da novidade: A, B, C ou D?** Muda P1, P2 e P9 inteiros.
2. **`E*`:** sugestão 0,75 e 0,50 — os dois já estão resolvidos no manifesto
   (beta 8,14-21,00 e 31,17-58,25 em escopo local).
3. **`beta` fixo após a primeira consolidação ou re-resolvido a cada fronteira:**
   re-resolvido é o que habilita M3 e o agendamento de `E*`. Fixo é mais simples
   de auditar. Escolher um, não misturar.
4. **Seeds:** sugestão calibração {0,1,2}, confirmação 10 seeds declaradas em P1.
5. **P6 (meta 2) antes ou depois de P4:** P0 já mostrou `ratio` abaixo de 0,30 em
   seis camadas (L3 0,018, L21 0,009, L23 0,148, L22 0,270, e L19/L20 ~0,5 com PR
   caindo). Sugestão: fazer P6 **antes** de P4, em 3 seeds e ordem invertida —
   é medição pura, sem acurácia, e determina se as camadas finais precisam de
   tratamento separado antes de montar os braços da Meta 1.
6. **Qual nome o método novo recebe.** `SlowHeat` já está comprometido com o
   endpoint fechado do BERT. Se B for o caminho, o método precisa de nome novo
   para não herdar o resultado negativo.

## 9. Riscos específicos da rota "método novo"

| Risco | Probabilidade | Mitigação |
|---|---|---|
| Novidade lida como incremental (Opção C) | alta | ancorar em A1 (teorema) ou M3 (hiperparâmetro medido), não na combinação |
| M1/M2/M3 todos indistinguíveis do controle de LR reduzido | média | é resultado negativo publicável via Opção A; o controle é obrigatório e não opcional |
| Reivindicar primazia em MAX / importância por neurônio | alta — pressão de escrita | seção 7 do manuscrito já proíbe; atualizar a §7 junto com a §9 |
| Herdar o resultado negativo do BERT pelo nome `SlowHeat` | média | nome novo (item 6) |
| Venue principal exigir ganho sobre replay | alta | o diagnóstico de BERT já mostrou que esse caminho está fechado; planejar para workshop/venue menor |

## 10. O que não fazer

- Não comparar `beta=3` contra `beta=30`: remove quantidade diferente de
  plasticidade, e a diferença de aquisição é explicável por capacidade.
- Não escolher hiperparâmetro olhando acurácia (regra 1 do roadmap).
- Não usar `PR` agregado para nada que dependa de normalização — foi exatamente
  esse o bug de escopo de ~2 ordens.
- Não reaproveitar `randomize_slowheat_protection` (`split_clinc150.py:820`):
  ele põe `slow_heat = 1,0`, proteção hard, e não é um controle iso-E.
- Não rodar em GPU sem autorização.
- Não tratar o manifesto atual como válido: ele é anterior ao código de
  `heat_concentration`.
