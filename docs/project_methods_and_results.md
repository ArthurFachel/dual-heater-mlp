# DualHeat / Functional SlowHeat

## Visão do projeto, catálogo de métodos e análise dos resultados

**Data da análise:** 17 de agosto de 2026  
**Projeto:** `dual-heater` v0.2.0  
**Benchmark dos resultados fornecidos:** Split-MNIST class-incremental  
**Estado científico:** protótipo de pesquisa; os resultados sustentam conclusões exploratórias, não uma alegação de estado da arte.

---

## 1. Resumo executivo

O projeto investiga mecanismos de plasticidade em nível de neurônio para aprendizagem contínua. A ideia central do método atual, **Functional SlowHeat**, é identificar unidades importantes para tarefas anteriores e reduzir seletivamente suas atualizações, mantendo uma fração explícita da rede disponível para aprender tarefas novas.

O arquivo analisado contém 14 linhas não vazias, correspondentes a **13 métodos únicos**. A configuração `slowheat_replay_hidden_beta_30_budget_0.25` aparece duas vezes com métricas preditivas idênticas e apenas uma pequena diferença de tempo; ela foi tratada como duplicata de exportação, e não como uma seed adicional.

Principais resultados:

- **SlowHeat + DER++** obteve a maior acurácia média final: **84,708%**.
- **DER++** foi o melhor baseline sem SlowHeat: **81,984%**.
- SlowHeat + DER++ superou DER++ em **2,724 pontos percentuais** de acurácia e reduziu o forgetting em **4,360 p.p.**, uma redução relativa de **20,8%**.
- O mesmo SlowHeat acrescentado ao replay simples produziu ganho bem menor: **+0,300 p.p.** de acurácia e **−0,760 p.p.** de forgetting.
- O custo observado do SlowHeat foi alto: aproximadamente **69% mais tempo** que Replay ou DER++, apesar de apenas **0,20% de FLOPs estimados adicionais**. Isso sugere overhead de implementação, movimentação de estado e aplicação de máscaras que a contagem teórica de FLOPs não representa bem.
- **ER-ACE** teve o menor forgetting, **3,900%**, e BWT positivo, mas sua acurácia final de **71,128%** ficou abaixo de Replay, DER++ e suas combinações com SlowHeat.
- Vanilla, EWC, SI, LwF calibrada e A-GEM terminaram próximos de **20% de acurácia class-incremental**, indicando falha severa no cenário global de dez classes.
- A-GEM e LwF ainda mantiveram cerca de **90% de acurácia task-aware**. O problema principal desses dois métodos foi, portanto, a competição/calibração entre tarefas na cabeça global, e não apenas a perda da discriminação dentro de cada par de classes.
- O patamar de aproximadamente 20% não é o acaso uniforme de dez classes, que seria 10%. Ele é compatível com um modelo que termina dominado pela tarefa mais recente e acerta essencialmente um dos cinco pares.
- As larguras de IC95% do CSV usam aproximação normal e são compatíveis com **cinco seeds**. Com `n=5`, esses intervalos são otimistas; a inferência principal deve usar diferenças pareadas por seed e intervalo de Student ou bootstrap pareado.

Conclusão central: **o resultado mais promissor é SlowHeat + DER++, mas ele ainda é exploratório**. O candidato confirmatório congelado no repositório é SlowHeat + Replay contra Replay, contraste no qual o efeito observado neste CSV é pequeno e incerto.

---

## 2. Problema de pesquisa

Em aprendizagem contínua, um modelo aprende uma sequência de tarefas sem poder reiniciar seus parâmetros. O desafio é equilibrar:

- **estabilidade:** preservar o que já foi aprendido;
- **plasticidade:** continuar adquirindo conhecimento novo.

Se todos os parâmetros permanecerem muito plásticos, tarefas novas sobrescrevem soluções antigas. Se a proteção for excessiva, o modelo deixa de adquirir as tarefas seguintes. O projeto explora um meio-termo: estimar importância por unidade, consolidar essa evidência nas fronteiras de tarefa e modular o update efetivo do otimizador.

---

## 3. Evolução conceitual do projeto

### 3.1 DualHeat legado

`DualHeat` é o mecanismo histórico do projeto. Ele combina dois estados por neurônio:

- **fast heat:** estatística transitória usada para inibição lateral durante o treino;
- **slow heat:** média de importância que reduz gradientes de unidades consideradas importantes.

O fast heat atua apenas em modo de treino e modifica a saída por um fator dependente da atividade das outras unidades. O slow heat pode usar magnitude de ativação ou sensibilidade `|ativação × gradiente|`.

Limitações que motivaram a revisão:

- magnitude de ativação isolada não é invariável a reparametrizações que preservam a função;
- proteção apenas por linha não preserva completamente o caminho de um neurônio;
- multiplicar o gradiente bruto não equivale a multiplicar o update final do AdamW;
- não há orçamento explícito de capacidade livre;
- a inibição rápida altera a função durante treino, mas não durante avaliação.

DualHeat permanece no código para pesquisa histórica e ablações, mas **não é o método principal avaliado no CSV**.

**Proveniência:** DualHeat é uma proposta própria deste repositório e não possui, até esta revisão, artigo externo publicado. As referências conceitualmente relacionadas são o critério de Taylor de primeira ordem de [Molchanov et al. (ICLR 2017)](https://research.nvidia.com/publication/2017-04_pruning-convolutional-neural-networks-resource-efficient-inference) e os mecanismos de consolidação seletiva de [EWC](https://doi.org/10.1073/pnas.1611835114) e [SI](https://proceedings.mlr.press/v70/zenke17a.html); isso não significa que esses artigos descrevam DualHeat.

### 3.2 Functional SlowHeat

Functional SlowHeat é a direção atual. Para a pré-ativação `z_i` do neurônio `i` e a loss `L`, a utilidade instantânea é:

```text
u_i = sum_amostras |z_i * dL/dz_i|
u_normalizado,i = u_i / (média_j(u_j) + epsilon)
```

Essa estatística é acumulada durante a tarefa por uma média móvel. Em uma fronteira de tarefa, a evidência é consolidada por `max`, `mean` ou `sum`.

A regra principal usa:

```text
importance_memory_i <- max(importance_memory_i, task_ema_i)
```

Depois da consolidação, as unidades são ranqueadas. O número de unidades protegidas respeita:

```text
protected_count <= floor((1 - plasticity_budget) * número_de_unidades)
```

Com `plasticity_budget=0.25`, pelo menos 25% das unidades de cada camada permanecem totalmente livres.

Para uma unidade consolidada, o fator plástico é:

```text
m_i = 1 / (1 + beta * slow_heat_i)
```

Com `beta=30` e `slow_heat=1`, o update aplicado é aproximadamente `1/31` do update nativo.

**Fonte do método:** Functional SlowHeat é uma contribuição própria documentada no [manuscrito técnico local](../article/manuscript.md), ainda não uma publicação revisada por pares. O sinal `|z · dL/dz|` tem relação com critérios de saliência por Taylor de primeira ordem, como [Molchanov et al. (ICLR 2017)](https://research.nvidia.com/publication/2017-04_pruning-convolutional-neural-networks-resource-efficient-inference), mas a normalização, consolidação, budget, proteção fatorada e semântica do otimizador são definições locais.

### 3.3 Proteção fatorada de caminhos

Para o peso `W_l[i,j]`, o otimizador combina a importância do neurônio de destino com a do neurônio de origem:

```text
M_l[i,j] = min(m_destino_l[i], m_origem_l-1[j])
```

Assim, um neurônio importante protege tanto sua linha de entrada quanto suas colunas de saída na camada seguinte. Isso preserva caminhos funcionais sem armazenar uma importância para cada peso.

### 3.4 Otimizadores SlowHeat

`SlowHeatAdamW` e `SlowHeatSGD` deixam o otimizador nativo calcular momentum, precondicionamento e weight decay. Em seguida, aplicam a máscara ao delta completo:

```text
delta_nativo = theta_depois_do_step - theta_antes_do_step
theta_novo = theta_antes_do_step + M * delta_nativo
```

A política padrão, `follow_update`, também interpola estados tensoriais como `exp_avg`, `exp_avg_sq` e momentum. A política `native` mascara os parâmetros, mas deixa o estado interno seguir a trajetória nativa; ela existe como ablação.

Essa semântica é importante porque um fator multiplicativo persistente pode ser parcialmente cancelado pela normalização do AdamW. O weight decay desacoplado também pode mover um parâmetro mesmo quando seu gradiente bruto foi reduzido.

**Base do otimizador:** [Loshchilov e Hutter, “Decoupled Weight Decay Regularization” (ICLR 2019)](https://arxiv.org/abs/1711.05101). `SlowHeatAdamW` e `SlowHeatSGD` são wrappers próprios do projeto, não algoritmos descritos naquele artigo.

### 3.5 SlowHeat em camadas lineares, convolucionais e MLPs

O pacote implementa:

- `SlowHeatLinear`: utilidade por unidade de saída;
- `SlowHeatConv2d`: utilidade por canal de saída;
- `SlowHeatMLP`: composição de camadas SlowHeat com proteção opcional da cabeça de saída;
- adaptação de orçamento baseada em aquisição de validação;
- consolidação `max`, `mean` e `sum`;
- hard-freeze exato como controle.

### 3.6 DualHeat-LoRA experimental

`DualHeatLoRALinear` mantém o peso-base congelado e treina um delta de baixo posto `B @ A`. A proteção é aplicada ao tensor de saída do adaptador. Como `lora_A` é compartilhada entre todas as saídas, o mecanismo **não garante proteção independente por saída**. Ele deve ser tratado como protótipo, não como extensão validada do Functional SlowHeat.

**Base LoRA:** [Hu et al., “LoRA: Low-Rank Adaptation of Large Language Models” (ICLR 2022)](https://arxiv.org/abs/2106.09685). A combinação de LoRA com DualHeat é uma adaptação local e não faz parte do método publicado por Hu et al.

---

## 4. Protocolo experimental principal

### 4.1 Split-MNIST class-incremental

O protocolo principal separa MNIST em cinco tarefas:

```text
T1: dígitos 0 e 1
T2: dígitos 2 e 3
T3: dígitos 4 e 5
T4: dígitos 6 e 7
T5: dígitos 8 e 9
```

Configuração da suíte de baselines:

| Item | Valor |
|---|---:|
| Arquitetura | `784 → 256 → 128 → 10` |
| Cenário | class-incremental |
| Task ID na avaliação principal | não |
| Task ID na avaliação diagnóstica | sim, restringindo a decisão ao par da tarefa |
| Épocas por tarefa | 10 |
| Batch size | 128 |
| Treino por classe | 1.000 exemplos |
| Validação por classe | 200 exemplos |
| Teste por classe | 500 exemplos |
| Otimizador | AdamW |
| Learning rate | `1e-3` |
| Weight decay | `1e-4` |
| Replay | 20 amostras por classe; minibatch 64 |
| SlowHeat | `beta=30`, budget plástico `0.25`, `follow_update` |
| DER++ | `alpha=0.5`, `beta=0.5` |
| EWC | `lambda=100` |
| SI | `lambda=1`, `epsilon=0.1` |
| Distillation/LwF | temperatura 2 |

Dentro de cada seed, os métodos recebem inicialização treinável idêntica, os mesmos dados, os mesmos splits e a mesma ordem de minibatches. As classes futuras são mascaradas antes de aparecerem. SlowHeat recebe a fronteira de tarefas para consolidar importância.

O enquadramento task-, domain- e class-incremental segue a taxonomia de [van de Ven e Tolias, “Three scenarios for continual learning”](https://arxiv.org/abs/1904.07734). As métricas BWT e FWT têm como referência [Lopez-Paz e Ranzato, “Gradient Episodic Memory for Continual Learning” (NeurIPS 2017)](https://papers.nips.cc/paper/2017/hash/f87522788a2be2d171666752f97ddebb-Abstract.html).

### 4.2 Três noções de fairness

O repositório separa três comparações:

1. **Mesmas épocas:** o conjunto que corresponde ao CSV fornecido. Métodos com replay processam mais exemplos, pois acrescentam lotes da memória.
2. **Mesmo número de exemplos:** limita a soma de exemplos atuais e de replay por tarefa.
3. **Custo observado e FLOPs estimados:** reporta tempo, exemplos, FLOPs, memória de replay e logits armazenados.

O CSV fornecido **não é uma comparação de mesmo número de exemplos**: Vanilla processou 102.400 exemplos, enquanto Replay, DER++ e as combinações SlowHeat processaram 143.360.

---

## 5. Catálogo dos métodos presentes nos resultados

### 5.1 Vanilla

MLP treinado sequencialmente com AdamW, sem replay e sem regularização de aprendizagem contínua. É o controle inferior para medir esquecimento catastrófico.

**Fonte:** não é um método de continual learning publicado; é o controle de fine-tuning sequencial. O otimizador vem de [Loshchilov e Hutter (AdamW, ICLR 2019)](https://arxiv.org/abs/1711.05101), construído sobre [Kingma e Ba (Adam, ICLR 2015)](https://arxiv.org/abs/1412.6980).

### 5.2 Replay

Mantém uma memória episódica balanceada com 20 exemplos por classe. Em cada step, concatena o minibatch atual ao minibatch de replay e calcula cross-entropy ponderada pelo número de exemplos de cada parte.

**Fonte principal:** [Chaudhry et al., “On Tiny Episodic Memories in Continual Learning” (2019)](https://arxiv.org/abs/1902.10486).

### 5.3 DER++

Combina três termos:

```text
L = CE_atual + alpha * MSE(logits_replay, logits_armazenados)
             + beta  * CE_replay
```

Ele preserva rótulos e respostas antigas do modelo. Usa a mesma memória de imagens do replay e acrescenta os logits armazenados.

**Fonte principal:** [Buzzega et al., “Dark Experience for General Continual Learning: a Strong, Simple Baseline” (NeurIPS 2020)](https://papers.nips.cc/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html).

### 5.4 ER-ACE

Restringe a loss do lote atual às classes da tarefa corrente e calcula a loss de replay sobre todas as classes vistas. O objetivo é reduzir interferência assimétrica entre classes antigas e novas.

**Fonte principal:** [Caccia et al., “New Insights on Reducing Abrupt Representation Change in Online Continual Learning” (ICLR 2022)](https://openreview.net/forum?id=N8MaByOzUfb).

### 5.5 A-GEM

Calcula separadamente o gradiente atual e o gradiente de referência da memória. Quando o produto interno é negativo, projeta o gradiente atual para remover o componente conflitante.

**Fonte principal:** [Chaudhry et al., “Efficient Lifelong Learning with A-GEM” (ICLR 2019)](https://arxiv.org/abs/1812.00420). O precursor é [GEM, de Lopez-Paz e Ranzato (NeurIPS 2017)](https://papers.nips.cc/paper/2017/hash/f87522788a2be2d171666752f97ddebb-Abstract.html).

### 5.6 EWC

Estima uma Fisher diagonal online a partir dos gradientes e adiciona uma penalidade quadrática ao redor dos parâmetros consolidados:

```text
L_total = L_atual + lambda/2 * sum_i F_i * (theta_i - theta_i*)²
```

**Fontes:** [Kirkpatrick et al., “Overcoming catastrophic forgetting in neural networks” (PNAS 2017)](https://doi.org/10.1073/pnas.1611835114). Como o runner acumula uma Fisher diagonal online, a variante também é relacionada ao online EWC de [Schwarz et al., “Progress & Compress” (ICML 2018)](https://proceedings.mlr.press/v80/schwarz18a.html).

### 5.7 Synaptic Intelligence — SI

Acumula, durante a trajetória de treino, a contribuição `−gradiente × deslocamento`. Na fronteira da tarefa, transforma essa trajetória em importância sináptica e penaliza alterações futuras.

**Fonte principal:** [Zenke, Poole e Ganguli, “Continual Learning Through Synaptic Intelligence” (ICML 2017)](https://proceedings.mlr.press/v70/zenke17a.html).

### 5.8 LwF calibrada

Usa o modelo da tarefa anterior como professor e aplica distillation às classes antigas, sem armazenar imagens antigas. Os pesos das losses antiga e nova dependem da fração de classes antigas e novas.

**Fontes:** [Li e Hoiem, “Learning without Forgetting” (ECCV 2016)](https://doi.org/10.1007/978-3-319-46493-0_37) e a base de distillation de [Hinton, Vinyals e Dean (2015)](https://arxiv.org/abs/1503.02531). A ponderação “calibrated” por fração de classes é uma escolha local do runner.

### 5.9 Replay balanceado

Dá peso 0,5 à loss atual e 0,5 à loss de replay, independentemente do tamanho dos dois minibatches. No Replay comum, os pesos são proporcionais às quantidades de exemplos.

**Fonte:** derivada local de [Experience Replay com memória episódica](https://arxiv.org/abs/1902.10486); não foi identificado um artigo próprio que corresponda exatamente a essa regra `0.5/0.5` do repositório.

### 5.10 Replay com mais épocas

Executa 20 épocas por tarefa em vez de 10. É um controle de orçamento computacional, não um algoritmo novo de retenção.

**Fonte:** controle local baseado em [Experience Replay](https://arxiv.org/abs/1902.10486); não possui artigo independente.

### 5.11 Replay com early stopping

Pode executar até 30 épocas, com paciência de três épocas, restaurando o melhor estado segundo a média de validação sobre as tarefas vistas.

**Fontes:** [Experience Replay](https://arxiv.org/abs/1902.10486) e o princípio de early stopping por validação de [Prechelt, “Automatic early stopping using cross validation” (1998)](https://doi.org/10.1016/S0893-6080(98)00010-0). A paciência e o critério exatos são locais.

### 5.12 SlowHeat + Replay, hidden-only

Nome completo:

```text
slowheat_replay_hidden_beta_30_budget_0.25
```

- usa replay;
- mede utilidade funcional nas camadas ocultas;
- aplica proteção suave com `beta=30`;
- mantém pelo menos 25% das unidades plásticas;
- deixa a cabeça de saída comum e totalmente plástica;
- usa máscaras fatoradas e estado do AdamW em `follow_update`.

A cabeça livre é relevante porque a competição global entre dez classes precisa ser recalibrada à medida que novas classes aparecem.

**Fontes:** [manuscrito local de Functional SlowHeat](../article/manuscript.md) + [Experience Replay](https://arxiv.org/abs/1902.10486). A combinação é própria deste projeto.

### 5.13 SlowHeat + DER++, hidden-only

Nome completo:

```text
slowheat_derpp_hidden_beta_30_budget_0.25
```

Usa a loss completa do DER++ para gerar o sinal funcional do SlowHeat. A memória, os rótulos, os logits armazenados e os hiperparâmetros `alpha=0.5` e `beta=0.5` são os mesmos do DER++. O teste é exploratório e o contraste relevante é SlowHeat + DER++ menos DER++.

**Fontes:** [manuscrito local de Functional SlowHeat](../article/manuscript.md) + [artigo de DER++](https://papers.nips.cc/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html). A combinação não é apresentada como método independente no artigo de DER++.

---

## 6. Métricas

Se `A[t,k]` é a acurácia da tarefa `k` depois de treinar até o estágio `t`:

### Acurácia média final — maior é melhor

```text
ACC = média_k A[T-1,k]
```

É o endpoint class-incremental principal.

### Average forgetting — menor é melhor

```text
F_k = max_{l=k,...,T-1} A[l,k] - A[T-1,k]
Forgetting = média das tarefas antigas
```

### Backward transfer — maior é melhor

Compara o desempenho final de cada tarefa antiga com seu desempenho logo depois de aprendê-la. Valor negativo indica deterioração; valor positivo indica melhora posterior.

### Forward transfer — maior é melhor

Compara a acurácia antes de treinar uma tarefa futura com o desempenho do modelo aleatório pareado.

### Acurácia task-aware — diagnóstico

Restringe a decisão às duas classes da tarefa. Não é a métrica principal porque fornece informação de tarefa durante a avaliação.

### Classifier gap — menor é melhor

```text
classifier_gap = task_aware_accuracy - class_incremental_accuracy
```

Um gap alto com task-aware alta indica que a representação ainda separa cada par, mas a cabeça global não calibra corretamente a competição entre tarefas.

---

## 7. Resultados consolidados

Os valores abaixo são médias sobre cinco seeds, inferidas da relação entre desvio-padrão e largura do IC no CSV. `±` representa a **meia largura do IC95% normal reportado pelo runner**, em pontos percentuais.

| Método | ACC final ↑ | Forgetting ↓ | BWT ↑ | Task-aware ↑ | Gap ↓ | Tempo (s) ↓ |
|---|---:|---:|---:|---:|---:|---:|
| **SlowHeat + DER++** | **84,708 ± 1,687** | 16,625 ± 2,197 | −16,625 | 98,452 | **13,744** | 4,524 |
| **DER++** | 81,984 ± 1,071 | 20,985 ± 1,356 | −20,985 | **98,668** | 16,684 | 2,672 |
| **SlowHeat + Replay** | 76,824 ± 1,605 | 27,070 ± 1,969 | −27,070 | 98,272 | 21,448 | 4,365–4,397 |
| **Replay + early stopping** | 76,820 ± 1,348 | 26,350 ± 1,640 | −26,350 | 98,148 | 21,328 | **1,940** |
| **Replay balanceado** | 76,740 ± 1,409 | 27,500 ± 1,740 | −27,500 | 98,372 | 21,632 | 2,570 |
| **Replay** | 76,524 ± 1,327 | 27,830 ± 1,692 | −27,830 | 98,280 | 21,756 | 2,585 |
| **Replay, 20 épocas** | 75,732 ± 1,409 | 28,985 ± 1,787 | −28,985 | 98,196 | 22,464 | 5,104 |
| **ER-ACE** | 71,128 ± 1,433 | **3,900 ± 0,870** | **+46,655** | 98,444 | 27,316 | 2,620 |
| **A-GEM** | 19,704 ± 0,061 | 99,285 ± 0,152 | −99,285 | 90,536 | 70,832 | 2,960 |
| **LwF calibrada** | 19,608 ± 0,100 | 99,190 ± 0,189 | −99,190 | 90,428 | 70,820 | 2,413 |
| **EWC** | 19,600 ± 0,094 | 99,230 ± 0,090 | −99,230 | 68,176 | 48,576 | 2,730 |
| **Vanilla** | 19,588 ± 0,066 | 99,285 ± 0,069 | −99,285 | 67,040 | 47,452 | 1,954 |
| **SI** | 19,568 ± 0,064 | 99,235 ± 0,121 | −99,235 | 69,896 | 50,328 | 3,007 |

Observações:

- BWT e forgetting coincidem em módulo para quase todos os métodos deste arquivo. Isso indica trajetórias sem recuperação relevante depois do pico. ER-ACE é a exceção: forgetting baseado no melhor ponto e BWT baseado na diagonal contam histórias diferentes.
- O FWT variou aproximadamente de `−0,1244` a `−0,1189`, com intervalos largos. Nenhum método apresentou evidência útil de transferência para tarefas ainda não treinadas.

---

## 8. Comparações diretas

### 8.1 SlowHeat + DER++ versus DER++

| Métrica | DER++ | SlowHeat + DER++ | Diferença |
|---|---:|---:|---:|
| Acurácia final | 81,984% | 84,708% | **+2,724 p.p.** |
| Forgetting | 20,985% | 16,625% | **−4,360 p.p.** |
| Redução relativa de forgetting | — | — | **20,8%** |
| Task-aware | 98,668% | 98,452% | −0,216 p.p. |
| Classifier gap | 16,684% | 13,744% | **−2,940 p.p.** |
| Tempo | 2,672 s | 4,524 s | **+69,3%** |
| Exemplos do learner | 143.360 | 143.360 | 0 |
| FLOPs estimados | 201,924 G | 202,332 G | +0,20% |
| Memória de replay | 628.800 B | 628.800 B | 0 |
| Logits armazenados | 8.000 B | 8.000 B | 0 |

Interpretação: o ganho ocorre principalmente na retenção class-incremental e na redução do gap da cabeça. A acurácia task-aware praticamente não muda. Isso sugere que SlowHeat preservou melhor a compatibilidade global entre representações antigas e a cabeça livre, em vez de melhorar a discriminação interna das tarefas.

O custo é a principal desvantagem. A diferença entre FLOPs estimados é pequena, mas o tempo cresce muito. O provável gargalo está em snapshots de parâmetros/estados, aplicação de máscaras e overhead Python, não em multiplicações da rede.

### 8.2 SlowHeat + Replay versus Replay

| Métrica | Replay | SlowHeat + Replay | Diferença |
|---|---:|---:|---:|
| Acurácia final | 76,524% | 76,824% | **+0,300 p.p.** |
| Forgetting | 27,830% | 27,070% | −0,760 p.p. |
| Redução relativa de forgetting | — | — | 2,7% |
| Task-aware | 98,280% | 98,272% | −0,008 p.p. |
| Classifier gap | 21,756% | 21,448% | −0,308 p.p. |
| Tempo | 2,585 s | 4,365–4,397 s | aproximadamente **+69%** |
| FLOPs estimados | 201,924 G | 202,332 G | +0,20% |

Interpretação: o efeito é pequeno diante da variação entre seeds. Os ICs marginais se sobrepõem fortemente. Sem as diferenças pareadas por seed, esse CSV não sustenta que SlowHeat + Replay seja melhor que Replay.

### 8.3 DER++ versus Replay

DER++ aumentou a acurácia em **5,460 p.p.**, reduziu o forgetting em **6,845 p.p.** e reduziu o classifier gap em **5,072 p.p.**, com apenas **3,3% mais tempo observado**, os mesmos exemplos processados e apenas 8.000 bytes adicionais para logits.

Dentro deste protocolo, DER++ é o baseline com melhor relação entre desempenho e custo.

### 8.4 Early stopping versus Replay

Early stopping aumentou a acurácia em apenas **0,296 p.p.**, mas reduziu:

- tempo em **25,0%**;
- exemplos processados em **27,0%**;
- FLOPs estimados em **27,0%**.

Como as diferenças de acurácia são pequenas e incertas, early stopping aparece como a opção mais eficiente entre as variantes de Replay avaliadas.

### 8.5 Mais épocas versus Replay

Dobrar de 10 para 20 épocas:

- dobrou exemplos e FLOPs;
- aumentou o tempo em **97,4%**;
- reduziu a acurácia em **0,792 p.p.**;
- aumentou o forgetting em **1,155 p.p.**.

Mais treino não resolveu a interferência e pode ter reforçado o viés para dados recentes.

### 8.6 Replay balanceado versus Replay

As diferenças foram pequenas: **+0,216 p.p.** de acurácia e **−0,330 p.p.** de forgetting. Os intervalos marginais se sobrepõem quase completamente. Não há evidência suficiente de vantagem prática.

### 8.7 ER-ACE: retenção forte, desempenho global intermediário

ER-ACE apresentou forgetting de apenas **3,900%** e BWT de **+46,655%**, mas sua acurácia final foi **5,396 p.p. abaixo de Replay** e **10,856 p.p. abaixo de DER++**.

Isso mostra por que forgetting não deve ser interpretado isoladamente. Um método pode esquecer pouco porque sua trajetória de aquisição e recuperação é diferente, sem terminar com a melhor solução global. O endpoint primário continua sendo a acurácia média final.

### 8.8 A-GEM e LwF: representação preservada, cabeça global falha

A-GEM e LwF terminaram próximos de 20% class-incremental, mas alcançaram aproximadamente 90,5% task-aware. Seus classifier gaps, cerca de 70,8 p.p., são os maiores da tabela.

Diagnóstico: os modelos ainda distinguem as duas classes quando a identidade da tarefa é conhecida, mas não calibram corretamente logits de tarefas diferentes. Isso aponta para viés da cabeça global e competição entre classes, não para destruição completa das representações.

### 8.9 EWC e SI

EWC e SI não melhoraram a acurácia class-incremental sobre Vanilla. A task-aware accuracy também ficou bem abaixo dos métodos com replay. Neste protocolo e com estes hiperparâmetros, a regularização paramétrica sem exemplos antigos não foi suficiente para preservar nem a representação nem a calibração global.

### 8.10 Fronteira desempenho–tempo

Considerando apenas as médias de acurácia e tempo, três configurações formam a fronteira prática observada:

| Perfil | Método | ACC final | Tempo |
|---|---|---:|---:|
| Econômico | Replay + early stopping | 76,820% | 1,940 s |
| Equilibrado | DER++ | 81,984% | 2,672 s |
| Maior acurácia | SlowHeat + DER++ | 84,708% | 4,524 s |

ER-ACE não entra nessa fronteira por acurácia e tempo, mas permanece singular quando a prioridade é forgetting. Como o tempo depende da máquina e a amostra tem cinco seeds, essa leitura é descritiva, não uma garantia de ordenação futura.

---

## 9. Análise estatística e qualidade dos dados

### 9.1 O que o IC do CSV representa

O agregador do projeto calcula:

```text
meia_largura_IC95 = 1.96 * desvio_padrão / sqrt(n)
```

Os valores fornecidos são exatamente compatíveis com `n=5`. Esse é um intervalo normal aproximado, não um intervalo t de Student.

Para cinco seeds, o quantil t com quatro graus de liberdade é aproximadamente 2,776, contra 1,96. Portanto, um IC t marginal seria cerca de **41,6% mais largo**. O IC normal do CSV deve ser apresentado como resumo descritivo, não como garantia forte de precisão.

### 9.2 Por que a análise deve ser pareada

Os experimentos compartilham seed, inicialização e minibatches. O teste correto usa, para cada seed:

```text
d_seed = métrica_candidato(seed) - métrica_referência(seed)
```

Depois analisa a média de `d_seed`. Médias, desvios e ICs separados por método não informam a correlação entre os pares. Por isso, sobreposição ou não sobreposição de ICs marginais não substitui:

- t pareado;
- bootstrap pareado;
- contagem de sinais por seed;
- tamanho de efeito e intervalo da diferença.

O repositório já implementa essas análises para o protocolo confirmatório, mas o CSV entregue não contém os valores por seed nem o resumo pareado.

### 9.3 Duplicata detectada

`slowheat_replay_hidden_beta_30_budget_0.25` aparece duas vezes. Todas as métricas são idênticas, exceto o tempo:

- primeira linha: `4,365334689 s`;
- segunda linha: `4,397215460 s`.

Isso parece resultar da concatenação de duas exportações da mesma configuração. As linhas não devem ser combinadas como se fossem dez seeds. O relatório usa a primeira para deltas numéricos e mostra o tempo como faixa.

### 9.4 Proveniência ausente no arquivo fornecido

O CSV não inclui:

- IDs das seeds;
- hash do commit;
- configuração completa;
- matrizes de acurácia por estágio e tarefa;
- diferenças pareadas;
- indicação da seção que produziu cada linha.

Os custos e a lista de métodos indicam que a maior parte das linhas vem da suíte de baselines com mesmas épocas, enquanto SlowHeat + DER++ vem da seção exploratória separada. Para publicação, essas origens precisam permanecer explicitamente separadas.

---

## 10. Custo computacional e memória

| Família | Exemplos do learner | FLOPs estimados | Memória episódica | Logits |
|---|---:|---:|---:|---:|
| Vanilla, EWC, SI | 102.400 | cerca de 144–145 G | 0 | 0 |
| LwF calibrada | 102.400 no learner; 184.320 no total | 182,693 G | 0 | 0 |
| Replay, DER++, ER-ACE, A-GEM e variantes | 143.360 | 201,924 G | 628.800 B | 0 |
| DER++ e SlowHeat + DER++ | 143.360 | 201,924–202,332 G | 628.800 B | 8.000 B |
| SlowHeat + Replay | 143.360 | 202,332 G | 628.800 B | 0 |
| Replay, 20 épocas | 286.720 | 403,849 G | 628.800 B | 0 |
| Replay + early stopping | média de 104.653 | média de 147,405 G | 628.800 B | 0 |

A memória reportada para replay é aproximadamente 0,629 MB em unidade decimal. O custo permanente adicional do SlowHeat é pequeno por unidade, mas o otimizador atual tira snapshots temporários de parâmetros e estados protegidos a cada step. Esse custo temporário não aparece na coluna de memória de replay.

---

## 11. Métodos e ablações implementados, mas ausentes deste CSV

### 11.1 Variantes SlowHeat do runner Split-MNIST

| Nome | Propósito |
|---|---|
| `slowheat` | SlowHeat básico, com proteção da saída por padrão |
| `slowheat_adaptive` | adapta o budget com acurácia de validação |
| `slowheat_native_state` | máscara parâmetros, mas mantém estado nativo do otimizador |
| `slowheat_unidirectional` | proteção somente por linha, sem fatoração entre camadas |
| `slowheat_unbudgeted` | remove a garantia de capacidade plástica |
| `slowheat_none` | registra o otimizador, mas nunca consolida; controle de wiring |
| `hard_freeze` | congela exatamente unidades consolidadas |
| `slowheat_replay` | combinação genérica SlowHeat + replay |
| `slowheat_distillation` | SlowHeat combinado com distillation |
| `slowheat_hidden_beta_30_budget_0.25` | SlowHeat hidden-only sem replay |
| `slowheat_replay_hidden_adaptive_beta_30_budget_0.25` | replay com budget adaptativo |
| `slowheat_replay_partial_output_beta_30_budget_0.25` | proteção forte nas ocultas e parcial na saída |
| `slowheat_replay_hidden_beta_30_budget_0.25_calibrated` | acrescenta offset antigo/novo escolhido na validação |

O parser também aceita nomes estruturados como `slowheat_beta_10_budget_0.25` ou `slowheat_distillation_hidden_beta_30_budget_0.25`.

### 11.2 Controles adicionais

| Nome | Propósito |
|---|---|
| `distillation` | professor congelado, sem replay |
| `replay_global_lr_reduction` | replay com LR global multiplicado por `1/31` |
| `reduced_lr` | controle de LR reduzido no benchmark sintético |
| `slowheat_max`, `slowheat_mean`, `slowheat_sum` | estratégias de consolidação no sintético |
| `slowheat_max_sgd` | SlowHeat com SGD |
| `slowheat_max_legacy_adamw` | hook no gradiente bruto, implementação histórica |
| `slowheat_max_native_state` | estado do AdamW não acompanha a máscara |
| `slowheat_max_unidirectional` | proteção apenas por linha |
| `slowheat_max_unbudgeted` | sem reserva mínima de plasticidade |

### 11.3 Generalização planejada pelo runner

O protocolo completo inclui:

- cinco ordens fixas de classes no Split-MNIST;
- MLPs `256–128`, `512–256` e `512–512–256`;
- memórias de 5, 10, 20, 50 e 100 exemplos por classe;
- Permuted-MNIST domain-incremental;
- Split CIFAR-100;
- TinyImageNet local em formato `ImageFolder`.

O CSV fornecido não contém resultados dessas extensões.

---

## 12. Validação de engenharia

A suíte atual contém **95 testes automatizados, todos aprovados nesta revisão**. Eles cobrem, entre outros pontos:

- invariância da utilidade funcional à reescala recíproca ReLU;
- utilidade zero para unidade ReLU morta;
- orçamento mínimo de capacidade;
- proteção fatorada de linhas e colunas;
- hard-freeze exato;
- máscara aplicada ao delta final de AdamW e SGD;
- políticas `follow_update` e `native`;
- checkpoints que falham de forma segura quando máscaras não são registradas;
- inicialização pareada e agenda determinística;
- métricas class-incremental e task-aware;
- agregação multi-seed, retomada e diferenças pareadas;
- execução dos baselines em runners pequenos.

Os testes validam contratos de implementação. Eles não demonstram eficácia científica nem substituem replicação em benchmarks maiores.

---

## 13. Limitações e ameaças à validade

1. **Apenas cinco seeds no CSV.** A incerteza é alta, especialmente para os métodos SlowHeat.
2. **Sem dados pareados no arquivo.** Não é possível calcular o teste correto da diferença.
3. **Mistura de seções.** SlowHeat + DER++ é exploratório e não pertence ao contraste confirmatório congelado.
4. **Fairness por épocas, não por exemplos.** Métodos com replay processam 40% mais exemplos que Vanilla.
5. **Um benchmark simples.** Split-MNIST não demonstra escalabilidade para visão complexa ou linguagem.
6. **Fronteiras de tarefa conhecidas.** SlowHeat é boundary-aware, o que limita comparações com métodos task-free.
7. **Cabeça compartilhada sensível a viés.** Vários métodos preservam desempenho task-aware, mas falham globalmente.
8. **Overhead de tempo elevado.** A implementação atual do otimizador não é fundida e pode não escalar.
9. **IC normal com amostra pequena.** Os intervalos apresentados são estreitos demais em relação a um IC t.
10. **Artefatos brutos ausentes do repositório.** Sem resultados por seed, ambiente e hash, a auditoria fica incompleta.
11. **Documentação parcialmente desatualizada.** Alguns textos ainda afirmam que baselines especializados não foram implementados ou que a suíte não foi executada, embora o código já contenha os métodos e o CSV fornecido contenha resultados.

---

## 14. Conclusões sustentadas

Os dados permitem afirmar que, nesta execução exploratória:

- replay é essencial para alcançar bom desempenho class-incremental;
- DER++ superou substancialmente o replay simples com custo adicional pequeno;
- SlowHeat + DER++ alcançou o melhor resultado médio e reduziu forgetting e classifier gap em relação a DER++;
- SlowHeat + Replay teve efeito pequeno em relação a Replay;
- ER-ACE minimizou forgetting, mas não maximizou acurácia final;
- A-GEM e LwF preservaram informação task-aware, porém sofreram forte descalibração global;
- dobrar as épocas do replay não melhorou o resultado;
- early stopping reduziu custo sem perda média aparente;
- o SlowHeat atual tem overhead de tempo muito maior que seu overhead aritmético estimado.

Os dados **não** permitem afirmar ainda que:

- SlowHeat é superior de forma geral a DER++, Replay ou outros métodos;
- o ganho de SlowHeat + DER++ é estatisticamente confirmado;
- SlowHeat + Replay supera Replay no endpoint confirmatório;
- o método escala para CIFAR-100, TinyImageNet, redes convolucionais ou transformers;
- a redução de forgetting, sozinha, representa melhor aprendizagem contínua.

---

## 15. Próximos passos recomendados

1. **Executar a confirmação pré-registrada de 20 seeds** para Replay versus `slowheat_replay_hidden_beta_30_budget_0.25`, sem alterar hiperparâmetros depois de observar resultados.
2. **Exportar diferenças pareadas** com média, IC t, bootstrap pareado, contagem de sinais e tamanho de efeito.
3. **Pré-registrar uma confirmação separada de SlowHeat + DER++**, já que essa combinação é hoje o resultado mais promissor, mas surgiu como análise exploratória.
4. **Repetir no orçamento de mesmo número de exemplos**, evitando confundir método com exposição adicional a dados.
5. **Perfilar o otimizador SlowHeat** e eliminar snapshots/cópias desnecessários antes de testar modelos maiores.
6. **Arquivar artefatos completos:** configuração, seed, matriz de acurácia, curvas, custo, ambiente, hash do commit e CSV agregado.
7. **Investigar o classifier gap** com matrizes de confusão, distribuição de logits antigos/novos e calibração por estágio.
8. **Executar as ablações causais:** hidden-only versus saída protegida, budget fixo/adaptativo, proteção fatorada/row-only e `follow_update`/`native`.
9. **Validar generalização** em ordens alternativas, memórias diferentes, Permuted-MNIST e Split CIFAR-100 antes de qualquer alegação ampla.
10. **Deduplicar e validar exportações**, rejeitando nomes repetidos ou incluindo explicitamente a origem de cada seção.

---

## 16. Estrutura do repositório

```text
src/dual_heater/
  dual_heat.py       DualHeat legado
  slow_heat.py       Functional SlowHeat linear, convolucional e MLP
  optim.py           SlowHeatAdamW e SlowHeatSGD
  lora.py            adaptação DualHeat-LoRA experimental
  metrics.py         métricas de aprendizagem contínua

experiments/
  split_mnist.py                 runner principal e baselines
  split_mnist_suite.py           fairness, ablações e generalização
  confirmatory_split_mnist.py    confirmação pré-registrada
  confirmatory_statistics.py     estatística pareada
  synthetic_cl.py                benchmark sintético determinístico
  visual_generalization.py       Permuted-MNIST, CIFAR-100 e TinyImageNet

docs/               contratos, protocolo e registro experimental
notebooks/           execução interativa dos protocolos
tests/               95 testes aprovados
article/             manuscrito técnico em desenvolvimento
```

---

## 17. Recomendação final

O projeto possui uma base de engenharia cuidadosa e um mecanismo bem definido, especialmente na semântica do update final, proteção fatorada e orçamento de capacidade. O resultado **SlowHeat + DER++** justifica uma nova confirmação dedicada. Entretanto, o contraste confirmatório atualmente pré-registrado — SlowHeat + Replay versus Replay — apresenta neste arquivo um efeito pequeno, enquanto o custo de tempo é grande.

A interpretação mais responsável é: **Functional SlowHeat é um complemento promissor para DER++, ainda não uma melhoria confirmada e geral em aprendizagem contínua**.

---

## 18. Mapa completo de fontes por identificador do código

Esta seção evita dois erros comuns: deixar uma variante sem referência e atribuir um artigo externo a uma ablação criada apenas para este projeto.

| Identificador ou família | Classificação bibliográfica | Fonte correta |
|---|---|---|
| `vanilla` | controle local de fine-tuning sequencial | [Adam](https://arxiv.org/abs/1412.6980) e [AdamW](https://arxiv.org/abs/1711.05101) |
| `replay` | implementação de método publicado | [On Tiny Episodic Memories in Continual Learning](https://arxiv.org/abs/1902.10486) |
| `replay_balanced` | variação local de Replay | [Replay](https://arxiv.org/abs/1902.10486); ponderação `0.5/0.5` local |
| `replay_more_epochs` | controle local de Replay | [Replay](https://arxiv.org/abs/1902.10486); orçamento de 20 épocas local |
| `replay_early_stopping` | Replay + política de treino | [Replay](https://arxiv.org/abs/1902.10486) e [early stopping](https://doi.org/10.1016/S0893-6080(98)00010-0) |
| `replay_global_lr_reduction` | controle local de Replay | [Replay](https://arxiv.org/abs/1902.10486) e base do [AdamW](https://arxiv.org/abs/1711.05101); fator `1/31` local |
| `derpp` | implementação de método publicado | [Dark Experience for General Continual Learning](https://papers.nips.cc/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html) |
| `er_ace` | implementação de método publicado | [New Insights on Reducing Abrupt Representation Change](https://openreview.net/forum?id=N8MaByOzUfb) |
| `agem` | implementação de método publicado | [Efficient Lifelong Learning with A-GEM](https://arxiv.org/abs/1812.00420); precursor [GEM](https://papers.nips.cc/paper/2017/hash/f87522788a2be2d171666752f97ddebb-Abstract.html) |
| `ewc` | implementação online de método publicado | [EWC original](https://doi.org/10.1073/pnas.1611835114) e [online EWC em Progress & Compress](https://proceedings.mlr.press/v80/schwarz18a.html) |
| `si` | implementação de método publicado | [Continual Learning Through Synaptic Intelligence](https://proceedings.mlr.press/v70/zenke17a.html) |
| `distillation` | aplicação local de técnica publicada | [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) |
| `lwf_calibrated` | LwF com ponderação local | [Learning without Forgetting](https://doi.org/10.1007/978-3-319-46493-0_37) + [distillation](https://arxiv.org/abs/1503.02531); calibração local |
| `slowheat` | método próprio | [manuscrito técnico local](../article/manuscript.md); relação conceitual com [Taylor de primeira ordem](https://research.nvidia.com/publication/2017-04_pruning-convolutional-neural-networks-resource-efficient-inference) |
| `slowheat_adaptive` | ablação própria | [manuscrito local](../article/manuscript.md); controlador de budget específico do projeto |
| `slowheat_native_state` | ablação própria | [manuscrito local](../article/manuscript.md) + base [AdamW](https://arxiv.org/abs/1711.05101) |
| `slowheat_unidirectional` | ablação própria | [manuscrito local](../article/manuscript.md); remove a fatoração proposta pelo projeto |
| `slowheat_unbudgeted` | ablação própria | [manuscrito local](../article/manuscript.md); remove o budget proposto pelo projeto |
| `slowheat_none` | controle de wiring próprio | [manuscrito local](../article/manuscript.md); sem consolidação |
| `hard_freeze` | controle próprio | [manuscrito local](../article/manuscript.md); binariza a proteção consolidada |
| `slowheat_replay` | combinação própria | [SlowHeat local](../article/manuscript.md) + [Replay](https://arxiv.org/abs/1902.10486) |
| `slowheat_distillation` | combinação própria | [SlowHeat local](../article/manuscript.md) + [distillation](https://arxiv.org/abs/1503.02531) |
| `slowheat_derpp_hidden_beta_30_budget_0.25` | combinação própria exploratória | [SlowHeat local](../article/manuscript.md) + [DER++](https://papers.nips.cc/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html) |
| `slowheat_hidden_beta_30_budget_0.25` | configuração própria estruturada | [manuscrito local](../article/manuscript.md) |
| `slowheat_replay_hidden_beta_30_budget_0.25` | configuração própria estruturada | [SlowHeat local](../article/manuscript.md) + [Replay](https://arxiv.org/abs/1902.10486) |
| `slowheat_replay_hidden_adaptive_beta_30_budget_0.25` | ablação própria | [SlowHeat local](../article/manuscript.md) + [Replay](https://arxiv.org/abs/1902.10486) |
| `slowheat_replay_partial_output_beta_30_budget_0.25` | ablação própria | [SlowHeat local](../article/manuscript.md) + [Replay](https://arxiv.org/abs/1902.10486) |
| `slowheat_replay_hidden_beta_30_budget_0.25_calibrated` | ablação própria | [SlowHeat local](../article/manuscript.md) + [Replay](https://arxiv.org/abs/1902.10486); offset de logits local |
| nomes `slowheat[_replay|_distillation][_hidden]_beta_X_budget_Y` | configurações geradas pelo parser | fontes dos componentes correspondentes; o nome completo não representa um artigo independente |
| `reduced_lr` | controle sintético local | base do otimizador [AdamW](https://arxiv.org/abs/1711.05101) |
| `slowheat_max`, `slowheat_mean`, `slowheat_sum` | ablações próprias de consolidação | [manuscrito local](../article/manuscript.md) |
| `slowheat_max_sgd` | ablação de otimizador | [manuscrito local](../article/manuscript.md); referência moderna para momentum: [Sutskever et al. (ICML 2013)](https://proceedings.mlr.press/v28/sutskever13.html) |
| `slowheat_max_legacy_adamw` | implementação histórica própria | [manuscrito local](../article/manuscript.md) + [AdamW](https://arxiv.org/abs/1711.05101) |
| `slowheat_max_native_state` | ablação própria | [manuscrito local](../article/manuscript.md) + [AdamW](https://arxiv.org/abs/1711.05101) |
| `slowheat_max_unidirectional` | ablação própria | [manuscrito local](../article/manuscript.md) |
| `slowheat_max_unbudgeted` | ablação própria | [manuscrito local](../article/manuscript.md) |
| `DualHeatLinear`, `DualHeatMLP` | protótipos próprios legados | sem artigo próprio; ver [código](../src/dual_heater/dual_heat.py), [Taylor de primeira ordem](https://research.nvidia.com/publication/2017-04_pruning-convolutional-neural-networks-resource-efficient-inference), [EWC](https://doi.org/10.1073/pnas.1611835114) e [SI](https://proceedings.mlr.press/v70/zenke17a.html) apenas como contexto |
| `DualHeatLoRALinear` | adaptação própria de método publicado | [LoRA](https://arxiv.org/abs/2106.09685) + [código local](../src/dual_heater/lora.py); DualHeat-LoRA não aparece no artigo original de LoRA |
| `SlowHeatAdamW`, `SlowHeatSGD` | wrappers próprios | [manuscrito local](../article/manuscript.md), [AdamW](https://arxiv.org/abs/1711.05101) e [momentum/SGD](https://proceedings.mlr.press/v28/sutskever13.html) |

### 18.1 Referências bibliográficas principais

1. **Buzzega, P. et al. (2020).** [Dark Experience for General Continual Learning: a Strong, Simple Baseline](https://papers.nips.cc/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html). NeurIPS 2020.
2. **Caccia, L. et al. (2022).** [New Insights on Reducing Abrupt Representation Change in Online Continual Learning](https://openreview.net/forum?id=N8MaByOzUfb). ICLR 2022.
3. **Chaudhry, A. et al. (2019).** [Efficient Lifelong Learning with A-GEM](https://arxiv.org/abs/1812.00420). ICLR 2019.
4. **Chaudhry, A. et al. (2019).** [On Tiny Episodic Memories in Continual Learning](https://arxiv.org/abs/1902.10486).
5. **Hinton, G., Vinyals, O. e Dean, J. (2015).** [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531).
6. **Hu, E. J. et al. (2022).** [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). ICLR 2022.
7. **Kingma, D. P. e Ba, J. (2015).** [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980). ICLR 2015.
8. **Kirkpatrick, J. et al. (2017).** [Overcoming catastrophic forgetting in neural networks](https://doi.org/10.1073/pnas.1611835114). PNAS 114(13), 3521–3526.
9. **Li, Z. e Hoiem, D. (2016).** [Learning without Forgetting](https://doi.org/10.1007/978-3-319-46493-0_37). ECCV 2016, 614–629.
10. **Lopez-Paz, D. e Ranzato, M. (2017).** [Gradient Episodic Memory for Continual Learning](https://papers.nips.cc/paper/2017/hash/f87522788a2be2d171666752f97ddebb-Abstract.html). NeurIPS 2017.
11. **Loshchilov, I. e Hutter, F. (2019).** [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101). ICLR 2019.
12. **Molchanov, P. et al. (2017).** [Pruning Convolutional Neural Networks for Resource Efficient Inference](https://research.nvidia.com/publication/2017-04_pruning-convolutional-neural-networks-resource-efficient-inference). ICLR 2017.
13. **Prechelt, L. (1998).** [Automatic early stopping using cross validation: quantifying the criteria](https://doi.org/10.1016/S0893-6080(98)00010-0). Neural Networks 11(4), 761–767.
14. **Schwarz, J. et al. (2018).** [Progress & Compress: A scalable framework for continual learning](https://proceedings.mlr.press/v80/schwarz18a.html). ICML 2018.
15. **Sutskever, I. et al. (2013).** [On the importance of initialization and momentum in deep learning](https://proceedings.mlr.press/v28/sutskever13.html). ICML 2013.
16. **van de Ven, G. M. e Tolias, A. S. (2019).** [Three scenarios for continual learning](https://arxiv.org/abs/1904.07734).
17. **Zenke, F., Poole, B. e Ganguli, S. (2017).** [Continual Learning Through Synaptic Intelligence](https://proceedings.mlr.press/v70/zenke17a.html). ICML 2017.
18. **Projeto DualHeat (2026).** [Functional SlowHeat: manuscrito técnico local](../article/manuscript.md). Documento de pesquisa ainda não revisado por pares.
