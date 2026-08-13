# DualHeat: Dual-Timescale Heat Regulation for Continual Learning

## Análise Teórica

**Arthur Fachel** — MALTA Lab, PUCRS

---

## 1. Introdução e Motivação

Redes neurais artificiais sofrem de **catastrophic forgetting** (McCloskey & Cohen, 1989): aprender uma nova tarefa destrói o conhecimento adquirido em tarefas anteriores. Métodos existentes atacam o problema por três frentes:

1. **Regularização** — penaliza mudanças em parâmetros importantes (EWC, SI, MAS)
2. **Replay** — armazena e reutiliza exemplares antigos (ER, GEM, A-GEM)
3. **Arquitetura** — aloca parâmetros dedicados por tarefa (Progressive NNs, PackNet, HAT)

EWC (Elastic Weight Consolidation, Kirkpatrick et al. 2017) é o mais influente dos métodos de regularização. Ele estima a importância de cada peso via Fisher Information Matrix e adiciona um termo quadrático de penalidade na loss. O custo é alto: a Fisher é O(n²) no número de parâmetros, e mesmo aproximações diagonais exigem uma passada adicional pelo dataset antigo.

**DualHeat** propõe uma alternativa mais simples: usar a **magnitude de ativação de cada neurônio** como proxy de importância, e regular o aprendizado em duas escalas temporais simultâneas — uma rápida (inibição lateral no forward) e uma lenta (modulação do learning rate no backward). O custo é O(n) por camada, não requer passadas extras sobre dados antigos, e reduz o forgetting em 34% em testes controlados.

---

## 2. Formulação Matemática

### 2.1 Notação

Seja uma camada linear com `N` neurônios de saída. Para cada neurônio `i` no passo de treino `t`:

| Símbolo | Definição |
|---------|-----------|
| `z_i(t)` | Pré-ativação: `Σⱼ w_ij(t) · x_j(t) + b_i(t)` |
| `output_i(t)` | Saída pós-inibição |
| `heat_fast_i(t)` | Calor rápido (EMA pós-inibição) |
| `heat_slow_i(t)` | Calor lento (média amostral) |
| `α` | Decaimento do EMA rápido |
| `γ` | Força da inibição lateral |
| `δ` | Decaimento ativo do fast heat |
| `β` | Força da proteção EWC no LR |

### 2.2 Equações do Sistema

**Passo 1 — Pré-ativação:**

$$z_i(t) = \sum_j w_{ij}(t) \cdot x_j(t) + b_i(t)$$

Idêntica a uma camada linear padrão.

**Passo 2 — Inibição Lateral Divisiva:**

$$output_i(t) = \frac{z_i(t)}{1 + \gamma \cdot \frac{1}{N-1} \sum_{j \neq i} heat\_fast_j(t-1)}$$

A inibição é **lateral** (cada neurônio é inibido pela média dos _outros_) e **divisiva** (escala o sinal proporcionalmente, diferente da inibição subtrativa que pode inverter o sinal). O heat usado é do passo anterior (`t-1`), evitando dependência circular.

**Passo 3 — Fast Heat (EMA + Decay Ativo):**

$$heat\_fast_i(t) = \max\left(0, \;\alpha \cdot heat\_fast_i(t-1) + (1-\alpha) \cdot |output_i(t)| - \delta\right)$$

Três componentes:
- **α · heat(t-1)**: memória — carrega o histórico
- **(1-α) · |output(t)|**: nova evidência — pós-inibição, reflete contribuição real
- **−δ**: decay ativo — drena heat quando o neurônio está silencioso

O clamp em 0 impede valores negativos.

**Passo 4 — Slow Heat (Média Amostral Verdadeira):**

$$heat\_slow_i(t) = heat\_slow_i(t-1) + \frac{|output_i(t)| - heat\_slow_i(t-1)}{t}$$

Média aritmética de `|output|` desde o início do treino. Cada observação pesa `1/t`, então o estimador converge para a média verdadeira (invejado, consistente).

**Passo 5 — EWC Gradient Hook:**

$$w_{ij}(t+1) = w_{ij}(t) - \frac{\eta}{1 + \beta \cdot heat\_slow_i(t)} \cdot \frac{\partial \mathcal{L}}{\partial w_{ij}}$$

$$b_i(t+1) = b_i(t) - \frac{\eta}{1 + \beta \cdot heat\_slow_i(t)} \cdot \frac{\partial \mathcal{L}}{\partial b_i}$$

O learning rate efetivo do neurônio `i` é reduzido por `1/(1 + β·heat_slow_i)`. Neurônios com alta ativação média (importantes) aprendem mais devagar.

### 2.3 Ordem de Execução por Passo

```
1.  Calcula z_i(t)                    (pré-ativação)
2.  Calcula output_i(t)               (usa heat_fast de t-1)
3.  Atualiza heat_fast_i(t)           (com |output_i(t)|)
4.  Atualiza heat_slow_i(t)           (com |output_i(t)|)
5.  Backward: gradiente escala por    1/(1 + β·heat_slow_i)
6.  Optimizer.step()                  (LR já escalado pelo hook)
```

---

## 3. Análise do Estado Estacionário

### 3.1 Fast Heat sem Clamp

Ignorando o operador `max(0, ·)`, o estado estacionário do fast heat é:

$$heat\_fast^* = \alpha \cdot heat\_fast^* + (1-\alpha) \cdot |output| - \delta$$

Resolvendo:

$$heat\_fast^* \cdot (1-\alpha) = (1-\alpha) \cdot |output| - \delta$$

$$\boxed{heat\_fast^* = |output| - \frac{\delta}{1-\alpha}}$$

### 3.2 Threshold de Atividade

O clamp em zero introduz um limiar:

$$heat\_fast_i = 0 \quad \text{se} \quad |output_i| < \frac{\delta}{1-\alpha}$$

Interpretação: neurônios com ativação média pós-inibição abaixo de `δ/(1-α)` têm heat zero e não inibem ninguém. Isso é uma forma de **esparsidade induzida por limiar** — apenas neurônios ativos participam da competição.

Para `α=0.93, δ=0.04` (default CL):

$$\frac{0.04}{1-0.93} = \frac{0.04}{0.07} \approx 0.571$$

Neurônios com `|output| < 0.571` são ignorados pela inibição lateral.

### 3.3 Fast Heat com Inibição Ativa

Quando `heat_fast > 0`, o output é reduzido pela inibição lateral. O estado estacionário acoplado é:

$$|output_i| = \frac{|z_i|}{1 + \gamma \cdot \overline{heat\_fast}_{j \neq i}}$$

$$heat\_fast_i^* = \max\left(0, \frac{|z_i|}{1 + \gamma \cdot \overline{heat\_fast}_{j \neq i}} - \frac{\delta}{1-\alpha}\right)$$

Onde `z_i` é a pré-ativação (sem inibição). Isso forma um sistema de equações acopladas: o heat de cada neurônio depende do output (que depende do heat dos outros). No equilíbrio, neurônios mais fortes (maior `|z|`) mantêm heat positivo e inibem os mais fracos.

### 3.4 Slow Heat

O slow heat é um estimador da média de `|output|`:

$$\mathbb{E}[heat\_slow_i] = \mathbb{E}[|output_i|]$$

Como `t → ∞`, `heat_slow_i` converge para a verdadeira média de `|output_i|` (lei forte dos grandes números). O LR efetivo converge para:

$$\eta_{eff,i} \rightarrow \frac{\eta}{1 + \beta \cdot \mathbb{E}[|output_i|]}$$

---

## 4. Relação com Métodos Existentes

### 4.1 EWC (Elastic Weight Consolidation)

**EWC** adiciona um termo quadrático à loss:

$$\mathcal{L}_{total}(\theta) = \mathcal{L}_{new}(\theta) + \frac{\lambda}{2} \sum_k F_k (\theta_k - \theta_{A,k}^*)^2$$

Onde `F_k` é o elemento diagonal da Fisher Information Matrix.

**DualHeat** substitui a penalidade na loss por uma modulação do learning rate:

$$\eta_k \rightarrow \frac{\eta}{1 + \beta \cdot heat\_slow_{parent}(k)}$$

Onde `heat_slow_{parent}(k)` é o slow heat do neurônio de saída a que o peso `k` pertence.

**Diferenças fundamentais:**

| Aspecto | EWC | DualHeat |
|---------|-----|----------|
| Custo | O(n_params) — cada peso tem Fisher | O(n_neurons) — agrega por neurônio |
| Medida de importância | Fisher Information (segunda derivada) | Média de ativação (primeiro momento) |
| Mecanismo | Penalidade na loss | Modulação do LR |
| Dados extras | Requer forward+backward no dataset antigo | Nenhum — estatísticas do próprio batch |
| Armazenamento | Guarda pesos ótimos + Fisher por tarefa | Nenhum — heat_slow é running statistic |

### 4.2 SI (Synaptic Intelligence)

SI (Zenke et al. 2017) mede importância pelo acúmulo do produto gradiente-mudança ao longo da trajetória de treino:

$$\omega_k = \sum_{t} \frac{\partial \mathcal{L}}{\partial \theta_k} \cdot \Delta \theta_k$$

E penaliza mudanças em pesos com `ω_k` alto.

DualHeat difere em:
- SI é per-weight (custo O(n_params)), DualHeat é per-neuron (O(n_neurons))
- SI rastreia o histórico de gradientes; DualHeat rastreia ativação pós-inibição
- SI precisa armazenar `ω_k` entre tarefas; DualHeat mantém running statistics

### 4.3 MAS (Memory Aware Synapses)

MAS (Aljundi et al. 2018) define importância como a sensibilidade da saída ao peso:

$$\Omega_k = \frac{1}{N} \sum_{x} \left\|\frac{\partial \|F(x)\|^2}{\partial \theta_k}\right\|$$

A relação com DualHeat: MAS mede quanto a saída muda se o peso muda — uma aproximação local da curvatura. DualHeat mede o quanto o neurônio contribui pra saída (`|output|`). Ambas são proxies de importância, mas MAS opera na saída da rede enquanto DualHeat opera na ativação do neurônio.

### 4.4 Normalização Divisiva (Carandini & Heeger, 2012)

O modelo clássico de normalização:

$$R_i = \gamma \cdot \frac{D_i}{\sigma^n + \sum_j D_j^n}$$

DualHeat usa uma forma simplificada:

$$output_i = \frac{z_i}{1 + \gamma \cdot mean\_heat\_others}$$

Diferenças:
- Normalização clássica usa a ativação **atual** de todos os neurônios no denominador; DualHeat usa o **heat acumulado** (EMA histórico)
- Normalização clássica é instantânea; DualHeat tem memória (o heat reflete atividade recente)
- O decay ativo (δ) não existe na normalização clássica

### 4.5 ALIF (Adaptive Leaky Integrate-and-Fire)

Neurônios ALIF têm um threshold adaptativo que aumenta com a atividade recente e decai lentamente — análogo ao fast heat que sobe com atividade e decai com δ. Diferenças:
- ALIF adapta o threshold de disparo; DualHeat adapta a inibição lateral
- ALIF é auto-inibição por neurônio; DualHeat é inibição lateral entre neurônios
- ALIF não tem contraparte EWC (slow heat)

### 4.6 Resumo Comparativo

| Método | Custo | Medida de Importância | Proteção | Forward Effect | Dados extras |
|--------|-------|----------------------|----------|----------------|--------------|
| EWC | O(n_params) | Fisher Information | Penalidade na loss | Nenhum | Dataset antigo |
| SI | O(n_params) | Path integral ∇θ·Δθ | Penalidade na loss | Nenhum | Trajetória do gradiente |
| MAS | O(n_params) | ∂||F(x)||²/∂θ | Penalidade na loss | Dataset antigo |
| DualHeat | **O(n_neurons)** | **Média de ativação** | **Modulação LR** | **Inibição lateral** | **Nenhum** |

DualHeat é o único que (a) atua no forward (inibição) E no backward (LR), (b) tem custo por neurônio em vez de por peso, e (c) não requer passadas extras sobre dados.

---

## 5. Por que a Média de Ativação Funciona como Proxy de Importância?

### 5.1 Intuição

Um neurônio com alta ativação média `|output|` contribui consistentemente para a saída da rede. Mudar seus pesos tem alto impacto na loss — exatamente a definição de "importância" em EWC. A média de ativação é um proxy barato para essa sensibilidade.

### 5.2 Conexão com a Fisher Information

Para uma rede com saída `f(x; θ)` e loss quadrática `ℒ = ½(f(x; θ) - y)²`, a Fisher Information para um peso `w_ij` é:

$$\mathcal{F}_{ij} = \mathbb{E}_{x,y}\left[\left(\frac{\partial \mathcal{L}}{\partial w_{ij}}\right)^2\right]$$

Pela regra da cadeia:

$$\frac{\partial \mathcal{L}}{\partial w_{ij}} = \frac{\partial \mathcal{L}}{\partial z_i} \cdot x_j$$

Para o neurônio `i` com ativação `|output_i|` após non-linearidade:

$$\frac{\partial \mathcal{L}}{\partial z_i} \propto \delta_i \cdot f'(z_i)$$

Onde `δ_i` é o erro propagado. A intuição central do DualHeat é que:

**Neurônios com `|output_i|` consistentemente alto têm, em média, `(∂L/∂z_i)` alto — porque a saída deles afeta a loss diretamente através de todas as conexões downstream.**

Não é uma equivalência matemática exata (Fisher != ativação), mas é uma **correlação empírica** que se sustenta porque:
1. A ativação média correlaciona com a variância do gradiente (Fisher)
2. É monotônica com o número de caminhos de gradiente que passam pelo neurônio
3. O custo por neurônio vs por peso é a vantagem prática

### 5.3 Justificativa via Sensibilidade

Seja `f: ℝ^D → ℝ^C` a função da rede. A importância do neurônio `i` pode ser medida por:

$$I_i = \mathbb{E}_{x}\left[\left\|\frac{\partial \|f(x)\|}{\partial b_i}\right\|\right]$$

Onde `b_i` é o bias do neurônio. Pela regra da cadeia:

$$\frac{\partial \|f(x)\|}{\partial b_i} = \frac{\partial \|f(x)\|}{\partial output_i} \cdot f'(z_i)$$

Se assumirmos que `∂‖f‖/∂output_i` e `f'(z_i)` são correlacionados com `|output_i|` (ativações altas têm mais impacto downstream), então `I_i` é aproximadamente proporcional a `|output_i|`.

DualHeat usa essa proxy e a aplica de duas formas:
- **No forward**: inibição lateral força a rede a não depender de poucos neurônios
- **No backward**: redução do LR protege neurônios que se mostraram importantes

---

## 6. Dinâmica do Sistema Acoplado

### 6.1 Ciclo de Retroalimentação Negativa

O sistema DualHeat é um **sistema dinâmico com realimentação negativa**:

```
heat_fast ↑  →  inibição lateral ↑  →  output ↓  →  post_mag ↓  →  heat_fast ↓
     ↑                                                                    |
     └────────────────────────────────────────────────────────────────────┘
```

Esse ciclo tem um ponto fixo estável (seção 3.1). Sem o decay ativo `δ`, o heat poderia oscilar entre 0 e um valor alto com amortecimento fraco. O decay `δ` funciona como termo dissipativo que garante convergência.

### 6.2 Competição Lateral

A inibição lateral cria competição do tipo **soft winner-take-all**:

$$output_i = \frac{z_i}{1 + \gamma \cdot \overline{heat\_fast}_{j \neq i}}$$

Quando o neurônio `k` está muito ativo, seu `heat_fast_k` sobe, aumentando a média dos "outros" para todos os neurônios. Isso reduz o output de todos exceto o próprio `k`. O resultado é que neurônios que disparam consistentemente forte suprimem os mais fracos — forçando a rede a distribuir representações.

Para tarefas diferentes, neurônios diferentes vencem a competição, criando **vias neurais especializadas por tarefa**. Quando uma nova tarefa começa:
1. Os neurônios que dominavam a tarefa antiga estão "quentes" (heat alto)
2. Eles inibem fortemente os demais
3. Para aprender a nova tarefa, a rede precisa recrutar neurônios que não foram inibidos — ou forçar os existentes a se adaptar
4. O EWC slows updates nos neurônios quentes (importantes), forçando o recrutamento de neurônios novos ou antes silenciosos

### 6.3 Trade-off Forgetting × Plasticidade

O parâmetro `β` controla o trade-off fundamental:

| `β` | Efeito | Problema |
|-----|--------|----------|
| 0 | Sem proteção EWC | Forgetting alto (vanilla) |
| Baixo (0.2-0.5) | Proteção leve | Pode não proteger o suficiente |
| Médio (1.0-3.0) | Proteção moderada | **Sweet spot** para CL |
| Alto (5.0+) | Proteção forte | Não consegue aprender tarefas novas (LR efetivo ~0.09 do original) |

O parâmetro `γ` controla a pressão competitiva:

| `γ` | Efeito | Problema |
|-----|--------|----------|
| 0 | Sem inibição | Sem especialização por neurônio |
| Baixo (0.5-1.0) | Competição leve | Mistura entre tarefas |
| Médio (1.5-3.0) | Competição moderada | Boa separação |
| Alto (5.0+) | Competição agressiva | Destrói sinal em feed-forward |

No regime CL ótimo encontrado (`γ=2.0, β=2.0`), a competição é forte o suficiente para segregar tarefas, e a proteção EWC evita que os neurônios segregados sejam sobrescritos.

---

## 7. Limitações e Trabalho Futuro

### 7.1 Limitações Teóricas

1. **Proxy imperfeito**: média de ativação não é equivalente a Fisher Information. Pode haver neurônios com ativação baixa mas alta importância downstream (ex: neurônios que gateiam informação via multiplicação)

2. **Agregação por neurônio**: pesos diferentes do mesmo neurônio podem ter importâncias drasticamente diferentes. EWC consegue proteger pesos individuais; DualHeat protege todos os pesos de um neurônio uniformemente

3. **Slow heat sem esquecimento**: a média amostral `1/t` converge mas nunca esquece. Um neurônio que foi importante no passado distante mas irrelevante agora ainda tem LR reduzido

4. **Não atua na última camada**: por design, a última camada é um `nn.Linear` padrão sem heat. A última camada é a mais propensa a forgetting.

### 7.2 Limitações Empíricas

1. Testado apenas em MLPs rasas (128→64→20). Comportamento em CNNs, ResNets, ou Transformers não foi verificado

2. Dataset sintético com 20 classes Gaussianas separáveis. Testes em benchmarks padrão (CIFAR-100, MiniImageNet) são necessários

3. Task-incremental apenas. Class-incremental (sem task oracle) e domain-incremental não foram testados

### 7.3 Extensões Possíveis

1. **Fisher híbrida**: usar média de ativação como prior, mas calibrar com uma estimativa esparsa de Fisher quando disponível

2. **Slow heat com forget**: substituir `1/t` por EMA lento `η` para permitir que o slow heat esqueça neurônios que deixaram de ser importantes

3. **Heat por head em transformers**: adaptar o mecanismo para attention heads (cada head seria um "neurônio" com seus próprios heats)

4. **Inibição topológica**: em vez de inibição lateral completa (todos contra todos), usar inibição baseada em topologia (vizinhos mais próximos)

---

## 8. Conclusão

DualHeat é um mecanismo de regularização neural de **dois tempos** que opera simultaneamente no forward (inibição lateral divisiva) e no backward (modulação do LR por neurônio). Ele combina ideias conhecidas (normalização divisiva, EWC, EMA tracking) em uma arquitetura nova com as seguintes propriedades:

1. **Custo O(n_neurons)**: escala melhor que EWC/SI/MAS (O(n_params))
2. **Sem dados extras**: não requer passadas sobre datasets antigos
3. **Efeito duplo**: inibição no forward + proteção no backward
4. **Decay ativo**: cria limiar natural de atividade

Em experimentos controlados com split de 20 classes em 4 tasks, DualHeat reduz o forgetting em 34% (0.731 → 0.482) mantendo accuracy média similar à vanilla, superando a versão com auto-inibição (v1) que não conseguia competir com a vanilla.

O mecanismo não tem equivalente exato na literatura. É uma contribuição original que merece benchmarks mais amplos antes de publicação formal.

---

## Referências

- Aljundi, R., et al. (2018). Memory Aware Synapses: Learning what (not) to forget. *ECCV*.
- Carandini, M. & Heeger, D. J. (2012). Normalization as a canonical neural computation. *Nature Reviews Neuroscience*.
- Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting in neural networks. *PNAS*.
- McCloskey, M. & Cohen, N. J. (1989). Catastrophic interference in connectionist networks. *Psychology of Learning and Motivation*.
- Zenke, F., et al. (2017). Continual Learning Through Synaptic Intelligence. *ICML*.
