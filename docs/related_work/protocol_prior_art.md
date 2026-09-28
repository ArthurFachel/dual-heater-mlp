# Levantamento de prioridade — protocolo de pareamento por plasticidade medida

> Documento de busca. As perguntas e os critérios de decisão abaixo foram
> escritos **antes** da primeira busca, pela mesma razão que um pré-registro
> existe: impedir que o resultado da busca seja reinterpretado para preservar a
> contribuição desejada.

**Estado:** CONCLUÍDO em 28/09/2026. Veredito na última seção.
**Bloqueia:** toda a redação do artigo, e a justificativa de QB-2.

---

## A pergunta que decide a âncora

**Existe trabalho publicado que pareia braços experimentais por plasticidade
efetiva medida (não nominal) ao comparar métodos de proteção seletiva?**

"Plasticidade efetiva medida" significa: uma quantidade calculada a partir do
passo de otimização realmente aplicado, usada para igualar braços antes de
comparar acurácia — e não um hiperparâmetro declarado (`beta`, `lambda`,
`budget`) assumido como equivalente entre braços.

## Critérios de decisão, declarados antes da busca

| Achado | Consequência para o artigo |
|---|---|
| **Nada parecido** | Âncora de protocolo se mantém. Redação segue o plano. |
| **Existe pareamento por capacidade/parâmetros, mas não por plasticidade do passo** | Âncora se mantém, com posicionamento explícito contra esse trabalho na §9. |
| **Existe pareamento por plasticidade medida, em outro domínio** (ex.: poda, quantização) | Âncora enfraquece. A contribuição vira "transporte + validação em CL", e a §1 precisa dizer isso. |
| **Existe pareamento por plasticidade medida em continual learning** | **Âncora cai.** `goals/proximo_passo_artigo.md` §3 precisa ser reescrito antes de qualquer redação. |

## Alvos mínimos de busca

### Prioridade 1 — o protocolo (existencial)

Termos, porque a literatura de CL raramente usa a palavra "plasticidade":

- `iso-plasticity` / `isoplasticity`
- `matched plasticity` / `plasticity-matched`
- `effective step size matching` / `effective learning rate control`
- `capacity-controlled ablation` continual learning
- `gradient mask budget` continual learning
- `equal-capacity control` catastrophic forgetting
- `learning rate control` ablation "selective protection"
- `effective plasticity` neural network continual

### Prioridade 2 — a família LoRA-CL (afeta só os resultados negativos)

- **O-LoRA** — ortogonalização de subespaço por tarefa; colide com o mecanismo `rank`
- **InfLoRA** — subespaço livre de interferência; colide com `rank` e `slice`
- **CorDA**, **MoRAL**, **SAPT** e o que a busca revelar

### Prioridade 3 — vizinhos conceituais já citados na §9 do manuscrito

EWC, SI, SLNID, HAT, UCB, Neuron Activation Importance. Verificar se **algum
deles** reporta um controle de learning rate pareado. Se reportarem, isso muda
a força da afirmação "o controle é rotineiramente omitido".

---

## Resultados

Busca executada em 28/09/2026. 21 consultas cobrindo as três prioridades.

### Prioridade 1 — o protocolo

| Trabalho | Ano | Venue | O que faz | Parea por plasticidade medida? | Colide com |
|---|---|---|---|---|---|
| **Mirzadeh et al., Understanding the Role of Training Regimes in Continual Learning** ([2006.06958](https://arxiv.org/abs/2006.06958)) | 2020 | NeurIPS | Mostra que learning rate decay, dropout e batch size alteram o esquecimento por alargarem os mínimos locais por tarefa. Diz explicitamente que esses fatores "podem ter outros fatores confundidores para reduzir o esquecimento catastrófico" | **Não.** Estuda o LR como *fator causal do esquecimento*, não como *controle pareado entre braços de métodos*. Varia o regime e observa o efeito; não mede plasticidade efetiva nem iguala braços por ela | Nada. **É o antecedente mais próximo e fortalece a premissa:** alguém já mostrou que o regime de treino confunde a medida, e mesmo assim a prática de pareamento não se estabeleceu |
| **Lyle et al., Understanding Plasticity in Neural Networks** ([2303.01486](https://arxiv.org/abs/2303.01486)) | 2023 | ICML | Define plasticidade de forma mensurável (capacidade de mudar predições) e liga a perda de plasticidade à curvatura do loss landscape | **Não.** Mede plasticidade como *objeto de estudo* (por que ela se perde ao longo do treino), não como *variável de controle* para igualar braços antes de comparar acurácia | Nada. Oferece vocabulário e legitimidade à ideia de plasticidade mensurável |
| **Dohare et al., Loss of Plasticity in Deep Continual Learning** ([Nature](https://www.nature.com/articles/s41586-024-07711-7)) | 2024 | Nature | Quantifica prevalência da perda de plasticidade; propõe medidas que a quantificam | **Não.** Mesmo padrão: plasticidade como fenômeno medido, não como controle experimental | Nada |
| **Hsu et al., Re-evaluating Continual Learning Scenarios** ([1810.12488](https://arxiv.org/abs/1810.12488)) | 2018 | NeurIPS CL Workshop | Uniformiza cenários de avaliação e mostra que métodos simples competem com complexos | **Não.** Padroniza *cenário* (task-IL/domain-IL/class-IL), não *plasticidade* | Nada. Precedente de "artigo cuja contribuição é protocolo de avaliação" — útil como modelo de posicionamento |
| **van de Ven & Tolias, Three Scenarios for Continual Learning** ([1904.07734](https://arxiv.org/abs/1904.07734)) | 2019 | arXiv / Nature MI 2022 | Idem, taxonomia de cenários | **Não** | Nada. Mesmo precedente de posicionamento |
| **SLCA / SLCA++** ([2303.05118](https://arxiv.org/html/2303.05118), [2408.08295](https://arxiv.org/html/2408.08295v1)) | 2023/24 | ICCV | "Slow Learner": reduz seletivamente o LR do backbone em relação ao classificador | **Não, e é o quase-hit mais perigoso.** Reduz LR de forma seletiva, mas como **método proposto**, não como controle. Não há braço que iguale plasticidade efetiva entre SLCA e as baselines | Conceitualmente vizinho do `lr_control`. Precisa ser citado na §9 |
| **Zhang et al., Optimal Rates / A Statistical Theory of Regularization-Based CL** ([2406.06213](https://arxiv.org/html/2406.06213v1)) | 2024 | arXiv | Análise estatística de CL baseado em regularização em regressão linear | **Não.** Teórico, sem protocolo experimental de pareamento | Nada |

### Prioridade 2 — família LoRA-CL

| Trabalho | Ano | Venue | O que faz | Colide com |
|---|---|---|---|---|
| **LoRA-FA** ([2308.03303](https://arxiv.org/abs/2308.03303)) | 2023 | arXiv / OpenReview | **Congela `A` e treina só `B`** — exatamente o mecanismo do braço `exact`. Motivação declarada: memória de ativação, não esquecimento. Afirma revelar "estrutura assimétrica e colapsável" na atualização do LoRA, implicando que **um dos fatores pode ser congelado sem sacrificar expressividade** | **O mecanismo do braço `exact` não é novo.** Ver análise abaixo |
| **LoRI** (citado por LoRA-FA) | 2025 | — | Congela `A` e usa múltiplos `B` para adaptação multi-tarefa | Mesmo mecanismo, aplicação multi-tarefa |
| **O-LoRA** ([2310.14152](https://arxiv.org/abs/2310.14152)) | 2023 | EMNLP Findings | Restringe atualizações a subespaços ortogonais entre tarefas | Mecanismo `rank`. **Precisa ser citado ao reportar que `rank` falhou** |
| **InfLoRA** ([2404.00228](https://arxiv.org/abs/2404.00228)) | 2024 | CVPR | Reparametriza pesos pré-treinados em subespaço que elimina interferência da tarefa nova sobre as antigas | Mecanismos `rank` e `slice` |
| **CURLoRA** ([2408.14572](https://arxiv.org/html/2408.14572v1)) | 2024 | arXiv | Decomposição CUR para estabilizar fine-tuning contínuo | Vizinho; não colide diretamente |
| **GS-LoRA** | 2024 | CVPR | Esquecimento contínuo seletivo (remover informação de propósito) | Problema diferente (unlearning) |

### Prioridade 3 — vizinhos da §9

Nenhum dos trabalhos clássicos verificados (EWC, SI, SLNID, HAT, UCB) reporta braço de
controle com learning rate pareado. A afirmação "o controle é rotineiramente omitido"
**se sustenta**.

---

## Veredito

**Linha 2 da tabela de critérios, com uma ressalva séria na Prioridade 2.**

### Sobre o protocolo (a pergunta existencial): a âncora se mantém

Nenhum dos 21 resultados de busca mostra trabalho que **pareie braços experimentais por
plasticidade efetiva medida** antes de comparar acurácia em continual learning. O padrão
encontrado na literatura é consistente e diferente:

- plasticidade aparece como **fenômeno a ser estudado** (Lyle 2023, Dohare 2024) ou como
  **alvo a ser preservado** (regularização regenerativa, isometria dinâmica);
- learning rate aparece como **fator causal investigado** (Mirzadeh 2020) ou como
  **componente de método proposto** (SLCA);
- protocolos de avaliação padronizam **cenário** (Hsu 2018, van de Ven 2019), não
  plasticidade.

O antecedente mais próximo, Mirzadeh et al. (NeurIPS 2020), **fortalece** a premissa do
artigo em vez de derrubá-la: mostrou em 2020 que o regime de treino é um confundidor do
esquecimento, e mesmo assim o pareamento por plasticidade não virou prática. Isso é
exatamente a lacuna que o protocolo preenche, e dá ao artigo uma citação forte para a
frase "o confundidor é conhecido e ainda assim não é controlado".

SLCA é o quase-hit e precisa ser tratado com cuidado na §9: ele **reduz LR seletivamente**,
o que se parece com o `lr_control`. A diferença é categórica e deve ser dita
explicitamente — em SLCA isso é o **método proposto**, aqui é o **braço de controle que
testa se o método é só isso**.

### Sobre o mecanismo `exact`: não é novo, e isso muda a redação

**LoRA-FA (2023) já congela `A` e treina só `B`.** O braço `exact` reimplementa, sem
saber, um método publicado. Três consequências, todas obrigatórias:

1. O artigo **não pode** apresentar "congelar `A`" como mecanismo próprio. Precisa citar
   LoRA-FA e LoRI e posicionar-se como *avaliação* desse mecanismo sob protocolo
   controlado, não como proposta dele.
2. **Isto não derruba o resultado confirmado de 28/09.** A contribuição ali é a
   comparação `exact − lr_control` sob plasticidade pareada — que LoRA-FA não faz, porque
   a motivação de LoRA-FA é memória de ativação e não esquecimento. Avaliar um mecanismo
   conhecido sob um protocolo novo continua sendo resultado.
3. **Afeta diretamente QB-2 e pode encolhê-lo.** LoRA-FA afirma que congelar um fator
   **não sacrifica expressividade** (estrutura assimétrica colapsável, a atualização
   reformulável como regressão linear de camada única). Se essa afirmação se sustenta, a
   explicação alternativa que QB-2 existe para eliminar — *"`exact` esquece menos porque
   tem menos capacidade"* — já tem uma resposta teórica na literatura. **Ler
   [2308.03303](https://arxiv.org/abs/2308.03303) na íntegra antes de gastar as 2h de
   GPU**, e registrar o achado na tabela K de `goals/protocol_lora_rank_matched.md` antes
   da run. QB-2 pode passar de "necessário" a "confirmação empírica de uma propriedade já
   argumentada", o que é um experimento de valor menor.

### Consequência prática

Âncora de protocolo **mantida**. A redação segue, com três citações agora obrigatórias:
Mirzadeh et al. (2020) como antecedente do confundidor, SLCA como vizinho a distinguir, e
LoRA-FA/LoRI como origem do mecanismo `exact`. Os três mecanismos negativos (`rank`,
`leak`, `slice`) precisam citar O-LoRA e InfLoRA ao serem reportados.

### Limitação desta busca

Feita por busca web, não por varredura sistemática de anais. O backend de extração de
página estava indisponível, então os abstracts foram lidos via resultados de busca e não
no texto integral. Os riscos residuais são: trabalho que use vocabulário completamente
diferente para a mesma ideia, e trabalho muito recente ainda não indexado. Antes da
submissão, vale uma passada em anais de CoLLAs e nos workshops de CL de NeurIPS/ICML.
