# Levantamento de prioridade — protocolo de pareamento por plasticidade medida

> Documento de busca. As perguntas e os critérios de decisão abaixo foram
> escritos **antes** da primeira busca, pela mesma razão que um pré-registro
> existe: impedir que o resultado da busca seja reinterpretado para preservar a
> contribuição desejada.

**Estado:** em execução.
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

_(preencher durante a busca — uma linha por trabalho relevante)_

| Trabalho | Ano | Venue | O que faz | Parea por plasticidade medida? | Colide com |
|---|---|---|---|---|---|

## Veredito

_(preencher ao fim, escolhendo uma das quatro linhas da tabela de critérios)_
