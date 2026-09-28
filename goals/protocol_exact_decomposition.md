# Protocolo congelado — decomposição do braço `exact`

> Pré-registro. Escrito **antes** de qualquer seed desta decomposição.
> Emendas só valem antes da primeira seed e devem ser commitadas antes do disparo.

**Estado:** congelado em 28/09/2026. Nenhuma seed executada.

---

## D1. Pergunta

O braço `exact` faz **duas** coisas simultaneamente:

1. congela a down-projection `A` — isto é **LoRA-FA** (arXiv 2308.03303, 2023),
   técnica publicada, depois reusada por **LoRA-Null** (arXiv 2503.02659, 2025)
   com a mesma motivação de preservação de conhecimento que a nossa;
2. mascara as linhas de `B` com um `SlowHeatFFNTracker` — isto é nosso.

O contraste confirmado `exact − lr_control` (forgetting −0,0921, mediana
−0,0998, 9−/1+, p = 0,02148, `E_eff` = 0,85) mede a **soma** dos dois. Não
existe hoje nenhum braço que separe as parcelas.

Pergunta: **quanto do ganho do `exact` vem de LoRA-FA e quanto vem do SlowHeat?**

## D2. Braços (congelados)

| braço | `A` | máscara em `B` | o que é |
|---|---|---|---|
| `vanilla` | treinável | não | LoRA padrão |
| `frozen_a_control` | **congelada** | **não** | LoRA-FA puro |
| `exact` | congelada | sim | LoRA-FA + SlowHeat |

`frozen_a_control` está em `UNMASKED_METHODS` e fora de `TRACKED_METHODS`: não
constrói tracker, não registra binding, não toca o otimizador. A única
diferença para `exact` é a maquinaria SlowHeat.

O congelamento de `A` é dirigido por `FROZEN_A_METHODS`, um conjunto
compartilhado, para que os dois braços não possam divergir por edição
acidental. Há teste que falha se essa via for duplicada.

## D3. Contrastes e família confirmatória

| contraste | isola |
|---|---|
| `frozen_a_control − vanilla` | efeito de **LoRA-FA sozinho** |
| `exact − frozen_a_control` | efeito do **SlowHeat dado LoRA-FA** |

**Família confirmatória: 2.** Holm sobre os dois. Endpoint primário:
**forgetting** (mesmo endpoint da confirmação original do `exact`, para que os
resultados sejam comparáveis). FAA como secundário, reportado sem correção e
declarado exploratório.

Teste: sinal exato bilateral, zeros descartados. n = 10 → p mínimo atingível
2/2^10 = 0,00195, com folga suficiente para α = 0,05 após Holm.

## D4. Seeds

Banda 5.000.003+, **disjunta das 243 seeds já reservadas** (verificado
mecanicamente contra todos os artefatos em `results/`):

```
5000003, 5025016, 5050029, 5075042, 5100055,
5125068, 5150081, 5175094, 5200107, 5225120
```

Registradas em código e verificadas por teste.

## D5. Pareamento

Todos os braços compartilham data stream, tokenização, otimizador, `rank`,
`alpha` (scaling = alpha/r constante) e orçamento de passos. `frozen_a_control`
tem **menos parâmetros treináveis** que `vanilla` — é inerente ao LoRA-FA e
**não** deve ser "corrigido" aumentando o rank, porque isso mudaria o mecanismo
sob teste. A contagem de parâmetros de cada braço é reportada na tabela.

`exact` e `frozen_a_control` têm exatamente a mesma contagem de parâmetros
treináveis, então o contraste que nos interessa já é pareado em capacidade.

## D6. Desfechos e o que cada um significa

**(a) `frozen_a_control − vanilla` significativo, `exact − frozen_a_control` não.**
Todo o ganho do `exact` é LoRA-FA. A contribuição do SlowHeat em LoRA é nula.
Consequência: o `exact` sai do artigo como método e entra como replicação
creditada de LoRA-FA. Este é o desfecho que o restante da evidência torna
plausível — os quatro mecanismos finos já ficaram abaixo do `vanilla`, e a
ablação de critério (28/09) mostrou que `|z·dL/dz|` empata com `|z|`.

**(b) Ambos significativos.**
LoRA-FA ajuda e o SlowHeat adiciona sobre ele. Reivindicação defensável e
honesta: "SlowHeat adiciona X p.p. sobre LoRA-FA", com LoRA-FA citado como
baseline publicado.

**(c) `exact − frozen_a_control` significativo, o outro não.**
O ganho é do SlowHeat, e congelar `A` sozinho não basta. Melhor desfecho
possível para a tese; exigiria explicar por que LoRA-FA não ajuda aqui.

**(d) Nenhum significativo.**
O `exact` não se sustenta em 10 seeds. Fecha a linha LoRA.

Os quatro são publicáveis. (a) e (d) são negativos e **serão reportados**.

## D7. Regras invioláveis

- não rodar seeds extras para empurrar qualquer contraste à significância;
- não recalcular Holm sobre família reduzida;
- não trocar o endpoint primário depois de ver resultado;
- não apagar resultados negativos;
- emendas só antes da primeira seed, commitadas antes do disparo.

## D8. Custo

**A medir com 1 seed antes de disparar as 10.** Estimativa por analogia já
falhou duas vezes neste projeto (erro de 4,7×), então o custo entra aqui como
número medido ou não entra.

Host: Qwen, GPU. Requer autorização explícita do usuário antes do disparo.

## D9. O que esta decomposição NÃO decide

Não diz se o SlowHeat funciona em geral — diz se ele adiciona algo **sobre
LoRA-FA, neste host, neste benchmark**. O resultado do BERT (+12,23 pp de
`magnitude` sobre aleatório, 10/10 seeds) mostra que a proteção seletiva
funciona lá; a geometria do LoRA, com `A` compartilhada, é diferente.

Também não decide o critério de importância: isso foi medido separadamente em
`protocol_importance_criterion_ablation.md`.
