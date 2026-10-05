# Material salvo do manuscrito anterior

**Origem:** `article/manuscript.md`, arquivado em
`article/_archive/manuscript_functional_slowheat.md` (commit deste movimento).
**Item 5.2** de `goals/roadmap_icml_ijcnn.md`: *"canibalizar
`article/manuscript.md` (seções 3 e 4.4 encaixam direto)"*.

Texto copiado **sem alteração de conteúdo**. As notas de enquadramento estão
marcadas como tal e são minhas, não do original.

---

## A. Por que escalar gradiente bruto não é escalar o update do AdamW

> **Enquadramento no ICML.** Isto é a Seção 3 do manuscrito anterior. Encaixa na
> **Seção 2** (effective plasticity): é o argumento de por que um fator aplicado
> ao gradiente não controla plasticidade sob otimizador adaptativo — exatamente
> o que o L2 depois **mediu** (`docs/results/sgd_plasticity_results.md`:
> `E > 1` era artefato do AdamW).
>
> O manuscrito anterior tinha o argumento analítico. A Fase 2 tem a medição em
> 12 seeds. **Juntos são a Seção 2;** separados, o argumento é teoria sem dado e
> a medição é dado sem explicação.

Um hook sobre o gradiente bruto aplica:

```text
g_i <- m_i * g_i
```

Para SGD puro, sem momentum nem weight decay, isso escala diretamente o update
do parâmetro. Sob normalização tipo Adam, um fator positivo persistente afeta o
primeiro momento aproximadamente de forma linear e o segundo aproximadamente de
forma quadrática:

```text
m_t proporcional a c * g
v_t proporcional a c^2 * g^2
m_t / sqrt(v_t) aproximadamente cancela c
```

O cancelamento não é exato em todo regime transiente, mas invalida a
interpretação do hook como taxa de aprendizado efetiva garantida. O AdamW também
aplica weight decay desacoplado fora do gradiente bruto.

O otimizador de pesquisa corrigido primeiro deixa o AdamW ou o SGD calcular o
delta nativo completo, e só então aplica a máscara de plasticidade:

```text
Delta_native = theta_after_native_step - theta_before_step
Delta_applied = M * Delta_native
theta <- theta_before_step + Delta_applied
```

Esse contrato é diretamente testável:

- máscara `1` reproduz o otimizador nativo;
- máscara `0` bloqueia movimento de gradiente **e** de weight decay;
- máscara `0.1` produz um décimo do update nativo final;
- o estado do otimizador sobrevive a ida e volta de checkpoint.

A política padrão `follow_update` aplica a mesma interpolação a deltas de estado
do otimizador com valor tensorial. A política `native` mantém a evolução dos
momentos sem máscara, como ablação explícita. Contadores escalares de passo do
AdamW permanecem globais, uma limitação de usar o layout nativo de estado do
otimizador do PyTorch.

---

## B. Correção da restauração do modo de avaliação

> **Enquadramento no ICML.** Isto é a Seção 4.4 do manuscrito anterior. É o
> **precedente** da Seção 3: a primeira vez em que um bug de instrumento
> invalidou runs já executadas, e o projeto descartou em vez de reinterpretar.
>
> A Seção 3 do ICML narra o caso R-E, em que uma métrica errada passou por cinco
> gates. Este episódio é o anterior da mesma família e estabelece que descartar
> é a prática da casa, não uma reação pontual.

Uma auditoria em 02/09/2026 encontrou que a avaliação de acurácia no fim de
época chamava `model.eval()` sem restaurar o modo anterior do learner. Hooks de
importância funcional só acumulam enquanto o modelo está em modo de treino. Num
sweep de SlowHeat que não contivesse nenhum método FastHeat, a importância era
portanto acumulada apenas durante a primeira época de cada estágio. Sweeps que
continham um método FastHeat acabavam restaurando todos os learners na época
seguinte, o que expunha uma dependência inválida da composição do sweep.

A avaliação passou a usar um contexto sem efeito colateral que restaura o modo
anterior, e acurácia class-incremental e task-aware compartilham os mesmos
forward passes. Testes de regressão verificam a restauração de modo, a
independência da presença de um método FastHeat e a concordância exata entre
`vanilla` e o controle SlowHeat sem consolidação.

A confirmação congelada de Split-MNIST contém apenas Replay e SlowHeat+Replay,
então **qualquer run feita antes desta correção é inválida**. Suas seeds
congeladas, hiperparâmetros, endpoint e análise permanecem inalterados; uma run
válida precisa usar o código corrigido e um diretório de saída novo.

As duas runs de confirmação versionadas satisfazem esse requisito. O contexto de
avaliação sem efeito colateral (`experiments/evaluation.py`) entrou no commit
`d5b22ad` (03/09/2026), que é o commit registrado no manifesto de ambiente da
primeira run; a segunda usa `f2f7616` (04/09/2026). As duas escreveram em
diretórios novos.

---

## C. O que NÃO foi salvo, e por quê

O manuscrito anterior tem 604 linhas e 12 seções. O item 5.2 identificou duas
como aproveitáveis direto. As demais ficam no arquivo pelos seguintes motivos:

| seção do original | destino |
|---|---|
| 1 Motivation, 2 Method | são sobre o **SlowHeat como mecanismo**; o ICML é sobre o instrumento de plasticidade. Reescrever, não reusar |
| 4.1–4.3, 4.5 | protocolo e confirmação de Split-MNIST — evidência de outro artigo |
| 5 Historical Diagnostic Pilot | o próprio original marca como **superseded**; não citar |
| 6 Convolutional Benchmarks | exploratório, 10 seeds, sem pré-registro |
| 7 BERT/CLINC150 | é **corpo do IJCNN** pela regra de não-sobreposição |
| 8 Hard versus Soft | resultado negativo de regime; candidato à Seção 5, mas precisa de reescrita, não de cópia |
| 9–12 | related work, experimentos pendentes, safe claims, reprodutibilidade — todos desatualizados pela Fase 2 |

**Nada foi apagado.** O original está em
`article/_archive/manuscript_functional_slowheat.md` com cabeçalho explicando o
arquivamento.
