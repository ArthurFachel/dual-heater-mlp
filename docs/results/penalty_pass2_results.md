# Passada 2 — Q1 falsificada, mas num regime de quase-chance

**Data:** 2026-10-05
**Protocolo:** `goals/protocol_penalty_pass2.md`, congelado em `22461f5`,
**antes** da primeira seed da banda 9.500.011+.
**Artefatos:** `results/penalty_pass2/` — 12 manifests, agregado em
`aggregate.json`.
**Agregador:** `scripts/analyze_penalty_pass2.py` (11 testes, 10 mutações mortas).
**Antecedentes:** `docs/results/lambda_sweep_results.md` (L3, que forneceu o
escalar), `docs/results/sgd_plasticity_results.md` (L2).

> ⚠️ **Primeira passada da Fase 2 que lê acurácia.** As anteriores foram
> mecanismo-only. O resultado abaixo é o resultado.

---

## Resumo em uma linha

O MAS a 30× bate o controle pareado em esquecimento, unânime em 12/12 seeds,
`p = 0,00049` — **e os três arms terminaram perto do nível de chance**, o que
torna o contraste não interpretável como preservação de conhecimento.

## Desenho

| item | valor |
|---|---|
| host | Split-MNIST class-IL, MLP (256, 128), 5 tarefas de 2 classes |
| otimizador | SGD puro, `lr = 3e-3` |
| arms | `vanilla`, `mas` (λ = 30, 30× Hsu), `lr_control` (`lr × 0,8541`) |
| `lr_scale` | `0,8541056558955461`, congelado do L3 antes de qualquer acurácia |
| seeds | 12, banda 9.500.011+ |
| primário | `mas − lr_control` em esquecimento médio |
| família | 1 comparação, sem Holm |
| custo | ~8 min de CPU, 4 shards paralelos |

## Resultado

| arm | esquecimento | dp | acurácia final | dp |
|---|---:|---:|---:|---:|
| `vanilla` | 0,8381 | 0,0556 | 0,1487 | 0,0243 |
| `mas` | **0,7536** | 0,0645 | **0,1976** | 0,0269 |
| `lr_control` | 0,7937 | 0,0584 | 0,1601 | 0,0294 |

### Veredito das três predições

| # | predição | veredito | evidência |
|---|---|---|---|
| **Q1** | `mas − lr_control` **não** é significativo | **FALSIFICADA** | −0,0400, **12−/0+**, `p = 0,00049` |
| **Q2** | `lr_control − vanilla` reduz esquecimento | **CONFIRMADA** | −0,0445, `p = 0,00049`. O controle é um controle |
| **Q3** | `mas` tem acurácia **menor** que `vanilla` | **FALSIFICADA** | 0,1976 contra 0,1487 — o MAS ficou **acima**, em 11/12 seeds |

A ordenação `mas < lr_control < vanilla` em esquecimento vale em **12 de 12
seeds**, sem exceção.

## A ressalva que domina a leitura

**Os três arms terminaram em quase-chance.** Com 5 tarefas de 2 classes em
class-IL, o nível de chance ao fim da sequência é 0,10. As acurácias finais
foram 0,149 / 0,198 / 0,160 — **o melhor arm não chega a duas vezes o chão**.

Isso não invalida a aritmética do contraste, mas muda o que ele significa:

> Um contraste de esquecimento entre modelos que mal aprenderam não mede
> preservação de conhecimento. Mede quanto cada arm deixou de adquirir.

O esquecimento de 0,84 do `vanilla` é consistente com um modelo que aprende cada
tarefa e perde quase tudo na seguinte — o colapso total típico de class-IL sem
replay. Os 0,75 do `mas` são um colapso ligeiramente menor, não uma retenção.

**O mecanismo mais provável para a vantagem do MAS neste regime é trivial:** a
30× a força publicada, a penalidade segura os pesos perto da âncora com força
suficiente para que o modelo mude menos — e um modelo que muda menos tanto
esquece menos quanto aprende menos. Que a acurácia tenha subido em vez de cair
(Q3 falsificada) é compatível com isso: num regime onde o vanilla colapsa para
quase-chance, qualquer coisa que impeça o colapso completo melhora as duas
métricas ao mesmo tempo. **Não é evidência de consolidação seletiva.**

### Por que o desenho não previu isso

O §D.2 do protocolo exigia reportar a acurácia final justamente para pegar o
caso "reduziu esquecimento colapsando a aquisição", e o agregador tinha a
verificação `flags_acquisition_collapse`. Mas ela **compara arms entre si**, e
por isso é cega para o caso em que todos colapsam juntos: a ordenação pode ser
perfeitamente consistente, com `p` de 0,00049, e os três modelos mal terem
aprendido.

A verificação que faltava (`flags_near_chance_regime`, comparando o melhor arm
contra o nível de chance) foi escrita **depois de ver os dados** e está marcada
como tal no código. Ela não muda nenhum número desta passada — muda o que o
relatório é obrigado a dizer sobre eles.

## Interpretação

**O que esta passada estabelece:** no regime testado, o MAS a 30× produz menos
esquecimento que um controle que remove a mesma plasticidade sem consolidação,
de forma unânime e com margem clara sobre o limiar.

**O que ela não estabelece:**

1. **Nada sobre o MAS publicado.** `mas_lambda = 30` é 30× o de Hsu et al. O L3
   escolheu esse ponto porque é o menor com controle construível, não porque
   alguém o propôs.
2. **Nada sobre preservação de conhecimento**, pelo argumento de quase-chance
   acima. Para separar "consolidação" de "mudou menos" seria preciso um regime
   onde o `vanilla` de fato aprende — mais épocas, lr maior, ou replay.
3. **Nada sobre EWC ou SI.** O L3 mostrou que para eles não existe λ com
   controle construível; não estão nesta passada.
4. **Nada sob AdamW**, onde nenhum controle é construível (G8).

## Limites

1. **Regime de quase-chance**, discutido acima. É o limite dominante.
2. **Um host, um λ, um otimizador, um lr.**
3. **O pareamento é exato na média, aproximado por seed.** O `lr_scale` é um
   escalar único; o `E` do MAS variou entre 0,829 e 0,879 no L3.
4. **12 seeds detectam efeitos grandes.** Um efeito em 9 de 12 seeds daria
   `p = 0,146` e seria reportado como nulo.
5. **O `E` foi medido no L3, não aqui.** Os hiperparâmetros são os mesmos por
   construção, verificado por teste, mas a medição é de outra run.

## O que isto significa para o artigo

**Não muda a Seção 4, e é importante dizer por quê.** O eixo do ICML é o
instrumento e o que ele torna visível, não a eficácia de nenhum método. Esta
passada acrescenta:

> *Quando o contraste pareado finalmente é construível — e ele só é para um dos
> três métodos, a 30 vezes a força publicada — o método supera o controle em
> esquecimento, unanimemente. Mas no regime em que isso foi medido nenhum dos
> arms escapou do nível de chance, de modo que o contraste não separa
> consolidação de simples redução de mudança. A dificuldade de construir o
> controle e a dificuldade de interpretá-lo têm a mesma origem: forças de
> penalidade altas o bastante para produzir um controle com poder são altas o
> bastante para dominar o treino.*

Isso é mais honesto e mais útil que um resultado positivo limpo, porque é o
padrão que a Seção 4 vem documentando desde a passada 1.

**O que NÃO sustenta:** nenhuma afirmação de superioridade do MAS sobre
qualquer baseline, em qualquer configuração publicada.

## Nota de método

Q1 foi declarada como **nula** no §E, e foi falsificada. Isso é registrado como
tal: a predição errou. O §E.1 também declarava, antes das seeds, que um Q1
falsificado "exigiria replicação antes de qualquer afirmação forte" — e a
ressalva de quase-chance é motivo adicional para não fazer nenhuma.

Q3 também foi falsificada, e na direção oposta à esperada. As duas falsificações
apontam para a mesma coisa: o desenho assumiu um regime de treino que não era o
regime real.

## Registro de alterações

| data | alteração |
|---|---|
| 2026-10-05 | resultado da passada 2, 12 seeds, ~8 min de CPU. Q1 falsificada (12/12, p = 0,00049), Q2 confirmada, Q3 falsificada. Acrescentada ao agregador a verificação `flags_near_chance_regime`, escrita **após** ver os dados, que sinaliza que nenhum arm escapou do nível de chance. Nenhum número da passada muda por causa dela. |
