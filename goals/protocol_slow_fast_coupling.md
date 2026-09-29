# Protocolo congelado — acoplamento `slow → fast` em LoRA

> Pré-registro. Escrito **antes** de qualquer implementação e de qualquer seed.
> Emendas só valem antes da primeira seed e devem ser commitadas antes do disparo.

**Estado:** congelado em 28/09/2026. Nada implementado, nenhuma seed executada.

---

## E1. Motivação

O componente `fast` (inibição lateral) nunca rodou em LoRA:
`lora_slowheat.py` não menciona `FastHeat` nenhuma vez. A metade "dual" do
dual-heat é, em LoRA, código não exercido.

Testar `fast` sozinho não se sustenta: ele não protege nada. É um operador de
*forward* sobre ativações, enquanto retenção em CL é governada por *o que o
backward preserva*. Qualquer efeito seu sobre esquecimento é necessariamente
**mediado** pelo SlowHeat.

Logo, a pergunta não é "onde encaixar o fast", mas **como os dois se acoplam**.

## E2. As duas vias de acoplamento

Os operadores vivem em espaços distintos e não somam:

```
fast:  z ← z · g(f)                       (forward, ativações)
slow:  ∇ ← ∇ · 1/(1 + β·s)                (backward, gradientes)
```

O único ponto de contato possível é o **sinal de importância**.

**Via (a), `fast → slow`.** O que o slow mede já passou pelo gate:
`s ← EMA(|z·g(f)| · |dL/d(z·g(f))|)`. É o que `DualHeatLoRALinear` (código
morto, `lora.py:163-171`) implementa: `post_mag` é tomado **depois** do gate e
alimenta `fast_heat` e `slow_heat` ao mesmo tempo.

**Esta via está em grande parte pré-refutada.** A ablação de critério
(28/09, 10 seeds) mostrou que trocar `|z·dL/dz|` por `|z|` não muda o resultado
(−0,17 pp, 5+/5, p = 1,00) e que os rankings têm variância praticamente idêntica
(2,9124e-03 vs 2,9070e-03). Se *o que* se mede não importa, concentrar o `z`
antes de medir dificilmente importará. A via (a) **não será testada**.

**Via (b), `slow → fast`.** Não existe em nenhum host. Verificado:
`set_external_scale` é usado apenas para `global_topk` no BERT
(`bert.py:627,643`), nunca para transportar estado do slow.

É a via testada aqui.

## E3. Formulação

Unidades já consolidadas ficam **isentas da competição**:

```
g_i = 1 / (1 + γ · (1 - p_i) · mean_others(f))

p_i = get_lr_scales()_i = 1 / (1 + β · slow_heat_i)   ∈ (0, 1]
```

- `p_i → 1` (unidade plástica, nunca consolidada): compete normalmente,
  recuperando o gate padrão;
- `p_i → 0` (unidade protegida): `(1-p_i) → 1`... **atenção ao sinal**.

Correção da convenção: `get_lr_scales` devolve **plasticidade**, não proteção.
Valor alto = plástica. Portanto a isenção correta é:

```
g_i = 1 / (1 + γ · p_i · mean_others(f))
```

Unidade protegida (`p_i → 0`) fica isenta; unidade plástica (`p_i → 1`) compete
com força total. Esta é a forma congelada. Um teste deve travar a convenção,
porque trocar o sinal inverte a hipótese silenciosamente.

Semântica: *o que já foi aprendido não disputa espaço de representação com o
que está sendo aprendido agora.*

Antes da primeira consolidação `slow_heat = 0`, logo `p_i = 1` e o gate reduz
exatamente ao FastHeat padrão. A tarefa 1 é, por construção, idêntica ao
baseline — propriedade que deve ser verificada por teste bit-a-bit.

## E4. Ponto de inserção

Saída do delta do adaptador, `B·z`, **não** a saída da camada.

Justificativa: é o único espaço onde proteção mostrou efeito (`exact` mascara
linhas de `B`), e preserva `delta = 0` na inicialização, mantendo o modelo base
recuperável. Aplicar na saída da camada `h` violaria a regra registrada em
`lora_slowheat.py` ("PEFT continua sendo a autoridade do forward") e tornaria
`h ≠ W·x` mesmo com adaptador zerado.

## E5. Braços

| braço | fast | slow | acoplamento |
|---|---|---|---|
| `vanilla` | não | não | — |
| `exact` | não | sim | — |
| `fast_only` | sim | não | — |
| `dual_uncoupled` | sim | sim | nenhum (γ fixo) |
| `dual_coupled` | sim | sim | **`p_i` modula γ** |

`fast_only` existe para documentar que o componente sozinho não sustenta
retenção — previsão explícita, não hipótese.
`dual_uncoupled` é o controle que isola o **acoplamento** do mero fato de
haver dois mecanismos. Sem ele, `dual_coupled − exact` confundiria as duas
coisas, que é exatamente o erro que o braço `frozen_a_control` corrigiu no
`exact`.

## E6. Família confirmatória

| contraste | isola |
|---|---|
| `dual_coupled − exact` | o dual acrescenta algo ao slow sozinho |
| `dual_coupled − dual_uncoupled` | **o acoplamento** acrescenta algo |

**Família de 2**, Holm, sinal exato bilateral, endpoint primário **forgetting**
(mesmo dos demais protocolos LoRA, para comparabilidade). `fast_only − vanilla`
é reportado como descritivo, **fora** da família.

10 seeds, banda a reservar e verificar disjunta no momento da implementação.

## E7. Falsificadores (declarados antes)

O resultado nulo é o desfecho esperado. Estes dois medidores decidem se ele
terá **explicação mecânica** ou será apenas "não funcionou":

**F1 — absorção por `B`.** Em LoRA, escalonamento diagonal na saída do delta é
replicável escalando linhas de `B`. O otimizador pode desfazer o gate.
Medir `‖B‖_F` por camada ao longo do treino em `dual_coupled` e `exact`.
**Se `‖B‖` crescer proporcionalmente à inibição acumulada, a absorção está
confirmada** e o nulo é explicado: o gate é cancelado pelos pesos treináveis.

Há uma assimetria que torna o desfecho não-óbvio: quem poderia cancelar a
isenção é justamente a unidade cujo gradiente o slow atenua. F1 mede se essa
assimetria basta.

**F2 — o gate muda o ranking?** `top_k_overlap` entre o ranking de importância
do slow com e sem fast ativo. **Se o overlap passar de 0,95, o fast não altera
quem é protegido**, e qualquer efeito observado vem de outro lugar — o que
tornaria uma diferença positiva suspeita, não animadora.

F2 usa `aggregate_ranking_degeneracy()`, já implementada e validada na ablação
de critério.

## E8. Desfechos

**(a) Ambos os contrastes nulos, F1 confirma absorção.** O fast é cancelado
pelos pesos do adaptador. Resultado negativo com **explicação mecânica** —
publicável, e fecha a linha dual-em-LoRA definitivamente.

**(b) Ambos nulos, F1 não confirma absorção.** O gate sobrevive mas não muda
retenção. Evidência de que a competição de ativações é irrelevante para
esquecimento neste regime.

**(c) `dual_coupled − dual_uncoupled` positivo.** O acoplamento importa. Único
desfecho que sustenta reivindicação de mecanismo, e o mais improvável.

**(d) `dual_coupled − exact` positivo mas `− dual_uncoupled` nulo.** Ter dois
mecanismos ajuda, o acoplamento não. Reivindicação fraca, e honestamente
reportável como tal.

Todos serão reportados.

## E9. Expectativa registrada

**Previsão: desfecho (a) ou (b).** Três resultados independentes de 28/09
apontam na mesma direção: `lr_control` empata com máscara seletiva; o critério
de importância não importa; `frozen_a_control` aparentemente supera `exact`
(1 seed). O padrão é que **restringir o adaptador não ajuda em LoRA**.

A via (b) é diferente em espécie — realoca em vez de restringir — mas isso a
torna *não-refutada*, não promissora.

Registrar a previsão antes serve para que um resultado positivo seja tratado
com a desconfiança apropriada, e não como confirmação de intuição.

## E10. O que isto NÃO decide

- Não diz se o DualHeat funciona em geral: em MLP/BERT o `fast` age sobre
  ativações reais desde o passo 1, enquanto em LoRA multiplica `delta = 0` na
  inicialização e só ganha efeito conforme `B` cresce.
- Não testa a via (a), declarada pré-refutada em E2.
- Não testa inserção em `z = A·x` nem na saída da camada.

## E11. Pré-condição de execução

**Não disparar antes de a decomposição do `exact`
(`protocol_exact_decomposition.md`) fechar.** Se aquele resultado mostrar que
o ganho é todo de LoRA-FA, o braço `exact` aqui deixa de ser uma linha de base
significativa e este protocolo precisa ser reconsiderado — possivelmente com
`frozen_a_control` no lugar de `exact` como referência.

Execução em GPU requer autorização explícita do usuário.
