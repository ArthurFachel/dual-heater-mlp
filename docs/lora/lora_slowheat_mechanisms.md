# Aplicação 3 — Três mecanismos SlowHeat-em-LoRA

**Fonte:** `src/dual_heater/lora_slowheat.py` · **Testes:** `tests/test_lora_slowheat.py` (46)
**Estado:** implementados e verificados por mutação. **Nenhum dos três superou o
vanilla** no benchmark de 10 domínios × 10 seeds.

## 1. O problema

`DualHeatLoRALinear` (`src/dual_heater/lora.py`) aplica o hook de proteção ao
tensor `delta` do adaptador, no espaço de saída. Como `lora_A` é compartilhada
entre todas as linhas de saída, um update induzido por uma saída plástica ainda
altera o delta de uma saída protegida através de `lora_B`. A docstring do
próprio módulo declara isso: "o hook não garante proteção independente por
saída".

`build_exact_slowheat_lora` (`src/dual_heater/bert.py:1055`) resolve congelando
`A` inteira. A proteção passa a ser exata, ao custo de toda a plasticidade do
subespaço de entrada: todas as tarefas futuras ficam restritas ao mesmo
subespaço.

Os três mecanismos em `src/dual_heater/lora_slowheat.py` exploram o espaço
entre esses dois pontos. Cada um responde de forma diferente à pergunta *qual
unidade a proteção cobre*.

## 2. Os mecanismos

### `rank` — proteção no espaço de rank

A unidade protegida deixa de ser o neurônio de saída (uma unidade do modelo
base) e passa a ser a direção do gargalo, que é a unidade funcional própria do
LoRA:

```text
z = A x              # [B, T, r]
delta = B z * scaling
importancia_j = |z_j * dL/dz_j|         # r unidades, nao d_ff
```

Proteger `j` mascara a linha `j` de `A` (produtora) e a coluna `j` de `B`
(consumidora). É o mesmo padrão produtor/consumidor de `qwen.py:430-444`.

**Exatidão:** sob máscara hard, o termo `B[:, j] z_j` fica congelado para todas
as saídas, porque `z_j` depende apenas de `A[j, :]`. A proteção é exata **por
componente**, não por saída: a saída total continua se movendo pelas direções
livres. Testado em `test_rank_hard_mask_freezes_the_protected_component_exactly`.

O teorema A1 reescrito nesse espaço fica `E >= (r − P) / r`, enunciado bem mais
forte que a versão no espaço de saída, já que `r` é pequeno.

**Custo:** o tracker tem `r` unidades em vez de `intermediate_size`. Para
`r=16` contra `d_ff=4864`, são duas ordens de magnitude a menos de estado.

**Armadilha descoberta (testada):** com a inicialização padrão do LoRA (`B = 0`),
a importância no gargalo é **identicamente zero**, porque
`dL/dz = B^T dL/d(delta) = 0`. O estimador não coleta evidência nenhuma até `B`
sair da inicialização. Não é bug, é propriedade do estimador, e não deve ser
confundido com "não há direções importantes". Assertado em
`test_bottleneck_importance_is_zero_while_b_is_still_zero`.

**Falsificador:** se a importância no gargalo for quase uniforme entre as `r`
direções, não há o que selecionar e o mecanismo degenera em LR reduzido.

### `leak` — máscara ciente do vazamento, com `A` treinável

Mantém a unidade no espaço de saída e fecha o vazamento em vez de proibir `A`
treinável:

```text
f_i = fator de plasticidade da saida i
a_j = min_{i : |B[i,j]| > tau} f_i      # combinacao 'min'
a_j = sum_i w_ij f_i,  w_ij = B[i,j]^2 / sum_i B[i,j]^2    # combinacao 'weighted'
```

A linha `j` de `A` só é tão plástica quanto a saída mais protegida que ela
alcança. A combinação por `minimum` é a convenção da casa (`dynamic_matrix_mask`).

**Exatidão:** com máscara hard e `tau=0`, qualquer saída protegida alcançada
por `B[i,j] != 0` zera a linha `j`, dando proteção exata por saída **com `A`
treinável** — exatamente o que `DualHeatLoRALinear` alega e não entrega.
Testado em `test_leak_hard_mask_protects_an_output_row_exactly_with_a_trainable_a`,
com guarda de mutação em `test_leak_without_the_a_bound_lets_the_protected_output_drift`.

**Falsificador previsto, e o que de fato aconteceu:** a previsão era que, uma vez
que `B` fica densa, quase toda linha `j` alcança alguma saída protegida, `a_j`
colapsa e o método degenera em "congelar A" com custo extra. Isso está assertado
para máscara hard em `test_leak_bound_collapses_once_b_is_dense`, onde uma única
saída protegida com `B` densa leva `leak_collapse_fraction` a 1,0.

Na run real, com máscara **soft**, `leak_collapse_fraction = 0,0` em todos os
estágios. O colapso **não** ocorreu — ele é específico da máscara hard. O
mecanismo sobreviveu ao seu falsificador previsto e ainda assim foi o pior braço
da tabela.

### `slice` — fatiamento do rank por fronteira

Particiona `r` entre tarefas. As dimensões de tarefas passadas ficam congeladas
nos dois fatores, com rank total fixo (sem crescimento do adaptador). É a
"Solução B" da doc de Transformers, mas com posto total fixo e alocação decidida
pelo controlador de plasticidade declarada, não por heurística.

**Exatidão:** total para as contribuições das tarefas anteriores, por
construção — uma fatia congelada não recebe gradiente nenhum. Isso também
significa que o mecanismo **não usa informação nenhuma sobre o que é
importante**: é alocação cega.

**Problema que invalida o braço na run de 10 tarefas:** com `r = 16` e 10
tarefas, cada tarefa recebe 1–2 direções, e `E_eff` medido foi **0,062**. O
braço não foi avaliado pelo mecanismo; foi penalizado por remoção de capacidade.

**Falsificador:** empata com "resetar o adaptador por tarefa e somar os deltas".
Se empatar, a alocação não está fazendo trabalho nenhum.

### Braços de referência

- `vanilla`: LoRA puro, sem tracker e sem máscara.
- `exact`: Solução A, `A` congelada e linhas de `B` mascaradas. Espelha
  `build_exact_slowheat_lora` num host Qwen. Ver
  [aplicação 2](lora_exact_producer_only.md).
- `lr_control`: remove a **mesma** quantidade de plasticidade, mas espalhada
  uniformemente via learning rate (`lr · E`) em vez de seletivamente via
  máscara. É o falsificador obrigatório: um mecanismo que não supera o
  `lr_control` não está fazendo nada que um escalar não faça.

## 3. Resultados (10 domínios, 10 seeds, sem pareamento de plasticidade)

Contrastes pareados contra vanilla, teste de sinal exato bicaudal:

| mecanismo | métrica | dif. média | p | vitórias | E_eff |
|---|---|---|---|---|---|
| `rank` | FAA | −0,0032 | 1,0000 | **5/10** | 0,645 |
| `rank` | forgetting | +0,0044 | 1,0000 | 5/10 | |
| `leak` | FAA | −0,0058 | 0,3438 | 3/10 | 0,659 |
| `leak` | forgetting | +0,0033 | 0,7539 | 6/10 | |
| `slice` | FAA | +0,0188 | 0,3438 | 7/10 | 0,062 |
| `slice` | forgetting | −0,0243 | 0,3438 | 7/10 | |

Nenhum sobrevive a Holm. `rank` é o resultado mais nulo possível (5/10, p=1,0).
`leak` teve o pior FAA médio e o maior pico de memória. `slice` tem o sinal
certo nas duas métricas, mas com 6% da plasticidade do vanilla — o que torna o
número notável e não interpretável ao mesmo tempo.

Detalhes completos, custo e ressalvas em
[lora_qwen_benchmark_results.md](lora_qwen_benchmark_results.md).

## 4. Decisões de implementação

**PEFT continua sendo a autoridade do forward.** A instrumentação só registra
forward hooks e lê os pesos de `lora_A`/`lora_B`. Nenhum forward de LoRA é
reimplementado, então dropout do adaptador, cast de dtype, adapters
desabilitados e adapters merged mantêm a semântica do PEFT.

**Os trackers não são submódulos do modelo.** PEFT e Transformers percorrem
`named_modules` ao salvar e ao decidir o que treinar; estado de mecanismo não é
nem uma coisa nem outra. `QwenLoRASlowHeat.to()` os co-loca com o modelo.

**A cabeça de classificação fica plástica e sem máscara em todos os braços.**
Congelar uma cabeça recém-inicializada torna uma run Class-IL estruturalmente
incapaz de aprender qualquer rótulo. A exceção é declarada, não acidental.

**fp32, não fp16.** O passo pontual mascarado do `SlowHeatAdamW` rejeita
explicitamente `GradScaler` (`optim.py:599-602`). Além disso, o config publicado
do Qwen declara `bfloat16`, que as Pascal não suportam nativamente, e a cabeça
criada pelo `modules_to_save` do PEFT nasce em fp32 — sem forçar o dtype, o
corpo e a cabeça divergem e o forward quebra.

## 5. Controlador de plasticidade declarada

`calibrate_to_target_plasticity(target)` resolve, por bisseção em cada
fronteira, o botão que faz a plasticidade **medida** igualar um alvo declarado.
O hiperparâmetro do método passa a ser plasticidade retida — grandeza medida
antes de qualquer acurácia ser lida.

Qual botão é resolvido depende do mecanismo:

- `exact`, `rank`, `leak`: `slow_strength` (monotonicamente decrescente em E).
- `slice`: o **piso** aplicado às direções congeladas. Sob partição estrita uma
  direção congelada vale exatamente zero, então a única forma de elevar E é
  relaxar esse piso — o que **abre mão da exatidão** e existe apenas para
  igualar custo.
- `vanilla`, `lr_control`: sem botão, E = 1 por definição.

Alvos abaixo do piso analítico `(N − P)/N` são inalcançáveis e o controlador
reporta `solved = 0.0` em vez de aproximar em silêncio.

## 6. Verificação

46 testes, todos passando. A suíte passou na primeira execução, o que a torna
**não verificada, não correta**. Foram injetadas cinco mutações no código de
produção:

| Mutação | Resultado |
|---|---|
| `rank`: eixo das máscaras invertido (linhas <-> colunas) | 4 testes falharam |
| `leak`: binding em `A` removido | 2 testes falharam |
| `slice`: fatia congelada invertida | 2 testes falharam |
| `rank`: tracker no espaço de saída em vez do gargalo | 6 testes falharam |
| máscara de validade ignorada no hook | **sobreviveu** |

A quinta sobreviveu porque o efeito é duplamente garantido: atenção causal mais
pooling no último token não-pad já zeram o gradiente das posições de padding, de
modo que remover a máscara não muda nenhum número end-to-end. O contrato foi
então assertado estruturalmente no call site
(`test_the_hook_forwards_the_scoped_validity_mask_to_the_tracker`), e a mutação
passou a morrer. O teste comportamental de padding permanece como guarda de
regressão, não como evidência de que a máscara funciona.

## 7. Protocolo do benchmark

`experiments/qwen_lora_slowheat.py` (uma seed) e `experiments/qwen_lora_sweep.py`
(multi-seed, multi-GPU). Split-CLINC150 class-incremental, primeiros `--tasks`
domínios oficiais. Todos os braços compartilham stream de dados, tokenização,
família de otimizador, learning rate, número de épocas e código de avaliação
idênticos; só o mecanismo de mascaramento muda. Tokens idênticos entre braços
(205.570 ± 568) servem de checksum desse pareamento.

Métricas por braço: FAA, forgetting médio e BWT (código compartilhado em
`dual_heater.metrics`), segundos de wall-clock, tokens de treino (padding
excluído), e pico de memória do alocador CUDA.

Diagnósticos por estágio: `effective_plasticity`, `protected_unit_fraction`,
`leak_collapse_fraction` e `frozen_rank_dims`.

## 8. Limitações que devem acompanhar qualquer resultado

- **Os braços da primeira run não são pareados em plasticidade efetiva.** Eles
  usam o mesmo `slow_strength` e o mesmo budget nominal, mas `E_eff` medida
  difere muito entre mecanismos (0,062 a 0,923), e a ordenação do FAA segue
  `E_eff` quase monotonicamente. A run iso-plasticidade corrige isso.
- **Só FFN.** GQA no Qwen2.5-0.5B (14 heads de query, 2 de KV) impede um tracker
  indexado por head de endereçar Q, K e V com um vetor só.
- **Prioridade não verificada.** O-LoRA, InfLoRA e a família de LoRA para CL não
  foram levantados. Vários desses trabalhos particionam ou ortogonalizam
  subespaço por tarefa, o que toca diretamente `slice` e em parte `rank`. A §7
  do manuscrito proíbe reivindicação de primazia sem essa checagem.

## Referências

- [Índice das aplicações de LoRA](lora_applications.md)
- [Resultados do benchmark](lora_qwen_benchmark_results.md)
- [Contrato do Functional SlowHeat](../mechanisms/functional_slowheat.md)
- [SlowHeat em Transformers, seção 12 (mascaramento LoRA)](../mechanisms/functional_slowheat_transformers.md)
- [Semântica do otimizador](../mechanisms/optimizer_semantics.md)
- [LoRA](https://arxiv.org/abs/2106.09685)
