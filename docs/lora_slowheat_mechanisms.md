# SlowHeat em LoRA: três mecanismos

Status: **protótipo implementado e testado; resultados exploratórios de uma
seed**. Nenhum número aqui sustenta comparação inferencial entre braços.

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
entre esses dois pontos.

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

**Custo:** o tracker tem `r` unidades em vez de `intermediate_size`. Para
`r=8` contra `d_ff=4864`, são três ordens de magnitude a menos de estado.

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

**Falsificador, e ele dispara:** uma vez que `B` fica densa, quase toda linha
`j` alcança alguma saída protegida, `a_j` colapsa e o método degenera em
"congelar A" com custo extra. Isso não é hipótese: está assertado em
`test_leak_bound_collapses_once_b_is_dense`, onde uma única saída protegida com
`B` densa leva `leak_collapse_fraction` a 1,0. O runner mede essa fração por
estágio; **ela deve ser reportada junto com qualquer resultado desse braço.**

### `slice` — fatiamento do rank por fronteira

Particiona `r` entre tarefas. As dimensões de tarefas passadas ficam congeladas
nos dois fatores, com rank total fixo (sem crescimento do adaptador).

**Exatidão:** total para as contribuições das tarefas anteriores.

**Falsificador:** empata com "resetar o adaptador por tarefa e somar os deltas".
Se empatar, a alocação não está fazendo trabalho nenhum.

### Braços de referência

- `vanilla`: LoRA puro, sem tracker e sem máscara.
- `exact`: Solução A, `A` congelada e linhas de `B` mascaradas. Espelha
  `build_exact_slowheat_lora` num host Qwen.

## 3. Decisões de implementação

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

## 4. Verificação

`tests/test_lora_slowheat.py`, 29 testes. A suíte passou na primeira execução,
o que a torna **não verificada, não correta**. Foram injetadas cinco mutações no
código de produção:

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

## 5. Protocolo do benchmark

`experiments/qwen_lora_slowheat.py`. Split-CLINC150 class-incremental, primeiros
`--tasks` domínios oficiais, uma seed, uma GPU. Todos os braços compartilham
stream de dados, tokenização, família de otimizador, learning rate, número de
épocas e código de avaliação idênticos; só o mecanismo de mascaramento muda.

Métricas por braço: FAA, forgetting médio e BWT (código compartilhado em
`dual_heater.metrics`), segundos de wall-clock, tokens de treino (padding
excluído), e pico de memória do alocador CUDA.

Diagnósticos por estágio: `effective_plasticity` (valor médio da máscara sobre
todos os elementos mascarados), `protected_unit_fraction`,
`leak_collapse_fraction` e `frozen_rank_dims`.

## 6. Limitações que devem acompanhar qualquer resultado

- **Uma seed.** Não sustenta comparação inferencial entre braços. É medição de
  viabilidade e custo.
- **Os braços não são pareados em plasticidade efetiva.** Eles usam o mesmo
  `slow_strength` e o mesmo budget nominal, mas `effective_plasticity` medida
  difere entre mecanismos, então qualquer diferença de acurácia é parcialmente
  atribuível a quanta plasticidade foi removida. Parear via
  `solve_strength_for_plasticity_scoped` é pré-requisito para qualquer
  afirmação causal.
- **Falta o controle de LR reduzido.** Sem um braço `lr * E` sem máscara, não se
  pode distinguir o mecanismo de um simples agendamento de learning rate.
- **Só FFN.** GQA no Qwen2.5-0.5B (14 heads de query, 2 de KV) impede um tracker
  indexado por head de endereçar Q, K e V com um vetor só.
- **`leak` provavelmente colapsa.** Ver `leak_collapse_fraction` por estágio
  antes de interpretar qualquer número desse braço.
- **Prioridade não verificada.** O-LoRA, InfLoRA e a família de LoRA para CL não
  foram levantados. Vários desses trabalhos particionam ou ortogonalizam
  subespaço por tarefa, o que toca diretamente `slice` e em parte `rank`. A §7
  do manuscrito proíbe reivindicação de primazia sem essa checagem.

## Referências

- [Contrato do Functional SlowHeat](functional_slowheat.md)
- [SlowHeat em Transformers, seção 12 (mascaramento LoRA)](functional_slowheat_transformers.md)
- [Semântica do otimizador](optimizer_semantics.md)
- [LoRA](https://arxiv.org/abs/2106.09685)
