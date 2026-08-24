# Functional SlowHeat em Transformers e LLMs

Este documento descreve uma adaptação do Functional SlowHeat para blocos de
atenção e feed-forward de Transformers. A implementação deve começar em um
Transformer pequeno e só avançar para LLMs depois que o otimizador mascarado
deixar de depender de snapshots completos dos parâmetros e momentos.

## 1. Escolha das unidades funcionais

Um Transformer não possui uma única definição natural de neurônio. A proposta
usa duas unidades complementares:

1. **unidade intermediária da FFN**, incluindo a unidade composta de SwiGLU;
2. **cabeça de atenção**, com opção futura de granularidade por dimensão da
   cabeça.

Na primeira implementação, as dimensões do residual stream não devem ser
protegidas diretamente. Elas participam de residual, normalização, embeddings,
QKV, FFN e possivelmente de uma cabeça de linguagem com pesos amarrados. A
proteção correta exigiria registrar todas essas ramificações ao mesmo tempo.

## 2. Fluxo geral por tarefa

O ciclo de vida é igual ao Functional SlowHeat original:

```text
forward da tarefa
    -> observar ativações funcionais
backward
    -> calcular |z * dL/dz|
    -> reduzir batch e tokens
    -> normalizar por grupo de unidades
    -> atualizar task_ema
fronteira de tarefa
    -> consolidar max/mean/sum
    -> aplicar capacity budget
próxima tarefa
    -> mascarar o delta final do otimizador
```

Tokens de padding, posições ignoradas da loss e exemplos inválidos devem ser
excluídos da redução de utilidade.

## 3. Feed-Forward Network padrão

Considere:

```text
pre = up_proj(x)
h = activation(pre)
y = down_proj(h)
```

### Passo 1 — observar a unidade intermediária

O melhor ponto de observação é `h`, pois é a representação realmente consumida
por `down_proj`:

```text
h.shape = [batch, tokens, d_ff]
```

### Passo 2 — calcular a utilidade

```text
u[j] = sum_{b,t valid} |h[b,t,j] * dL/dh[b,t,j]|
```

Normalizar sobre `d_ff`:

```text
u_norm[j] = u[j] / max(mean(u), eps)
```

### Passo 3 — consolidar e aplicar budget

Com budget plástico `p_ff`:

```text
max_protected_ff = floor((1 - p_ff) * d_ff)
```

Selecionar as unidades de maior evidência e produzir `m_ff[j]`.

### Passo 4 — proteger os parâmetros

Uma unidade FFN `j` controla:

- a linha `j` de `up_proj.weight`;
- `up_proj.bias[j]`, se existir;
- a coluna `j` de `down_proj.weight`.

Na primeira versão, não aplicar fator de destino às linhas de `down_proj`, pois
essas linhas produzem o residual stream, que ainda não possui tracker próprio:

```text
M_up[j,i] = m_ff[j]
M_down[o,j] = m_ff[j]
```

## 4. FFN com SwiGLU ou GEGLU

Considere:

```text
g = gate_proj(x)
u = up_proj(x)
h = silu(g) * u
y = down_proj(h)
```

A unidade funcional é o produto `h[j]`. Medir `gate_proj` e `up_proj`
isoladamente não representa todo o caminho utilizado por `down_proj`.

### Passo 1 — observar o produto

```text
utility[j] = sum_{b,t valid} |h[b,t,j] * dL/dh[b,t,j]|
```

### Passo 2 — criar um grupo de parâmetros

Quando a unidade `j` for protegida, aplicar seu fator a:

```text
gate_proj.weight[j, :]
gate_proj.bias[j]
up_proj.weight[j, :]
up_proj.bias[j]
down_proj.weight[:, j]
```

Os dois produtores e o consumidor devem compartilhar o mesmo identificador de
unidade. O registrador deve aceitar múltiplas linhas produtoras para um único
tracker.

## 5. Atenção multi-head

Considere as projeções:

```text
Q = q_proj(x)
K = k_proj(x)
V = v_proj(x)
head_output = attention(Q, K, V)
y = o_proj(concat(head_output))
```

### Granularidade recomendada inicialmente

Usar uma importância por cabeça:

```text
importance.shape = [num_heads]
```

Isso preserva a estrutura natural de Q/K/V e reduz a complexidade do
mapeamento. A granularidade por dimensão pode ser adicionada depois.

### Passo 1 — remodelar as projeções

```text
Q.shape = [batch, tokens, num_query_heads, head_dim]
K.shape = [batch, tokens, num_kv_heads, head_dim]
V.shape = [batch, tokens, num_kv_heads, head_dim]
```

### Passo 2 — coletar sinais locais

Para cada cabeça, podem ser observados quatro sinais:

```text
u_q[h] = sum |Q_h * dL/dQ_h|
u_k[h] = sum |K_h * dL/dK_h|
u_v[h] = sum |V_h * dL/dV_h|
u_o[h] = sum |head_output_h * dL/dhead_output_h|
```

Todas as somas reduzem batch, tokens e `head_dim`.

### Passo 3 — produzir uma importância da cabeça

A regra conservadora inicial é:

```text
u_head[h] = max(u_q[h], u_k[h], u_v[h], u_o[h])
```

Também devem ser avaliadas média e soma como ablações. Antes da combinação, os
quatro sinais devem ser normalizados para impedir que um tensor domine apenas
por escala numérica.

### Passo 4 — aplicar o budget de cabeças

```text
max_protected_heads = floor((1 - p_attention) * num_heads)
```

Como o número de cabeças costuma ser pequeno, o arredondamento é relevante.
Por exemplo, com 32 cabeças e `p_attention=0.25`, no mínimo oito permanecem
plásticas.

### Passo 5 — mascarar Q/K/V

Para a cabeça `h`, as linhas correspondentes são:

```text
rows = h * head_dim : (h + 1) * head_dim
```

Aplicar `m_head[h]` a essas linhas de:

```text
q_proj.weight
k_proj.weight
v_proj.weight
```

e aos respectivos biases.

### Passo 6 — mascarar a projeção de saída

As features concatenadas da cabeça `h` ocupam as colunas:

```text
columns = h * head_dim : (h + 1) * head_dim
```

Aplicar o mesmo fator a:

```text
o_proj.weight[:, columns]
```

Assim, uma cabeça importante protege tanto suas projeções produtoras Q/K/V
quanto o caminho que transporta sua saída para o residual stream.

## 6. QKV fundido

Algumas implementações armazenam Q, K e V em um único parâmetro:

```text
in_proj_weight.shape = [q_size + k_size + v_size, hidden_size]
```

A máscara deve ser construída por fatias:

```text
mask[q_slice] = expanded_query_head_mask
mask[k_slice] = expanded_key_head_mask
mask[v_slice] = expanded_value_head_mask
```

O checkpoint deve registrar os offsets e a configuração de cabeças. Não basta
registrar que existe uma máscara com determinado shape, pois modelos MHA, GQA e
MQA podem ter o mesmo tamanho total com semânticas diferentes.

## 7. GQA e MQA

Em Grouped-Query Attention, vários query heads compartilham um KV head.

### Mapeamento

```text
queries_per_kv = num_query_heads / num_kv_heads
kv_head(q_head) = q_head // queries_per_kv
```

A importância de um KV head deve agregar todos os query heads que o consomem:

```text
u_kv[g] = max u_query_related[h] for h mapped to g
```

Usar `max` é conservador: se qualquer query head depender fortemente daquele
KV head compartilhado, suas linhas K/V são protegidas.

## 8. Granularidade por dimensão da cabeça

Uma versão posterior pode manter:

```text
importance.shape = [num_heads, head_dim]
```

Isso oferece budget mais fino, mas introduz novos cuidados:

- dimensões internas podem sofrer rotações ou mudanças de base;
- RoPE opera sobre pares de dimensões de Q/K;
- kernels e tensor parallel frequentemente particionam blocos maiores.

Com RoPE, o menor grupo seguro é um par:

```text
(dimension 2k, dimension 2k + 1)
```

O budget deve selecionar pares completos, nunca uma única coordenada do par.

## 9. Consolidação e orçamento em modelos grandes

### Buffers persistentes

Os buffers por unidade têm custo baixo:

```text
O(sum d_ff + sum num_heads)
```

Mesmo em modelos grandes, esse custo é muito menor que os parâmetros.

### Budget por camada

É a opção mais simples e garante capacidade em cada bloco, mas trata unidades
com custos diferentes como equivalentes.

### Budget ponderado

Uma alternativa mais fiel é ponderar cada unidade pelo número de parâmetros
que controla:

```text
cost_ff_unit ~= 2 * hidden_size
cost_attention_head ~= 4 * hidden_size * head_dim
```

O budget passa a reservar uma fração de custo plástico, não apenas uma contagem
de unidades. Isso exige seleção do tipo knapsack ou uma aproximação por razão
`importance / cost`.

### Budget hierárquico recomendado

Para LLMs:

1. mínimo plástico por camada;
2. mínimo plástico por família: FFN e atenção;
3. redistribuição global da capacidade restante;
4. limites para impedir uma camada totalmente protegida ou totalmente livre.

## 10. Treino distribuído

### Data parallel

Cada rank observa batches diferentes. Antes de atualizar a EMA global:

1. acumular numerador por unidade;
2. acumular contagem de tokens válidos;
3. executar `all_reduce` dos dois;
4. calcular a média global;
5. normalizar e atualizar a EMA de forma idêntica em todos os ranks.

Normalizar localmente antes do `all_reduce` muda a ponderação quando os ranks
possuem quantidades diferentes de tokens válidos.

### Tensor parallel

Quando QKV ou FFN estão particionados pela dimensão de unidades:

- cada rank pode manter utilidade de suas unidades locais;
- um budget local é barato, mas não garante top-k global;
- top-k global exige troca de importâncias ou um algoritmo distribuído;
- metadados de checkpoint devem usar índices globais das unidades.

### Pipeline parallel

Cada estágio pode consolidar suas próprias camadas, mas a chamada de fronteira
de tarefa precisa ser sincronizada entre todos os estágios.

## 11. Interação com activation checkpointing e atenção fundida

O tracker atual fecha sobre `z.detach()` em um hook. Em blocos com activation
checkpointing, essa referência pode reter storage e reduzir a economia de
memória pretendida.

A solução de produção deve usar uma destas estratégias:

1. operação autograd customizada que combine observação e redução;
2. recomputar a ativação necessária no backward;
3. salvar ativação quantizada ou em baixa precisão;
4. instrumentar apenas subconjuntos de passos para estimativa amostral.

Não se deve materializar nem instrumentar a matriz de atenção `[T, T]`. Q, K,
V e a saída da cabeça possuem custo linear em tokens e permitem preservar SDPA
ou FlashAttention com menos interferência.

## 12. Mascaramento LoRA

LoRA representa o update como:

```text
delta_weight = B @ A
A.shape = [rank, input_features]
B.shape = [output_features, rank]
```

### Por que mascarar somente B não basta

Para a saída `i`:

```text
delta_weight[i, :] = B[i, :] @ A
```

Mesmo que `B[i, :]` fique congelada, uma alteração em `A` muda essa saída. A
matriz `A` é compartilhada por todas as linhas.

### Solução A — A congelado

1. inicializar ou pré-treinar `A`;
2. congelar `A` antes do continual learning;
3. treinar apenas `B`;
4. aplicar o fator da saída à linha correspondente de `B`.

Com `A` fixo, proteger `B[i, :]` protege exatamente a linha efetiva da saída.
A limitação é restringir todas as tarefas futuras ao mesmo subespaço de
entrada.

### Solução B — bancos LoRA expansíveis

Para cada fase de capacidade:

1. congelar os bancos antigos `(A_old, B_old)`;
2. criar um banco novo `(A_new, B_new)`;
3. manter `B_new[i, :] = 0` para saídas já protegidas;
4. permitir updates apenas nas linhas plásticas;
5. consolidar e congelar o banco na próxima fronteira.

Isso oferece proteção exata das saídas antigas, mas o rank total cresce.

### Solução C — LoRA agrupado

Criar `A_g` e `B_g` independentes por:

- cabeça de atenção;
- grupo de cabeças;
- bloco de unidades FFN.

Quando o grupo for protegido, congelar ambos os fatores. A proteção é exata no
nível do grupo, não por dimensão individual.

### Recomendação

Começar com `A` congelado e `B` mascarado. Depois comparar com LoRA agrupado por
cabeça. Bancos expansíveis devem ser avaliados somente quando houver uma
política explícita de crescimento máximo de rank.

## 13. Otimizador em bilhões de parâmetros

O otimizador atual recupera o delta nativo clonando parâmetros e estados. Isso
é apropriado como implementação de referência, mas não para LLMs.

Se parâmetro e dois momentos forem FP32, os snapshots representam cerca de:

```text
3 tensors * 4 bytes * parameter_count
```

ou aproximadamente 12 GB temporários por bilhão de parâmetros registrados.

A implementação escalável deve aplicar a máscara dentro do kernel do
otimizador:

```text
native_moment_delta = compute_adam_moment_delta(...)
moment += mask * native_moment_delta

native_parameter_delta = compute_adamw_delta(...)
parameter += mask * native_parameter_delta
```

Requisitos adicionais:

- `foreach` ou kernel Triton/CUDA fundido;
- máscaras broadcast sem expandir para um tensor por peso;
- compatibilidade com parâmetros e estados sharded;
- checkpoint com IDs estruturais, não apenas posições em `param_groups`;
- teste explícito de `torch.compile` e graph breaks.

## 14. Esqueleto de módulos

```python
class SlowHeatFFNTracker(nn.Module):
    def observe(self, hidden, valid_tokens=None):
        # Registra |hidden * grad_hidden| reduzido em batch/tokens.
        ...


class SlowHeatAttentionTracker(nn.Module):
    def observe_qkv(self, q, k, v, valid_tokens=None):
        ...

    def observe_head_output(self, head_output, valid_tokens=None):
        ...


class TransformerMaskRegistry:
    def register_swiglu(self, tracker, gate_proj, up_proj, down_proj):
        ...

    def register_attention(self, tracker, q_proj, k_proj, v_proj, o_proj):
        ...
```

O tracker deve ser separado do `nn.Linear`, pois uma unidade SwiGLU controla
duas projeções produtoras e uma consumidora.

## 15. Plano de implementação no repositório

### Etapa A — Transformer pequeno, apenas FFN

- criar `FunctionalSlowHeatTracker` independente de camada;
- instrumentar a ativação intermediária de uma FFN ReLU/GELU;
- registrar linhas de `up_proj` e colunas de `down_proj`;
- validar em classificação sequencial pequena.

### Etapa B — SwiGLU

- observar o produto gated;
- agrupar `gate_proj`, `up_proj` e `down_proj`;
- testar que nenhuma das duas linhas produtoras escapa da proteção.

### Etapa C — atenção por cabeça

- começar com MHA sem QKV fundido;
- observar Q/K/V e saída da cabeça;
- proteger blocos de linhas e colunas;
- adicionar QKV fundido e depois GQA.

### Etapa D — LoRA

- substituir o protótipo baseado em magnitude por utilidade funcional;
- implementar `A` congelado e `B` mascarado;
- comparar LoRA agrupado por cabeça;
- medir rank útil e crescimento por tarefa.

### Etapa E — escala

- substituir snapshots por update fundido;
- integrar data e tensor parallel;
- medir throughput, pico de memória e graph breaks;
- somente então aumentar o número de parâmetros.

## 16. Testes mínimos de aceitação

1. Padding e tokens ignorados não alteram a utilidade.
2. A FFN produz um vetor `[d_ff]` de utilidade finita.
3. Uma unidade SwiGLU protege as duas projeções de entrada e a coluna de saída.
4. Uma cabeça protegida mascara Q, K, V e as colunas corretas de O.
5. QKV fundido produz as mesmas máscaras da versão com projeções separadas.
6. GQA agrega corretamente vários query heads por KV head.
7. O budget mantém capacidade mínima em FFN e atenção separadamente.
8. Máscara 1 coincide com AdamW nativo; máscara 0 bloqueia weight decay.
9. `follow_update` impede acúmulo local de momentos protegidos.
10. Com `A` congelado, atualizar uma linha plástica de `B` não altera saídas
    protegidas.
11. Checkpoint rejeita uma topologia com número de heads, offsets ou GQA
    incompatíveis.
12. O controle sem consolidação coincide com o Transformer vanilla.

## 17. Limitações que devem acompanhar os resultados

- GELU, SiLU, softmax e normalização limitam a invariância por reescala.
- Importância de cabeça é uma aproximação local e depende da granularidade.
- Budget por unidade não equivale a budget de parâmetros ou memória.
- Instrumentação pode reduzir ganhos de FlashAttention e checkpointing.
- Resultados em um Transformer pequeno não demonstram escalabilidade em LLMs.
- LoRA com `A` treinável compartilhado não oferece proteção independente por
  saída.

## Referências

- [Contrato do Functional SlowHeat](functional_slowheat.md)
- [Semântica do otimizador](optimizer_semantics.md)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [LoRA](https://arxiv.org/abs/2106.09685)
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- [GQA](https://arxiv.org/abs/2305.13245)
- [Documentação de MultiheadAttention do PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)
- [Documentação de SDPA do PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
