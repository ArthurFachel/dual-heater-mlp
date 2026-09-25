# Functional SlowHeat em RNNs, GRUs e LSTMs

Este documento descreve como adaptar o Functional SlowHeat para arquiteturas
recorrentes. A principal diferença em relação a MLPs é que os mesmos parâmetros
são reutilizados em todos os passos de tempo. A importância deve agregar essas
utilizações sem consolidar ou aplicar uma EMA dependente da ordem temporal.

## 1. Definição da unidade funcional

### RNN simples

Para:

```text
a_t = W_ih x_t + W_hh h_{t-1} + b
h_t = activation(a_t)
```

a unidade protegida é a coordenada `i` do estado oculto.

### LSTM

A unidade é a coordenada `i` da célula de memória, agrupando seus quatro gates:

```text
input gate
forget gate
candidate gate
output gate
```

Proteger apenas um gate não protege a função da célula completa.

### GRU

A unidade é a coordenada oculta `i`, agrupando as linhas correspondentes aos
três componentes empacotados da GRU.

## 2. Por que a acumulação temporal precisa mudar

O tracker atual usa uma EMA toda vez que seu hook dispara. Em uma recorrência
desenrolada, há uma ocorrência do mesmo módulo por tempo, e o backward costuma
percorrer essas ocorrências em ordem reversa.

Se a EMA for atualizada por ocorrência:

- tempos posteriores e anteriores recebem pesos diferentes;
- o resultado depende da ordem do backward;
- sequências mais longas realizam mais updates da EMA;
- batches com comprimentos diferentes tornam-se difíceis de comparar.

Por isso, a contribuição temporal deve ser acumulada com operações
comutativas:

```text
step_sum[i] += contribution_t[i]
step_count += valid_items_t
```

Somente depois de agregar a sequência ou o optimizer step deve-se atualizar a
EMA da tarefa:

```text
step_mean = step_sum / max(step_count, 1)
normalized = step_mean / max(mean(step_mean), eps)
task_ema = decay * task_ema + (1 - decay) * normalized
```

## 3. Fluxo passo a passo em uma RNN simples

### Passo 1 — inicializar o estado

Para hidden size `H`:

```text
importance_memory: [H]
task_ema:           [H]
slow_heat:          [H]
step_sum:           [H]
step_count:         escalar
```

### Passo 2 — executar a célula em cada tempo

```text
for t in sequence:
    a_t = linear_ih(x_t) + linear_hh(h_previous)
    h_t = activation(a_t)
```

Na primeira implementação, observar `h_t`, pois ele é tanto saída funcional
quanto fonte da recorrência seguinte.

### Passo 3 — coletar utilidade temporal

Quando o backward alcançar `h_t`:

```text
contribution_t[i] = sum_b valid[b,t] * abs(
    h_t[b,i] * dL/dh_t[b,i]
)
```

Adicionar a `step_sum` sem atualizar a EMA imediatamente.

### Passo 4 — finalizar a observação do batch

Depois do backward completo:

```text
tracker.finish_backward()
```

Esse método:

1. divide pela quantidade de posições válidas;
2. normaliza entre unidades;
3. atualiza `task_ema` uma única vez;
4. limpa `step_sum` e `step_count`.

O loop de treino deve falhar de forma segura se `optimizer.step()` for chamado
sem `finish_backward()` quando houver estatística pendente.

## 4. Máscaras fatoradas da RNN simples

### Peso input-hidden

Para:

```text
W_ih.shape = [H, input_size]
```

a máscara de destino é:

```text
M_ih[i,j] = m_hidden[i]
```

Se a entrada vier de outra camada SlowHeat compatível:

```text
M_ih[i,j] = min(m_hidden[i], m_input[j])
```

### Peso hidden-hidden

Para:

```text
W_hh.shape = [H, H]
```

a mesma unidade aparece como destino e fonte recorrente:

```text
M_hh[i,j] = min(m_hidden[i], m_hidden[j])
```

Isso protege:

- a linha `i`, que calcula a próxima ativação da unidade;
- a coluna `j`, que transporta a influência da unidade para o futuro.

### Bias

```text
M_bias[i] = m_hidden[i]
```

### Cabeça de saída

Para `y_t = W_out h_t`:

```text
M_out[o,i] = min(m_output[o], m_hidden[i])
```

Se a saída não possuir tracker, usar apenas `m_hidden[i]` nas colunas.

## 5. Consolidação na fronteira de tarefas

Depois do último backward da tarefa:

1. chamar `finish_backward()` para qualquer estatística pendente;
2. consolidar `task_ema` em `importance_memory`;
3. aplicar o budget por hidden layer;
4. produzir `slow_heat` e `m_hidden`;
5. limpar somente o acumulador da tarefa;
6. resetar ou destacar o estado recorrente conforme o protocolo.

```text
importance_memory = max(importance_memory, task_ema)
max_protected = floor((1 - plasticity_budget) * H)
m_hidden = 1 / (1 + beta * slow_heat)
```

Não se deve consolidar em cada timestep. Os timesteps pertencem à mesma
utilização compartilhada dos parâmetros dentro de uma tarefa.

## 6. Functional SlowHeat em LSTM

Uma LSTM calcula, de forma simplificada:

```text
i_t = sigmoid(W_ii x_t + W_hi h_{t-1} + b_i)
f_t = sigmoid(W_if x_t + W_hf h_{t-1} + b_f)
g_t = tanh(   W_ig x_t + W_hg h_{t-1} + b_g)
o_t = sigmoid(W_io x_t + W_ho h_{t-1} + b_o)
c_t = f_t * c_{t-1} + i_t * g_t
h_t = o_t * tanh(c_t)
```

### Opção 1 — utilidade do estado oculto

```text
u_h[i] = sum_{b,t valid} |h_t[b,i] * dL/dh_t[b,i]|
```

É a opção mais simples, mas pode subestimar a memória transportada por `c_t`.

### Opção 2 — estado oculto mais estado de célula

```text
u_cell[i] = sum |h_t[i] * dL/dh_t[i]|
          + lambda_c * sum |c_t[i] * dL/dc_t[i]|
```

Essa é a definição funcional recomendada para a segunda implementação.
`lambda_c` deve ser declarado e testado; uma alternativa é normalizar os dois
sinais separadamente antes da soma.

### Opção 3 — importância por gates

Coletar utilidade nas quatro preativações e agregar:

```text
u_cell[i] = max(u_input[i], u_forget[i], u_candidate[i], u_output[i])
```

Essa variante é conservadora e facilita diagnosticar qual gate torna uma
célula importante.

### Comparação necessária

As três opções devem ser tratadas como hipóteses experimentais. Não há motivo
para assumir que uma delas é universalmente superior antes de ablações.

## 7. Mapeamento dos pesos empacotados da LSTM

No PyTorch, sem projeção:

```text
weight_ih.shape = [4 * H, input_size]
weight_hh.shape = [4 * H, H]
bias_ih.shape   = [4 * H]
bias_hh.shape   = [4 * H]
```

Para uma célula `i`, as quatro linhas são:

```text
i
i + H
i + 2H
i + 3H
```

O fator deve ser repetido na ordem usada pelo framework:

```text
gate_row_factor = concat(
    m_hidden,
    m_hidden,
    m_hidden,
    m_hidden,
)
```

### Máscara input-hidden

```text
M_ih[row,j] = min(gate_row_factor[row], m_input[j])
```

ou somente `gate_row_factor[row]` quando a entrada não é protegida.

### Máscara hidden-hidden

```text
M_hh[row,j] = min(gate_row_factor[row], m_hidden[j])
```

### Biases

```text
M_bias_ih = gate_row_factor
M_bias_hh = gate_row_factor
```

Embora os dois biases sejam somados na função, ambos precisam acompanhar a
mesma proteção.

## 8. LSTM com projeção

Quando `proj_size > 0`, o estado de célula possui dimensão `H`, mas o estado
oculto recorrente possui dimensão `P = proj_size`.

Isso cria dois tipos de unidade:

- célula de memória `[H]`;
- coordenada projetada `[P]`.

A implementação deve usar trackers separados ou uma regra explícita de
agregação. A projeção:

```text
weight_hr.shape = [P, H]
```

recebe máscara fatorada:

```text
M_hr[p,h] = min(m_projected[p], m_cell[h])
```

Os pesos recorrentes consomem o estado projetado:

```text
weight_hh.shape = [4 * H, P]
M_hh[row,p] = min(gate_cell_factor[row], m_projected[p])
```

Essa variante não deve ser inferida apenas a partir dos shapes; o checkpoint
precisa registrar `hidden_size` e `proj_size`.

## 9. GRU

Para GRU, repetir o fator oculto três vezes na ordem de gates usada pela
implementação:

```text
gate_row_factor = concat(m_hidden, m_hidden, m_hidden)
```

Aplicar:

```text
M_ih[row,j] = min(gate_row_factor[row], m_input[j])
M_hh[row,j] = min(gate_row_factor[row], m_hidden[j])
```

Como algumas formulações aplicam o reset gate antes ou depois da transformação
recorrente, a adaptação deve ser testada contra a semântica exata do backend
utilizado.

## 10. Sequências variáveis e PackedSequence

Padding não pode contribuir para importância.

Com tensor padded:

```text
valid.shape = [batch, time]
contribution *= valid[..., None]
```

Com `PackedSequence`, a redução pode operar diretamente sobre `packed.data`,
mas ainda é necessário preservar a correspondência entre estados internos e
tempos quando a utilidade inclui `c_t` ou gates.

O denominador deve ser a quantidade real de itens válidos, não
`batch * max_length`.

## 11. BPTT completo e truncado

### BPTT completo

Captura contribuições de toda a sequência, mas o custo de memória cresce
linearmente com o comprimento.

### Truncated BPTT

Em cada janela:

1. executar forward e backward;
2. acumular a estatística daquela janela;
3. chamar `finish_backward()`;
4. destacar `h` e `c` antes da próxima janela;
5. continuar atualizando a mesma `task_ema`.

A importância passa a representar o horizonte truncado. Unidades úteis apenas
para dependências mais longas podem ser subestimadas. O tamanho da janela deve
ser salvo no protocolo e mantido igual entre métodos.

## 12. Fronteiras de tarefa e estado recorrente

Existem três políticas possíveis:

### Reset

```text
h = 0
c = 0
```

É a política mais limpa para tarefas independentes.

### Detach

```text
h = h.detach()
c = c.detach()
```

Mantém o contexto numérico sem permitir gradientes entre tarefas. Deve ser
usado apenas quando o stream realmente é contínuo.

### Estado específico por tarefa

Salvar estados separados por tarefa exige task ID e altera o cenário de
avaliação. Não deve ser usado em Class-IL sem ser declarado como informação
oracle adicional.

A política de estado precisa ser aplicada depois da consolidação e registrada
nos resultados.

## 13. Implementação fundida versus célula explícita

`nn.LSTM` e `nn.GRU` normalmente usam kernels fundidos. Eles são rápidos, mas
não expõem facilmente todas as preativações, gates e células intermediárias
necessárias para o tracker.

### Primeira implementação recomendada

Usar `SlowHeatRNNCell`/`SlowHeatLSTMCell` em um loop Python explícito. Isso
permite validar:

- utilidade temporal;
- agrupamento dos gates;
- máscaras recorrentes;
- padding;
- fronteiras e truncamento.

### Implementação de produção

Depois da validação científica:

1. criar `torch.autograd.Function` customizada;
2. acumular a estatística no backward do kernel;
3. evitar materializar contribuições para todos os tempos;
4. preservar suporte a packed sequences e baixa precisão;
5. comparar numericamente contra a célula explícita.

Um hook apenas no tensor de saída de `nn.LSTM` pode não enxergar toda a
dependência recorrente interna de estados anteriores, especialmente quando a
loss usa somente a última saída. Ele não deve ser considerado equivalente sem
um teste de adjoints internos.

## 14. Mascaramento no otimizador

O mesmo contrato do Functional SlowHeat permanece:

```text
theta_applied = theta_before + M * (theta_native - theta_before)
```

Em `follow_update`:

```text
state_applied = state_before + M * (state_native - state_before)
```

Para pesos recorrentes, uma máscara forte atua em todos os tempos e pode ter
efeito dinâmico maior do que em uma camada feed-forward. `beta` precisa de
calibração separada para RNN, GRU e LSTM.

## 15. Esqueleto de implementação

```python
class TemporalSlowHeatTracker(nn.Module):
    def begin_backward_window(self):
        self.step_sum.zero_()
        self.step_count.zero_()

    def observe(self, state, valid=None):
        detached_state = state.detach()

        def collect(grad):
            contribution = detached_state.float().abs() * grad.float().abs()
            if valid is not None:
                contribution *= valid[..., None]
            self.step_sum.add_(contribution.sum(dim=0))
            self.step_count.add_(valid.sum() if valid is not None else state.shape[0])
            return grad

        state.register_hook(collect)

    def finish_backward(self):
        signal = self.step_sum / self.step_count.clamp_min(1)
        self.update_task_ema(signal)
        self.step_sum.zero_()
        self.step_count.zero_()
```

Em uma implementação real, múltiplos hooks temporais podem atualizar buffers
no mesmo stream CUDA. É necessário testar sincronização, determinismo e
compatibilidade com compilação.

## 16. Plano de implementação no repositório

### Etapa A — RNN simples

- criar `TemporalSlowHeatTracker`;
- implementar `SlowHeatRNNCell` com `tanh`;
- criar máscara simétrica/fatorada de `weight_hh`;
- suportar padding e `finish_backward()`;
- validar em sequências curtas sintéticas.

### Etapa B — LSTMCell

- implementar quatro gates explicitamente;
- comparar utilidade de `h`, `h+c` e gates;
- repetir fatores nas quatro faixas empacotadas;
- adicionar biases duplos compatíveis com PyTorch.

### Etapa C — GRU e multilayer

- agrupar três gates da GRU;
- registrar conexões entre camadas recorrentes;
- tratar dropout somente entre layers;
- budgets separados por layer e direção.

### Etapa D — bidirecional e projeções

- tracker separado para forward e reverse;
- mapeamento explícito para concatenação das direções;
- trackers de célula e projeção em LSTMP.

### Etapa E — desempenho

- comparar célula explícita e kernel fundido vanilla;
- criar backward customizado somente depois de validar a semântica;
- medir tempo e pico de memória por tamanho de sequência;
- testar BPTT completo e truncado com protocolos pareados.

## 17. Testes mínimos de aceitação

1. A acumulação produz o mesmo resultado independentemente da ordem em que as
   contribuições temporais são adicionadas.
2. Padding não contribui para utilidade nem para a contagem.
3. Sequências idênticas padded e packed produzem importâncias equivalentes.
4. A máscara de `weight_hh` protege simultaneamente linha de destino e coluna
   de origem.
5. Uma célula LSTM protegida mascara as quatro linhas de gates.
6. Os dois biases da LSTM recebem a mesma máscara de gate.
7. BPTT truncado não consolida prematuramente entre janelas.
8. A fronteira de tarefa limpa/destaca o estado conforme a política declarada.
9. Máscara 1 coincide com o otimizador nativo.
10. Máscara 0 bloqueia gradiente, momentum e weight decay.
11. Checkpoint restaura trackers, budget, tamanho oculto, gates e direção.
12. O controle sem consolidação coincide com a RNN/LSTM vanilla.

## 18. Limitações que devem acompanhar os resultados

- `tanh` e sigmoid não preservam a invariância por reescala de redes ReLU.
- A importância observada depende do horizonte de BPTT.
- Proteger parâmetros não preserva o valor do estado oculto para novas entradas.
- Uma coordenada protegida ainda interage com unidades plásticas pela dinâmica.
- Células explícitas podem ser muito mais lentas que kernels fundidos.
- Fronteiras de tarefa e política de reset são informação adicional do
  protocolo e precisam ser declaradas.

## Referências

- [Contrato do Functional SlowHeat](functional_slowheat.md)
- [Semântica do otimizador](optimizer_semantics.md)
- [Documentação de RNN do PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.RNN.html)
- [Documentação de LSTM do PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
- [Documentação de GRU do PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html)
- [Documentação de PackedSequence do PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.rnn.PackedSequence.html)
