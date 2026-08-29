# Functional SlowHeat em redes convolucionais

Este documento descreve como adaptar o Functional SlowHeat para redes neurais
convolucionais. O objetivo é preservar a semântica do método existente:

- utilidade funcional de primeira ordem `|z * dL/dz|`;
- consolidação nas fronteiras de tarefas;
- orçamento mínimo de capacidade plástica;
- proteção fatorada das conexões de entrada e saída;
- mascaramento do delta final do otimizador e, opcionalmente, de seus estados.

O repositório já contém `SlowHeatConv2d`, mas a implementação atual cobre
principalmente convoluções densas simples. Residual, normalização, convoluções
agrupadas e topologias com ramificações ainda exigem um registrador de
conectividade mais geral.

## 1. Definição da unidade funcional

Em uma camada convolucional, a unidade protegida será um **canal de saída**.
Cada canal corresponde a um filtro compartilhado por todas as posições
espaciais da imagem.

Para uma saída convolucional:

```text
z.shape = [batch, output_channels, height, width]
```

a utilidade instantânea do canal `c` é:

```text
u[c] = sum_{b,h,w} |z[b,c,h,w] * dL/dz[b,c,h,w]|
```

Se houver padding de sequências ou imagens com regiões inválidas, a redução
deve usar uma máscara de validade:

```text
u[c] = sum valid[b,h,w] * |z[b,c,h,w] * dL/dz[b,c,h,w]|
```

A soma sobre posições é coerente com o compartilhamento de pesos: o mesmo
filtro é reutilizado em todas elas, portanto a evidência do filtro deve agregar
todas as suas utilizações.

## 2. Fluxo passo a passo durante uma tarefa

### Passo 1 — inicializar o estado por canal

Para uma convolução com `C_out` canais, manter:

```text
importance_memory: [C_out]
task_ema:           [C_out]
slow_heat:          [C_out]
task_step:          escalar
```

Os vetores devem ser buffers do módulo para participarem de `state_dict()` sem
serem tratados como parâmetros treináveis.

### Passo 2 — calcular a convolução normalmente

O forward não muda:

```text
z = conv2d(x, weight, bias, stride, padding, dilation, groups)
```

Nenhuma inibição deve ser aplicada ao forward. O SlowHeat controla a
plasticidade dos parâmetros, não a função usada em inferência.

### Passo 3 — registrar a observação funcional

Durante treino, registrar um hook no tensor `z`. Quando o backward produzir
`dL/dz`, calcular a contribuição por canal:

```text
contribution = abs(z.detach()) * abs(grad.detach())
signal = contribution.sum(dim=(0, 2, 3))
```

Em `float16` ou `bfloat16`, a multiplicação e a redução devem ser promovidas a
`float32` para reduzir overflow e perda de precisão.

### Passo 4 — normalizar dentro da camada

Normalizar a utilidade pelos canais da própria camada:

```text
normalized[c] = signal[c] / max(mean(signal), eps)
```

Essa normalização remove fatores globais de batch e escala da loss. Ela não
torna diretamente comparáveis canais de camadas diferentes.

### Passo 5 — acumular evidência da tarefa

Atualizar `task_ema` com a mesma regra usada pelas camadas lineares:

```text
task_ema = decay * task_ema + (1 - decay) * normalized
```

Um forward sem backward não deve alterar a importância.

## 3. Consolidação na fronteira de tarefas

Ao terminar a tarefa `k`, chamar `consolidate()` em todas as camadas
convolucionais instrumentadas.

### Passo 1 — fundir evidência persistente

Estratégia principal:

```text
importance_memory = max(importance_memory, task_ema)
```

As estratégias `mean` e `sum` permanecem disponíveis como ablações.

### Passo 2 — aplicar o orçamento de plasticidade

Com `C_out` canais e orçamento plástico `p`:

```text
max_protected = floor((1 - p) * C_out)
```

Selecionar no máximo `max_protected` canais com maior evidência positiva. Os
demais permanecem completamente plásticos.

### Passo 3 — derivar o fator de aprendizado

Para cada canal selecionado:

```text
m[c] = 1 / (1 + beta * slow_heat[c])
```

Para canais não selecionados:

```text
m[c] = 1
```

O budget garante uma fração de canais livres, não uma fração exata de
parâmetros livres. Canais de camadas distintas podem controlar quantidades de
pesos muito diferentes.

### Passo 4 — limpar somente a estatística corrente

Após consolidar:

```text
task_ema = 0
task_step = 0
```

`importance_memory` e `slow_heat` devem permanecer no checkpoint.

## 4. Construção das máscaras fatoradas

### Convolução densa

Para:

```text
weight.shape = [C_out, C_in, K_h, K_w]
```

a máscara é:

```text
M[o,i,kh,kw] = min(m_destination[o], m_source[i])
```

Isso produz dois efeitos:

1. `m_destination[o]` protege o filtro inteiro que cria o canal `o`;
2. `m_source[i]` protege todos os kernels que consomem o canal anterior `i`.

O bias recebe apenas o fator de destino:

```text
M_bias[o] = m_destination[o]
```

### Convolução agrupada

Para `groups > 1`, a segunda dimensão do peso contém apenas os canais de
entrada daquele grupo. O mapeamento deve ser explícito:

```text
group_out = o // (C_out / groups)
local_input = i_local
global_input = group_out * (C_in / groups) + local_input

M[o,i_local,kh,kw] = min(
    m_destination[o],
    m_source[global_input],
)
```

Não se deve simplesmente remodelar um vetor global de `C_in` sobre a segunda
dimensão do peso.

### Convolução depthwise

Quando `groups == C_in`, cada canal de saída está ligado a um único canal de
entrada, ou a um pequeno conjunto quando há multiplicador depthwise. A máscara
deve preservar essa correspondência em vez de criar conexões inexistentes.

### Transição convolução para linear

Se um tensor `[B, C, H, W]` for achatado antes de uma camada linear, o fator do
canal precisa ser repetido para todas as posições pertencentes a ele:

```text
source_factor_flat = repeat_interleave(m_channel, H * W)
```

Isso pressupõe flatten em ordem NCHW. Adaptive pooling antes do flatten torna o
mapeamento menor e estável.

## 5. Normalização e blocos residuais

### BatchNorm

Os parâmetros treináveis por canal devem acompanhar a máscara:

```text
BatchNorm.weight[c] -> m[c]
BatchNorm.bias[c]   -> m[c]
```

Entretanto, `running_mean` e `running_var` não são atualizados pelo otimizador.
Mesmo com peso congelado, essas estatísticas podem mudar a função de um canal
protegido. As opções são:

1. congelar as estatísticas após consolidação;
2. manter estatísticas específicas por tarefa;
3. usar GroupNorm/LayerNorm sem estatísticas correntes;
4. declarar BatchNorm não protegido como ablação.

### Residual

Em um bloco residual, um canal pode alimentar simultaneamente a branch
principal e a skip connection. Toda aresta parametrizada que consome esse
canal deve receber sua máscara de origem.

Na soma residual, os canais precisam compartilhar a mesma identidade
semântica. Se uma branch usa projeção `1x1`, essa projeção deve ser registrada
como consumidora e produtora explícita.

O registrador sequencial atual não é suficiente para essa topologia. A futura
API deve representar um grafo de conexões, por exemplo:

```python
registry.connect(source=conv1, target=conv2)
registry.connect(source=input_channels, target=skip_projection)
registry.merge(sources=[conv2, skip_projection], target=residual_channels)
```

## 6. Mascaramento no otimizador

Em cada passo de uma tarefa posterior:

1. o AdamW/SGD calcula seu update nativo completo;
2. o SlowHeat obtém o delta nativo;
3. aplica `M * delta_native`;
4. restaura o parâmetro usando apenas o delta mascarado;
5. em `follow_update`, aplica a mesma interpolação aos momentos.

```text
theta_applied = theta_before + M * (theta_native - theta_before)
```

Isso deve incluir weight decay. Mascarar apenas o gradiente bruto não possui a
mesma semântica sob AdamW.

## 7. Esqueleto de implementação

```python
class SlowHeatConv2d(FunctionalSlowHeatMixin, nn.Conv2d):
    def forward(self, x):
        z = super().forward(x)
        if self.training and z.requires_grad:
            detached_z = z.detach()

            def collect(grad):
                contribution = detached_z.float().abs() * grad.float().abs()
                signal = contribution.sum(dim=(0, 2, 3))
                self.update_task_importance(signal)
                return grad

            z.register_hook(collect)
        return z
```

Para produção, a multiplicação e a redução devem ser fundidas ou processadas
em blocos para evitar materializar um tensor FP32 completo do feature map.

## 8. Estado da implementação no repositório

Implementado em `SlowHeatConv2d`, `SlowHeatCNN` e `SlowHeatVGG11`:

- API completa da `nn.Conv2d`, incluindo tuplas, dilatação, grupos e modos de
  padding;
- utilidade funcional por canal, máscara espacial opcional e promoção para
  FP32 antes da multiplicação em baixa precisão;
- CNN sequencial com duas convoluções, pooling adaptativo e cabeça linear;
- VGG11 adaptada para CIFAR com oito convoluções, cinco max-pools, pooling
  adaptativo `1x1` e cabeça linear, sem BatchNorm;
- registro automático das arestas Conv→Conv e Conv→Linear;
- máscaras fatoradas corretas para convoluções densas, grouped e depthwise;
- repetição dos fatores por posição no flatten NCHW;
- mascaramento do delta final e dos estados por `SlowHeatAdamW` e
  `SlowHeatSGD`.

Normalização affine, estatísticas correntes de BatchNorm e um registrador em
grafo para residual/fan-out continuam como extensões futuras.

### Plano incremental

### Etapa A — CNN sequencial mínima (concluída)

- completar a compatibilidade de `SlowHeatConv2d` com `nn.Conv2d`;
- aceitar tuplas em `kernel_size`, `stride` e `padding`;
- adicionar `dilation`, `groups`, `padding_mode`, `device` e `dtype`;
- criar `SlowHeatCNN` com duas convoluções e uma cabeça linear;
- registrar conexões Conv→Conv e Conv→Linear explicitamente.

### Etapa B — normalização

- implementar máscara para GroupNorm e BatchNorm affine;
- escolher política explícita para estatísticas correntes do BatchNorm;
- testar checkpoint com buffers de normalização e de SlowHeat.

### Etapa C — topologias modernas

- residual com projeção `1x1`;
- convoluções grouped e depthwise (concluído para cadeias sequenciais);
- concatenação de canais;
- fan-out para múltiplos consumidores.

### Etapa D — benchmark

- manter o protocolo Class-IL do Split-CIFAR;
- substituir a MLP sobre pixels achatados por uma CNN verdadeira;
- comparar vanilla, SlowHeat sem consolidação, row-only e fatorado;
- medir acurácia, forgetting, BWT, tempo e pico de memória.

## 9. Testes mínimos de aceitação

1. A utilidade tem shape `[C_out]` e permanece finita em baixa precisão.
2. Reescalar reciprocamente canais de uma cadeia Conv–ReLU–Conv preserva a
   utilidade dentro da tolerância numérica.
3. Um canal ReLU morto recebe utilidade zero.
4. O budget mantém a fração plástica solicitada.
5. A linha do filtro produtor e a coluna consumidora recebem a mesma proteção.
6. A máscara de grouped convolution respeita os limites de cada grupo.
7. Máscara 1 reproduz exatamente o otimizador nativo.
8. Máscara 0 bloqueia gradiente, momentum e weight decay.
9. Checkpoint restaura importância, budget e metadados da topologia.
10. O controle sem consolidação reproduz a CNN vanilla com inicialização e
    batches pareados.

## 10. Limitações que devem acompanhar os resultados

- A utilidade é uma aproximação local de primeira ordem, não uma prova causal.
- A proteção por canal perde informação espacial.
- O budget de canais não equivale a um budget de parâmetros ou FLOPs.
- BatchNorm pode alterar canais protegidos por meio de estatísticas correntes.
- A eficácia em CNNs só estará estabelecida depois de uma avaliação com CNN
  real; os protocolos CIFAR atuais usam imagens achatadas em uma MLP.

## Referências

- [Contrato do Functional SlowHeat](functional_slowheat.md)
- [Semântica do otimizador](optimizer_semantics.md)
- [Protocolo Split-CIFAR atual](split_cifar.md)
- [Molchanov et al., Taylor pruning de feature maps](https://arxiv.org/abs/1611.06440)
- [Documentação de Conv2d do PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
