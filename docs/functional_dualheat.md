# Functional DualHeat

Functional DualHeat é definido neste repositório como:

```text
FastHeat de ativação + Functional SlowHeat
```

Ele não é a implementação histórica de `DualHeatLinear`/`DualHeatMLP`. Essas
APIs continuam inalteradas apenas para compatibilidade com checkpoints e
experimentos antigos.

## FastHeat

Para uma ativação oculta `h`, cada unidade ou canal mantém um buffer
`fast_heat` não treinável. A magnitude pós-ativação é normalizada dentro da
camada:

```text
a_i      = mean(|h_i|)
r_i      = a_i / (mean_j(a_j) + eps)
others_i = mean_{j != i}(fast_heat_j)
gate_i   = 1 / (1 + gamma * others_i)
h'_i     = gate_i * h_i
fast_heat_i <- ReLU(alpha*fast_heat_i + (1-alpha)*(r_i-delta))
```

O forward usa o estado anterior e somente depois atualiza o buffer. Em
`train()` o gate é aplicado e atualizado. Em `eval()` o gate continua aplicado,
mas o estado fica congelado; validação e teste não alteram o modelo. FastHeat
não é reiniciado nas fronteiras de tarefa e é salvo pelo `state_dict`.

Os gates aparecem apenas nas ativações ocultas: após cada ativação do MLP;
após cada ReLU e antes do pooling na CNN pequena e VGG11; e após o
`GroupNorm+ReLU` do stem e cada `soma residual+ReLU` no ResNet18. A cabeça de
logits nunca recebe gate.

As classes públicas são `FunctionalDualHeatMLP`, `FunctionalDualHeatCNN`,
`FunctionalDualHeatVGG11` e `FunctionalDualHeatResNet18`. Cada modelo expõe
`get_fast_states()` e `reset_fast_heat()`.

## Piloto e benchmarks

O piloto usa três seeds fixas, cinco épocas por tarefa e a grade completa de
`alpha={0.90,0.97}`, `gamma={0.5,2.0}` e `delta={0.0,0.5}` em VGG11 e ResNet18.
A seleção consulta exclusivamente a acurácia Class-IL final de validação. O
manifesto congelado é obrigatório para os benchmarks principais de 13 métodos.

```bash
python run_all_tests.py \
  --num-seeds 10 \
  --sections functional-dualheat-pilot \
    split-cifar10-vgg11-functional-dualheat \
    split-cifar10-resnet18-functional-dualheat \
  --device cpu --no-download
```

Os novos diretórios de saída são distintos dos sweeps SlowHeat anteriores. Por
arquitetura, o relatório aplica Holm aos quatro contrastes primários
`DualHeat - SlowHeat`; métricas secundárias incluem FastHeat contra vanilla,
forgetting, task-aware, classifier gap, sinais por seed, bootstrap, tempo,
FLOPs e bytes de estado. Todo o protocolo permanece exploratório até a execução
das dez seeds pareadas.
