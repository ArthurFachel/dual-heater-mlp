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

Os diretórios de saída são distintos dos sweeps SlowHeat anteriores. Por
arquitetura, o relatório aplica Holm aos quatro contrastes primários
`DualHeat - SlowHeat`; métricas secundárias incluem FastHeat contra vanilla,
forgetting, task-aware, classifier gap, sinais por seed, bootstrap, tempo,
FLOPs e bytes de estado.

## Resultados pareados concluídos

Os benchmarks VGG11 e ResNet18 foram concluídos com dez seeds pareadas e 13
métodos. Todos os checks de épocas, exemplos atuais, replay e histórico de
seleção foram registrados como iguais em cada contraste. Os próprios artefatos
mantêm o status `exploratory_paired_benchmark`.

Diferença de acurácia final Class-IL, sempre `DualHeat - SlowHeat`:

| Arquitetura | Par | Diferença média | IC95% t pareado | p ajustado de Holm |
|---|---|---:|---:|---:|
| VGG11 | `dualheat - slowheat` | -0,00483 | [-0,00892; -0,00074] | 0,08030 |
| VGG11 | `dualheat_lpr - slowheat_lpr` | +0,01430 | [0,00283; 0,02577] | 0,08030 |
| VGG11 | `dualheat_classifier_expander - slowheat_classifier_expander` | -0,01295 | [-0,02866; 0,00276] | 0,19027 |
| VGG11 | `dualheat_scroll - slowheat_scroll` | -0,00146 | [-0,02223; 0,01931] | 0,87717 |
| ResNet18 | `dualheat - slowheat` | -0,00007 | [-0,01252; 0,01238] | 1,00000 |
| ResNet18 | `dualheat_lpr - slowheat_lpr` | -0,00457 | [-0,01393; 0,00479] | 1,00000 |
| ResNet18 | `dualheat_classifier_expander - slowheat_classifier_expander` | -0,00496 | [-0,01825; 0,00833] | 1,00000 |
| ResNet18 | `dualheat_scroll - slowheat_scroll` | -0,00247 | [-0,01946; 0,01452] | 1,00000 |

Nenhum contraste permaneceu significativo a 5% após Holm. O único ganho médio
de FastHeat foi no par VGG11 com LPR, +1,43 ponto percentual, mas seu p ajustado
foi 0,08030. O resultado sustenta uma conclusão negativa/indeterminada: FastHeat
não trouxe melhoria robusta nas duas arquiteturas sob este protocolo.

Artefatos-fonte:

- `results/split_mnist_protocol/functional_dualheat_pilot/pilot_results.json`;
- `results/split_mnist_protocol/functional_dualheat_pilot/selected_fastheat_config.json`;
- `results/split_mnist_protocol/split_cifar10_vgg11_functional_dualheat/functional_dualheat_analysis.json`;
- `results/split_mnist_protocol/split_cifar10_resnet18_functional_dualheat/functional_dualheat_analysis.json`.

O manifesto selecionou `fast_decay=0.9`, `fast_strength=0.5` e
`fast_threshold=0.0` usando somente validação. A auditoria posterior observou
que o loader valida schema, status e pertinência à grade, mas não recomputa o
vencedor a partir de um hash imutável do piloto. Isso limita a força de uma
alegação de pré-registro.

Para o catálogo completo de métodos, consulte [methods_catalog.md](methods_catalog.md).
