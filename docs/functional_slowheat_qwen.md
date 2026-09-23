# Qwen2 SlowHeat: escopo, achados e limites

Estado: implementação e verificação em CPU concluídas; smoke em GPU executado.
Runs de calibração e de ordem/seed existem, mas usam um número de passos já
revogado e não foram agregadas, portanto não há resultado empírico de continual
learning publicável.

## 1. O que existe

| Arquivo | Papel |
|---|---|
| `src/dual_heater/qwen.py` | Host `SlowHeatQwen2ForSequenceClassification` + `QwenSlowHeatConfig` |
| `src/dual_heater/transformer.py` | Helpers de capacidade e máscara agora compartilhados por BERT e Qwen |
| `tests/test_slow_heat_qwen.py` | 37 testes em CPU, modelos Qwen2 minúsculos e aleatórios |
| `experiments/qwen_slowheat_smoke.py` | Smoke no checkpoint real (executado; ver seção 6) |
| `experiments/qwen_capacity_diagnostic.py` | Diagnóstico de capacidade e concentração; 8 testes |
| `experiments/qwen_iso_plasticity.py` | Runner da ablação iso-plasticidade; 42 testes |
| `experiments/capacity_calibration.py` | Aritmética de capacidade, `dominant_unit_overlap`, `importance_profile` |

Documentos relacionados: [`qwen_capacity_calibration.md`](qwen_capacity_calibration.md)
e [`qwen_iso_plasticity_ablation.md`](qwen_iso_plasticity_ablation.md). O
protocolo congelado da ablação está em `goals/protocol_iso_plasticity.md`.

O refactor moveu `dynamic_matrix_mask`, `merge_task_importance`,
`apply_global_capacity`, `apply_hierarchical_capacity` e `apply_family_capacity`
de `bert.py` para `transformer.py`. O BERT mantém os nomes privados históricos
como aliases e seus 44 testes seguem verdes, então o comportamento do BERT não
mudou.

## 2. Escopo: somente a FFN gated

Duas restrições arquiteturais do Qwen2 definem o escopo.

**MLP gated.** O bloco é `down_proj(act(gate_proj(x)) * up_proj(x))`. A unidade
funcional é o produto elementwise, de 4864 dimensões no Qwen2.5-0.5B. O tracker
observa a entrada de `down_proj` via `forward_pre_hook`.

**GQA bloqueia a atenção.** O Qwen2.5-0.5B tem 14 heads de query e apenas 2 de
key/value. Q, K e V não compartilham hidden size, então um único vetor de
importância indexado por head não endereça as três projeções. O
`SlowHeatAttentionTracker` assume que compartilham. Estender exige um tracker
ciente de GQA que mapeie cada grupo KV para seus query heads. Isso não foi
feito.

Diferenças estruturais frente ao BERT:

- sem biases nas projeções da FFN, então não há bindings de bias;
- três matrizes por bloco em vez de duas: `gate_proj` e `up_proj` são produtoras
  (máscara nas linhas), `down_proj` é consumidora (máscara nas colunas);
- `score` (cabeça de classificação) é inicializada aleatoriamente. Congelá-la
  tornaria uma run Class-IL estruturalmente incapaz de aprender qualquer
  rótulo, então `keep_score_plastic=True` a mantém treinável e sem máscara. Essa
  isenção é reportada explicitamente por `exempt_parameter_names()` e contada
  em separado em `mask_coverage_summary()`, para que uma exceção estrutural
  necessária não se confunda com vazamento acidental.

## 3. Achado: o estimador é invariante ao longo da cadeia gated

A hipótese inicial era que observar a saída de `act_fn` (a posição usada no
BERT) mediria o tensor errado e ordenaria mal as unidades. **Isso está errado.**

Para um produto elementwise `z = a * u`, vale `dL/da = dL/dz * u`, logo

```text
|a * dL/da| = |a * u * dL/dz| = |z * dL/dz|
```

e simetricamente para `u`. O estimador `|z * dL/dz|` produz o mesmo sinal
normalizado seja observado em `act(gate)`, em `up` ou no produto. Verificado
numericamente: diferença máxima de 2,4e-7 entre os três pontos.

Consequência prática: a escolha do ponto de hook **não** afeta o ranking das
unidades. O `forward_pre_hook` em `down_proj` continua sendo a posição correta
por outro motivo — a entrada de `down_proj` é exatamente o tensor que a máscara
de colunas de `down_proj.weight` multiplica, então tracker e máscara endereçam o
mesmo objeto. Mas nenhum bug de ranking está sendo evitado por essa escolha.

O teste `test_functional_importance_is_invariant_across_the_gated_product_chain`
fixa essa propriedade.

## 4. Verificação por mutação

Os 40 testes passaram na primeira execução, o que é motivo para desconfiar, não
para comemorar. Dez mutações foram injetadas no código de produção para medir se
a suíte discrimina de fato:

| Mutação | Detectada |
|---|---|
| hook em `act_fn` em vez da entrada de `down_proj` | não (invariância — ver seção 3) |
| máscara de `down_proj` usa linhas em vez de colunas | sim, 5 testes |
| `up_proj` sem binding | sim, 3 testes |
| máscara de validade descartada no host | sim, após adicionar teste |
| máscara de validade descartada no tracker | sim |
| `freeze_unbound` mantém tudo treinável | sim, 4 testes |
| assinatura ignora `slowheat_config` | sim, 2 testes |
| fator hard invertido | sim |
| `capacity_scope` forçado a `local` | sim, após adicionar teste |
| `capacity_scope` forçado a `global` | sim, após adicionar teste |
| budget global protege tudo | sim, 2 testes |

Três mutações sobreviveram à primeira suíte e motivaram testes novos:
discriminação entre escopos de capacidade (`local` dá cota por camada,
`global`/`hierarchical` concentram proteção onde há utilidade) e propagação da
máscara de atenção.

A primeira mutação permanece indetectável por ser matematicamente indiferente. A
posição do hook é fixada estruturalmente em
`test_tracker_is_bound_to_the_gated_product_tensor`.

## 5. Nota sobre o teste de padding

`test_padding_tokens_do_not_change_functional_utility` passa mesmo sem a máscara
de validade. Atenção causal somada ao pooling no último token não-pad já zera o
gradiente nas posições de padding: a norma do gradiente por token em uma
sequência `[2,7,9,3,0,0]` é `[0, 0, 0, 0.17, 0, 0]`. O teste é mantido como
regressão contra mudanças futuras no pooling, não como evidência de que a
máscara funciona. Essa evidência está em
`test_validity_mask_actually_excludes_masked_tokens`, que exercita o tracker
diretamente.

## 6. Custo: estimado e medido

Números derivados da config, **não medidos**:

- parâmetros totais do Qwen2.5-0.5B: cerca de 494M;
- envelope FFN-only com `freeze_unbound_parameters=True`: 24 camadas x
  (2 x 4864 x 896 + 896 x 4864) = cerca de 318M treináveis, mais a cabeça
  `score`;
- full fine-tune fp32 com AdamW: aproximadamente 8,3 GB.

**Medição real** (`results/run_logs/results_gpu_memory_smoke.log`, GTX 1080 Ti,
`--budget 0.50`, batch 2):

| Grandeza | Valor |
|---|---:|
| Pico alocado | 5,447 GiB |
| Pico reservado | 6,033 GiB |
| Unidades protegidas | 58.368 |
| 10 passos | 3,17 s |

A estimativa de 8,3 GB era pessimista: o consumo real é cerca de 35% menor e a
folga na 1080 Ti de 11 GB é confortável, não pequena.

## 7. O que ainda não foi feito

- nenhuma integração com `experiments/split_clinc150.py`;
- nenhuma agregação entre seeds das runs de calibração existentes, nem
  diferenças pareadas;
- nenhum registro de Gate 1, Gate 2 ou Gate 3;
- o critério declarado de `qwen_capacity_calibration.md`
  (`--min-effective-plasticity`) nunca foi aplicado numa run;
- a confirmação de 10 seeds x 10 domínios x 120 passos não produziu manifestos;
- `docs/qwen_layer_anomaly.md`, prometido como entregável de P6, não existe,
  embora as 6 runs de medição estejam em `results/qwen_layer_anomaly/`.

`docs/bert_slowheat_diagnostic_results.md` registra que o SlowHeat em BERT
superou fine-tuning sequencial mas não superou replay, e conclui que não se deve
abrir nova grade para resgatar o mesmo endpoint. Levar o mecanismo ao Qwen é uma
pergunta distinta — outra arquitetura, outra escala, MLP gated — e não uma
continuação confirmatória daqueles diagnósticos. Para que continue sendo uma
pergunta distinta, o protocolo, as métricas e as seeds precisam ser congelados
antes da primeira run, e não escolhidos depois de ver os números.
