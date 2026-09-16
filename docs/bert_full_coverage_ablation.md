# Ablação de cobertura completa do SlowHeat em BERT

Status: **implementada e testada, ainda sem resultados experimentais**. Este
documento fixa a semântica do preset `--full-coverage-variants` antes de qualquer
execução cara. A implementação está em `src/dual_heater/bert.py`; o registro e o
runner estão em `experiments/split_clinc150.py`.

## Escopo

A cobertura completa se aplica somente a `BertForSequenceClassification` da
Hugging Face. Ela não implica suporte a RoBERTa, SwiGLU, GQA, QKV fundido,
decoders, activation checkpointing ou LoRA de cobertura completa.

As unidades científicas são:

- uma unidade por neurônio intermediário da FFN;
- uma unidade por cabeça de atenção;
- uma unidade por dimensão oculta na saída de embeddings e nas junções
  residuais;
- uma unidade por dimensão da saída do pooler;
- uma unidade por logit de classe.

O protocolo SlowHeat persistido usa schema 3. Checkpoints SlowHeat de schema 2
são rejeitados em vez de serem interpretados como se tivessem a nova topologia.
A telemetria de Heat usa schema 2 e expõe listas para as cinco famílias:
`attention`, `ffn`, `residual`, `pooler` e `classifier`.

## Grafo de parâmetros e ownership

Para a camada de encoder `l`, o grafo usado para derivar as máscaras é:

```text
embedding output E
  -> Q_l/K_l/V_l -> attention heads H_l -> attention dense -> attention residual A_l
  -> intermediate FFN F_l -> output dense -> block residual R_l
R_l becomes the input node for layer l+1
R_last -> pooler P -> classifier C
```

Cada parâmetro recebe no máximo uma `PlasticityMaskBinding`. Uma matriz pode,
porém, receber fatores dos dois endpoints dentro dessa única máscara.

| Parâmetro | Fator de linha/saída | Fator de coluna/entrada |
|---|---|---|
| pesos dos embeddings de palavra/posição/tipo | nenhum | E, quando embeddings estão habilitados |
| peso/bias da LayerNorm dos embeddings | E, quando LayerNorm está habilitada | nenhum |
| peso de Q/K/V na camada `l` | H_l, quando atenção está habilitada | residual de entrada, quando residual está habilitado |
| bias de Q/K/V | H_l, quando atenção está habilitada | nenhum |
| peso da dense de saída da atenção | A_l, quando residual está habilitado | H_l expandido, quando atenção está habilitada |
| bias da dense de saída da atenção | A_l, quando residual está habilitado | nenhum |
| peso/bias da LayerNorm de saída da atenção | A_l, quando LayerNorm está habilitada | nenhum |
| peso da dense intermediária | F_l, quando FFN está habilitada | A_l, quando residual está habilitado |
| bias da dense intermediária | F_l, quando FFN está habilitada | nenhum |
| peso da dense de saída da FFN | R_l, quando residual está habilitado | F_l, quando FFN está habilitada |
| bias da dense de saída da FFN | R_l, quando residual está habilitado | nenhum |
| peso/bias da LayerNorm de saída da FFN | R_l, quando LayerNorm está habilitada | nenhum |
| peso da dense do pooler | P, quando pooler está habilitado | R_last, quando residual está habilitado |
| bias da dense do pooler | P, quando pooler está habilitado | nenhum |
| peso do classificador | C, quando classificador está habilitado | P, quando pooler está habilitado |
| bias do classificador | C, quando classificador está habilitado | nenhum |

### Regra conservadora entre endpoints

Para fatores de linha `r` e coluna `c`, a máscara matricial é:

```python
torch.minimum(r.reshape(-1, 1), c.reshape(1, -1))
```

A operação `minimum` garante que qualquer endpoint protegido possa restringir o
update completo do peso, incluindo weight decay no `SlowHeatAdamW`. Uma média
permitiria que o endpoint mais plástico enfraquecesse a proteção do outro; um
produto introduziria supressão adicional e confundiria as escalas dos endpoints.
Quando só um endpoint participa, seu vetor é transmitido por broadcast; quando
nenhum participa, o parâmetro fica sem binding.

## Métodos do preset

Todos os nove métodos SlowHeat abaixo usam capacidade `hierarchical`,
`fast_heat=None`, nenhum replay e `freeze_unbound_parameters=False`. Portanto,
todos os parâmetros do modelo continuam treináveis. Os budgets padrão de FFN,
atenção, residual e pooler são 0,25; o primeiro estudo não calibra separadamente
os dois budgets novos.

| Identificador | Definição exata |
|---|---|
| `vanilla` | BERT sequencial com AdamW, sem trackers ou máscaras SlowHeat |
| `slowheat_full_coverage` | habilita embeddings, LayerNorm, residual, atenção, FFN, pooler e classificador |
| `slowheat_all_minus_embeddings` | cobertura completa, mas remove o fator de embeddings (`track_embeddings=False`) |
| `slowheat_all_minus_layernorm` | cobertura completa, mas não mascara os parâmetros afins de LayerNorm (`protect_layer_norm=False`) |
| `slowheat_all_minus_residual` | cobertura completa, mas remove fatores residuais de linhas e colunas (`track_residual=False`) |
| `slowheat_all_minus_attention` | cobertura completa, mas remove o fator específico por cabeça (`track_attention=False`) |
| `slowheat_all_minus_ffn` | cobertura completa, mas remove o fator por neurônio intermediário (`track_ffn=False`) |
| `slowheat_all_minus_pooler` | cobertura completa, mas remove o fator específico do pooler (`protect_pooler=False`) |
| `slowheat_all_minus_classifier` | cobertura completa, mas remove o fator específico por logit (`protect_classifier=False`) |
| `slowheat_ffn_attention` | habilita somente FFN e atenção; usa a mesma seleção de famílias do `slowheat` histórico, mas com o escopo hierárquico exigido pelo novo preset |

“Menos uma família” significa remover o fator daquela família, não garantir que
todos os tensores do módulo fiquem sem máscara. Por exemplo, sem atenção, as
colunas de Q/K/V ainda podem receber o fator residual; sem FFN, endpoints
residuais ainda podem mascarar pesos da FFN; sem classificador, as colunas do
peso do classificador ainda podem receber o fator do pooler. O teste da ablação
deve, portanto, conferir bindings e endpoints concretos, não apenas o nome do
método.

## Treinável não significa mascarado

`trainable_parameters` conta elementos com `requires_grad=True`.
`mask_coverage` descreve a cobertura estrutural por bindings:

- `trainable_parameter_count`: elementos treináveis;
- `masked_parameter_count`: elementos de parâmetros treináveis que possuem uma
  máscara SlowHeat, sem expandir máscaras broadcast;
- `masked_fraction`: razão entre as duas contagens;
- `binding_count`: quantidade de parâmetros vinculados.

Nas variantes estendidas, remover uma família restaura o update AdamW nativo nos
parâmetros que deixam de ter qualquer endpoint, em vez de congelá-los. Assim, a
suíte mantém o mesmo número de parâmetros treináveis, mas **não é pareada por
custo de máscara**: `masked_parameter_count` muda entre variantes. Toda
comparação deve incluir `mask_coverage`; uma diferença de resultado não isola
somente uma hipótese funcional se a quantidade de parâmetros mascarados também
mudou.

## Padding e granularidade

Trackers de embeddings, FFN, atenção e junções residuais recebem a mesma máscara
`attention_mask` de forma `[B,T]`. Tokens de padding são excluídos da redução de
`|z * dL/dz|`; caso contrário, diferentes comprimentos de padding alterariam a
importância e a seleção de capacidade sem acrescentar informação da tarefa.
Pooler e classificador observam tensores por exemplo, `[B,H]` e `[B,C]`, e por
isso não usam máscara de token.

Embedding é protegido por coordenada oculta em todas as linhas de vocabulário,
posição e tipo, não por token. LayerNorm não possui tracker próprio: seus
parâmetros afins reutilizam a importância da junção residual correspondente.

## Por que o primeiro estudo exclui FastHeat e replay

FastHeat altera ativações no forward, e replay altera a distribuição de dados e
o orçamento de exemplos. Incluí-los confundiria a pergunta inicial — quais
famílias de fatores SlowHeat explicam o comportamento da cobertura completa —
com mecanismos adicionais. O preset mantém ambos ausentes; estudos posteriores
podem adicioná-los como fatores separados e pareados.

## Métricas e artefatos obrigatórios

Inspecione, por seed e de forma agregada quando houver seeds suficientes:

- FAA (`final_average_accuracy`) e forgetting;
- acurácia task-aware e a matriz por tarefa/estágio;
- macro-F1;
- `classifier_gap`, definido como FAA task-aware menos FAA Class-IL;
- `trainable_parameters`;
- `mask_coverage`, sobretudo `masked_parameter_count` e `masked_fraction`;
- `peak_memory` e tempo observado.

O runner usa validação por padrão (`evaluate_test=False`). Matrizes de teste,
macro-F1 e `classifier_gap` só devem ser consultados no fluxo confirmatório após
o manifesto ter sido congelado. Uma única seed é diagnóstico exploratório, não
evidência estatística.

## Smoke exploratório

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 -m experiments.split_clinc150 \
  --full-coverage-variants \
  --device cpu \
  --seeds 0 \
  --epochs-per-task 1 \
  --batch-size 2 \
  --output-dir results/bert_full_coverage_smoke
```

Este comando pode carregar da rede o modelo e o dataset configurados. Ele **não
faz parte da validação automatizada** e não deve ser executado sem aprovação,
pois pode realizar downloads e uma execução longa. Os testes automatizados usam
configurações BERT minúsculas e dados locais sintéticos, sem GPU ou rede.

## Fluxo confirmatório pretendido

1. Faça calibração somente no split de validação; não consulte teste para
   escolher força, budgets, épocas ou métodos.
2. Produza um manifesto congelado com o candidato, o protocolo, o
   modelo/dataset/revisões, a ordem das tarefas e os hiperparâmetros antes da
   avaliação final.
3. Reserve seeds e ordens de tarefa disjuntas das usadas na exploração e na
   calibração.
4. Execute o conjunto confirmatório sem alterar o manifesto e só então habilite
   a avaliação de teste.
5. Reporte matrizes completas, dispersão entre seeds, `mask_coverage`, parâmetros
   treináveis e memória de pico junto às métricas principais.

Até esse fluxo ser concluído, a implementação é experimental e não sustenta uma
alegação de eficácia.

## Limitações

- Dimensões residuais dependem da base: coordenar endpoints locais é consistente
  para uma parametrização BERT fixa, mas não prova invariância a rotações ou
  reparametrizações da representação.
- As ablações de atenção e FFN removem importância específica da família, não
  toda máscara incidente nos módulos correspondentes.
- Pooler e classificador permanecem acoplados por endpoints: retirar um fator
  não necessariamente libera toda a matriz entre ambos.
- Os hooks residuais retêm ativações destacadas até o backward; o pico de memória
  deve ser medido antes de escalar para BERT-base.
- Activation checkpointing e LoRA de cobertura completa estão fora de escopo.

## Referências internas

- [Functional SlowHeat em Transformers](functional_slowheat_transformers.md)
- [Catálogo atual de métodos](methods_catalog.md)
- [Resultados históricos BERT/CLINC150](bert_clinc150_results.md)
- [Dashboard e telemetria](live_dashboard.md)
