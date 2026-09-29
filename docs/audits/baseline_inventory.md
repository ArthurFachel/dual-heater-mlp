# Inventário de baselines por host

Tarefa 1.1 de `goals/roadmap_icml_ijcnn.md`. **Levantamento por regex sobre
`experiments/` e `src/dual_heater/`, com leitura de código nos pontos citados.**
Data: 2026-09-29, commit `9a0c551`.

Motivo: um paper de CL sem baselines padrão no host principal é rejeitado por
isso sozinho. Antes de planejar o que implementar, é preciso saber o que existe.

---

## Tabela de cobertura (host × baseline)

| baseline | Split-MNIST (MLP) | Split-CLINC150 (BERT-mini) | Qwen LoRA | CIFAR/ResNet |
|---|---|---|---|---|
| vanilla (sequencial) | sim | sim | sim | sim |
| replay (ER) | sim | sim | **não** | sim |
| A-GEM | sim | não | não | não |
| **EWC** | **sim** | **não** | **não** | **não** |
| **SI** | **sim** | **não** | **não** | **não** |
| **MAS** | **sim** (29/09, ver abaixo) | não | não | não |
| **LwF** | parcial (ver nota) | não | não | não |
| DER/DER++ | citado no agregador, não implementado como método | não | não | não |
| SlowHeat / DualHeat (casa) | sim | sim | sim | sim |
| `lr_control` (falsificador de plasticidade) | parcial | não | **sim** | não |
| `frozen_a_control` (LoRA-FA) | n/a | n/a | sim | n/a |

> **Atualização 29/09, commits `e6a427e` e `12b7d05`.** A penalidade quadrática
> foi extraída de `experiments/split_mnist.py` para `src/dual_heater/ewc.py`
> (EWC, MAS, Fisher por exemplo, consolidação), com equivalência numérica
> contra a implementação histórica pinada em
> `tests/test_ewc_extraction_equivalence.py`. MAS foi implementado e ligado ao
> runner do MLP. As linhas BERT e Qwen desta tabela **continuam vazias** — o
> módulo existe e é reutilizável, mas nenhum host transformer o consome ainda.
>
> MAS fica deliberadamente **fora** dos registros visuais
> (`ALL_VISUAL_METHODS`, 33 métodos) pelo mesmo motivo que
> `hard_freeze_replay`: os sweeps já rodados foram executados com aquele
> conjunto, e acrescentar um 34º método mudaria silenciosamente o que uma
> re-execução de um sweep publicado roda. MAS entra em suíte apenas pelo
> piloto da Fase 2.4, cujo protocolo precisa ser pré-registrado antes da
> primeira seed. Guarda em `tests/test_visual_generalization.py`.
>
> Pelo mesmo motivo, `mas_lambda`/`mas_decay` só entram no `config_payload`
> quando `"mas"` está em `config.methods`: caso contrário o sha256 do
> pré-registro congelado em `experiments/confirmatory_split_mnist.py` mudaria.
> Verificado — o hash segue `015b3162...`, inalterado.

---

## Correções ao que o roadmap assumia

O roadmap (Tarefa 1.1) previa encontrar EWC "apenas em
`experiments/split_mnist.py:183`". A busca mostra um quadro diferente em dois
pontos, e ambos mudam o tamanho da Fase 1:

1. **EWC no MLP é mais completo do que uma linha de registro.** Há uma
   implementação inteira, com Fisher empírico *por exemplo* (não o quadrado do
   gradiente da média — o erro que a própria skill do projeto lista como
   auditoria obrigatória):

   | peça | local |
   |---|---|
   | registro do método | `experiments/split_mnist.py:183` |
   | penalidade quadrática genérica | `experiments/split_mnist.py:1263` `_parameter_penalty` |
   | Fisher empírico por exemplo | `experiments/split_mnist.py:1303` `_accumulate_empirical_fisher` |
   | consolidação + âncoras + decay | `experiments/split_mnist.py:1331` `_consolidate_ewc_importance` |
   | soma na loss | `experiments/split_mnist.py:2008` |
   | hiperparâmetros | `ewc_lambda=100.0`, `ewc_decay=1.0` (linhas 382-383) |

2. **SI também já existe no MLP**, o que o roadmap não previa:
   `si_lambda`/`si_epsilon` (384-385), caminho `si_path` (1852, 2101),
   consolidação com deslocamento (2258-2265), soma na loss (2013).
   A Tarefa 1.4 ("repetir para SI e MAS") é, no MLP, só MAS.

3. **`_parameter_penalty` já é a forma `Σ Ω_i (θ_i − θ*_i)²` compartilhada por
   EWC, SI e MAS.** O DRY que a Tarefa 1.4 pede já está feito no host MLP; o que
   falta é extraí-lo de `split_mnist.py` para um módulo reutilizável pelos
   hosts BERT e Qwen. Note a diferença de convenção que precisa ser preservada
   na extração: EWC entra com fator `0.5 * lambda` (linha 2009), SI entra com
   `si_lambda` sem o meio (linha 2014).

4. **MAS não existe.** Os acertos de `mas` na primeira varredura eram
   substring (`schemas`, `formas`). Precisa ser escrito do zero.

5. **LwF é parcial.** `lwf_calibrated` (linha 185) é `MethodSpec(distillation=True)`
   — destilação de logits de classes antigas
   (`_old_class_distillation_loss`, 1283). Não é o LwF completo com
   temperatura e cabeça por tarefa. Reportar como "destilação calibrada", não
   como LwF, ou implementar o LwF de fato.

---

## Consequência para a Fase 1

O bloco de código é **maior** do que o roadmap estimou em um eixo e **menor**
em outro:

- **Menor:** EWC e SI não precisam ser escritos, precisam ser *extraídos*. A
  penalidade compartilhada existe e já está correta.
- **Maior:** a extração tem de atravessar três hosts com arquiteturas
  diferentes, e o host Qwen treina LoRA — onde "âncora do parâmetro" significa
  a matriz do adaptador, não o peso base, e isso é uma decisão de método que
  precisa ser declarada e não inferida.

Ordem que isso sugere para a Fase 1, substituindo a do roadmap:

1. Extrair `ewc_penalty` como módulo (`src/dual_heater/ewc.py`), com os testes
   da Tarefa 1.2, e **fazer `split_mnist.py` passar a usá-lo** — assim a
   extração é validada contra uma implementação que já roda, em vez de ser
   código novo sem uso.
2. MAS no mesmo módulo, reusando a penalidade (`fisher=omega`).
3. Só então ligar ao host BERT (`split_clinc150.py`), que é onde o teste de
   mordida da Tarefa 1.3 importa.

---

## Comando de reprodução

```bash
cd /mnt/B-SSD/fachel/dual-heater-mlp
grep -rniE "\bewc\b|\bsi_(lambda|importance|anchors|epsilon|path)\b|['\"]mas['\"]|\blwf\b|['\"]agem['\"]|['\"]der\+?\+?['\"]" \
  --include=*.py src/ experiments/ | grep -v test
```
