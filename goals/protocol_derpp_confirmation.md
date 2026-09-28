# Protocolo congelado — confirmação de SlowHeat+DER++ em Split-MNIST

> Pré-registro, escrito **antes** de qualquer seed desta confirmação. Não é
> editado depois de ver acurácia; alterações exigem commit anterior à run e
> linha nova na tabela K.

**Estado: CONGELADO em 28/09/2026**, antes da primeira seed.

---

## A. O que motiva

`docs/architectures/arch_mlp.md`, evidência E3 (suíte `dualheat_pairs`,
exploratória, 10 seeds, Holm por dataset):

| Contraste | Referência | Candidato | Diferença | p Holm | Sinais |
|---|---:|---:|---:|---:|---|
| vs DER++ | 82,40% | 85,58% | **+3,182 pp** | <0,0001 | 10+/0− |

É o resultado exploratório mais forte do host MLP e **não tem pré-registro
próprio**. A confirmação congelada existente (declarada em código, em
`experiments/confirmatory_split_mnist.py`, via `CONFIRMATORY_SEEDS` e as
constantes de endpoint) cobre apenas `SlowHeat+Replay − Replay`.

Sem esta confirmação, a única alegação de eficácia pré-registrada do projeto é
"melhor que replay simples", que é um alvo fraco. Com ela, passa a ser "melhor
que DER++", um baseline forte.

## B. Pergunta confirmatória

Acrescentar Functional SlowHeat ao DER++ melhora a acurácia média final
class-incremental em Split-MNIST, em relação ao DER++ sozinho?

## C. Hipótese declarada antes da run

*`slowheat_derpp_hidden_beta_30_budget_0.25` supera `derpp` na acurácia média
final, com a diferença pareada por seed positiva na maioria das seeds.*

Hipótese nula: as diferenças pareadas se distribuem simetricamente em torno de
zero. **Resultado nulo é publicável** e encerra a alegação; não haverá tentativa
de resgate com grade nova, outro dataset ou outro endpoint.

## D. Decisões congeladas

| # | Decisão | Valor congelado | Justificativa |
|---|---|---|---|
| D1 | Braços | `derpp` e `slowheat_derpp_hidden_beta_30_budget_0.25` — exatamente dois | a pergunta é um contraste único; braço a mais infla multiplicidade sem responder nada |
| D2 | Seeds | `2_000_003, 2_025_011, 2_050_017, 2_075_037, 2_100_043, 2_125_059, 2_150_061, 2_175_087, 2_200_089, 2_225_097, 2_250_101, 2_275_103, 2_300_119, 2_325_127, 2_350_133, 2_375_149, 2_400_151, 2_425_163, 2_450_169, 2_475_179` (20) | banda própria, verificada disjunta das confirmatórias do Split-MNIST (104729–586429), das exploratórias (múltiplos de 11), das do LoRA (700001–925097) e da banda de QB-2 (1000003+) |
| D3 | Endpoint primário | acurácia média final class-incremental, contraste `slowheat_derpp − derpp` | mesmo endpoint primário do resto do projeto |
| D4 | Endpoints secundários | average forgetting, BWT, tempo de parede | reportados sempre, nunca promovidos a primário |
| D5 | Teste | t pareado **e** sinal exato bicaudal, **ambos reportados** | a confirmação existente mostrou que eles podem discordar (task-aware: t dá 0,043, sinal dá 0,115). Reportar os dois impede escolher o favorável depois |
| D6 | Correção múltipla | nenhuma — família de **uma** comparação primária | declarado antes; os secundários de D4 são descritivos e não entram em família |
| D7 | Critério de sucesso | `p < 0,05` **nos dois testes** de D5 **e** diferença média positiva | mais estrito que o padrão, por causa da discordância conhecida entre os testes |
| D8 | Hiperparâmetros | `beta = 30`, `budget = 0,25`, `lr = 1e-3`, idênticos ao exploratório | **não** re-tunar. Re-tunar depois de ver o exploratório é escolher hiperparâmetro pela acurácia |

## E. Cenário

| Item | Valor |
|---|---|
| Dataset | Split-MNIST, 5 tarefas de 2 classes |
| Cenário | class-incremental, cabeça global compartilhada |
| Modelo | MLP `[256, 128]` |
| Pareamento | por seed: inicialização, partições, minibatches e índices de replay idênticos entre braços |
| Otimizador | `lr = 1e-3` nos dois braços |
| Hardware | CPU |

## F. Verificação obrigatória antes da run

1. As 20 seeds de D2 não pertencem a nenhuma banda reservada — o guard-rail de
   `experiments/dualheat_pairs.py:105` deve aceitá-las;
2. `DERPP_CONFIRMATORY_SEEDS` existe em
   `experiments/confirmatory_split_mnist.py` e bate com D2;
3. `tests/test_split_mnist.py::test_derpp_confirmatory_seeds_are_registered_and_disjoint`
   passa;
4. suíte de testes verde;
5. este arquivo commitado **antes** da primeira seed.

## G. Análise declarada

```text
para cada seed s: d_s = FAA(slowheat_derpp, s) − FAA(derpp, s)
testes: t pareado bicaudal E sinal exato bicaudal
confirma se: ambos p < 0,05 E media(d_s) > 0
```

Sem análise intermediária, sem parada antecipada, sem inspeção seed a seed antes
das 20 terminarem.

## H. O que este protocolo não resolve

- **Um único benchmark.** Split-MNIST tem teto alto; `+3 p.p.` ali pode não
  transferir. A inversão de sinal na CNN
  ([proximo_passo_cnn.md](proximo_passo_cnn.md), C-3) mostra que o efeito do
  SlowHeat depende tanto do método base quanto do seletor de memória.
- **Sem tuning por método.** `lr = 1e-3` fixo para os dois braços. Um revisor
  pode alegar que o DER++ está sub-ajustado. Limitação declarada, não resolvida —
  tunar por método custa semanas nas 3 GPUs Pascal disponíveis.
- **Não isola o SlowHeat.** Como em todo contraste algoritmo-contra-algoritmo do
  projeto, o que se mede é `DER++ + SlowHeat` contra `DER++`, não o efeito do
  SlowHeat em isolamento.

## K. Registro de alterações

| Data | Alteração | Antes da run? |
|---|---|---|
| 28/09/2026 | criação e congelamento. D1 a D8 fechados. | sim — nenhuma seed executada |
