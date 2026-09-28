# Protocolo congelado — confirmação da inversão de sinal por seletor de replay

> Pré-registro. Escrito **antes** de qualquer seed desta confirmação. Não é
> editado depois de ver acurácia; alterações exigem commit anterior à run e
> linha nova na tabela K.

**Estado: CONGELADO em 28/09/2026**, antes da primeira seed.
**Alvo de submissão:** IJCNN 2027 (deadline 31/01/2027, 6 páginas IEEE).

---

## A. O que motiva

A análise exploratória do sweep de 50 seeds
([../docs/results/replay_selection_50seed_analysis.md](../docs/results/replay_selection_50seed_analysis.md))
encontrou que **o sinal do contraste inverte conforme o seletor de memória**,
dentro do mesmo dataset e do mesmo método base. Em `split_cifar100`,
`slowheat_derpp − derpp` na acurácia média final:

| seletor | média | sinais | p (sinal, bruto) |
|---|---:|---:|---:|
| `loss` | **+0,0035** | 44+/6− | 3,2e-08 |
| `hybrid` | +0,0011 | 32+/18− | 0,065 |
| `first` | −0,0035 | 13+/37− | 9,4e-04 |
| `representative` | **−0,0050** | 7+/43− | 2,1e-07 |

Duas direções opostas, ambas com dezenas de seeds concordando, separadas apenas
pela escolha de um componente que a maioria dos artigos não reporta.

**Esse material é exploratório por construção.** Os dados precediam a pergunta,
e o próprio artefato carrega `status: exploratory_not_independent_confirmation`.
Nenhuma reanálise dos mesmos 50 seeds muda isso. Este protocolo existe para
gastar seeds novas num contraste declarado antes de vê-las.

## B. Pergunta confirmatória

Em `split_cifar100`, o contraste `slowheat_derpp − derpp` na acurácia média
final tem **sinal oposto** sob o seletor `loss` e sob o seletor
`representative`?

## C. Hipótese declarada antes da run

*Sob `loss`, a diferença pareada por seed é positiva na maioria das seeds; sob
`representative`, negativa na maioria das seeds.*

Hipótese nula: pelo menos um dos dois contrastes não difere de zero, ou ambos
têm o mesmo sinal.

**Os três desfechos e o que cada um significa, declarados agora:**

| desfecho | leitura | publicável? |
|---|---|---|
| ambos significativos, sinais opostos | a inversão replica; é o resultado do artigo | sim — é a tese |
| ambos significativos, mesmo sinal | a inversão era artefato dos 50 seeds originais | sim — retratação honesta do achado exploratório |
| um ou nenhum significativo | o efeito é menor do que o exploratório sugeria | sim — reportar como não replicado |

**Nenhum desfecho autoriza trocar de dataset, de seletor ou de endpoint para
resgatar o achado.** Um resultado nulo encerra a linha e vai para o artigo
como tal.

## D. Decisões congeladas

| # | Decisão | Valor congelado | Justificativa |
|---|---|---|---|
| R1 | Dataset | `split_cifar100`, apenas | é onde a inversão é mais limpa no exploratório; um dataset mantém a família pequena |
| R2 | Seletores | `loss`, `representative` e **`first`** | `loss` e `representative` são os dois extremos da inversão e formam a família confirmatória. `first` é **exigido estruturalmente** pelo runner (`replay_selection_sweep.py:397`): é o seletor de referência de `paired_differences_vs_first`, sem o qual o relatório não é construído. Seus contrastes são reportados como **exploratórios**, fora da família de R8 |
| R3 | Contraste confirmatório | `slowheat_derpp_hidden_beta_30_budget_0.25 − derpp` | é o par que produz a inversão. **O runner executa a matriz fixa completa** (6 métodos, `SWEEP_METHODS`) e não aceita filtro; os demais métodos são computados e reportados, mas **só este contraste é confirmatório** |
| R4 | Seeds | 20, banda `3_000_003, 3_025_011, 3_050_017, 3_075_037, 3_100_043, 3_125_059, 3_150_061, 3_175_087, 3_200_089, 3_225_097, 3_250_101, 3_275_103, 3_300_119, 3_325_127, 3_350_133, 3_375_149, 3_400_151, 3_425_163, 3_450_169, 3_475_179` | banda própria, disjunta de toda seed já usada no repositório (ver seção G) |
| R5 | Endpoint primário | `final_average_accuracy`, contraste `slowheat_derpp − derpp`, por seletor | mesmo endpoint primário do sweep exploratório |
| R6 | Endpoints secundários | `average_forgetting`, `backward_transfer`, tempo de parede, bytes de replay | reportados sempre, nunca promovidos |
| R7 | Teste | sinal exato bicaudal sobre as 20 diferenças pareadas por seed, um por seletor | mesmo teste do resto do projeto; sem suposição de normalidade |
| R8 | Correção múltipla | Holm sobre a família de **2** (um contraste por seletor), α = 0,05 | família declarada antes da run |
| R9 | Critério de sucesso | ambos com `p_Holm < 0,05` **e** sinais das medianas opostos | é a inversão que precisa replicar, não cada contraste isoladamente |
| R10 | Hiperparâmetros | idênticos ao sweep exploratório: `beta = 30`, `budget = 0,25`, e o `config_payload` registrado em `sweep_report.json` para `split_cifar100` | re-tunar depois de ver o exploratório seria escolher hiperparâmetro pela acurácia |

## E. Por que 20 seeds e não 50

O teto de significância do teste de sinal bicaudal é `2/2^n`, e sob Holm numa
família de `m` o melhor `p` atingível é `m * 2/2^n`:

| n | família = 2 | veredito |
|---|---:|---|
| 10 | 3,9e-03 | viável |
| **20** | **3,8e-06** | **folga de três ordens de grandeza** |
| 50 | 1,8e-15 | desnecessário |

Com 20 seeds o desenho tem folga ampla para detectar um efeito consistente, e
os efeitos exploratórios (44+/6− e 7+/43− em 50 seeds) são fortes o bastante
para sobreviver à redução. Não há razão para gastar 50.

## F. Custo medido, não estimado

Do `sweep_report.json` exploratório, somando `elapsed_seconds` médio para
`split_cifar100` sobre os **3 seletores de R2** e a **matriz fixa completa de 6
métodos** que o runner executa: **5,3 min por seed**.

**20 seeds ≈ 2,0 h de GPU** (1 Pascal). Sem download (os dados de CIFAR-100 já
estão em `data/cifar-100-python`).

Decomposição do custo por seed, do `sweep_report.json` exploratório (medido em
CUDA): 2 métodos sem memória rodam uma vez (0,52 min); 4 métodos com memória
rodam uma vez por seletor (1,80 min × 3 seletores = 5,40 min). Total 5,9 min por
seed.

> **Duas correções registradas antes de qualquer análise.**
>
> 1. A estimativa inicial de ~40 min supunha que o runner aceitaria um filtro de
>    métodos. Ele não aceita — executa `SWEEP_METHODS` inteiro.
> 2. A estimativa seguinte de 1,8 h somou mal (tratou os 4 métodos de memória
>    como se rodassem uma vez, não uma vez por seletor) e, pior, comparou tempos
>    de CUDA com uma execução em CPU. **Uma tentativa em CPU foi iniciada e
>    abandonada aos 19 min**, com fator CPU/GPU medido em 4,3× e projeção de
>    8,5 h. A árvore parcial está preservada em
>    `results/_abandoned_cpu_partial_replay_selector/` (8 runs de learner do
>    grupo `no_memory`, nenhuma análise feita sobre ela).
>
> **Nenhuma decisão científica foi tocada por esses erros** — seeds, contraste,
> família e critério permanecem como congelados. O device passa a ser `cuda`
> para todas as 20 seeds, sem mistura.

## G. Verificação obrigatória antes da run

1. As 20 seeds de R4 não colidem com nenhuma banda já usada: confirmatórias do
   Split-MNIST (104729–586429), exploratórias (múltiplos de 11), LoRA
   (700001–925097), QB-2 (1000003+), M1 (2000003+), **e as 50 seeds do sweep
   exploratório** (`sweep_report.json → seeds`, banda 6314085–2101606466 — atenção,
   esta se sobrepõe numericamente a outras bandas e precisa ser checada como
   conjunto, não por intervalo);
2. o guard-rail de `experiments/replay_selection_sweep.py:360` aceita as seeds;
3. `DERPP_SELECTOR_CONFIRMATORY_SEEDS` existe em código e bate com R4;
4. suíte de testes verde;
5. este arquivo commitado **antes** da primeira seed.

## H. Análise declarada

```text
para cada seletor s em {loss, representative}:
    para cada seed k: d_k = FAA(slowheat_derpp, s, k) − FAA(derpp, s, k)
    p_s = sinal exato bicaudal sobre {d_k}
p_loss, p_repr := Holm(p_loss, p_repr)

confirma a inversao se:
    p_loss < 0,05  E  p_repr < 0,05
    E  sign(mediana d | loss) != sign(mediana d | representative)
```

Sem análise intermediária, sem parada antecipada, sem inspeção seed a seed
antes das 20 terminarem.

## I. O que este protocolo não resolve

- **Não isola o SlowHeat.** Conforme o `comparison_warning` do artefato
  original: *"Every ranked learner selects its own memory"* — cada learner
  ranqueia as próprias imagens, então o contraste é algoritmo-completo contra
  algoritmo-completo. A inversão é real como fenômeno de avaliação; a
  atribuição causal ao mecanismo não está estabelecida.
- **Um dataset, um par de métodos.** `split_cifar10` também inverte no
  exploratório, mas com outro par de seletores (`first` −0,0078 vs `hybrid`
  +0,0006). Se essa generalização entrar no artigo, entra como exploratória e
  assim declarada — este protocolo não a confirma.
- **Não explica o mecanismo da inversão.** Por que `loss` e `representative`
  produzem memórias que revertem o sinal é uma pergunta aberta; o artigo
  reporta o fenômeno e a sua consequência metodológica.
- **Não mede relevância prática.** As diferenças são de milésimos. O achado é
  sobre a **direção** do contraste ser instável, não sobre o tamanho do ganho —
  e o artigo precisa dizer isso explicitamente para não parecer que vende
  +0,35 p.p. como avanço.

## J. Regra editorial herdada

Sempre reportar `average_forgetting` ao lado de `final_average_accuracy`. No
exploratório há casos em que os dois discordam (`split_cifar100`/`representative`
contra replay ganha acurácia **e** esquece mais), e citar só um inverteria a
leitura.

## K. Registro de alterações

| Data | Alteração | Antes da run? |
|---|---|---|
| 28/09/2026 | criação e congelamento. R1 a R10 fechados. | sim — nenhuma seed executada |
| 28/09/2026 | **emenda antes da primeira seed.** R2 passa a incluir `first` (exigência estrutural do runner, `replay_selection_sweep.py:397`, que o usa como referência de `paired_differences_vs_first`); R3 reformulado de "métodos" para "contraste confirmatório", porque o runner executa a matriz fixa de 6 métodos sem filtro; seção F corrigida de ~40 min para 1,8 h. **A família confirmatória de R8 permanece 2** (`loss` e `representative`), inalterada. Nenhum critério estatístico foi tocado. | sim — nenhuma seed executada |
| 28/09/2026 | **correção de device e custo, antes de qualquer análise.** A run passa a ser executada em `cuda` (1 Pascal), não em CPU. Motivo: os tempos de referência do artefato exploratório foram medidos em CUDA, e a projeção em CPU dava 8,5 h contra as 2,0 h em GPU. A tentativa em CPU foi abandonada aos 19 min e a árvore parcial preservada em `results/_abandoned_cpu_partial_replay_selector/`, sem análise. Seção F reescrita com a decomposição correta do custo. **R1 a R10 inalterados** — nenhuma seed, contraste, família ou critério foi tocado. | sim — nenhuma análise feita |
