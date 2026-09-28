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
| R2 | Seletores | `loss` e `representative`, exatamente dois | são os dois extremos da inversão; `first` e `hybrid` ficam fora da família confirmatória |
| R3 | Métodos | `derpp` e `slowheat_derpp_hidden_beta_30_budget_0.25` | o par que produz o contraste |
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

Do `sweep_report.json` exploratório, `elapsed_seconds` médio para
`split_cifar100` com 2 seletores e todos os métodos de memória é **3,6 min por
seed**. Restringindo aos 2 métodos de R3, o custo real fica abaixo disso.

**20 seeds × ~2 min ≈ 40 min de CPU.** Cabe numa sessão, sem GPU.

Isso torna a confirmação barata o suficiente para que um resultado nulo não
represente perda material — o que é exatamente a condição em que um
pré-registro é fácil de honrar.

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
