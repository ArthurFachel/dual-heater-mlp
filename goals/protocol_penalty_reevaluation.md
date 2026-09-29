# Protocolo congelado — re-avaliação de métodos de penalidade sob plasticidade pareada

> Pré-registro. Escrito **antes** da primeira seed e antes de qualquer execução
> de GPU. Não é editado depois de ver acurácia; alterações exigem commit
> anterior à run e linha nova na tabela §K.

**Estado: CONGELADO em 2026-09-29.**
**Estado de execução: NÃO AUTORIZADO.** Nenhuma seed foi rodada. Ver §J.
**Instrumento:** `goals/protocol_plasticity_generalization.md` (congelado antes
deste documento, em `3de06ba`), implementado em `src/dual_heater/plasticity.py`.
**Origem:** Fase 2.4 de `goals/roadmap_icml_ijcnn.md`; é a Seção 4 do artigo.

---

## A. Pergunta

A literatura de consolidação por penalidade (EWC, SI, MAS) reporta ganhos de
retenção contra um baseline sequencial que **não é pareado em plasticidade**.
Cada um desses métodos remove plasticidade do update; o baseline não remove
nenhuma. A pergunta é:

*Removida a mesma quantidade média de plasticidade — medida por
`plasticity_ratio()` — um método de penalidade reduz o esquecimento mais do que
uma redução uniforme do learning rate?*

Não é "EWC ajuda?". Isso é compatível com "qualquer freio ajuda". É "a **forma**
da remoção importa, sob medição comensurável?".

## B. Hipótese declarada antes da run

*Para cada método `M ∈ {ewc, si, mas}`, a diferença pareada por seed
`forgetting(M) − forgetting(lr_control_M)` é negativa na maioria das seeds.*

Hipótese nula: as diferenças pareadas se distribuem simetricamente em torno de
zero.

**Os dois desfechos são reportáveis e nenhum encerra o artigo.** Se nenhum
método bate seu `lr_control` pareado, isso é um resultado negativo publicável e
é exatamente o coração da Seção 4. O artigo é sobre o protocolo de medição, não
sobre um método vencer.

## C. Arms

| arm | o que é | plasticidade |
|---|---|---|
| `vanilla` | fine-tuning sequencial, sem penalidade | 1,0 por construção |
| `ewc` | Fisher empírico diagonal por exemplo (`src/dual_heater/ewc.py`) | medida |
| `si` | caminho do gradiente (Zenke et al., ICML 2017) | medida |
| `mas` | sensibilidade da saída, sem rótulo (Aljundi et al., ECCV 2018) | medida |
| `lr_control_ewc` | sem penalidade, `lr × E_ewc` | = `E_ewc` |
| `lr_control_si` | sem penalidade, `lr × E_si` | = `E_si` |
| `lr_control_mas` | sem penalidade, `lr × E_mas` | = `E_mas` |

**O `lr_control` é obrigatório e é pareado POR MÉTODO, não um só para os três.**
Um `lr_control` único forçaria os três métodos ao mesmo `E`, que é precisamente
a grandeza que o instrumento existe para medir em vez de assumir. O `E` de cada
método é lido do passo-sombra do próprio método (§E), sem consultar acurácia.

> **Emenda de 2026-09-29, alinhada a G1' do protocolo do instrumento.** `E_M` é
> agora a **razão de normas** `‖Δ_native‖/‖Δ_unpenalized‖`, não a razão média
> por elemento. A calibração mediu razão por elemento `> 1` em `ewc` e `mas`,
> o que tornaria `lr_scale > 1` — um aumento de learning rate, não um controle.
> Ver `goals/protocol_plasticity_generalization.md` §B.2 e §K.
>
> **Se `E_M > 1` mesmo sob a razão de normas** (G8), o método é declarado
> **não-pareável**: seu contraste contra `vanilla` é reportado como secundário,
> o `lr_control` correspondente **não é construído**, e a comparação sai da
> família confirmatória — com a redução de `m` declarada AQUI, antes das seeds,
> e não depois de ver quem venceu.

### C.1 A ordem obrigatória de duas passadas

O `lr_scale` de um `lr_control` depende do `E` medido do seu método, que só
existe depois de rodar o método. Logo:

1. **Passada 1 (medição):** rodar `vanilla`, `ewc`, `si`, `mas` com o
   passo-sombra ativo. Ler `plasticity_ratio` e as diagnósticas do manifest.
   **Nenhum endpoint de acurácia é lido nesta passada.**
2. **Congelar** os três `lr_scale` num commit, com os manifests citados.
3. **Passada 2 (controles):** rodar os três `lr_control` com os escalares
   congelados, nas mesmas seeds.

Ler acurácia entre 1 e 2 contaminaria a escolha do escalar. O runner deve
recusar a passada 2 se o commit que congela os escalares não existir.

## D. Hosts

**Split-MNIST (MLP) e Split-CIFAR100 (CNN).** Os dois já têm EWC, SI e MAS
completos e testados.

**NÃO** o host BERT, **NÃO** o host Qwen. A Tarefa 1.3 do roadmap (EWC no host
BERT) é paralela a este piloto, não anterior: ela cobre baselines que um
reviewer cobra no host transformer, e não está no caminho crítico da Seção 4.
Ver `docs/audits/baseline_inventory.md` para a tabela host × baseline.

## E. Instrumentação da plasticidade

`measure_shadow_step()` (`src/dual_heater/plasticity.py`), por passo amostrado:

| campo | fonte |
|---|---|
| `norm_ratio` | **métrica primária G1'** do protocolo do instrumento |
| `plasticity_ratio` | diagnóstica D1 (era a primária até a emenda de 29/09) |
| `direction_cosine` | diagnóstica D2 |

**Taxa de amostragem: TODOS os passos com âncora (1/1).** Emendado em
2026-09-29; ver §K linha 3 e a ressalva abaixo. `E_M` deixa de ser uma
estimativa amostral e passa a ser a **média exata** sobre os passos em que a
penalidade esteve ativa. Custo: 3 passos em vez de 1 em cada passo medido.

> **A taxa original era 1/50 e foi revogada por insuficiência de potência,
> medida exaustivamente e sem ler acurácia.** A justificativa de 1/50 no texto
> congelado era custo (3× por passo medido); a medição mostrou que esse custo
> não existe nesta escala — 12 seeds × 4 arms custam ~10 min de CPU a 1/5 e
> ~20 min a 1/1.
>
> A medição de TODOS os 128 passos com âncora de uma seed (`7000003`, MAS) deu:
>
> | quantidade | valor |
> |---|---:|
> | `E` exato (128 passos) | 1,0124 |
> | desvio-padrão ENTRE passos | 0,0772 |
> | faixa observada | 0,804 a 1,213 |
> | IC95% da média com n=3 | [0,923, 1,095], largura 0,171 |
> | IC95% da média com n=36 | [0,991, 1,034], largura 0,043 |
> | efeito a medir, `abs(1 − E)` | **0,012** |
>
> Ou seja: a 1/50 o ruído da estimativa é ~14× o efeito, e mesmo agregando as
> 12 seeds (36 amostras) o intervalo continua ~3× mais largo que o efeito. A
> série a 1/50 reportou `E_mas = 0,957` (pareável por G8) enquanto o valor
> exato é 1,015 (**não** pareável) — a classificação G8 dependia do ruído.
> Congelar `lr_scale = 0,957` seria construir o controle sobre uma flutuação
> amostral.
>
> **Por que isto não é seleção pós-hoc:** a evidência é a variância do próprio
> instrumento sobre um único arm, medida exaustivamente, sem nenhum contraste
> entre métodos e sem nenhum endpoint de acurácia. A alternativa — manter 1/50
> — não produziria um resultado diferente a nosso favor; produziria um `E`
> cujo intervalo de confiança contém tanto "pareável" quanto "não pareável",
> ou seja nenhuma conclusão. Amostrar exaustivamente remove a estimativa do
> caminho, não escolhe o seu valor.
>
> **As runs feitas a 1/50 e a 1/5 são descartadas**, não reinterpretadas, e
> ficam arquivadas em `results/_abandoned_penalty_pass1_sampling_1_50/`.

`E_M` de um método é a **média das razões de normas sobre todos os passos com
âncora daquela seed**, e entra no manifest com o número de passos e o
desvio-padrão entre passos.

### E.1 O portão de comensurabilidade

`direction_cosine` decide se o pareamento é enunciável. Um método com
`cos ≈ 1,0` faz um escalonamento quase diagonal e é comparável a um
`lr_control`. Um método com `cos` baixo **gira** o update, e parear por `E`
sozinho seria enganoso.

**Limiar declarado agora: `cos < 0,9`.** Para um método abaixo disso, o
contraste contra seu `lr_control` é reportado com a ressalva explícita de que a
intervenção não é diagonal, e o artigo diz isso na tabela. **O contraste não é
descartado e o método não é removido da família** — remover um arm depois de ver
uma diagnóstica é seleção pós-hoc, mesmo que a diagnóstica não seja acurácia.

## F. Endpoint primário e família confirmatória

**Endpoint primário:** esquecimento médio (average forgetting), como definido em
`src/dual_heater/metrics.py`, diferença pareada por seed.

**Família confirmatória — exatamente 6 comparações:**

| # | contraste | host |
|---|---|---|
| 1 | `ewc − lr_control_ewc` | Split-MNIST |
| 2 | `si − lr_control_si` | Split-MNIST |
| 3 | `mas − lr_control_mas` | Split-MNIST |
| 4 | `ewc − lr_control_ewc` | Split-CIFAR100 |
| 5 | `si − lr_control_si` | Split-CIFAR100 |
| 6 | `mas − lr_control_mas` | Split-CIFAR100 |

**Teste:** sinal bilateral sobre diferenças pareadas por seed.
**Correção:** Holm sobre os 6.
**Limiar:** `p_Holm < 0,05`.

Os contrastes `M − vanilla` e a acurácia média final são **secundários**,
reportados sempre, **fora** da família confirmatória. Acurácia média final vai
na tabela principal junto com o esquecimento, para que ganho de retenção não
esconda colapso de aquisição.

## G. Banda de seeds

**Banda declarada: 7.000.003 + espaçamento primo, 12 seeds:**

```
7000003, 7025011, 7050017, 7075037, 7100043, 7125059,
7150061, 7175087, 7200089, 7225097, 7250101, 7275103
```

Disjunta de todas as bandas já gastas, verificado por
`tests/test_penalty_reevaluation_seeds.py`:

| banda | uso |
|---|---|
| 700.001+ | confirmação LoRA |
| 2.000.003+ | confirmação DER++ |
| 3.000.003+ | confirmação de seletor de replay |
| 4.000.003+ | ablação de critério de importância |
| 5.000.003+ | decomposição do `exact` |
| 6.000.003+ | plasticidade pareada |

## H. Piso de significância — o desenho limpa a própria barra

Com `n` seeds unânimes e família de tamanho `m` sob Holm, o menor `p_Holm`
atingível é `m × 2 / 2^n`. Calculado **antes** de lançar:

| n | m = 6 | limpa 0,05? |
|---|---|---|
| 8 | 0,0469 | sim, por margem mínima |
| 10 | 0,0117 | sim |
| **12** | **0,0029** | **sim, com folga de 17×** |

`n = 12` é a escolha: a 10 seeds um único sinal invertido já deixaria o piso em
0,0234 e a margem some rápido. A 12, o desenho sobrevive a duas seeds contrárias
e ainda pode reportar significância.

## I. Custo — a medir, não a estimar

**Não estimado neste documento de propósito.** O procedimento declarado:

1. Rodar **uma seed de calibração fora da banda** (`7999991`), um host, todos os
   arms da passada 1, cronometrada.
2. Multiplicar por arms × seeds ÷ GPUs, contando a passada 2 separadamente.
3. Lembrar que a avaliação Class-IL cresce como `T(T+1)/2`: extrapolar de um
   prefixo curto subestima.
4. **Reportar o número ao Fachel antes de pedir autorização para a run cheia.**

A seed de calibração é lida para **wiring e custo apenas** — que o
`plasticity_ratio` aparece no manifest, que os arms propagam, quanto tempo leva.
Seus endpoints são `n = 1` e **não têm nenhum valor estatístico**; citá-los como
resultado é proibido.

## J. Portão de autorização

**Nada neste protocolo roda sem "pode rodar" explícito do Fachel.** Isso inclui
a seed de calibração da §I.

Ao lançar:

```bash
setsid nohup <comando> < /dev/null > results/<run>/log 2>&1 & disown
ps -eo pid,ppid,sess,cmd | grep <padrao> | grep -v grep   # exige PPID=1
```

Uma run já morreu por SIGTERM aos 34 minutos por não estar destacada. Verificar
`PPID=1` **é** parte do lançamento, não uma conferência opcional.

## K. Registro de alterações

| data | alteração |
|---|---|
| 2026-09-29 | documento congelado, antes de qualquer seed e de qualquer GPU |
| 2026-09-29 | **`E_M` passa a ser a razão de normas** (§C, §E), alinhado à emenda G1' de `protocol_plasticity_generalization.md`. Motivo: razão por elemento `> 1` em `ewc` e `mas` na calibração tornaria `lr_scale > 1`. Acrescentada a regra do método **não-pareável** (G8) e a ressalva de potência do §E (3 amostras/arm/seed a 1/50). Evidência mecanismo-only, `n=1`, seed fora da banda, nenhuma acurácia lida. Nenhuma seed confirmatória gasta até aqui. |
| 2026-09-29 | **Taxa de amostragem 1/50 → 1/1** (§E). Motivo: a medição exaustiva dos 128 passos com âncora de uma seed deu dp entre passos de 0,0772 contra um efeito `abs(1−E)` de 0,0124 — o ruído da estimativa a 1/50 é ~14× o efeito, e a 36 amostras ainda é ~3×. A classificação G8 do MAS invertia entre a série a 1/50 (`E = 0,957`, pareável) e o valor exato (`E = 1,015`, não pareável). A justificativa original de 1/50 era custo, e o custo medido é ~20 min de CPU para as 12 seeds. Evidência mecanismo-only: variância de um único arm, sem contraste entre métodos e sem nenhuma acurácia lida. **As runs a 1/50 e 1/5 foram descartadas e arquivadas**, não reinterpretadas. Nenhuma seed confirmatória de acurácia gasta até aqui; a passada 2 não havia começado. |
