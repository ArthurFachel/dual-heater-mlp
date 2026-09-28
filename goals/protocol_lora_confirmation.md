# Protocolo congelado — confirmação do LoRA exato produtor-only

> Pré-registro. Este arquivo é escrito **antes** da run confirmatória e não é
> editado depois de ver qualquer acurácia dela. Se um valor mudar, o commit que
> o muda tem de vir antes da run correspondente, e a mudança fica registrada na
> tabela K.

**Estado: CONGELADO em 25/09/2026**, antes de qualquer seed confirmatória ter
sido executada.

As seeds e o conjunto de braços estão registrados em código
(`LORA_CONFIRMATORY_SEEDS` e `PREREGISTERED_ARMS` em
`experiments/qwen_lora_sweep.py`) e a flag `--confirmatory` **recusa** qualquer
desvio: subconjunto de seeds, braço a mais, ou ausência do alvo de
plasticidade. Coberto por `tests/test_lora_confirmation_guard.py`.

---

## A. O que motiva esta confirmação

Duas runs exploratórias de 10 seeds cada, em Split-CLINC150 com Qwen2.5-0.5B,
estão reportadas em
[docs/lora/lora_qwen_benchmark_results.md](../docs/lora/lora_qwen_benchmark_results.md).
A segunda pareia todos os braços mascarados em `E = 0,85` e inclui o controle
de learning rate.

Achado exploratório que motiva confirmar:

| contraste | métrica | diferença | p (sinal exato) | a favor |
|---|---|---|---|---|
| exact − lr_control | forgetting | −0,1190 | 0,0020 | 10/10 |
| exact − lr_control | FAA | +0,0810 | 0,0215 | 9/10 |
| exact − vanilla | forgetting | −0,0807 | 0,0020 | 10/10 |
| exact − vanilla | FAA | +0,0478 | 0,0215 | 9/10 |

Achado que motiva **descartar** três braços: `rank`, `leak` e `slice` não
superam `lr_control` significativamente (p ≥ 0,11) e os três ficam **abaixo do
vanilla** em FAA mesmo sob plasticidade pareada.

## B. Pergunta confirmatória

Dada a mesma quantidade de plasticidade efetiva removida, proteger
**seletivamente** as unidades importantes reduz o esquecimento mais do que
reduzir o learning rate **uniformemente**?

Note que a pergunta não é "proteger ajuda?" (respondida pelo contraste contra
vanilla, que é compatível com "qualquer freio ajuda"), e sim "a **distribuição**
da proteção importa?".

## C. Hipótese declarada antes da run

*`exact` reduz o forgetting médio em relação a `lr_control` na mesma
plasticidade efetiva, com a diferença pareada por seed negativa na maioria das
seeds.*

Hipótese nula: as diferenças pareadas se distribuem simetricamente em torno de
zero. Resultado nulo é publicável e encerra a linha; não há tentativa de salvar
o achado exploratório.

---

## D. Decisões congeladas

| # | Decisão | Valor congelado | Justificativa |
|---|---|---|---|
| A1 | Braços | `vanilla`, `exact`, `lr_control` — exatamente três | `rank`, `leak` e `slice` falharam sob controle honesto; mantê-los inflaria a correção múltipla sem chance realista de efeito |
| A2 | Seeds confirmatórias | `700001, 725009, 750019, 775037, 800053, 825059, 850061, 875089, 900089, 925097` (banda própria; ver K, 25/09) | declaradas **antes** de qualquer acurácia; disjuntas das confirmatórias do Split-MNIST, das exploratórias (múltiplos de 11) e das do protocolo iso-plasticidade. Registradas em código como `LORA_CONFIRMATORY_SEEDS` e protegidas por guard-rail |
| A3 | Endpoint primário | forgetting médio, contraste `exact − lr_control` | é o único contraste que isola distribuição de quantidade; forgetting é o endpoint com o maior efeito exploratório |
| A4 | Endpoints secundários | FAA (mesmo contraste); forgetting e FAA vs vanilla | reportados sempre, nunca promovidos a primário depois de vistos |
| A5 | Teste estatístico | sinal exato bicaudal sobre as 10 diferenças pareadas por seed | sem suposição de normalidade em n=10; é o mesmo teste do sweep exploratório |
| A6 | Correção múltipla | Holm sobre a família de 2 comparações do endpoint primário e secundário no contraste vs lr_control (α=0,05) | a família encolheu de 8 para 2 **porque os braços foram declarados antes**, não porque o resultado foi visto |
| A7 | Plasticidade alvo | `E = 0,85`, re-resolvida por bisseção em cada fronteira | mesmo valor da run exploratória; alcançável por todos os braços |
| A8 | Critério de sucesso | p < 0,025 (limiar de Holm mais estrito na família de 2) no endpoint primário **e** sinal na direção predita | declarado antes; um p entre 0,025 e 0,05 é registrado como "não confirmado" |

### Regra sobre A6, que é onde este pré-registro compra rigor

Reanalisar as **seeds exploratórias já executadas** com a família reduzida de 2
comparações seria escolher a correção depois de ver o resultado, exatamente o
que a correção existe para impedir. O afrouxamento do limiar (de 0,00625 para
0,025) só é legítimo porque:

1. o conjunto de braços está declarado **antes** de rodar as seeds novas;
2. as seeds confirmatórias nunca foram executadas com nenhum braço de LoRA;
3. este arquivo está commitado antes da run.

## E. Cenário e modelo

| Item | Valor |
|---|---|
| Modelo | `Qwen/Qwen2.5-0.5B`, `Qwen2ForSequenceClassification`, `num_labels=150` |
| Precisão | fp32 em memória e no treino (Pascal CC 6.x não tem bf16 nativo; `SlowHeatAdamW` rejeita `GradScaler`) |
| Adaptador | LoRA `r=16`, `alpha=32`, alvos `gate_proj`, `up_proj`, `down_proj` |
| Dataset | `clinc/clinc_oos:plus`, Class-IL por domínio |
| Tarefas | 10, na ordem de `CLINC150_DOMAINS` |
| Dados por tarefa | 750 treino / 300 validação (50 e 20 por classe) |
| Épocas por tarefa | 3 |
| `batch_size` | 8 |
| `max_length` | 64 |
| `learning_rate` | 1e-4 |
| Cabeça de classificação | treinável e sem máscara em todos os braços |
| Hardware | uma seed inteira por GPU, sem DDP |

**A ordem das tarefas é parte do protocolo** e é passada explicitamente, nunca
inferida de uma contagem.

## F. Braços

| Braço | Definição | `E` | Papel |
|---|---|---|---|
| `vanilla` | LoRA puro, sem tracker e sem máscara | 1,00 | âncora de forgetting |
| `exact` | `A` congelada, linhas de `B` mascaradas | 0,85 | o candidato |
| `lr_control` | `lr × 0,85`, **sem máscara nenhuma** | 1,00 | o falsificador |

O `lr_control` é **obrigatório**. Sem ele o contraste volta a ser contra
vanilla, que não separa "proteger ajuda" de "proteger seletivamente ajuda".

Medido na run exploratória: `lr_control` é o **pior** braço de todos, pior que
o vanilla (FAA 0,5684 contra 0,6016). Isso o torna um teste não-trivial e não
um alvo fácil posicionado abaixo de todos.

## G. Endpoints

Reportados por braço e por seed, com diferenças **pareadas por seed** (nunca
por posição na lista) e contagem de sinais:

| Endpoint | Definição | Papel |
|---|---|---|
| Forgetting médio | `mean_{k<T-1} (max_l A[l,k] − A[T-1,k])` | **primário**, contraste vs `lr_control` |
| FAA | `mean_k A[T-1,k]` | secundário |
| BWT | transferência para trás | descritivo |
| `E_eff` por fronteira | plasticidade medida | verificação de pareamento |
| Custo | tempo de parede, pico de memória, tokens | descritivo |

O pareamento por seed é propriedade testada em
`tests/test_lora_sweep_aggregate.py`, com mutação verificada.

## H. Verificação obrigatória antes da run

0. a run é lançada com `--confirmatory`, que pina seeds e braços e recusa
   desvio silencioso;
1. `E_eff = 0,8500` (tolerância 1e-3) para `exact` em **todas** as fronteiras e
   todas as seeds;
2. `lr_control` não registra máscara nenhuma (`mask_bindings()` vazio) e tem
   `effective_learning_rate == 8,5e-5`;
3. `vanilla` tem `E = 1,0` e `lr = 1e-4`;
4. tokens de treino idênticos entre os três braços dentro de cada seed
   (checksum do pareamento de dados);
5. `claim_scope` do manifesto declara pareamento e presença do controle;
6. suíte completa verde.

## I. Análise declarada

```text
para cada seed s: d_s = forgetting(exact, s) − forgetting(lr_control, s)
teste: sinal exato bicaudal sobre {d_s}
confirma se: p < 0,025 E mediana(d_s) < 0
```

Sem análise intermediária, sem parada antecipada, sem inspeção seed a seed
antes das 10 terminarem. As 10 seeds rodam e o teste é aplicado uma vez.

## J. O que este protocolo não resolve

- **Custo de aquisição do `exact`.** Ele treina metade dos parâmetros do
  adaptador (4.214.016 contra 6.769.920, r=16) e chega mais baixo ao fim da primeira
  tarefa. `E` mede plasticidade retida da máscara, não capacidade do adaptador,
  então a leitura *esquece menos porque aprendeu menos* permanece possível e
  deve ser reportada junto com o resultado.
- **Generalização.** Um modelo, um `r`, um benchmark, um ponto de plasticidade.
- **Prioridade.** O-LoRA, InfLoRA e a família de LoRA para continual learning
  não foram levantados. Nenhuma reivindicação de novidade antes dessa checagem,
  independentemente do resultado desta run.

## K. Registro de alterações

| Data | Alteração | Antes da run? |
|---|---|---|
| 25/09 | criação e congelamento. A1 a A8 fechados. | sim — nenhuma seed confirmatória executada com braço de LoRA |
| 25/09 | **correção de A2, antes da primeira run.** A lista original começava em 104729, que é a **primeira seed confirmatória do Split-MNIST** (`CONFIRMATORY_SEEDS` em `experiments/confirmatory_split_mnist.py`); o guard-rail do repositório abortou o lançamento, corretamente. O erro expôs um problema maior: as outras nove eram primos arbitrários, não registrados em lugar nenhum, e portanto não protegidos contra uso exploratório futuro. Substituídas por uma banda própria (700001+), registradas em código como `LORA_CONFIRMATORY_SEEDS` e protegidas por guard-rail com teste. | sim — a run abortou na validação de argumentos, nenhum modelo foi carregado e nenhum endpoint foi lido |

| 25/09 | correção factual na seção J: a contagem de parâmetros treináveis do braço `exact` estava copiada do smoke com rank 8 (2.174.208 contra 3.452.160); os valores corretos para `r=16`, lidos dos manifestos, são 4.214.016 contra 6.769.920. **Não** toca em nenhuma decisão A1–A8, endpoint ou critério — J é descritiva, sobre o que o protocolo não resolve. | run em execução; nenhum endpoint lido, nenhum critério alterado |

| 28/09 | **registro pós-run, sem alteração de critério.** A run concluiu 10/10 seeds e a verificação dos manifestos expôs que a **seção E descreve dois valores que não foram os executados**: `alpha` consta como 32 mas foi **16**, e `max_length` consta como 64 mas foi **48**. A confirmatória usou exatamente o mesmo config da run exploratória que a motivou (verificado campo a campo nos dois conjuntos de manifestos), então nenhum braço foi favorecido e a comparação entre runs permanece válida. É erro de transcrição no protocolo, não desvio de execução. Registrado aqui em vez de corrigido em silêncio na seção E. **Nenhuma decisão A1–A8, endpoint ou critério é tocada.** | **não** — posterior à run. Por isso fica como registro, e a seção E abaixo mantém o texto original com esta nota anexada |

### Valores executados, para referência (28/09)

| item | seção E | executado nas 10 seeds |
|---|---|---|
| `alpha` | 32 | **16** |
| `max_length` | 64 | **48** |

Todos os demais campos da seção E batem com os manifestos: modelo, fp32,
`r=16`, alvos `gate_proj`/`up_proj`/`down_proj`, 10 tarefas, 750/300 por tarefa,
3 épocas, batch 8, `lr=1e-4`, cabeça treinável, uma seed por GPU.

**Resultado da run:** [docs/lora/lora_confirmation_results.md](../docs/lora/lora_confirmation_results.md).
Endpoint primário confirmado (`p = 0,02148 < 0,025`, mediana negativa, 9/10).

A partir daqui, qualquer alteração exige commit anterior à run correspondente e
uma linha nova nesta tabela.
