# Efeito de piso: um endpoint pré-registrado com variância zero

**Origem:** calibração do §I de `goals/protocol_long_sequence_criterion.md`,
executada em 05/10/2026 para medir custo.
**Status:** **exploratório.** Uma seed (`11_900_003`), fora da banda
confirmatória, GTX 1080 Ti, 167 s. **Não é um resultado confirmatório e não
pode ser citado como tal.**
**Artefatos:** `results/long_sequence_criterion_calibration/`.

---

## O que aconteceu

A calibração existia para medir tempo de parede. Mediu: 167 s por seed, 0,46 h
projetadas para as dez, dentro do teto de 4 h do §I.

E mostrou que o endpoint primário congelado em S5 — `task1_retention`, a
retenção da primeira tarefa — **vale exatamente 0,0000 nos quatro braços**.

| braço | T1 aquisição | T1 retenção | T1 esquecimento | FAA | FAA / chance |
|---|---:|---:|---:|---:|---:|
| `vanilla` | 0,8133 | **0,0000** | 0,8133 | 0,1867 | 14,00× |
| `slowheat_random_hard` | 0,8133 | **0,0000** | 0,8133 | 0,1767 | 13,25× |
| `slowheat_magnitude_hard` | 0,8133 | **0,0000** | 0,8133 | 0,1733 | 13,00× |
| `slowheat_hard` | 0,8133 | **0,0000** | 0,8133 | 0,1853 | 13,90× |

A família confirmatória declarada em S7 são dois contrastes pareados sobre esse
endpoint. Ambos dão diferença **identicamente zero**, em todas as seeds, por
construção. Um teste de sinal sobre zeros não tem poder: as dez seeds
produziriam `p = 1,00` garantido, e isso não seria evidência de empate — seria
ausência de medição.

`task1_forgetting` herda o mesmo problema. Com a aquisição idêntica entre braços
(0,8133, porque a tarefa 1 é treinada antes de qualquer proteção entrar em
ação) e a retenção zerada, o esquecimento da tarefa 1 é a mesma constante nos
quatro.

## Não é ruído, e não é regime degenerado

Duas explicações alternativas, as duas descartadas pelos próprios artefatos.

**Não é ruído.** Em `T = 2`, 10 seeds, o mesmo endpoint tem desvio-padrão entre
seeds de 0,045 a 0,082 e amplitude entre braços de 0,354:

| braço | T1 retenção `T = 2` | sd entre seeds |
|---|---:|---:|
| `slowheat_magnitude_hard` | 0,4423 | 0,0817 |
| `slowheat_hard` | 0,4407 | 0,0809 |
| `slowheat_random_hard` | 0,3200 | 0,0630 |
| `vanilla` | 0,0887 | 0,0447 |

Em `T = 5` a amplitude entre braços é **0,0000**. O endpoint não ficou ruidoso;
parou de variar.

**Não é regime degenerado.** O gate do §G.1 compara a FAA do melhor braço contra
a chance e exige ≥ 4×. O observado é **14×**. Os modelos aprendem: a acurácia na
última tarefa fica entre 0,80 e 0,91. É um regime saudável em que uma métrica
específica saturou.

## O mecanismo: saturação progressiva por distância temporal

`per_task_forgetting` mostra onde a informação está e onde ela acabou:

| braço | tarefa 1 | tarefa 2 | tarefa 3 | tarefa 4 |
|---|---:|---:|---:|---:|
| `slowheat_hard` | 0,813 | 0,773 | 0,690 | **0,423** |
| `slowheat_magnitude_hard` | 0,813 | 0,767 | 0,707 | **0,430** |
| `slowheat_random_hard` | 0,813 | 0,803 | 0,797 | **0,467** |
| `vanilla` | 0,813 | 0,880 | 0,807 | **0,670** |

A tarefa 1 está saturada no teto para todos. A tarefa 2 quase. A tarefa 3
começa a separar. **A tarefa 4 discrimina com folga** — 0,423 contra 0,670, um
intervalo de 0,247.

O decaimento da tarefa 1, braço a braço:

```
slowheat_hard   0.813 -> 0.527 -> 0.250 -> 0.097 -> 0.000
magnitude_hard  0.813 -> 0.520 -> 0.257 -> 0.077 -> 0.000
random_hard     0.813 -> 0.430 -> 0.140 -> 0.013 -> 0.000
vanilla         0.813 -> 0.040 -> 0.033 -> 0.000 -> 0.000
```

Três coisas, nessa ordem:

1. **Os braços de fato diferem** — em `T = 2` a separação é grande (0,527 contra
   0,040) e é ela que sustenta R-A.
2. **A diferença encolhe a cada tarefa** e some no estágio 4.
3. **O zero final é comum a todos**, inclusive ao braço que começou dez vezes
   acima do `vanilla`.

A informação sobre a tarefa 1 não desapareceu de repente: ela foi consumida pela
interferência das tarefas seguintes até cair abaixo da resolução do conjunto de
validação (300 exemplos; um acerto vale 0,0033).

Generalizando: num stream de `T` tarefas, um endpoint ancorado na tarefa `k`
tem `T - k` tarefas de interferência. **A tarefa 1 é a que sofre mais, e por
isso satura primeiro.** Ancorar o primário nela é escolher deliberadamente o
ponto de medição com menor vida útil.

## Por que nenhum gate pegou

O §G.1 foi escrito antes das seeds, de propósito, com a lição da passada 2
(`6970361`: `p = 0,00049` com os três arms perto da chance). Ele vigia FAA
contra a chance. **Passou com 14× e não podia detectar isto**, porque olha outra
métrica: a FAA é a média da diagonal final, dominada pelas tarefas recentes, que
estão vivas.

Nem o verificador pré-endpoint resolve. V5 checa a variância do *ranking de
importância* — a diagnóstica da §F — e ela está saudável (3,0e-03, mesma ordem
do `T = 2`). V6 checa a sobreposição entre braços. V7 checa o regime. **Nenhum
checa se o endpoint primário tem variância.**

Esta é a segunda ocorrência do padrão que o item 5.1 de
`goals/roadmap_icml_ijcnn.md` registra sobre R-E:

> um resultado confirmatório com Holm, sinal exato e 5 gates descreveu condições
> que não existiam, e nenhum gate podia detectar — todos verificavam a métrica,
> e era a métrica que estava errada

E é de espécie diferente, o que a torna informativa em vez de repetitiva:

| | R-E (28/09) | aqui (05/10) |
|---|---|---|
| a métrica está | **calculada errada** (escopo do denominador) | **calculada certa** |
| o problema é | implementação | o endpoint não responde neste regime |
| detectável por | revisão de código | só medindo a dispersão no regime alvo |

O segundo caso é o mais perigoso dos dois: não há bug para encontrar. O código
está correto, o teste está verde, a mutação morre, e o número sai zero porque a
pergunta não tem resposta ali.

## Correção do que foi dito antes

Num relato anterior desta sessão eu afirmei que "a coluna `forgetting` ordena os
braços" e que isso sustentaria W1 e W2. **A afirmação misturou duas métricas.**

- `task1_forgetting` — derivada do endpoint primário — é a mesma constante
  (0,8133) nos quatro braços. **Não ordena nada.**
- `average_forgetting` — média sobre as quatro tarefas — ordena: 0,6750
  (`funcional`) < 0,6792 (`magnitude`) < 0,7200 (`aleatório`) < 0,7925
  (`vanilla`).

A ordenação existe, mas é de outra métrica, e vem das tarefas recentes, não da
tarefa 1.

## O que isto NÃO autoriza

- **Não autoriza trocar o endpoint primário.** Escolher `average_forgetting`
  depois de ver que ele funciona é pesca de endpoint, e seria pior que o
  problema original. Qualquer mudança exige pré-registro novo, commitado antes,
  declarando que a emenda veio de uma calibração que leu acurácia.
- **Não autoriza nenhuma afirmação sobre W1 ou W2.** Uma seed, sem correção
  múltipla, fora da banda. A ordenação de `average_forgetting` é uma **pista**,
  não evidência.
- **Não invalida R-A nem R-B.** Aqueles resultados são `T = 2`, onde o endpoint
  tem amplitude 0,354 entre braços e sd de 0,045–0,082 entre seeds. Eles medem o
  que dizem medir.
- **Não é um claim sobre a literatura.** A observação de que trabalhos de CL
  reportam retenção da primeira tarefa em sequências longas sem declarar o piso
  é uma **conjectura**; não foi feita revisão sistemática.

## Consequência imediata

As dez seeds da banda 12.000.017+ **não foram gastas**. Rodá-las produziria
`p = 1,00` por construção e queimaria a banda num resultado vazio.

A Fase 3 fica congelada no estado atual: protocolo commitado, scripts prontos,
custo medido (0,46 h). Retomável em 28 minutos de GPU se o endpoint for
re-registrado.

## O que pode entrar no ICML, e com que ressalva

A Seção 3 do ICML (`5.1` do roadmap) é o estudo de caso do próprio resultado do
projeto. Este episódio é um segundo caso, de espécie diferente, e o eixo
candidato é:

> Um endpoint de retenção tem vida útil finita no comprimento da sequência. A
> primeira tarefa é a que satura primeiro, e um protocolo pré-registrado que a
> escolha como primário pode descobrir isso só depois de congelar.

**Ressalva que acompanha obrigatoriamente:** uma seed exploratória, um host, um
benchmark. Se este eixo entrar no artigo, precisa de medição confirmatória da
dispersão do endpoint em função de `T` — o que é um protocolo novo, não uma
releitura deste.

Decidir se vai para a Seção 3 (caso central) ou para a Seção 6 (limitações)
depende do que a Seção 3 já carrega. Não é decidido aqui.
