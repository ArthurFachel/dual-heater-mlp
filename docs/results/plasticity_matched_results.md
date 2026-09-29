# Plasticidade pareada sobre a superfície — resultado

**Pré-registro:** `goals/protocol_plasticity_matched.md` (congelado em `d9b57fc`,
antes da primeira seed).
**Execução:** 29/09/2026, 10 seeds, 71,4 min no caminho crítico, 3 GPUs
(2× GTX 1080 Ti + Titan Xp), lançada destacada.
**Artefatos:** `results/plasticity_matched/`.

---

## Pergunta

A confirmação de 28/09 reportou `exact − lr_control` como **pareado em
plasticidade efetiva**, com a verificação H1 exigindo `E = 0,8500` em todas as
seeds. A verificação passou. A afirmação era falsa mesmo assim:
`effective_plasticity()` media a máscara apenas sobre os parâmetros que carregam
binding, e os 2.555.904 parâmetros de `A`, congelados e com fator de update
exatamente zero, não entravam na média.

Medido sobre a superfície treinável de referência, o endpoint primário comparou
**E = 0,532 contra E = 0,850** — 32 pontos de plasticidade removida na direção
que favorece o braço tratado.

A pergunta desta run: **o efeito sobrevive quando os dois braços são pareados
sobre a superfície?**

## Desenho

Três braços, 10 seeds pareadas, banda nova (6.000.003+), disjunta de todas as
já gastas.

| braço | `A` | lr | treináveis | `E` superfície | papel |
|---|---|---:|---:|---:|---|
| `vanilla` | treinável | 1,000e-4 | 6.769.920 | 1,000000 | descritivo (P10, fora da família) |
| `lr_control` | treinável | **6,225e-5** | 6.769.920 | **0,622462** | falsificador pareado |
| `frozen_a_control` | **congelada** | 1,000e-4 | 4.214.016 | **0,622462** | tratamento (LoRA-FA) |

Os dois braços do contraste removem **a mesma fração de plasticidade da mesma
superfície** — um por congelamento, outro por learning rate. É a única
diferença entre eles.

O `lr` do controle é aritmética pura: `4.214.016 / 6.769.920 = 0,6224617129892229`,
derivado sem consultar acurácia.

## Integridade antes do endpoint

As oito verificações H1–H8 rodaram **antes** de qualquer endpoint ser olhado
(`scripts/verify_plasticity_matched.py`):

```
OK: 10 seeds, 8/8 verificações passaram
```

Incluindo: pareamento de `E` exato entre os dois braços (diferença 0,0 em todas
as seeds), ambos no alvo declarado, tokens de treino pareados dentro de cada
seed, e — o modo de falha que já mordeu este projeto — confirmação de que o
`lr_control` **realmente recebeu** o fator de escala, e de que o
`frozen_a_control` **não** recebeu.

## Resultado

Endpoint primário **forgetting**, sinal exato bicaudal, Holm sobre família de 2:

| contraste | média | mediana | sinais | p | p Holm | |
|---|---:|---:|---:|---:|---:|---|
| `frozen_a_control − lr_control` (forgetting) | **−0,1260** | −0,1287 | **10−/0+** | 0,00195 | **0,00391** | ✓ |
| `frozen_a_control − lr_control` (FAA) | **+0,0992** | +0,1105 | **0−/10+** | 0,00195 | **0,00391** | ✓ |

**Desfecho "sobrevive" do pré-registro.** O critério P9 (p_Holm < 0,025 **e**
mediana negativa) é satisfeito, e ambos os endpoints atingem o **piso de Holm**
para n=10 (0,0039) — não há p menor possível com 10 seeds nesta família.

Unânime: 10/10 seeds na mesma direção nos dois endpoints, sem uma única
inversão.

### Valores absolutos

| braço | forgetting | FAA | aquisição últ. tarefa | retenção T0 |
|---|---:|---:|---:|---:|
| `vanilla` | 0,4125 | 0,5879 | 0,9230 | 0,0490 |
| `lr_control` | 0,4741 | 0,5275 | 0,9177 | 0,0427 |
| `frozen_a_control` | **0,3480** | **0,6267** | 0,8950 | **0,0930** |

O ganho de retenção **não** vem de colapso de aquisição: o
`frozen_a_control` perde 2,8 pp de aquisição na última tarefa (0,8950 contra
0,9230 do vanilla) e ganha 4,4 pp de retenção da primeira. O trade-off existe,
mas é favorável.

## O achado que o veredito binário esconde

Os contrastes descritivos contra o `vanilla` (P10, fora da família, sem
correção múltipla) mudam a interpretação:

| contraste descritivo | forgetting | sinais | p |
|---|---:|---:|---:|
| `frozen_a_control − vanilla` | −0,0644 | 8−/2+ | 0,109 |
| `lr_control − vanilla` | **+0,0616** | **9+/1−** | **0,0215** |

**O `lr_control` é pior que o vanilla.** Reduzir o learning rate uniformemente
em 37,75% **piora** o esquecimento (+6,16 pp), em 9 das 10 seeds.

Isso tem três consequências que precisam ser ditas juntas:

1. **O contraste primário é significativo em parte porque o comparador piorou.**
   Dos 12,60 pp de diferença, ~6,2 pp vêm do `lr_control` degradando abaixo do
   vanilla e ~6,4 pp do `frozen_a_control` melhorando acima dele. O efeito do
   tratamento contra a linha de base neutra (`frozen_a_control − vanilla`,
   −6,44 pp) **não atinge significância** (p = 0,109) nesta n.

2. **Isto reabre — e agora replica — o achado que 28/09 não replicou.** A run
   confirmatória de 28/09 mediu `lr_control − vanilla` com `E = 0,850` e deu
   nulo (p = 0,754). Esta run mede com `E = 0,622` e dá +0,0616 com p = 0,0215.
   A leitura coerente é que **o efeito depende da magnitude da redução**: cortar
   15% do lr não faz nada detectável; cortar 37,75% degrada. Não há contradição
   entre as duas runs — elas mediram condições diferentes.

3. **A conclusão do artigo fica mais interessante, não menos.** O resultado
   honesto não é "LoRA-FA funciona", é: *reduzir a superfície treinável e reduzir
   o learning rate removem a mesma quantidade de plasticidade efetiva e produzem
   resultados opostos.* A **distribuição** da restrição importa, mesmo quando a
   quantidade é idêntica. Esse é um achado mais forte para um paper de protocolo
   do que o veredito binário sozinho, porque mostra que o pareamento não é uma
   formalidade: ele revela estrutura que a comparação contra vanilla esconde.

## Consequência para o eixo de protocolo

O caso central do artigo **muda de forma**, e melhora.

A narrativa anterior seria "o instrumento derrubou nosso resultado". A narrativa
real é mais precisa e mais defensável:

> Um resultado confirmatório pré-registrado, com Holm, sinal exato e cinco
> gates, descreveu condições experimentais que não existiam. Corrigido o
> pareamento, o efeito **sobrevive** — mas a correção revela que o falsificador
> uniforme é ativamente pior que não fazer nada, o que significa que a
> comparação original media a soma de dois efeitos de sinais opostos sem saber.

O instrumento não derrubou a conclusão. Ele mostrou que a conclusão estava certa
**pelo motivo errado**, e que a magnitude reportada era a soma de um ganho real
com uma degradação do comparador.

## Custo

1.040 s/seed nos três braços, 71,4 min no shard crítico (4 seeds), ~53 min nos
outros dois. Sharding por seed, não por braço: os três braços de uma seed
compartilham GPU, stream de dados e ordenação, que é o que mantém o teste de
sinal pareado válido.

## Limites

- Um host (Qwen2.5-0.5B), um benchmark (Split-CLINC150), `r = 16`, 10 tarefas.
- **Congelar `A` remove 37,75% dos parâmetros treináveis E fixa as direções da
  down-projection.** Um controle que removesse a mesma fração de parâmetros
  *aleatórios* separaria as duas coisas; não está neste desenho. Declarado em
  §J do pré-registro **antes** da run, não descoberto depois.
- `frozen_a_control − vanilla` não atinge significância em n=10 (p = 0,109). A
  afirmação "LoRA-FA melhora sobre vanilla" **não** é sustentada por esta run;
  o que é sustentado é o contraste pareado.
- Não implementamos a correção de gradiente em forma fechada do LoRA-FA
  (`g_B = (r/α)²(AᵀA)⁻¹ g_B`); nosso `frozen_a_control` é LoRA-FA sem ela.
- O comportamento do `lr_control` em função da magnitude do corte é
  **observacional aqui** (dois pontos: 0,850 → nulo, 0,622 → degrada). Uma
  curva exigiria pré-registro próprio.

## Nota de método

O contraste que esta run executa custava um argumento de linha de comando
(`--learning-rate-scale`) que não existia. O `lr_control` só podia ser escalado
via `--target-plasticity`, que é o botão dos braços mascarados — e um protocolo
cujo tratamento congela em vez de mascarar não tem braço mascarado. Sem a flag,
o falsificador teria rodado **sem tratamento**, com manifest de aparência
normal. Seis mutações da cadeia CLI → config → runner → artefato foram mortas
por teste antes do congelamento; duas delas só morreram depois de extrair
`build_run_config` e `plasticity_record` como seams testáveis.
