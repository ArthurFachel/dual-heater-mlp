# Próximo passo do artigo — decisão de ancoragem e caminho crítico

> Documento de decisão, escrito em 25/09/2026 com a confirmação de LoRA em
> execução. Responde a uma pergunta única: **dado tudo que foi medido, qual é a
> próxima coisa a fazer para que exista um artigo submissível?**
>
> Complementa `opcoes_novidade_e_proximos_passos.md`, que lista *opções* de
> novidade. Aqui há uma recomendação e uma ordem.

---

## 1. Onde o artigo está hoje

O manuscrito tem 604 linhas, doze seções e uma lista de onze afirmações
seguras. O que ele **não** tem é uma contribuição declarada que sobreviva às
suas próprias ressalvas:

- **§9 fecha a porta da prioridade.** "No claim of being the first method to
  use neuron importance, MAX masks, lateral inhibition or importance-dependent
  plasticity is justified."
- **§11 fecha a porta da eficácia.** "SlowHeat outperforms established
  continual-learning baselines" está listado como **não sustentado**. O único
  contraste confirmado é contra Replay puro em Split-MNIST.
- **§8 derrubou a explicação favorita.** O regime de proteção não transfere: em
  cinco alvos não-Transformer, hard nunca vence soft.

O trabalho de LoRA das últimas duas sessões não mudou esse quadro na direção
que se esperava — mas mudou em outra direção, que é o ponto deste documento.

## 2. O que o trabalho de LoRA de fato produziu

### 2.1 Três mecanismos novos, três falhas

`rank`, `leak` e `slice` foram implementados, testados por mutação e medidos em
duas runs de 10 seeds. Sob plasticidade pareada em `E = 0,85`, os três ficam
**abaixo do vanilla** e não superam o controle de learning rate de forma
significativa. Detalhes em
[docs/lora/lora_slowheat_mechanisms.md](../docs/lora/lora_slowheat_mechanisms.md).

Isso é resultado negativo publicável, não desperdício. Mas **não é um método
novo**, e portanto não resolve o problema da §1.

### 2.2 Um achado metodológico que ninguém procurava

O braço `lr_control` — reduzir learning rate uniformemente na mesma
plasticidade — é o **pior de todos os braços, pior que não fazer nada**:

| braço | FAA | forgetting |
|---|---|---|
| vanilla (E=1,0) | 0,6016 | +0,3967 |
| lr_control (lr × 0,85) | 0,5684 | +0,4350 |

Reduzir LR uniformemente em 15% **piora** o esquecimento. Isso tem duas
consequências não triviais:

1. O falsificador padrão da literatura ("seu método é só um LR menor
   disfarçado") é um teste **mais difícil de passar do que se supunha**, porque
   o alvo não está abaixo de todo mundo — mas também é um teste que **não pode
   ser dispensado**, porque sem ele o contraste contra vanilla confunde "freiar
   ajuda" com "freiar seletivamente ajuda".
2. Qualquer trabalho que compare proteção seletiva apenas contra vanilla está
   medindo a coisa errada.

### 2.3 O efeito **cresce** sob controle mais rigoroso

Este é o resultado mais forte do projeto inteiro, e é contraintuitivo:

| contraste (braço `exact`) | run 1, sem pareamento | run 2, `E = 0,85` |
|---|---|---|
| forgetting vs vanilla | −0,0629 (p=0,002) | −0,0807 (p=0,002) |
| FAA vs vanilla | +0,0349 (p=0,11) | +0,0478 (p=0,02) |
| forgetting vs lr_control | — | **−0,1190 (p=0,002, 10/10)** |

Controles mais rigorosos normalmente encolhem efeitos. Este aumentou. A leitura
é que a primeira run **subestimava** o mecanismo, porque braços com mais
plasticidade retida apareciam artificialmente bem.

---

## 3. A decisão que trava todo o resto

O artigo pode se ancorar em três lugares. **Só um deles é alcançável com o que
existe.**

| Âncora | O que exigiria | Viável? |
|---|---|---|
| **Prioridade de mecanismo** | Provar que alguma construção é nova | **Não.** §9 já fechou, e os três mecanismos novos falharam |
| **Eficácia contra baselines** | Vencer DER++, ER-ACE, EWC, replay em múltiplos alvos | **Não com o hardware atual.** Exigiria a lista de 10 itens da §10 do manuscrito |
| **Protocolo de avaliação + resultados negativos** | O que já está medido, mais uma literatura honesta | **Sim** |

### Recomendação: ancorar no protocolo, não no mecanismo

A tese defensável é:

> *Avaliações de proteção seletiva confundem rotineiramente a **quantidade** de
> plasticidade removida com a **distribuição** dela. Propomos um protocolo que
> separa as duas — pareamento por plasticidade efetiva medida, mais um controle
> de learning rate na mesma plasticidade — e mostramos que ele muda conclusões
> em duas direções: efeitos aparentes desaparecem, e efeitos reais crescem.*

Evidência já disponível para sustentar isso:

- **A1**, o teorema `E >= (N−P)/N`, dá o piso analítico que torna o pareamento
  bem-definido e diz quando um alvo é inalcançável.
- **A2**, a construção iso-plasticidade por bisseção, é o instrumento.
- **A3**, o contrato de otimizador, explica por que mascarar gradiente bruto
  não é mascarar o passo — pré-requisito para o pareamento significar algo.
- Os **três mecanismos falhados** são a demonstração de que o protocolo tem
  poder de rejeição (não é um protocolo que aprova tudo).
- O **crescimento do efeito do `exact`** é a demonstração de que ele não é
  apenas conservador.
- O **`lr_control` pior que vanilla** é o achado que justifica o controle.

Isso é um artigo de metodologia com resultados empíricos, não um artigo de
método novo. É menor em ambição e **inteiramente sustentado** pelo que já foi
medido.

---

## 4. Caminho crítico

### Passo 1 — Levantamento de literatura (BLOQUEADOR, ~1 dia, sem GPU)

Nada mais deve ser escrito antes disto. Alvos mínimos:

- **O-LoRA** (ortogonalização de subespaço por tarefa) — colide com `rank`
- **InfLoRA** (subespaço livre de interferência) — colide com `rank` e `slice`
- Família LoRA-CL: **CorDA**, **MoRAL**, **SAPT**, e o que a busca revelar
- Para o protocolo: alguém já pareia por plasticidade efetiva medida? A busca
  tem que ser explícita sobre isso, porque é a âncora recomendada

Saída: uma seção de related work honesta e, crucialmente, **a resposta sobre se
o protocolo é novo**. Se já existir, a âncora da §3 cai e este documento precisa
ser reescrito antes de qualquer redação.

### Passo 2 — Fechar a confirmação de LoRA (em execução)

`goals/protocol_lora_confirmation.md`, 10 seeds pré-registradas, três braços.
Resultado esperado em ~1,5h. Dois desfechos, ambos utilizáveis:

- **Confirma** (p < 0,025): o artigo ganha um resultado confirmatório em
  Transformer, que hoje não tem.
- **Não confirma**: o artigo ganha uma demonstração de que um efeito
  exploratório com p=0,002 não sobreviveu à confirmação — que é exatamente o
  argumento a favor do protocolo.

**Não reabrir o pré-registro em nenhum dos casos.**

### Passo 3 — Resolver o confundimento de aquisição do `exact` (~2h de GPU)

Este é o buraco no único resultado positivo. O braço `exact` treina 4.214.016
parâmetros contra 6.769.920 dos outros (rank 16), porque congela `A` — 62% da
capacidade do adaptador. A leitura *ele
esquece menos porque aprendeu menos* **não está descartada**, e o pareamento em
`E` não a resolve — `E` mede plasticidade retida da máscara, não capacidade do
adaptador.

Experimento que decide: rodar `exact` com `r = 32` contra `vanilla` com
`r = 16`, o que iguala os parâmetros treináveis. Se o efeito sobreviver, a
explicação por capacidade cai. Se desaparecer, o único resultado positivo do
projeto era um artefato de capacidade — e isso precisa ser reportado.

Custo estimado: ~2h nas 3 GPUs. **É o experimento de maior valor por hora de
GPU no projeto hoje.** Deve ser pré-registrado antes de rodar.

### Passo 4 — Reescrever §9, §10 e §11 do manuscrito

Só depois dos passos 1 a 3:

- **§9** ganha a família LoRA-CL e o posicionamento do protocolo.
- **§10** encolhe drasticamente. A lista atual de dez itens descreve um artigo
  de eficácia que não vai existir; sob a âncora de protocolo, a maior parte
  deixa de ser pré-requisito de submissão e vira trabalho futuro.
- **§11** ganha as afirmações de LoRA, incluindo as negativas, e perde qualquer
  resquício de ambição de eficácia.

### Passo 5 — Decidir o veículo

Sob a âncora de protocolo, workshops de metodologia e reprodutibilidade em ML
são alvo mais adequado que trilha principal. Isso não é rebaixamento: é o
lugar onde "seu controle está errado e aqui está a medida disso" é
contribuição, e não fraqueza.

---

## 5. O que NÃO fazer

- **Não rodar mais mecanismos novos antes do passo 1.** Três já falharam sob
  controle honesto. Um quarto sem levantamento de literatura é aposta cega.
- **Não reanalisar as seeds exploratórias com a família de Holm reduzida.**
  Escolher a correção depois de ver o resultado é exatamente o que ela existe
  para impedir. Já está registrado em
  `goals/protocol_lora_confirmation.md`, A6.
- **Não perseguir eficácia contra baselines fortes com o hardware atual.** A
  §10 do manuscrito pede treze comparações com tuning declarado. Em 3 GPUs
  Pascal isso são semanas, e o resultado provável é "dentro do ruído".
- **Não apagar os resultados negativos.** `rank`, `leak` e `slice` permanecem
  no código, nos testes e nos docs. Sob a âncora de protocolo eles deixam de
  ser fracasso e viram evidência de poder de rejeição.

---

## 6. Resumo executivo

| # | Ação | Custo | Bloqueia |
|---|---|---|---|
| 1 | Levantamento O-LoRA / InfLoRA / LoRA-CL **e** protocolos de pareamento | ~1 dia, sem GPU | tudo |
| 2 | Fechar confirmação de LoRA | em execução | §11 |
| 3 | Pré-registrar e rodar `exact r=32` vs `vanilla r=16` | ~2h GPU | o único resultado positivo |
| 4 | Reescrever §9, §10, §11 | ~1 dia | submissão |
| 5 | Escolher veículo | — | — |

A decisão que precisa ser tomada por um humano, e que nenhum experimento
resolve: **aceitar a âncora de protocolo** (§3) em vez de continuar procurando
um método novo. As duas últimas sessões foram uma tentativa honesta de achar o
método; ela falhou três vezes seguidas sob controle rigoroso. O ativo que
sobrou é o rigor em si.

## Referências

- [Opções de novidade](opcoes_novidade_e_proximos_passos.md)
- [Pré-registro da confirmação de LoRA](protocol_lora_confirmation.md)
- [Protocolo iso-plasticidade](protocol_iso_plasticity.md)
- [Resultados do benchmark de LoRA](../docs/lora/lora_qwen_benchmark_results.md)
- [Os três mecanismos](../docs/lora/lora_slowheat_mechanisms.md)
- [Manuscrito](../article/_archive/manuscript_functional_slowheat.md)
