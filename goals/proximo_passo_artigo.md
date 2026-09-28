# Próximo passo do artigo — índice por arquitetura e decisões transversais

> Reescrito em 28/09/2026. A versão anterior
> ([_arquivo_proximo_passo_artigo_2026-09-25.md](_arquivo_proximo_passo_artigo_2026-09-25.md))
> tratava os quatro hosts como um bloco só e concluía "inviável" para o artigo
> inteiro a partir de uma exigência — vencer DER++/ER-ACE/EWC — que **só se
> aplica a MLP e CNN**. Esses baselines não existem em BERT nem em Qwen.
> Misturar os hosts produziu uma decisão errada de escopo.
>
> Este documento agora contém apenas o que é transversal. Cada host tem o seu.

---

## Os quatro documentos

| Host | Documento | O que sustenta | Precisa de GPU? |
|---|---|---|---|
| MLP | [proximo_passo_mlp.md](proximo_passo_mlp.md) | **eficácia local** confirmada e replicada (+0,87 p.p. vs Replay, 20 seeds) | não |
| CNN | [proximo_passo_cnn.md](proximo_passo_cnn.md) | **limite**: o sinal do efeito inverte conforme o método base | leve |
| BERT | [proximo_passo_bert.md](proximo_passo_bert.md) | **funcionalidade e limite**: porta para Transformer, perde para replay | não, host fechado |
| Qwen | [proximo_passo_qwen.md](proximo_passo_qwen.md) | **o protocolo** e o seu poder de rejeição | sim |

Cada documento é autônomo: tem a própria evidência, a própria âncora, o próprio
caminho crítico e a própria lista de proibições.

---

## 1. A tabela de baselines, que é a origem do erro anterior

| Baseline | MLP | CNN | BERT | Qwen |
|---|---|---|---|---|
| Vanilla / sequencial | sim | sim | sim | sim |
| Replay | sim | sim | sim | não |
| DER++ | sim | sim | **não** | **não** |
| ER-ACE | sim | sim | **não** | **não** |
| A-GEM, EWC, SI, LwF | sim | sim | **não** | **não** |
| `lr_control` (LR uniforme na mesma plasticidade) | não | não | não | **sim** |
| `permuted` (mesmo `E`, seleção embaralhada) | não | não | não | **sim** |

Duas leituras obrigatórias desta tabela:

1. **Exigir "vencer baselines estabelecidos" nos hosts Transformer é exigir
   implementá-los primeiro.** Em 3 GPUs Pascal, para um LLM, isso é outro
   projeto. Não é uma questão de rodar mais seeds.
2. **Os controles mais rigorosos do projeto só existem no Qwen.** `lr_control` e
   `permuted` são o que separa "quanta plasticidade foi removida" de "onde ela
   foi removida", e nenhum host clássico os tem. Isso é assimetria de rigor, e
   precisa ser declarada em vez de escondida.

**Regra editorial:** nenhuma tabela do manuscrito mistura hosts na mesma grade
de baselines. Células vazias seriam lidas como omissão.

---

## 2. Onde o artigo está hoje

O manuscrito tem 604 linhas, doze seções e uma lista de onze afirmações seguras.
O que ele não tem é uma contribuição declarada que sobreviva às próprias
ressalvas:

- **§9 fecha a porta da prioridade.** "No claim of being the first method to use
  neuron importance, MAX masks, lateral inhibition or importance-dependent
  plasticity is justified."
- **§11 fecha a porta da eficácia geral.** "SlowHeat outperforms established
  continual-learning baselines" está listado como **não sustentado**. O único
  contraste confirmado é contra Replay puro em Split-MNIST.
- **§8 derrubou a explicação favorita.** O regime de proteção não transfere: em
  cinco alvos não-Transformer, hard nunca vence soft.

Nada disso mudou. O que mudou é a leitura: essas portas se fecham em **hosts
diferentes**, e cada host tem uma âncora própria que continua de pé.

---

## 3. A decisão transversal que trava todo o resto

Com os hosts separados, as âncoras disponíveis são quatro, não uma:

| Âncora | Host | Viável? |
|---|---|---|
| Prioridade de mecanismo | qualquer | **Não.** §9 já fechou, e os três mecanismos novos do LoRA falharam |
| Eficácia local confirmada | MLP | **Sim**, já existe. Estreita: Replay, Split-MNIST, uma configuração |
| Limite de interação medido | CNN | **Sim**, já existe. É achado negativo/condicional |
| **Distribuição importa, sob pré-registro** | Qwen/LoRA | **Sim, confirmado em 28/09** (`p = 0,02148`), com uma explicação alternativa aberta (capacidade do adaptador, QB-2) |
| **Protocolo de avaliação + poder de rejeição** | Qwen | **Sim**, e é a única que organiza todas as outras |

### Recomendação: ancorar no protocolo, usar os hosts como demonstração

A tese defensável:

> *Avaliações de proteção seletiva confundem rotineiramente a **quantidade** de
> plasticidade removida com a **distribuição** dela. Propomos um protocolo que
> separa as duas — pareamento por plasticidade efetiva medida, mais um controle
> de learning rate na mesma plasticidade — e mostramos que ele muda conclusões
> em duas direções: efeitos aparentes desaparecem, e efeitos reais crescem.*

Os quatro hosts então têm papéis distintos e não competem:

- **MLP** mostra que o mecanismo faz alguma coisa, sob pré-registro e replicação.
- **CNN** mostra que o efeito não tem sinal fixo, o que motiva o protocolo.
- **BERT** mostra que a confusão quantidade/distribuição produziu uma conclusão
  otimista (o +22,92 p.p. atribuído ao regime hard) que depois foi refutada.
- **Qwen** constrói o protocolo e o exercita até ele rejeitar três mecanismos
  novos e uma hipótese central.

Evidência transversal que sustenta a tese:

- **A1**, o teorema `E >= (N−P)/N`, dá o piso analítico que torna o pareamento
  bem-definido e diz quando um alvo é inalcançável.
- **A2**, a construção iso-plasticidade por bisseção, é o instrumento.
- **A3**, o contrato de otimizador, explica por que mascarar gradiente bruto não
  é mascarar o passo — pré-requisito para o pareamento significar algo.
- Os **três mecanismos falhados** do LoRA demonstram poder de rejeição.
- O **crescimento do efeito do `exact`** sob controle mais rigoroso demonstra que
  o protocolo não é apenas conservador.
- O **`lr_control` pior que vanilla** justifica a existência do controle.
- A **morte de `iso − permutado` entre 2 e 10 tarefas** demonstra que conclusões
  de sequências curtas não transferem.

Isso é um artigo de metodologia com resultados empíricos, não um artigo de
método novo. É menor em ambição e **inteiramente sustentado** pelo que já foi
medido.

**Esta é a decisão que nenhum experimento resolve e que só o Fachel fecha.**

---

## 4. O único bloqueador transversal

### Levantamento de literatura sobre protocolos de pareamento — **RESOLVIDO em 28/09/2026**

**Veredito: a âncora da §3 se mantém.** Registro completo em
[../docs/related_work/protocol_prior_art.md](../docs/related_work/protocol_prior_art.md).

A pergunta existencial era: *alguém já pareia braços por plasticidade efetiva
medida?* Depois de 21 consultas, a resposta é **não**. O padrão da literatura é
consistente e diferente: plasticidade aparece como fenômeno estudado (Lyle 2023,
Dohare 2024) ou alvo a preservar; learning rate aparece como fator causal
investigado (Mirzadeh 2020) ou como componente de método proposto (SLCA);
protocolos de avaliação padronizam *cenário* (Hsu 2018, van de Ven 2019), não
plasticidade.

O antecedente mais próximo **fortalece** o artigo: Mirzadeh et al. (NeurIPS 2020)
mostrou que o regime de treino confunde a medida de esquecimento, e mesmo assim o
pareamento não virou prática. É exatamente a lacuna que o protocolo preenche.

**Três citações passaram a ser obrigatórias:**

| Citação | Onde | Por quê |
|---|---|---|
| Mirzadeh et al. 2020 | §1 e §9 | antecedente do confundidor; sustenta "o problema é conhecido e não é controlado" |
| SLCA / SLCA++ | §9 | quase-hit: reduz LR seletivamente, mas como **método**, não como **controle**. A distinção precisa ser explícita |
| O-LoRA, InfLoRA | onde `rank`, `leak`, `slice` forem reportados | mecanismos vizinhos aos três que falharam |

### Achado colateral que muda a redação do Qwen

**LoRA-FA (arXiv 2308.03303, 2023) já congela `A` e treina só `B`** — o mecanismo
do braço `exact`, com motivação diferente (memória de ativação, não esquecimento).
O artigo **não pode** apresentar isso como mecanismo próprio; precisa citar LoRA-FA
e LoRI e posicionar-se como *avaliação sob protocolo controlado*.

Isso **não derruba** o resultado confirmado de 28/09: a contribuição é a comparação
`exact − lr_control` sob plasticidade pareada, que LoRA-FA não faz.

Mas **afeta QB-2**: LoRA-FA argumenta que congelar um fator não sacrifica
expressividade. Se isso se sustentar, a explicação alternativa que QB-2 existe para
eliminar já tem resposta teórica publicada. Detalhes e a ação exigida em
[proximo_passo_qwen.md](proximo_passo_qwen.md), QB-2.

---

## 5. Regras editoriais que valem para os quatro hosts

- **Nunca reportar forgetting sem a acurácia final ao lado.** Na CNN, hard reduz
  forgetting em 8,34 p.p. e ainda perde 2,97 p.p. de acurácia; citar só
  forgetting inverteria a conclusão. O mesmo padrão aparece em CIFAR-100/MLP.
- **Nunca reanalisar seeds já executadas com uma família de correção diferente.**
  Escolher a correção depois de ver o resultado é exatamente o que ela existe
  para impedir.
- **Nunca promover endpoint secundário a primário depois de visto.**
- **Nunca citar contrastes no piso do acaso como resultado** (Split-MNIST
  vanilla, 19,64% contra 19,52% num piso de 20%).
- **Nunca citar ganho sobre fine-tuning sequencial sem regularização como
  competitivo** (Permuted-MNIST, +7,67 p.p.).
- **Nunca apagar resultado negativo.** Sob a âncora de protocolo eles deixam de
  ser fracasso e viram evidência de poder de rejeição.
- **Sempre declarar quando uma análise é exploratória por construção** (dados que
  já existiam em disco antes da pergunta).

---

## 6. Ordem de execução sugerida

**Atualizado em 28/09:** a confirmação do LoRA fechou (10/10 seeds,
**confirmado**, `p = 0,02148`) e as 3 GPUs estão livres. Isso muda a ordem: o
artigo passou a ter um resultado confirmatório em Transformer, e a única
explicação alternativa viva para ele é capacidade do adaptador — o que promove
QB-2 a prioridade máxima de GPU.

| # | Ação | Host | Custo | Estado |
|---|---|---|---|---|
| 1 | Levantamento de protocolos de pareamento | **transversal, bloqueador** | CPU, meio dia | ✅ **concluído 28/09** — âncora mantida |
| 2 | Rodar `exact r=26` vs `vanilla r=16` | Qwen QB-2 | **~2h GPU** | pré-registro congelado 28/09; **run pendente de autorização** |
| 3 | Registrar o desvio do braço `hard` na tabela K | Qwen QA-6 | texto | ✅ **concluído 28/09** |
| 4 | Analisar os agregados de CNN já em disco (50 seeds) | CNN C-3 | CPU, horas | ✅ **concluído 28/09** |
| 5 | Escrever `qwen_layer_anomaly.md` | Qwen QA-2 | CPU, horas | pendente |
| 6 | Pré-registrar e confirmar SlowHeat+DER++ vs DER++ | MLP M1 | CPU, horas | pré-registro congelado 28/09; **run pendente de autorização** |
| 7 | Reescrever §9, §10, §11 e depois §1 | manuscrito | CPU, dia | pendente — **desbloqueado** pelo item 1 |
| 8 | Escolher veículo | — | — | pendente |

Itens 1, 3, 4, 5 e 6 não usam GPU e podem correr em paralelo com o item 2.

### O que mudou em 28/09

O bloqueador caiu: o levantamento
([../docs/related_work/protocol_prior_art.md](../docs/related_work/protocol_prior_art.md))
confirmou que ninguém pareia braços por plasticidade efetiva medida, e o
antecedente mais próximo (Mirzadeh et al., NeurIPS 2020) **fortalece** a premissa.
A redação do manuscrito está desbloqueada.

Três achados colaterais, todos já registrados nos documentos de host:

1. **LoRA-FA já faz o que o braço `exact` faz.** O artigo precisa citá-lo e
   reposicionar `exact` como avaliação, não como proposta. Pode encolher QB-2 —
   ver `proximo_passo_qwen.md`.
2. **O desvio do braço `hard` é pior do que estava registrado**: alcança a run de
   120 passos, não só as obsoletas. A parte A do Qwen fica **sem resultado
   confirmatório**.
3. **O sweep de 50 seeds mostra um segundo confundidor**: o sinal do contraste
   inverte conforme o seletor de memória, dentro do mesmo dataset e método base.
   Reforça a tese de protocolo por um caminho independente do learning rate.

### Concluído

| Ação | Host | Resultado |
|---|---|---|
| Fechar a confirmação de LoRA | Qwen QB-1 | **confirmado** em 28/09: `exact − lr_control` em forgetting, `p = 0,02148 < 0,025`, mediana −0,0998, 9/10 seeds. [Registro](../docs/lora/lora_confirmation_results.md) |
| Remover 98,78 GB de checkpoints órfãos | CNN | `cache_derpp_10seeds` de 100 GB para 959 MB; 25 agregados de 50 seeds preservados e verificados |

---

## 7. Reescrita do manuscrito (passo 8 acima)

Só depois dos itens 1 a 7.

- **§7** ganha a ressalva de `hard_vs_soft` dentro dela, não só na §8.
- **§9** ganha a família LoRA-CL e o posicionamento do protocolo.
- **§10** encolhe drasticamente. A lista atual de dez itens descreve um artigo de
  eficácia que não vai existir; sob a âncora de protocolo a maior parte deixa de
  ser pré-requisito de submissão e vira trabalho futuro. **E precisa ser dividida
  por host**, porque os itens 1 e 4 (tuning e baselines) só se aplicam a MLP/CNN.
- **§11** ganha as afirmações de LoRA, incluindo as negativas, e perde qualquer
  resquício de ambição de eficácia geral.
- **§1** é reescrita por último, quando os números estiverem congelados.

## 8. Veículo

Sob a âncora de protocolo, workshops de metodologia e reprodutibilidade em ML
são alvo mais adequado que trilha principal. Isso não é rebaixamento: é o lugar
onde "seu controle está errado e aqui está a medida disso" é contribuição, e não
fraqueza.

## Referências

- [Opções de novidade](opcoes_novidade_e_proximos_passos.md)
- [Manuscrito](../article/manuscript.md)
- [Índice de resultados versionados](../docs/results/results_index.md)
- [Status de proveniência](../docs/audits/results_provenance_status.md)
- [Versão anterior, monolítica](_arquivo_proximo_passo_artigo_2026-09-25.md)
