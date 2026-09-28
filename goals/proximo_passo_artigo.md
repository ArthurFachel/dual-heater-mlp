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

### Levantamento de literatura sobre protocolos de pareamento (~meio dia, sem GPU)

A pergunta existencial: **alguém já pareia braços por plasticidade efetiva
medida?** Se sim, a âncora da §3 cai e este documento precisa ser reescrito
antes de qualquer redação.

A busca tem que ser explícita e com termos variados, porque a literatura de CL
raramente usa a palavra "plasticidade": *iso-plasticity*, *matched-capacity
control*, *effective step-size matching*, *gradient-mask budget*,
*capacity-controlled ablation*.

Esta busca bloqueia os quatro hosts. A busca da família LoRA-CL (O-LoRA,
InfLoRA, CorDA, MoRAL, SAPT) **não** é transversal: afeta só os resultados
negativos do Qwen e está em [proximo_passo_qwen.md](proximo_passo_qwen.md),
QB-3.

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

| # | Ação | Host | Custo |
|---|---|---|---|
| 1 | Levantamento de protocolos de pareamento | **transversal, bloqueador** | CPU, meio dia |
| 2 | Pré-registrar e rodar `exact r=32` vs `vanilla r=16` | Qwen QB-2 | **~2h GPU** |
| 3 | Registrar o desvio do braço `hard` na tabela K | Qwen QA-6 | texto |
| 4 | Analisar os agregados de CNN já em disco (50 seeds) | CNN C-3 | CPU, horas |
| 5 | Escrever `qwen_layer_anomaly.md` | Qwen QA-2 | CPU, horas |
| 6 | Pré-registrar e confirmar SlowHeat+DER++ vs DER++ | MLP M1 | CPU, horas |
| 7 | Reescrever §9, §10, §11 e depois §1 | manuscrito | CPU, dia |
| 8 | Escolher veículo | — | — |

Itens 1, 3, 4, 5 e 6 não usam GPU e podem correr em paralelo com o item 2.

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
