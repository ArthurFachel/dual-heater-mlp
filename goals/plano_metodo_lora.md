# Plano — desenvolver o método em LoRA

> Escrito em 28/09/2026, durante a run de confirmação do seletor.
> Documento de decisão, não de execução. Nenhuma seed foi gasta com base nele.

---

## 0. O achado que reordena tudo

Uma busca de prioridade antes de planejar encontrou **LoRA-Null**
([arXiv 2503.02659](https://arxiv.org/abs/2503.02659), 2025):

> *"if we freeze the values of the down-projection matrices during fine-tuning,
> it achieves even better preservation of the pre-trained world knowledge"*

Congelar `A` é exatamente o braço `exact`. E agora temos **duas** publicações
independentes do mesmo mecanismo:

| trabalho | mecanismo | motivação | colide com |
|---|---|---|---|
| LoRA-FA (2023) | congela `A`, treina `B` | memória | `exact` |
| **LoRA-Null (2025)** | congela `A` (+ init no espaço nulo) | **preservação de conhecimento / esquecimento** | **`exact`, e na mesma motivação** |

LoRA-FA era defensável como "mecanismo igual, motivação diferente". LoRA-Null
elimina essa defesa: mesmo mecanismo, **mesma motivação**, e publicado antes.

**Consequência dura: o braço `exact` não pode ser apresentado como método novo.**
O resultado confirmado de 28/09 (`exact − lr_control`, p = 0,0215) continua
válido e continua interessante — mas o que ele mede é *uma técnica publicada
avaliada sob um protocolo novo*, não *um método nosso que funciona*.

## 1. O espaço de mecanismos está saturado

Levantamento de 28/09 mais o desta sessão:

| trabalho | ideia |
|---|---|
| O-LoRA | subespaços ortogonais por tarefa |
| InfLoRA | atualização livre de interferência |
| LoRA-FA | congela `A` (memória) |
| LoRA-Null | congela `A` + init no espaço nulo (conhecimento) |
| OPLoRA | projeção ortogonal contra esquecimento |
| LoDA (ICML 2026) | decomposição de subespaço dirigida por tarefa |
| SHARE (2026) | subespaço compartilhado que cresce |
| LoRAC-IPC | composição com restrição em parâmetros críticos |

Há trabalho de 2026 em ICML e ECCV nessa linha, com rank adaptativo por tarefa
e camada. **Propor "mais um mecanismo de subespaço LoRA" é entrar na parte mais
concorrida e mais bem financiada da área, com três GPUs Pascal e um modelo de
0,5B.** Não é uma aposta sensata.

## 2. O que os nossos próprios dados já dizem sobre novos mecanismos

Três mecanismos foram construídos e medidos (`docs/lora/lora_slowheat_mechanisms.md`):

| mecanismo | FAA vs `lr_control` | p | FAA absoluta |
|---|---:|---:|---:|
| `slice` | +0,0191 | 0,754 | 0,5875 |
| `leak` | +0,0181 | 0,754 | 0,5865 |
| `rank` | +0,0124 | 0,109 | 0,5808 |
| — `vanilla` (referência) | — | — | **0,6016** |
| — `exact` (referência) | — | — | **0,6494** |

**Os três ficam abaixo do `vanilla`.** Nenhum é significativo. O `leak` chegou a
entregar o que o módulo original prometia (proteção exata por saída com `A`
treinável, com teste dedicado) e ainda assim perdeu.

O padrão é consistente: **o que funciona é a restrição dura e global (congelar
`A`); toda tentativa de proteção seletiva e fina fica abaixo de não fazer nada.**
Isso é um resultado sobre a geometria do LoRA, e é o quarto mecanismo falhado
que um quinto mecanismo teria que explicar antes de ser proposto.

## 3. Três caminhos, avaliados sem otimismo

### Caminho A — propor mais um mecanismo SlowHeat-em-LoRA

**Custo:** semanas de GPU. **Risco:** altíssimo.
Quatro mecanismos já falharam (`rank`, `leak`, `slice`, e o hook original).
O espaço está saturado por grupos maiores. Não há hipótese na mesa que explique
por que um quinto teria sucesso.

**Veredito: não recomendado.** Não existe razão técnica para esperar sucesso.

### Caminho B — `exact` como estudo de caso sob o protocolo

**Custo:** zero de GPU nova (o resultado já existe). **Risco:** baixo.
Reposicionar de "nosso método" para "técnica publicada (LoRA-FA/LoRA-Null)
avaliada sob iso-plasticidade com controle de LR". A contribuição é o protocolo
e o que ele revela: sob pareamento, o efeito do `exact` **cresce**
(−0,0629 → −0,0807 em forgetting), enquanto três alternativas finas somem.

**Veredito: é o que já está em curso, e é o uso honesto do material.**
Requer citar LoRA-FA e LoRA-Null com destaque, não em nota de rodapé.

### Caminho C — a pergunta que o nosso material responde e o campo não fez

Nenhum dos oito trabalhos da §1 controla plasticidade ao comparar. Todos
comparam mecanismo contra mecanismo com o mesmo LR nominal. A nossa medição
mostra que isso é um confundidor real: **reduzir LR uniformemente em 15% muda
FAA de forma comparável ao que os mecanismos alegam** (o `lr_control` é o braço
que derruba os três mecanismos finos).

A pergunta: **quanto dos ganhos publicados em LoRA-CL sobrevive ao pareamento
por plasticidade?**

**Custo:** médio (rodar O-LoRA / LoRA-Null / `exact` sob iso-`E` no nosso host).
**Risco:** médio, mas o resultado é publicável nas duas direções.
**Alinhamento:** é exatamente a âncora de protocolo do paper de ICML, aplicada à
família LoRA em vez de à família EWC/SI/MAS.

**Veredito: é o único caminho que gera contribuição nova em LoRA sem competir
em mecanismo.**

## 4. Recomendação

**Abandonar o desenvolvimento de novos mecanismos LoRA.** Seguir B + C:
o `exact` vira estudo de caso creditado, e o esforço novo vai para a
re-avaliação da família LoRA-CL sob plasticidade pareada.

Isso preserva todo o material medido, evita a competição perdida, e alimenta a
mesma âncora do artigo de ICML.

## 5. Se ainda assim quisermos um mecanismo: o pré-registro mínimo

Caso a decisão seja tentar, a regra da casa exige declarar antes:

1. **Hipótese explícita de por que este mecanismo difere dos quatro que
   falharam.** Sem isso, é tentativa e erro com nome de pesquisa.
2. **Falsificador declarado antes da run**, no padrão do `leak`
   (`test_leak_bound_collapses_once_b_is_dense`).
3. **Critério de parada:** se o mecanismo ficar abaixo do `vanilla` em FAA sob
   iso-`E`, a linha encerra. Não há segunda rodada de tuning.
4. **Braços obrigatórios:** `vanilla`, `lr_control`, `exact` e o novo, todos
   pareados em `E`. Comparar só contra `vanilla` é o erro que o `slice` expôs.
5. **Custo declarado antes**, e um resultado nulo publicado como tal.

## 6. O que fazer agora

| # | Ação | Custo |
|---|---|---|
| 1 | Citar LoRA-FA e LoRA-Null em `protocol_lora_confirmation.md` e no manuscrito; rebaixar `exact` de "método" para "técnica avaliada" | texto |
| 2 | Registrar em `docs/related_work/protocol_prior_art.md` que a prioridade 2 tem agora dois hits diretos | texto |
| 3 | Desenhar a re-avaliação da família LoRA-CL sob iso-`E` (Caminho C) | meio dia |
| 4 | Não iniciar mecanismo novo | — |
