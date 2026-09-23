# Status de proveniência: o que precisa ser re-executado antes da submissão

Auditoria de 22 de setembro de 2026. Este documento existe porque **96% dos
agregados versionados foram produzidos com árvore Git suja**, e essa é a maior
ameaça isolada a uma submissão.

| Medida | Valor |
|---|---:|
| Agregados em disco | 93 |
| Com `git dirty = true` | **89 (96%)** |
| Com pré-registro congelado | **2** (a mesma confirmação, executada duas vezes) |

"Árvore suja" significa que o commit registrado no `environment.json` não
descreve o código efetivamente executado, e o diff não foi fingerprintado.
A consequência prática: **não é possível reproduzir o resultado bit a bit a
partir da proveniência**, apenas aproximá-lo.

---

## Prioridades

### P1 — Confirmação Split-MNIST (crítica, resultado principal do artigo)

**Artefatos:** `results/protocol_post_eval_fix/confirmation/` e
`results/protocol_post_eval_fix_d5b22ad/confirmation/`.

**Situação:** ambas com árvore suja. É o resultado que sustenta o claim central
do artigo (+0,87 p.p., `p = 0,0052`).

**Atenuante forte:** as duas execuções são independentes, em commits
diferentes (`d5b22ad` e `f2f7616`), e reproduziram **todas** as métricas
científicas — apenas 21 de 100 campos escalares diferem, e todos são de custo.
Uma discrepância de código relevante entre as duas árvores sujas teria que ser
cientificamente neutra nos dois casos, o que é improvável.

**Ação recomendada:** re-executar as 20 seeds com árvore limpa e commit
fingerprintado. É a única re-execução realmente necessária antes da submissão.
Custo: 20 seeds × 2 métodos, poucos minutos por seed em CPU.

**Se não for re-executado:** declarar explicitamente a limitação na seção de
reprodutibilidade e citar a replicação independente como mitigação parcial.
Um revisor rigoroso ainda pode questionar.

### P2 — Contrastes pareados de 10 seeds (alta, evidência de generalidade)

**Artefatos:** `results/dualheat_pairs/{split_mnist,permuted_mnist,split_cifar10,split_cifar100}/`,
commit `be05068`, árvore suja.

**Situação:** sustentam os achados de interação com o método base — incluindo a
inversão de sinal entre ER-ACE (+4,15 p.p.) e DER++ (−1,13 p.p.) em CIFAR-10,
que é o achado mais interessante do projeto fora do MLP.

**Ação recomendada:** re-executar com árvore limpa se o achado de interação
entrar no artigo como contribuição. Se entrar apenas como observação
exploratória com ressalva, pode seguir.

### P3 — Manifesto do piloto FastHeat (média, defeito de desenho)

**Artefato:** `results/split_mnist_protocol/functional_dualheat_pilot/selected_fastheat_config.json`.

**Situação:** o loader valida schema, status e pertinência à grade, mas **não
recomputa o vencedor a partir de um hash imutável do piloto**. Isso não é
problema de árvore suja — é uma lacuna no mecanismo de pré-registro.

**Ação recomendada:** não alegar pré-registro para a seleção de FastHeat.
Descrever como seleção em validação, que é o que de fato é.

### P4 — Diagnósticos BERT (média, resultado negativo)

**Artefatos:** `results/bert_slowheat_review/` e os dois de replay, commit-base
`5e96093`, árvore suja.

**Situação:** sustentam o achado negativo (SlowHeat perde para replay).

**Atenuante:** resultados negativos são menos sensíveis a questionamento de
proveniência — a alegação é de que o método **não** ajudou, e ruído de
implementação favoreceria o contrário.

**Ação recomendada:** documentar a ressalva; re-execução é opcional.
**Atenção adicional:** o estudo de replay com memória 20 usa fingerprint de
fonte diferente (`ef17c0…`) e não pode ser agregado com os outros dois.

### P5 — Qwen (não aplicável ainda)

Nenhum resultado citável. As runs existentes usam 30 passos por tarefa, valor
revogado pelo próprio protocolo congelado. Qualquer execução futura deve
começar com árvore limpa — é a oportunidade de fazer certo desde o início.

---

## Resultados que NÃO devem ser citados em hipótese alguma

| Artefato | Motivo |
|---|---|
| `results/split_mniist_results.csv` e os outros CSVs soltos | sem seeds, sem commit, sem configuração |
| Seções 7–10 de `project_methods_and_results.md` | nenhum artefato em disco reproduz esses números |
| Seção 6 de `split_mnist_experiment_log.md` | os três valores divergem do único agregado de 20 seeds |
| `results/bert_10epoch/` | mistura duas sessões com metadados sobrescritos |
| Runs Qwen de 30 passos | valor revogado pela tabela K |
| Piloto sintético de 3 seeds | precede a implementação atual |

---

## Checklist antes de submeter

- [ ] P1 re-executado com árvore limpa, **ou** limitação declarada com a
      replicação independente citada como mitigação
- [ ] P2 decidido: re-executar ou rebaixar a observação exploratória
- [ ] P3: remover qualquer alegação de pré-registro para FastHeat
- [ ] Seção de reprodutibilidade declara o status de árvore suja honestamente
- [ ] Nenhum número da lista "não devem ser citados" aparece no manuscrito
- [ ] Cada tabela do artigo aponta para um artefato com commit registrado
- [ ] Tuning de learning rate por método executado ou declarado como ameaça
      (hoje todos os agregados usam `lr=1e-3` fixo)

---

## Como evitar reincidência

O runner já grava `environment.json` com commit e flag `dirty`. O que falta é
**recusar-se a executar** quando a árvore está suja em modo confirmatório, ou
no mínimo gravar o diff completo junto ao manifesto. Enquanto isso não existir,
toda execução futura deve ser precedida de `git status --porcelain` vazio.