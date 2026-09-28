# Próximo passo — BERT (CLINC150, full fine-tuning)

> Um de quatro documentos por arquitetura. Índice e decisões transversais em
> [proximo_passo_artigo.md](proximo_passo_artigo.md).
>
> Escopo: `SlowHeatBertForSequenceClassification` sobre
> `google/bert_uncased_L-4_H-256_A-4`. Fonte de evidência:
> [docs/architectures/arch_bert.md](../docs/architectures/arch_bert.md).

---

## 1. O que este host sustenta hoje

| Evidência | Resultado | Seeds |
|---|---|---|
| **B1** | hard aprendido supera fine-tuning sequencial em **+22,92 p.p.** (71,38% contra 48,47%) | 10+/0− |
| **B2** | ranking aprendido supera máscara aleatória pareada em **+9,37 p.p.** de retenção T1 | 10+/0− na retenção, 7/10 na acurácia |
| **B3** | contra replay (20 ex/classe), hard+replay **perde 0,83 p.p.** e custa 2,21× o tempo | 0+/3− |
| **B4** | gate de eficiência declarado antes da run: 3 de 4 critérios passaram, o de aquisição **reprovou** (−3,67 p.p. contra limite de −2,0) | 3 seeds |

Drift dos parâmetros protegidos é **exatamente zero**, o que valida a máscara
mecanicamente e é citável como verificação de implementação.

## 2. Os três limites que definem o host

**Duas tarefas não são continual learning.** Todo o B1–B4 usa os dois primeiros
domínios do CLINC150, somente validação. É diagnóstico de mecanismo.

**O +22,92 p.p. perdeu sua explicação.** A suíte `hard_vs_soft` testou os dois
regimes em 5 alvos não-Transformer sob protocolo congelado e árvore limpa: hard
nunca vence soft, e perde em CIFAR-100/MLP (−1,41 p.p., 10/10). Logo o ganho do
BERT **não é propriedade do congelamento binário**; é específico da arquitetura
ou do seu regime de capacidade, e o artigo tem que dizer isso.

**Contra uma baseline forte o método perde.** B3 é um resultado negativo
deliberadamente reportado, e B4 é um gate declarado antes da run que de fato
barrou a promoção de um resultado. Ambos são ativos metodológicos.

---

## 3. A âncora que este host sustenta

> *A integração do mecanismo em Transformer é funcional e o ranking de
> importância aprendido carrega sinal de retenção verificável, mas o ganho não
> sobrevive ao contraste contra replay, e o regime de proteção não explica o
> ganho observado.*

É uma âncora de **funcionalidade e limite**, não de eficácia. O host prova que o
mecanismo porta para Transformer e mede onde ele para de funcionar.

**Importante:** aqui não existe, nem poderia existir hoje, comparação contra
DER++, ER-ACE, A-GEM, EWC, SI ou LwF. Esses baselines só estão implementados
nas suítes MLP/CNN. O único baseline forte deste host é replay, e contra ele o
método perde. Qualquer exigência de "vencer baselines estabelecidos" neste host
é uma exigência de **implementar os baselines primeiro**, não de rodar mais
seeds.

---

## 4. Caminho crítico

### B-1 — Decidir se o host entra no artigo como está (decisão, sem custo)

Existe uma decisão registrada em `arch_bert.md` que o projeto já tomou:

> não executar SlowHeat+replay na sequência completa de dez tarefas e não abrir
> nova grade de beta, cobertura, memória ou budget para resgatar o mesmo
> endpoint.

Essa decisão continua correta e **não deve ser revista**. Ela significa que o
host está fechado para novos experimentos e entra no artigo com o que tem.

A única pergunta aberta é de posicionamento: B1–B4 entram como seção própria
(estado atual, §7 do manuscrito) ou como parágrafo dentro de uma seção de
limites? Isso depende da âncora global e está no hub.

### B-2 — Corrigir o texto da §7 do manuscrito (CPU, texto)

A §7 atual ainda diz *"This establishes that the Transformer integration is
functional and beneficial relative to sequential fine-tuning"* sem a ressalva de
`hard_vs_soft`, que só aparece na §8. Um leitor que pare na §7 sai com a
impressão errada.

Ação: mover a ressalva para o parágrafo do +22,92 p.p., com ponteiro para a §8.

### B-3 — Não reagrupar os fingerprints divergentes (CPU, verificação)

O estudo de replay com memória 20 usa fingerprint de fonte `ef17c0…` contra
`4bb938…` dos outros dois. Os valores **não podem** ser reagrupados como amostra
única. Verificar que nenhuma tabela do manuscrito faz isso.

### B-4 — Não citar as 11 runs históricas de dez tarefas (CPU, verificação)

Cada método tem uma seed, o teste foi consultado a cada estágio, e
`bert_10epoch` mistura duas sessões com metadados sobrescritos. Verificar que
nenhum número do manuscrito vem delas.

---

## 5. O que NÃO fazer aqui

- **Não escalar para 10 domínios.** A decisão está registrada e é correta:
  resgatar o mesmo endpoint com grade nova é p-hacking com passos extras.
- **Não atribuir o +22,92 p.p. ao regime hard.** Está refutado por 5 alvos.
- **Não comparar este host contra DER++/ER-ACE/EWC.** Não estão implementados
  aqui, e implementá-los para uma seção de diagnóstico de duas tarefas é um
  investimento desproporcional.
- **Não tratar B4 como falha.** Um gate declarado antes da run que reprova é a
  demonstração mais limpa do projeto de que os critérios não são escolhidos
  depois de ver os números.
- **Não promover os orçamentos de 5 e 10 exemplos.** São secundários e não podem
  ser promovidos post hoc.

---

## 6. Resumo

| # | Ação | Custo | Bloqueia |
|---|---|---|---|
| B-1 | Decidir posicionamento do host no artigo | decisão | §7 |
| B-2 | Mover a ressalva de `hard_vs_soft` para dentro da §7 | texto | honestidade da §7 |
| B-3 | Verificar que fingerprints divergentes não foram reagrupados | verificação | integridade |
| B-4 | Verificar que as 11 runs históricas não são citadas | verificação | integridade |

**Nenhuma GPU.** O host está experimentalmente fechado; o que resta é texto e
verificação.

## Referências

- [Evidência e limites do host](../docs/architectures/arch_bert.md)
- [Diagnóstico completo](../docs/results/bert_slowheat_diagnostic_results.md)
- [Ablação de cobertura](../docs/results/bert_full_coverage_ablation.md)
- [Contrato de Transformers](../docs/mechanisms/functional_slowheat_transformers.md)
- [Índice dos próximos passos](proximo_passo_artigo.md)
