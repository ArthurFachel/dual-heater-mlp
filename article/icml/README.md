# ICML 2027 — Plasticity-Matched Controls

**Título:** *Plasticity-Matched Controls: Re-evaluating CL Methods Against What
They Actually Constrain*

**Claim:** comparações em CL não controlam quanta plasticidade o método removeu.
Damos o instrumento, ele muda conclusões — inclusive as nossas, pré-registradas
e já confirmadas.

**Deadline:** ~22/01/2027 (**estimado** — CFP não publicada, rechecar
`goals/venue_deadlines.md`). 8 páginas + referências.

---

## Regra de não-sobreposição (do roadmap, inegociável)

R-A, R-B e R-D são **corpo do IJCNN** e entram aqui **só como linha de tabela**
citando o companion. R-C, R-E e o instrumento de plasticidade são corpo do ICML
e **não** aparecem no IJCNN. Citação cruzada: "companion paper, under review".

## Estrutura

| seção | pgs | conteúdo | fonte |
|---|---:|---|---|
| 1 Introduction | 1,0 | o confundimento de plasticidade | — |
| 2 Effective plasticity | 1,5 | definição, lema `E_sup ≤ E_bind`, generalização para penalidade | `docs/mechanisms/`, `src/dual_heater/plasticity.py` |
| 3 Case study: nosso próprio resultado | 1,5 | R-E + pareamento. **O caso central.** | `plasticity_matched_results.md`, `lora_confirmation_results.md` §R3 |
| 4 Re-evaluation | 2,0 | Fase 2: amplificação, G8, geometria por host | `penalty_pass1_*.md`, `sgd_plasticity_results.md`, `lambda_sweep_results.md`, `penalty_pass2_results.md` |
| 5 Negative results | 1,0 | R-C, R-F. R-A/R-B só como citação ao companion | `exact_decomposition_results.md` |
| 6 Limitations | 0,5 | não-unicidade, hosts, benchmarks | — |
| 7 Related Work | 0,5 | — | — |

## Ordem de escrita

**Seção 3 primeiro** (item 5.1 do roadmap): é o coração, e é a única que só
pode ser escrita por quem viveu o episódio.

## Material salvo do manuscrito anterior

`salvaged_from_manuscript.md` carrega as duas seções que o item 5.2 identificou
como aproveitáveis diretamente. O manuscrito original está em
`article/_archive/manuscript_functional_slowheat.md`, **não apagado**.

## Arquivos

```
article/icml/
  README.md                    este arquivo
  salvaged_from_manuscript.md  seções 3 e 4.4 do manuscrito anterior
  manuscript.md                (a escrever)
```
