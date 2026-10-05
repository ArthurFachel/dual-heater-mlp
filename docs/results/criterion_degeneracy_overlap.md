# Falsificador da §F recuperado — sobreposição top-k entre critérios

**Pré-registro:** `goals/protocol_importance_criterion_ablation.md` §F
(congelado em `a3f5117`, antes de qualquer seed).
**Run analisada:** `results/criterion_ablation/`, 10 seeds, 28/09/2026.
**Análise:** 05/10/2026, CPU, a partir dos checkpoints salvos.
**Artefato:** `results/criterion_ablation/degeneracy_overlap.json`.

---

## O que faltava

A §F do pré-registro declara **duas** diagnósticas obrigatórias por braço:

1. a variância do ranking normalizado;
2. a **sobreposição do conjunto top-k protegido** entre `magnitude` e
   `aleatório`, e entre `magnitude` e `funcional` (Jaccard).

E uma regra de interpretação congelada: se a sobreposição entre `magnitude` e
`aleatório` for **> 0,8**, o braço `magnitude` é declarado *degenerado* e
qualquer empate é reportado como **inconclusivo**, não como evidência.

**A run de 28/09 emitiu só a primeira.** O resultado publicado
(`criterion_ablation_results.md`) reporta a variância e conclui que "o empate é
real" a partir dela. A segunda métrica nunca foi calculada, e com ela a regra
congelada da §F nunca foi avaliada.

## Por que ela não saiu

Não foi esquecimento de reportar: foi uma impossibilidade estrutural do ponto
de medição. `run_split_clinc150` chama `aggregate_ranking_degeneracy` **sem
`reference`** (`experiments/split_clinc150.py:1476`), e sem um braço de
referência a função retorna apenas a variância — `top_k_overlap` sequer entra no
dicionário.

A causa é que a sobreposição é uma grandeza **entre braços**: ela compara os
conjuntos protegidos de dois runs separados. Um hook que roda dentro de um
único run não tem acesso ao outro. A implementação de `ranking_degeneracy_metrics`
aceita `reference` e o teste `tests/test_importance_criterion.py:192` exercita
esse caminho — mas **nenhum chamador de produção jamais o usou**.

Isso é uma lacuna de wiring, da mesma família do bug `importance_criterion`
registrado na §"Bugs corrigidos" do resultado de 28/09: o código existia, o
teste existia, e o caminho de produção não passava por ele.

## Como foi recuperada

Os checkpoints de `results/criterion_ablation/` guardam os buffers `slow_heat`
de cada tracker. `slow_heat` **é** o vetor que a máscara de plasticidade lê
(`src/dual_heater/bert.py:180`), então o seu suporte é o conjunto protegido que
o run de fato usou — não uma reconstrução dele.

`experiments/criterion_degeneracy.py` lê esses buffers e calcula o Jaccard por
tracker, com média **sobre trackers, não sobre unidades**: a seleção top-k opera
dentro de cada estado, então uma média ponderada por unidade esconderia um
tracker estreito degenerado atrás de um FFN largo saudável.

```bash
PYTHONPATH=src:. python scripts/analyze_criterion_degeneracy.py
```

## Resultado

| par | média | mínimo | máximo |
|---|---:|---:|---:|
| `magnitude` vs `aleatório` | **0,5748** | 0,5481 | 0,6731 |
| `magnitude` vs `funcional` | 0,8814 | 0,8758 | 0,8851 |

Por seed, `magnitude` vs `aleatório`:

| seed | Jaccard |
|---|---:|
| 4000003 | 0,5516 |
| 4025011 | 0,5483 |
| 4050017 | 0,5493 |
| 4075037 | 0,6731 |
| 4100043 | 0,5516 |
| 4125059 | 0,6100 |
| 4150061 | 0,5481 |
| 4175087 | 0,6135 |
| 4200089 | 0,5533 |
| 4225097 | 0,5493 |

Capacidade pareada entre braços verificada em **10/10 seeds**.

## Veredito

**0,5748 < 0,80. O falsificador da §F passa.**

O braço `magnitude` protegeu um conjunto próprio de unidades, não uma máscara
quase-aleatória com outro nome. A regra congelada **não** dispara, o empate de
R-B não é declarado inconclusivo, e as duas diagnósticas que a §F exigia agora
existem nos artefatos.

Isso **não é um achado novo** — é a conclusão já publicada, agora com a segunda
metade da evidência que o próprio pré-registro havia exigido. O que muda é que
antes a defesa de R-B apoiava-se só na variância; agora apoia-se nas duas
métricas declaradas.

## O que o 0,8814 significa, e o que não significa

`magnitude` e `funcional` selecionam **88,1% das mesmas unidades**. Isso é
consistente com o empate de −0,17 pp em retenção: os dois critérios protegem
quase o mesmo conjunto, então é esperado que produzam quase o mesmo resultado.

**Não é um segundo achado independente.** É a mesma observação medida na máscara
em vez de na acurácia, e tratá-la como confirmação adicional seria contar a
mesma evidência duas vezes.

O que ela acrescenta é mecanismo: o empate não vem de "os dois critérios são
igualmente ruins", vem de "os dois critérios concordam sobre quais unidades
importam". O componente do gradiente reordena a cauda, não o topo.

## Limites

- **Lido de checkpoints, não emitido pelo run.** O número é correto porque
  `slow_heat` é o estado que a máscara consumiu, mas a medição é post-hoc em
  relação à execução. Runs futuras devem emitir a métrica diretamente.
- **O estado lido é o do fim da sequência**, após a última consolidação. A §F
  pedia a métrica "por braço e por estágio"; com `T = 2` há uma consolidação
  intermediária cujo estado o checkpoint final não preserva.
- **Nada aqui toca os endpoints.** Nenhuma acurácia foi recalculada, nenhum
  contraste foi refeito. O resultado de 28/09 permanece exatamente como está.

## Consequência

`criterion_ablation_results.md` continua válido. A §F do seu pré-registro, que
estava cumprida pela metade, está cumprida inteira.

Para a extensão de sequência longa
(`goals/protocol_long_sequence_criterion.md`), a métrica passa a ser obrigatória
**antes** de olhar qualquer endpoint, via
`scripts/verify_long_sequence_criterion.py` (checagem V6).
