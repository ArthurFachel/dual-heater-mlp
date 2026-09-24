# Hard versus soft: resultados

**Estado:** suíte concluída em 23/09/2026; relatório do alvo CNN regenerado em
24/09/2026 (a run travou no passo de análise, não no treino — ver §6).

**Protocolo congelado:** [`hard_vs_soft_protection.md`](hard_vs_soft_protection.md) e
`results/hard_vs_soft/<alvo>/hard_vs_soft_protocol.json`
**Artefatos:** `results/hard_vs_soft/{split_mnist_mlp, permuted_mnist_mlp,
split_cifar10_mlp, split_cifar100_mlp, split_cifar10_cnn}/pair_report.{json,md}`
**Reprodução do relatório:**
`PYTHONPATH=. python -m experiments.hard_vs_soft --datasets <ds> --backbones <bb> --report-only`

10 seeds pareadas por alvo, árvore Git limpa no commit `6f4d12d` (registrado em
`queue.log` antes do primeiro passo). Holm sobre os quatro contrastes de
acurácia dentro de cada alvo.

---

## 1. A pergunta e a expectativa declarada

Regime de proteção estava perfeitamente confundido com arquitetura: Transformers
usam hard (congelamento binário), MLP e CNN usam soft (`1/(1+30h)`). A
expectativa foi registrada em `hard_vs_soft_protocol.json` **antes da execução**:

> O BERT mostrou hard > soft. Se o regime explica o resultado dos Transformers,
> hard deve vencer soft aqui também. Um resultado nulo ou invertido é
> informativo e será reportado.

O resultado é nulo ou invertido. Está sendo reportado.

## 2. Contraste primário: Hard − Soft (acurácia média final Class-IL)

| Alvo | Delta (pp) | IC95% t | p | p_Holm | seeds concordantes |
|---|---:|---|---:|---:|---|
| Split-MNIST / MLP | +0,498 | [−1,76; +2,75] | 0,63 | 1,00 | 5/10 |
| Permuted-MNIST / MLP | +0,875 | [−0,14; +1,89] | 0,084 | 0,167 | 7/10 |
| Split-CIFAR-10 / MLP | +1,114 | [−0,18; +2,41] | 0,084 | 0,167 | 7/10 |
| **Split-CIFAR-100 / MLP** | **−1,407** | [−1,82; −1,00] | 2,9e−05 | **2,9e−05** | **10/10** |
| Split-CIFAR-10 / CNN | −1,157 | [−2,53; +0,22] | 0,089 | 0,089 | 6/10 |

**Em nenhum dos cinco alvos o hard vence o soft de forma significativa.** O único
resultado que sobrevive a Holm é o CIFAR-100/MLP, onde o hard **perde**, com
concordância unânime das 10 seeds.

## 3. Contrastes secundários

| Alvo | Hard+replay − Soft+replay | Hard+replay − Replay | Hard − Vanilla |
|---|---:|---:|---:|
| Split-MNIST / MLP | −0,464 (ns) | +0,694 (ns) | +0,120 (ns) |
| Permuted-MNIST / MLP | **−1,232** (10/10) | −0,105 (ns) | **+16,985** (10/10) |
| Split-CIFAR-10 / MLP | **+2,273** (10/10) | **+2,548** (9/10) | +0,836 (ns) |
| Split-CIFAR-100 / MLP | **−4,010** (10/10) | **−3,191** (10/10) | **−1,825** (10/10) |
| Split-CIFAR-10 / CNN | **−3,531** (10/10) | **−6,222** (10/10) | **−2,969** (10/10) |

Negrito = sobrevive a Holm dentro do alvo.

O contraste 3 (hard+replay − replay) é a replicação do achado negativo do BERT:
**replica em 2 de 5 alvos** (CIFAR-100/MLP, CIFAR-10/CNN), **inverte em 1**
(CIFAR-10/MLP, +2,55 pp) e é nulo em 2. O achado do BERT não é uma regra geral.

## 4. Leitura

**1. O regime não explica o resultado dos Transformers.** A expectativa
declarada previa hard > soft nos hosts não-Transformer. O observado é nulo em
quatro alvos e invertido, com significância, no quinto. A vantagem do hard em
BERT é **específica da arquitetura ou do regime de capacidade**, não uma
propriedade do congelamento binário. Isso é uma afirmação mais forte que a
atual do manuscrito, e barata: não exige nova execução.

**2. O sinal da diferença acompanha a pressão de capacidade, não a arquitetura.**
Hard vence (marginalmente, sem significância) onde há folga — Split-MNIST,
Permuted-MNIST e CIFAR-10 em MLP — e perde onde a capacidade aperta:

| Alvo | backbone | tarefas × classes | exemplos/classe | Hard − Soft |
|---|---|---|---:|---:|
| Split-MNIST | MLP [256,128] | 5 × 2 | 1000 | +0,50 |
| Permuted-MNIST | MLP [512,256] | domain-IL × 10 | 1000 | +0,88 |
| Split-CIFAR-10 | MLP [1024,512] | 5 × 2 | 4000 | +1,11 |
| Split-CIFAR-100 | MLP [1024,512] | 10 × 10 | 400 | **−1,41** |
| Split-CIFAR-10 | CNN [32,64] | 5 × 2 | 4000 | −1,16 |

CIFAR-100 roda no **mesmo** backbone que CIFAR-10/MLP e inverte o sinal: dobra
as fronteiras, quintuplica as classes por tarefa e tem 1/10 dos exemplos. É o
contraste mais informativo da tabela justamente porque o backbone é constante.
Isso confirma o sinal preliminar dos dados contaminados de VGG11 (−2,78 pp) e
conecta com o eixo Qwen: **hard funciona quando sobra capacidade**. É a mesma
leitura que o teorema `E(beta,b) >= (N−P)/N` produz analiticamente — congelar
`P` unidades custa plasticidade proporcional, e o custo só é absorvível se
`N−P` ainda basta para a tarefa nova.

Ressalva: nesses pares, capacidade, número de fronteiras e volume de dados
variam juntos. A tabela ordena os alvos por dificuldade, não isola capacidade.
Isolar exigiria variar largura com tudo o mais fixo — barato em CPU e ainda não
feito.

**3. O confundimento foi removido, mas o par CNN não é iso-arquitetura.** O
contraste Hard − Soft na CNN é limpo (mesmas unidades, mesmo budget, muda só a
dureza). O que não se pode fazer é ler a diferença entre as *linhas* da tabela
como efeito de arquitetura: CIFAR-10/MLP e CIFAR-10/CNN diferem também em
épocas por tarefa (10 vs 5).

**4. Atenção à assimetria acurácia/forgetting.** Na CNN o hard reduz forgetting
(−8,34 pp contra vanilla) e ainda assim **perde** acurácia final (−2,97 pp).
Esquecer menos não ajudou porque a aquisição caiu junto. O mesmo padrão aparece
no CIFAR-100/MLP (−11,35 pp de forgetting, −1,83 pp de acurácia). Qualquer
leitura que cite só o forgetting inverteria a conclusão.

## 5. Limites

- **Exploratório, não confirmatório.** O protocolo foi congelado antes da
  execução e as seeds sorteadas por gerador determinístico, mas a suíte não é
  uma confirmação pré-registrada independente.
- **Sem tuning por método.** Todos os braços usam `lr=1e-3` e defaults fixos.
  Um hard com LR maior poderia compensar a plasticidade removida — é exatamente
  o braço de LR reduzido do eixo Qwen, e aqui ele não existe.
- **Holm é por alvo, não entre alvos.** As cinco famílias de quatro contrastes
  não são corrigidas conjuntamente. A afirmação "hard nunca vence" é descritiva
  sobre os cinco alvos, não um teste conjunto.
- **Acurácia absoluta baixa em CIFAR-10/MLP (≈17%) e CIFAR-100/MLP (≈4-6%).**
  Regime de esquecimento severo; diferenças de 1-2 pp sobre uma base próxima do
  acaso pesam menos do que a significância sugere.

## 6. Nota de proveniência do alvo CNN

As 10 seeds do `split_cifar10_cnn` foram treinadas em 23/09 (18:55-20:50) e o
passo de análise abortou: `summarize_pair_results` recusava qualquer backbone
diferente de MLP. O treino estava completo e íntegro em disco; só faltava o
relatório.

O relatório foi regenerado em 24/09 com `--report-only`, que lê
`seed_*/results.json` e o protocolo congelado sem retreinar e sem tocar em
nenhum artefato de seed. O cabeçalho do relatório agora nomeia a arquitetura
realmente treinada (antes dizia "MLP [1]" para qualquer alvo, o que teria
rotulado a CNN como MLP). `pair_report.json` traz
`analysis_provenance.rebuilt_from_existing_seeds = true` e o fingerprint do
código de análise, separado do `run_identity.json` do treino.

Mudanças: `experiments/dualheat_pairs.py` (parâmetro `allowed_backbones`,
cabeçalho por arquitetura), `experiments/hard_vs_soft.py` (`--report-only`),
`tests/test_hard_vs_soft.py` (3 testes novos).

## 7. O que isso muda nos outros documentos

- `docs/arch_cnn.md` e `docs/arch_mlp.md`: passam a ter resultado válido de
  proteção hard pós-correção de eval-mode. Os quatro agregados antigos com
  `hard_freeze` continuam não citáveis.
- `docs/arch_bert.md` e `article/manuscript.md`: o ganho do BERT não pode mais
  ser atribuído ao regime. A redação tem que dizer "específico da arquitetura
  e/ou do regime de capacidade".
- `goals/opcoes_novidade_e_proximos_passos.md`: reforça a Opção A. O eixo
  "capacidade determina se a proteção ajuda" agora tem evidência em MLP, CNN e
  Qwen, com um teorema por trás.
