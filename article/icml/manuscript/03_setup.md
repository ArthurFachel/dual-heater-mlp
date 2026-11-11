# 3 Hosts, arms and pre-registration

## 3.1 Hosts

| host | model | stream | protected surface | budget / factor | primary endpoint |
|---|---|---|---|---|---|
| MLP-S | MLP | Split-MNIST, class-IL, with a replay-family baseline B | hidden units | p = 0.25, beta = 30 [TODO: confirm soft vs hard in FROZEN_CONFIG] | final average accuracy (FAA), test |
| MLP-P | MLP | Permuted-MNIST | hidden units | hard | FAA |
| BERT | bert_uncased_L-4_H-256_A-4 | CLINC150 "plus" without OOS, class-IL, 2 tasks of 15 intents, 4 epochs/task, lr 5e-5, batch 32, seq 128 | FFN + attention | p = 0.50, soft beta = 30, criterion abs(z) | task-1 retention, validation |
| LM-410M | ConceptLM Pythia-410M [CITE ConceptLM], released weights | continued pre-training on FineWeb-Edu, importance measured on MiniPile; 125 steps (~4M tokens), lr 3e-5, grad. accum. 64, warmup 25 | FFN units | p = 0.75, soft beta = 30 | MiniPile test next-token loss (nats) at the last step |
| Qwen-FT | Qwen2.5-0.5B, full fine-tuning of FFN + head | CLINC150 "plus" as class-IL classification, 10 domains, 120 steps/task, lr 1e-5, batch 2, seq 64, fp32 | FFN units | hard, top 25% per layer frozen (E* = 0.75) | first-task retention |
| Qwen-LoRA | Qwen2.5-0.5B + LoRA r = 16 | Split-CLINC150, 10 tasks | LoRA A, B | frozen A; mask on rows of B | forgetting |
| Qwen-LoRA-Gen | Qwen2.5-0.5B + LoRA r = 8, alpha 16, FFN | SuperNI-15 (SAPT order 1) [CITE SAPT], generative, loss on answer tokens, 1 epoch/task, lr 1e-4, batch 2, greedy decoding up to 50 tokens, fp32 | LoRA A, B | frozen A; soft mask on rows of B (p = 0.50, beta = 3) | AP, ROUGE-L, as in SAPT |

The MLP and BERT hosts ran on CPU; LM-410M and Qwen ran on one GPU per seed
(GTX 1080 Ti or Titan Xp, fp32; Pascal has no bf16). All arms of a seed share a
process and a device. [TODO: MLP width/depth and Split-MNIST epochs from
FROZEN_CONFIG in experiments/confirmatory_split_mnist.py.]

**On the LM-410M host.** Continuing pre-training of Pythia on MiniPile, a subset
of its own pre-training corpus, cannot improve its test loss at any learning
rate we tried, so we train on a later corpus (FineWeb-Edu) and use MiniPile only
to measure forgetting of the original distribution. The effect is small in
absolute terms: the uniform arm forgets +0.0711 nats of MiniPile loss over the
run.

## 3.2 Pre-registration

Every confirmatory contrast in this paper followed the same procedure:

1. A protocol file with the question, arms, seeds, primary endpoint, test and
   decision rule is committed before the first seed runs. The commit hash is in
   Appendix A.
2. Seeds come from a band never used by any earlier run (checked against every
   config in the repositories).
3. The analysis script is written and validated on the earlier exploratory data
   before the run: it must reproduce the exploratory estimate exactly.
   Exception: for 3b the analysis script was written after the run finished.
   It computes the endpoints declared in the frozen protocol and its header
   says when it was written. The 3b epoch count (1 per task) came from a
   calibration on seed 950 whose acceptance threshold was amended post hoc
   (12/15 to 11/15 tasks acquired); the amendment is committed (6d487a5).
4. No seed is dropped. A crashed run is re-run with the same seed and the fact
   is reported. In F11 to F16 no seed crashed. [TODO: check PM and RC logs.]
5. Decision rule: paired two-sided t test (or exact sign test where declared),
   alpha = 0.05, Holm over the family when there is more than one contrast.
   CONFIRMED if the mean is in the predicted direction and p (adjusted) < 0.05;
   REFUTED if the 95% CI includes zero or the mean has the wrong sign.

Integrity checks run before any endpoint is read: matched E between arms, equal
protected counts, that the control actually received its scale and the method
arm did not, and for random arms that the drawn set differs from the selected
one and is constant (fixed) or re-drawn (permuted) as declared. A failed check
invalidates the run; it is not reinterpreted.

We report exploratory results where no confirmatory one exists, tagged E, and
never use them as the basis of a claim stated without that tag.

## 3.3 Families of runs

| id | host | contrast(s) | seeds | protocol commit |
|---|---|---|---|---|
| F11 | MLP-S | M - U for B in {replay, DER++, replay calibrated, ER-ACE} | 20 | e7a2ece |
| F12 | BERT | M - U on 5 domain pairs | 20 per pair | 77ebece |
| F13 | LM-410M | M - U | 10 (100-109) | f8526c8 |
| F14 | Qwen-FT | M - U | 10 (200-209) | c495d14 |
| F15 | Qwen-FT | M - R (re-drawn) | 10 (300-309) | dab50ed |
| F16 | Qwen-FT | M - R (fixed) | 10 (400-409) | cc81913 |
| PM | Qwen-LoRA | frozen_A - U, matched on the full surface | 10 | d9b57fc |
| RC | Qwen-LoRA | frozen_A - V; exact - frozen_A | 10 | f5d77ce |
| 3b | Qwen-LoRA-Gen | U - V with U matched to exact and to frozen_A (primary, Holm over 2); M - U and frozen_A - U (secondary) | 10 (1000-1009) | deea8d2 |

Exploratory sources (E): F3 (LM-410M, M - R and R - U, 6 seeds), F7/F8 (MLP,
U - B and M - U, 20 and 10 seeds), the BERT hard decomposition (10 seeds,
band 13.000.003+), and the LoRA rank comparison (Section 4.5, 10 seeds).
