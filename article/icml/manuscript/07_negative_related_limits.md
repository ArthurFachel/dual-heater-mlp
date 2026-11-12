# 7 Negative results

We list every pre-registered or decisive result that went against the method,
in one place, because a paper about controls is only credible if its own
method fails some of them.

| result | host | status | reading |
|---|---|---|---|
| exact - frozen_A = -0.0092 forgetting, 5+/5-, p = 1.000 | Qwen-LoRA | C, null | the mask adds nothing to LoRA-FA; 1.8x slower |
| original "confirmed" exact - lr control was matched on the wrong surface | Qwen-LoRA | C, invalidated | E 0.532 vs 0.850 (Section 6.1) |
| M - R(fixed) = +1.50 pp [-0.07, +3.07], p = 0.060 | Qwen-FT | C, REFUTED | selection under a hard mask is not robust |
| gradient term of the importance criterion: abs(z * dL/dz) - abs(z) = -0.17 pp, 5+/5- | BERT | C, null (Section 5.5) | the expensive part of the criterion buys nothing |
| exact - random_fixed on the generative stream = +0.06 pp AP [-0.14, +0.27], 5/10 | Qwen-LoRA-Gen | C, null bounded within 1 pp | selection of B rows adds nothing on the second stream either |
| our mask on DER++ under representative replay memory: -0.61 pp FAA, 1/20, p_Holm 8.0e-5; the pre-registered sign inversion between selectors did not replicate (high-loss: +0.20 pp, p_Holm 0.115) | CIFAR-replay | C, failed (RD) | with a good replay selector our mask costs accuracy |
| SlowHeat + ER-ACE FAA 0.7083 < ER-ACE 0.7109 | MLP-S | c | the uniform reduction costs ER-ACE more than selection gives back |
| mask timing pilot: freezing the mask after 1 or 3 boundaries, -1.31 and -1.30 pp FAA; head warm-up -2.90 pp | Qwen-FT | E, negative | none of the timing variants helps |
| lr control below vanilla at E = 0.85 did not replicate (p = 0.754) | Qwen-LoRA | c, null | can hurt (Section 6.1) or do nothing; we corrected 7 places in our docs that stated it as a rule |

The hard mask also trades: on Split-CIFAR-100 with an MLP, against the soft
mask, it lowers forgetting by 11.68 pp and lowers FAA by 1.41 pp, both 10/10
seeds (E, and at absolute FAA of 3.7% vs 5.1%, a near-floor regime). "Less
forgetting and higher accuracy" is not a claim this method supports in general.

# 8 Related work

**Learning less.** LoRA trades acquisition for retention relative to full
fine-tuning [CITE Biderman et al. 2024, LoRA Learns Less and Forgets Less], and
a random mask trained with a larger learning rate matches LoRA in fine-tuning
[CITE Xu and Zhang, ICML 2024, Random Masking Finds Winning Tickets]. Both point
at how much a method changes as the main variable. We make that variable a
control arm with a measured E and split the remainder into shape and selection.

**Training regime and plasticity.** That learning rate, batch size and dropout
alone change forgetting is known [CITE Mirzadeh et al. 2020, Understanding the
role of training regimes in CL]. Work on loss of plasticity studies the opposite
failure, networks that stop learning [CITE Dohare et al. 2024]. We use the
regime effect as a control: every masking method is compared against the same
reduction of plasticity with no structure.

**Parameter isolation and masking.** PackNet, HAT and Piggyback [CITE] freeze or
gate subsets of parameters per task. Utility-based perturbed gradient descent
(UPGD) [CITE Elsayed and Mahmood, ICLR 2024, arXiv 2404.00781] protects useful
units and perturbs the rest, close to our method. AWARe [CITE arXiv 2608.11758]
ranks neurons by activation saliency and freezes the top fraction with a binary
mask, which is the magnitude arm of Section 5.5. [TODO: SPARS, arXiv
2610.19876, one sentence.] None of these, to our knowledge, reports a
plasticity-matched uniform control. [TODO: verify this sentence against each
paper before submission.]

**Random masks as controls.** Recent work on LoRA finds random masks beating
gradient attribution on the B matrix [CITE arXiv 2610.12744]. Our BERT result
points the other way for a different reason: a random hard mask is a weak
control because its shape costs retention relative to a uniform scale
(Section 5.3), so selection looks larger against it than it is. Our generative
LoRA result agrees with theirs on B: random rows do as well as selected ones
(Section 5.4).

**Parameter-efficient CL.** LoRA-FA [CITE arXiv 2308.03303] freezes the LoRA
down-projection for memory; LoRA-Null [CITE arXiv 2503.02659] reuses it for
knowledge preservation. O-LoRA and InfLoRA [CITE] constrain adapter subspaces.
Our Section 6.1 finds that, on our host, the whole retention gain of a masked
LoRA comes from the LoRA-FA freeze.

**Pre-registration in ML.** [CITE pre-registration workshop papers, NeurIPS
2020 / 2021.] [TODO: one or two sentences.]

# 9 Limitations

- **Scale.** The largest model is 0.5B parameters. Full fine-tuning (Qwen-FT)
  was tested only on classification; generation was tested only under LoRA
  (SuperNI-15, 1 epoch per task).
- **One dataset per host**, few task sequences, short training (125 steps in
  LM-410M; 120 steps per task in Qwen).
- **Floors.** Qwen-FT retention is 4.8 to 9.0% absolute in the best arm, and
  BERT at T = 5 has zero retention (Section 6.2). Terms measured near a floor
  are compressed toward zero.
- **Missing cells.** Table 1 has no vanilla arm in LM-410M and only exploratory
  random arms in BERT and LM-410M. The shape and selection terms are
  confirmatory in MLP-S, MLP-P, Qwen-FT and the two LoRA settings. [TODO: update after the planned confirmatory runs.]
- **Hyperparameters** (beta = 30, budgets) were chosen in exploratory runs and
  not tuned per host or per baseline; the controls inherit them.
- **Non-uniqueness.** The order of the decomposition is a choice
  (Section 3.3). The ordering of arms within a host does not depend on it; the
  size of each term does.
- **Our LoRA-FA** omits the closed-form gradient correction of the original.
- **Devices.** LM-410M and Qwen seeds ran on two GPU models (1080 Ti, Titan
  Xp); arms of a seed always shared a device.
