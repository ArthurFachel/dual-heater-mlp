# Appendix B: Pre-registration protocols

Every confirmatory family in this paper ran under a written protocol committed to version control
before the first seed of that family; the commit hash is given in each subsection and in Appendix A.
Seeds came from bands that no earlier run had used, and the analysis script was written before the
run and, where earlier exploratory data existed, validated on those data. The exceptions are stated
where they apply: the analysis script of 3b was written after the run (Appendix A), 3c reuses the 3b
seeds by design, and F15 and F16 had no exploratory data for their new arm, so their scripts were
validated on earlier manifests and a smoke run. No seed is dropped; a failed run is re-run with the
same seed and the fact is reported. Where a protocol declares integrity gates, they are read before
the primary endpoint, and a failed gate invalidates the run instead of being reinterpreted.

Each subsection translates the frozen protocol (most were written in Portuguese); "not declared"
marks a field the protocol does not contain, and arm names are kept verbatim from the code. M is the
method, U the uniform update scale with the same mean factor, R the same mask on random units and V
vanilla. Outcomes are copied from Appendix A (for 3c, from the run's analysis output); nothing here
is recomputed.

## B.1 F11: selection vs matched uniform plasticity, Split-MNIST, four baselines

- **Setting.** MLP-S. FROZEN_CONFIG of experiments/confirmatory_split_mnist.py, unchanged, device
  CPU. All 12 arms run in the same process and on the same device. Runner
  scripts/run_confirm_uniform.py.
- **Question.** For each baseline B, does adding the method (selection by functional importance)
  give more than adding the same mean plasticity without selection (UniformPlasticityLevel)?
- **Arms.** For each B, the arms are B, method + B and B + uniform:
  - replay / slowheat_replay_hidden_beta_30_budget_0.25 / uniform_replay_hidden_beta_30_budget_0.25
  - derpp / slowheat_derpp_hidden_beta_30_budget_0.25 / uniform_derpp_hidden_beta_30_budget_0.25
  - replay_calibrated / slowheat_replay_hidden_beta_30_budget_0.25_calibrated /
    uniform_replay_hidden_beta_30_budget_0.25_calibrated
  - er_ace / slowheat_er_ace_hidden_beta_30_budget_0.25 / uniform_er_ace_hidden_beta_30_budget_0.25
- **Seeds.** 20 new, never-used seeds: [16000057, 16007976, 16015895, 16023814, 16031733, 16039652,
  16047571, 16055490, 16063409, 16071328, 16079247, 16087166, 16095085, 16103004, 16110923,
  16118842, 16126761, 16134680, 16142599, 16150518], i.e. 16000057 + 7919·k, k = 0..19. They are
  disjoint from CONFIRMATORY_SEEDS and from the F4/F7/F10 seeds.
- **Primary endpoint.** final_average_accuracy (FAA) on the test set at the end of the last task.
  The contrast is C_B = FAA(method + B) - FAA(B + uniform), paired by seed, for B in {replay, derpp,
  replay_calibrated, er_ace}.
- **Test and decision rule.** Two-sided paired t test per contrast, Holm over the 4 p-values, alpha
  0.05. Confirmed for B: mean C_B > 0 and Holm-adjusted p < 0.05. Refuted for B: the 95% CI of C_B
  includes 0 or the mean is <= 0. Analysis script: scripts/analyze_confirm_uniform.py, written
  before the run.
- **Integrity gates.** not declared.
- **Secondary endpoints.** Descriptive, without a family test: sign test, mean forgetting, (method +
  B) - B, and (B + uniform) - B.
- **Declared deviations and amendments.** not declared.
- **Protocol commit.** e7a2ece, docs/protocols/confirmatory_uniform_control.md.
- **Outcome.** CONFIRMED in 4 of 4 contrasts. M - U FAA: replay +0.62 [+0.25, +0.98] 17/20; DER++
  +1.91 [+1.39, +2.43] 19/20; replay cal. +1.49 [+1.21, +1.78] 20/20; ER-ACE +0.39 [+0.15, +0.64]
  15/20; Holm <= 4.3e-3.

## B.2 F12: BERT, soft mask (strength 30) vs uniform soft factor, 5 domain pairs

- **Setting.** BERT (google/bert_uncased_L-4_H-256_A-4), CLINC150, class-incremental, 2 tasks.
  Default SplitCLINC150Config: 4 epochs per task, lr 5e-5, batch 32, seq 128, task_limit 2. Budget
  0.50 on FFN and attention, device CPU. Each pair is moved to the front of the domain list (the
  same reorder function as F5; for banking -> credit_cards the order is the default one). Runner
  scripts/run_confirm_bert_soft30.py, one process per pair.
- **Question.** Does the method's soft mask (|z|, strength 30, budget 0.50) retain task 1 better
  than a control that applies the same mean soft factor without selection (uniform_soft)?
- **Arms.** All three run in the same process and on the same device, per pair: vanilla: reference,
  required by run_diagnostic; slowheat_magnitude_soft_b30: FFN + attention, strength 30, soft mask,
  magnitude criterion; slowheat_uniform_soft_b30: FFN + attention, strength 30, uniform_soft,
  magnitude criterion.
- **Pairs.** banking -> credit_cards, auto_and_commute -> travel, kitchen_and_dining -> home,
  small_talk -> meta, utility -> work.
- **Seeds.** 20 new seeds: [17000029, 17008040, 17016051, 17024062, 17032073, 17040084, 17048095,
  17056106, 17064117, 17072128, 17080139, 17088150, 17096161, 17104172, 17112183, 17120194,
  17128205, 17136216, 17144227, 17152238], i.e. 17000029 + 8011·k, k = 0..19. Band 17 is unused in
  the three repositories checked, and it is disjoint from 11-14.xxx.xxx and from 16.000.057+.
- **Primary endpoint.** task1_retention: accuracy on task 1 at the end of task 2, on the validation
  set (evaluate_test=False, as in the exploratory runs). The contrast is C_P =
  task1_retention(slowheat_magnitude_soft_b30) - task1_retention(slowheat_uniform_soft_b30), paired
  by seed, for each pair P.
- **Test and decision rule.** Two-sided paired t test of C_P per pair, Holm over the 5 p-values,
  alpha 0.05. Confirmed on pair P: mean > 0 and Holm-adjusted p < 0.05. Refuted on P: the 95% CI
  includes 0 or the mean is <= 0. Any other case is inconclusive. Overall reading: CONFIRMS if >= 4
  of the 5 pairs are confirmed, otherwise DOES NOT CONFIRM. Analysis script:
  scripts/analyze_confirm_bert_soft30.py, written and validated before the run.
- **Integrity gates.** not declared.
- **Secondary endpoints.** Descriptive, without a family test: FAA and T2 acquisition (method -
  uniform), sign test, method - vanilla, and uniform - vanilla.
- **Declared deviations and amendments.** Relative to the queued proposal, the runs use 5 processes
  with 4 threads each instead of 6 threads run serially. This changes only wall time; the 3 arms of
  each pair still share one process.
- **Protocol commit.** 77ebece.
- **Outcome.** CONFIRMED in 5 of 5 pairs. M - U T1 retention +4.07, +4.40, +3.20, +2.93, +5.20 pp;
  19-20/20; Holm <= 1.0e-6.

## B.3 F13: LM-410M FFN, soft mask (strength 30) vs matched uniform factor

- **Setting.** LM-410M: released ConceptLM Pythia-410M, continued training on FineWeb-Edu, with
  importance measured on MiniPile. Short protocol identical to scripts/ffn_soft_2x2.sh: --init
  released --data data/fineweb_edu --importance-data data/minipile --extra-eval data/minipile --lr
  3e-5 --grad-accum 64 --warmup 25 --max-steps 125 --eval-every 25 --log-every 5 --no-save. Runner
  scripts/f13_confirm_ffn_soft30.sh.
- **Question.** Does the method's soft FFN mask (top 25% of units per layer protected by importance
  on MiniPile, factor 1/(1+30h)) forget MiniPile less than the same mean factor applied without
  selection (--uniform-mask)?
- **Arms.** Both arms use the same code and protocol; running them on the same GPU model is not
  guaranteed. slowheat: --arm slowheat --families ffn --plasticity 0.75 --slow-strength 30; uniform:
  the same plus --uniform-mask.
- **Seeds.** 100, 101, ..., 109 (10 seeds). No args.json in runs/ uses a seed > 5. The seed fixes
  the data order and the permutation of the random arm (the random arm is not used here).
- **Primary endpoint.** D = minipile/test_ntp(slowheat) - minipile/test_ntp(uniform) at the final
  eval (split "final", step 125), paired by seed. Negative D means the method forgets less.
- **Test and decision rule.** Two-sided paired t test of D, alpha 0.05, single contrast without
  correction. CONFIRMED: mean D < 0 and p < 0.05. REFUTED: the 95% CI includes 0 or the mean
  is >= 0. Any other case is inconclusive. Analysis script: scripts/analyze_f13.py, written and validated
  before the run.
- **Integrity gates.** not declared.
- **Secondary endpoints.** Descriptive: sign test; FineWeb test_ntp (method - uniform, the cost on
  the new task); D relative to the uniform arm's forgetting (MiniPile at the final step minus
  MiniPile at step 0).
- **Declared deviations and amendments.** not declared.
- **Protocol commit.** f8526c8, docs/protocols/confirmatory_ffn_soft30.md.
- **Outcome.** CONFIRMED. M - U MiniPile loss -0.00764 nat [-0.00785, -0.00743], 10/10, p 3.5e-14;
  uniform forgetting +0.07111 nat; -10.7%.

## B.4 F14: Qwen-FT, hard mask vs matched reduced learning rate

- **Setting.** Qwen-FT: Qwen2.5-0.5B as a class-IL classifier on CLINC150.
  experiments/qwen_iso_plasticity.py on branch exp/qwen-hard-confirm (base aed2957 plus an --arms
  option that only filters the declared arms and records protocol.arms_filter). Flags: --domains
  banking credit_cards kitchen_and_dining home auto_and_commute travel utility work small_talk meta
  --steps-per-task 120 --learning-rate 1e-5 --batch-size 2 --max-length 64 --target-plasticity 0.75
  --arms hard_b0.75 reduced_lr vanilla --device cuda (fp32, local capacity, beta per_boundary). One
  process per seed, with all 3 arms in the same process and on the same GPU. Runner
  scripts/f14_confirm_qwen_hard.sh.
- **Motivation.** A post hoc analysis of the confirmatory run confirm120 (seeds 10..19, under an
  earlier protocol) gave hard_b0.75 - reduced_lr = +6.30 pp first-task retention [+3.42; +9.18],
  10/10. That contrast was not in the pre-registered family of that run.
- **Question.** Does the method's hard mask (25% of registered units frozen by importance,
  plasticity 0.75) retain the first task better than the matched reduced learning rate (lr x 0.75
  over the whole trainable surface, no mask)?
- **Arms.** hard_b0.75, reduced_lr, vanilla.
- **Seeds.** 200, 201, ..., 209 (10 seeds), never used on Qwen. confirm120 used 10-19.
- **Primary endpoint.** D = retention_first_task(hard_b0.75) - retention_first_task(reduced_lr),
  paired by seed.
- **Test and decision rule.** Two-sided paired t test of D, alpha 0.05, single contrast. CONFIRMED:
  mean D > 0 and p < 0.05. REFUTED: the 95% CI includes 0 or the mean is <= 0. Otherwise
  inconclusive. Analysis script: scripts/analyze_f14.py, written and validated on confirm120 before
  the run.
- **Integrity gates.** not declared.
- **Secondary endpoints.** Descriptive: sign test; FAA and last-task acquisition (hard -
  reduced_lr); hard - vanilla and reduced_lr - vanilla on retention.
- **Declared deviations and amendments.** not declared.
- **Protocol commit.** c495d14.
- **Outcome.** CONFIRMED. M - U retention +5.67 pp [+3.00, +8.33] 9/10, p 9.7e-4.

## B.5 F15: Qwen-FT, hard mask on selected vs re-drawn random units

- **Setting.** Qwen-FT, identical to F14 except for the arm list (branch exp/f15-hard-permuted, base
  d67a536 plus the opt-in arm --hard-permuted-control). Flags as in F14 with --hard-permuted-control
  --arms hard_b0.75 hard_permuted_b0.75 reduced_lr. One process per seed, all 3 arms on the same
  GPU. Motivation: hard differs from reduced_lr both in WHICH units lose plasticity and in HOW
  (binary); F15 isolates the first.
- **Question.** At the same effective plasticity (E = 0.75) and with the same binary mask, does
  freezing the 25% most important FFN units of each layer retain the first task better than freezing
  25% of units chosen at random?
- **Arms.**
  - hard_b0.75: as in F14 (top 25% by importance, per layer, binary mask).
  - hard_permuted_b0.75 (new): the same consolidation and budget. Right after each consolidation,
    each layer's importance (heat) vector is permuted with seed seed + 1000*boundary, so the number
    of frozen units per layer is identical and only their identity changes. The permutation is
    redone at every boundary, because consolidation rebuilds the vector from accumulated importance.
  - reduced_lr: as in F14. It replicates F14 on new seeds (secondary).
  - There is no vanilla arm, because F14 already anchored forgetting.
- **Seeds.** 300, 301, ..., 309 (10 seeds). Checked on 2026-11-09: no Qwen script, protocol,
  experiment or results directory uses 300-309.
- **Primary endpoint.** D = retention_first_task(hard_b0.75) -
  retention_first_task(hard_permuted_b0.75), paired by seed.
- **Test and decision rule.** Two-sided paired t test of D, alpha 0.05, single contrast without
  correction. CONFIRMED: mean D > 0 and p < 0.05 (selection matters under a hard mask). REFUTED: the
  95% CI includes 0 or the mean is <= 0 (the F14 gain comes from the binary or sparse form, not from
  the choice of units). Otherwise inconclusive. Analysis script: scripts/analyze_f15.py. No
  exploratory data existed for hard_permuted, so the script was validated on the F14 manifests (it
  reproduces +5.67 pp) and on a CPU smoke manifest with the three arms (keys and integrity check).
  There is no prior estimate of the effect size, and n = 10 matches F14.
- **Integrity gates.** permutation_reordered = True, and protected_units equal between hard and
  hard_permuted, at every boundary of every seed. If this fails, the run is invalid and is reported
  as such, without reinterpretation.
- **Secondary endpoints.** Descriptive: sign test; FAA and last-task acquisition (hard -
  hard_permuted); hard - reduced_lr (replication of F14) and hard_permuted - reduced_lr on
  retention.
- **Declared deviations and amendments.** not declared.
- **Protocol commit.** dab50ed.
- **Outcome.** CONFIRMED. M - R(re-drawn) +4.10 pp [+0.61, +7.59] 9/10, p 0.026.

## B.6 F16: Qwen-FT, hard mask on selected vs FIXED random units

- **Setting.** Qwen-FT, identical to F15 except for the arm list (branch exp/f16-hard-random-fixed,
  base 45f0611 plus --hard-random-fixed-control and a per-boundary digest of the protected set).
  Flags as in F14 with --hard-permuted-control --hard-random-fixed-control --arms hard_b0.75
  hard_random_fixed_b0.75 hard_permuted_b0.75. One process per seed, all 3 arms on the same GPU.
  Motivation: F15's control re-draws its set at every boundary, so no unit stays frozen throughout,
  while the hard set tends to be stable; a FIXED random set is the harder control.
- **Question.** With E = 0.75, a binary mask and the same count of protected units per layer, does
  freezing the most important units retain the first task better than freezing a random set drawn
  ONCE per run and kept at every boundary?
- **Arms.**
  - hard_b0.75: as in F14/F15.
  - hard_random_fixed_b0.75 (new): after each consolidation, the count k of protected units per
    layer equals that of hard (it comes from the same budget). The protected units are the first k
    of a random order per layer that depends only on (seed + 500000) and the layer index
    (fix_random_heat). The order is the same at every boundary, so the set is fixed (nested if k
    grows).
  - hard_permuted_b0.75: as in F15 (re-drawn at every boundary), used for the secondary fixed vs
    re-drawn contrast.
- **Seeds.** 400, 401, ..., 409 (10 seeds). Checked on 2026-11-09 by grep over scripts, goals,
  protocols and results of the 4 worktrees: no use.
- **Primary endpoint.** D = retention_first_task(hard_b0.75) -
  retention_first_task(hard_random_fixed_b0.75), paired by seed.
- **Test and decision rule.** Two-sided paired t test of D, alpha 0.05, single contrast. CONFIRMED:
  mean D > 0 and p < 0.05 (selection also beats the fixed random set). REFUTED: the 95% CI includes
  0 or the mean is <= 0. Otherwise inconclusive. Analysis script: scripts/analyze_f16.py. No
  exploratory data existed for the fixed arm, so the script was validated on the F15 manifests (it
  reproduces +4.10 pp) and on a short smoke run with the 3 arms (keys and integrity).
- **Integrity gates.** Not endpoints; any failure makes the run invalid, without reinterpretation: a
  hard mask and the same protected count in all 3 arms at every boundary; random_fixed = True, and
  protected_digest constant across boundaries in the fixed arm whenever the count did not change;
  the fixed arm's protected_digest differs from that of hard; permutation_reordered = True in
  hard_permuted.
- **Secondary endpoints.** Descriptive: sign test; FAA and acquisition (hard - fixed);
  hard_random_fixed - hard_permuted on retention (if > 0, the F15 control was weaker, as suspected);
  hard - hard_permuted (replication of F15).
- **Declared deviations and amendments.** not declared.
- **Protocol commit.** cc81913.
- **Outcome.** REFUTED. M - R(fixed) +1.50 pp [-0.07, +3.07] 8/10, p 0.060. Secondary: fixed -
  re-drawn +0.30 [-1.06, +1.66]; M - re-drawn +1.80 [+0.38, +3.22] 9/10.

## B.7 PM: plasticity matched on the trainable surface (Qwen-LoRA)

- **Setting.** Qwen-LoRA, copied verbatim from the manifest of the original LoRA confirmation:
  - model and training: Qwen/Qwen2.5-0.5B, Qwen2ForSequenceClassification, 150 labels, fp32;
  - LoRA: targets gate_proj, up_proj, down_proj; rank/alpha 16/16.0 (scaling 1.0 in all arms);
  - data and schedule: clinc/clinc_oos:plus, class-IL by domain, 10 tasks, 750 train / 300
    validation per task, 3 epochs, batch_size 8, max_length 48, base lr 1e-4;
  - hardware: one whole seed per GPU, no DDP.
- **Motivation (measurement defect).** An earlier confirmation reported exact - lr_control as
  matched in effective plasticity (E_eff = 0.8500 +- 1e-3), but E was averaged only over parameters
  with a mask binding, so the 2,555,904 frozen parameters of A were not counted. Over the trainable
  surface of vanilla r=16 (6,769,920 parameters), E is 0.532 for exact and 0.850 for lr_control.
  Lemma stated in the protocol: E_surface <= E_bindings, with equality iff nothing outside the
  bindings was frozen or rescaled.
- **Question.** Removing the same mean amount of plasticity from the trainable surface, does
  reducing the surface (freezing A, i.e. LoRA-FA) reduce forgetting more than reducing the learning
  rate uniformly? The protocol frames this as "does the shape of the removal matter under correct
  measurement?".
- **Hypothesis.** frozen_a_control reduces mean forgetting relative to lr_control_062, with a
  negative paired difference in most seeds. Null: the paired differences are symmetric around zero.
- **Arms.** Exactly three:
  - vanilla: A trainable, no mask, lr 1e-4, 6,769,920 trainable parameters, E 1.000.
  - lr_control_062: A trainable, no mask, lr 6.2246e-5, 6,769,920 trainable parameters, E 0.622462.
  - frozen_a_control: A frozen, no mask, lr 1e-4, 4,214,016 trainable parameters, E 0.622462.
  - Matching holds by arithmetic, not bisection: the controlled lr is 1e-4 x 0.6224617129892229 =
    6.224617e-5 (= 4,214,016/6,769,920). exact is dropped, because the earlier decomposition (RC)
    showed that its mask does not contribute.
- **Seeds.** New band 6_000_003, 6_025_019, 6_050_029, 6_075_041, 6_100_057, 6_125_069, 6_150_077,
  6_175_091, 6_200_107, 6_225_119. The band is disjoint from all bands already used, which was
  verified by test.
- **Primary endpoint.** Mean forgetting, contrast frozen_a_control - lr_control_062.
- **Test and decision rule.** Exact two-sided sign test over the 10 per-seed paired differences.
  Holm over a family of 2 (forgetting and FAA on the primary contrast), alpha 0.05; the Holm floor
  at n=10 is 2x2/1024 = 0.0039. Success criterion: p < 0.025 on the primary endpoint AND a negative
  median. vanilla is descriptive and outside the family. There is no interim analysis, no early
  stopping and no seed-by-seed inspection before all 10 seeds finish. No outcome authorizes extra
  seeds, a reduced family or promotion of an endpoint.
- **Integrity gates.** Mandatory verification before the run: tests/test_surface_plasticity.py green
  (9 tests) and tests/test_plasticity_matched_runner.py green (11 tests); all 6 declared mutations
  killed; masked_mean, masked_parameters and learning_rate_scale recorded in every arm's manifest;
  measured surface_plasticity within 1e-6 of 0.622462 in both treated arms, in every seed;
  effective_learning_rate = 6.224617e-5 for lr_control_062 and 1e-4 for the others, in every seed;
  identical tokens across arms within each seed; the seeds never run before with any LoRA arm.
- **Secondary endpoints.** FAA on the same contrast. surface_plasticity per arm is a check, not a
  result. frozen_a_control - vanilla and lr_control_062 - vanilla are descriptive.
- **Declared deviations and amendments.** None after freezing. The change log has a single entry
  (creation and freeze, before any seed; 6/6 mutations killed).
- **Declared limitations.** Freezing A removes 37.75% of the trainable parameters and also fixes the
  directions of the down-projection; a control that freezes the same fraction of random trainable
  parameters is not in this design. The run does not decide the method's mask and does not
  re-measure exact. One model, one benchmark, one r, one plasticity point.
- **Protocol commit.** d9b57fc (file frozen 2026-09-29).
- **Outcome.** CONFIRMED. frozen_A - lr control: forgetting -0.1260, FAA +9.92 pp, 10/10, p_Holm
  0.0039. Descriptive: lr control - vanilla forgetting +0.0616 9/10 p 0.0215; frozen_A - vanilla
  -0.0644 p 0.109.

## B.8 RC: decomposition of the exact arm (Qwen-LoRA)

- **Setting.** Qwen-LoRA (host Qwen, GPU). All arms share the data stream, tokenization, optimizer,
  rank, alpha (constant scaling alpha/r) and step budget. The remaining configuration is not
  restated in the protocol.
- **Motivation.** The confirmed contrast exact - lr_control (forgetting -0.0921, median -0.0998,
  9-/1+, p = 0.02148, E_eff = 0.85) measures the sum of two things: freezing A (LoRA-FA, a published
  technique) and masking the rows of B with the method's tracker. No existing arm separated them.
- **Question.** How much of the gain of exact comes from LoRA-FA, and how much from the method?
- **Arms.**
  - vanilla: A trainable, no mask on B (standard LoRA).
  - frozen_a_control: A frozen, no mask on B (pure LoRA-FA). It builds no tracker, registers no
    binding and does not touch the optimizer.
  - exact: A frozen, mask on B (LoRA-FA plus the method).
  - Freezing of A is driven by one shared set (FROZEN_A_METHODS), and a test fails if that path is
    duplicated. exact and frozen_a_control have the same trainable parameter count.
- **Seeds.** Band 5.000.003+, disjoint from the 243 seeds already reserved (checked mechanically
  against all artifacts in results/): 5000003, 5025016, 5050029, 5075042, 5100055, 5125068, 5150081,
  5175094, 5200107, 5225120. The seeds are registered in code and verified by test.
- **Primary endpoint.** Forgetting (the same endpoint as the original confirmation of exact).
- **Test and decision rule.** Family of 2 contrasts, frozen_a_control - vanilla (LoRA-FA alone) and
  exact - frozen_a_control (the method given LoRA-FA), with Holm over both. Exact two-sided sign
  test with zeros discarded; at n = 10 the minimum attainable p is 2/2^10 = 0.00195, which leaves
  enough margin for alpha = 0.05 after Holm. The declared outcomes are:
  - (a) only frozen_a_control - vanilla significant: all of the gain is LoRA-FA;
  - (b) both significant;
  - (c) only exact - frozen_a_control significant;
  - (d) neither significant.
  - All four will be reported. Inviolable rules: no extra seeds, no Holm recomputed over a reduced
    family, no change of primary endpoint after seeing results, no deletion of negative results, and
    amendments only before the first seed.
- **Integrity gates.** not declared.
- **Secondary endpoints.** FAA, reported without correction and declared exploratory.
- **Declared deviations and amendments.** None. The cost is to be measured on 1 seed before the 10
  are launched.
- **Protocol commit.** f5d77ce (file frozen 2026-09-28).
- **Outcome.** Declared outcome (a). frozen_A - vanilla -0.1118 (1+/9-, p_Holm 0.043); exact -
  frozen_A -0.0092 (5+/5-, p 1.000); FAA +0.69 pp; 611 s vs 336 s.

## B.9 3b: plasticity component of the gain in generative LoRA (Qwen-LoRA-Gen)

- **Setting.** Qwen-LoRA-Gen. experiments/superni_cl.py at the commit of the protocol (branch
  exp/3b-superni-runner). Model and data: Qwen2.5-0.5B + LoRA r 8 alpha 16 on the FFN; SuperNI-15
  from SAPT, order 1, full sets (train up to 1,000 and test up to 100 per task). Training and
  evaluation: 1 epoch per task, fp32, batch 2, eval batch 8, lr 1e-4, greedy generation up to 50
  tokens, --acquisition-margin 0.05 --min-vanilla-forgetting 0.05. Queue
  scripts/confirm_3b_queue.py, with one worker per GPU.
- **Question.** In continual instruction tuning, does a CL method's gain over vanilla survive when
  the control is a uniform learning rate with the SAME surface plasticity E?
- **Arms.** Five per seed:
  - vanilla, exact (the method on LoRA) and frozen_a_control (LoRA-FA) run in one process.
  - lrc_exact: lr_control with scale = surface_plasticity of exact for THE SAME SEED, read from that
    process's manifest (chained after it).
  - lrc_lorafa: lr_control with scale 0.6148148148148148 = 2039808/3317760, the structural surface
    plasticity of LoRA-FA, which does not depend on the seed.
- **Seeds.** 1000 to 1009 (a new band; the calibration used 950). No seed is dropped or replaced.
- **Primary endpoint.** AP and Fgt are computed as in SAPT (ROUGE-L, fraction scale). For X in
  {exact, frozen_a_control}: D_unmatched(X) = AP(X) - AP(vanilla) and D_matched(X) = AP(X) -
  AP(lr_control matched to X). The primary endpoint is D_unmatched(X) - D_matched(X) > 0, paired by
  seed. This equals AP(lr_control_X) - AP(vanilla), i.e. it asks whether the lr control alone
  already captures part of the gain.
- **Test and decision rule.** Two-sided paired t test over 10 seeds with a t-based 95% CI, Holm over
  the 2 methods, alpha 0.05; the number of seeds with each sign is reported. Declared reading: if
  the primary is positive and Holm-significant for X, the gain of X over vanilla is partly
  plasticity, and the fraction D_matched/D_unmatched is reported. If the S1 CI lies above zero, X's
  selection still pays at matched E; if it crosses zero, this is not confirmed. One calibration seed
  showed D_matched ~ +0.05 for both X, which enters no computation. Analysis script: see Declared
  deviations.
- **Integrity gates.** Checked before any endpoint is read:
  - V1. G-imp of exact passes in every seed. A failing seed stays in the report but is marked;
    failure in more than 2 seeds invalidates the whole confirmatory run.
  - V2. |E(exact) - E(lrc_exact)| <= 0.01 and |E(LoRA-FA) - E(lrc_lorafa)| <= 0.01 in every seed. By
    construction both should be 0, and anything else is a bug.
  - V3. vanilla G-aq >= 11/15 and vanilla Fgt >= 0.05 (the amended calibration criterion), reported
    per seed. These do not exclude seeds.
  - V4. A process exiting with a non-zero code is relaunched once with the same seed and arguments,
    and the relaunch is logged. A second failure leaves the seed missing, and the report says so.
- **Secondary endpoints.** Without family correction, all reported, none promoted later: S1
  D_matched(X) with paired 95% CI; S2 the same contrasts on Fgt; S3 exact - frozen_a_control on AP
  and Fgt (E not matched, descriptive); S4 ordering of the arms by mean AP, with and without
  matching.
- **Declared deviations and amendments.**
  - The epochs-per-task choice comes from a calibration whose criterion has a post hoc amendment,
    declared in goals/calibracao_3b.md (commits 6d487a5 and ea4d029).
  - V3 uses the amended criterion.
  - Any change after the commit of this file is to be declared in a new section at the end of the
    file; the frozen file contains no such section.
  - Not stated in the protocol: Appendix A records that the analysis script
    (scripts/analyze_3b_confirm.py @ 0e56045) was written after the run.
- **Protocol commit.** deea8d2, goals/protocol_3b_confirmatory.md.
- **Outcome.** Primary positive and Holm-significant for both X. U - V, AP: lrc_exact +0.0294
  [+0.0145, +0.0443] 10/10; lrc_lorafa +0.0289 [+0.0152, +0.0426] 10/10; p_Holm 2.0e-3 both;
  fraction D_matched/D_unmatched 0.673 / 0.668.

## B.10 3c: selection vs fixed random rows (Qwen-LoRA-Gen)

- **Setting.** Qwen-LoRA-Gen. Only the new arm runs, with exactly the arguments of the 3b main
  process: --order 1 --epochs-per-task 1 --batch-size 2 --eval-batch-size 8 --precision fp32
  --acquisition-margin 0.05 --min-vanilla-forgetting 0.05 --arms random_fixed (lr 1e-4, LoRA r 8
  alpha 16 on the FFN, budget 0.5, strength 3, runner defaults). One process per seed, queue
  scripts/queue_3c.py, with the same environment as 3b.
- **Motivation.** 3b measured plasticity and the gain of exact over the matched lr, but did not
  separate shape from selection. The protocol also states a fact measured on the 3b manifests:
  exact's mask on B averages 0.91 (0.903 to 0.915 over all 15 boundaries of the 10 seeds). exact
  removes only ~9% of B's plasticity, and the rest of its E (0.560 vs 0.615 for LoRA-FA) comes from
  freezing A. A null is therefore the expected result, and it is informative if bounded.
- **Question.** With A frozen and the same budget, strength and per-module distribution of mask
  factors, does choosing the rows of B by importance (exact) give more AP than protecting fixed
  random rows (random_fixed)?
- **Arms.** random_fixed (new) is exact with a single difference. After each consolidation, each
  module's heat values are sorted and assigned to the rows of ONE random order, drawn once per run
  with seed (seed + 500000) * 1009 + module index (the same offset as F16). The protected rows are
  order[:k] at every boundary (nested if k grows). Values only change rows, so the per-module
  multiset of factors is what the arm's own importance gives. vanilla, exact, frozen_a_control,
  lrc_exact and lrc_lorafa are NOT re-run; they are taken from 3b and paired by seed.
- **Seeds.** 1000 to 1009, the 3b seeds (see Declared deviations).
- **Primary endpoint.** Selection = AP(exact) - AP(random_fixed), per seed (AP as in SAPT, ROUGE-L,
  fraction scale).
- **Test and decision rule.** A single test without a family: two-sided paired t test over the 10
  seeds, t-based 95% CI, alpha 0.05, with sign counts reported. Declared reading:
  - primary > 0 with p < 0.05: selection pays over fixed random rows on this host (reported with the
    ~9% caveat);
  - primary < 0 with p < 0.05: fixed random rows beat selection, reported as a result;
  - 95% CI entirely within +-0.01 AP (1 pp): "selection null and bounded to 1 pp" on this host;
  - CI crossing zero and leaving +-0.01: an unbounded null, i.e. the run lacks the power to bound
    it.
  - Stated power, using the 3b S3 sd (0.0075): 80% for |effect| = 0.0075 AP, 85% for 0.008, 96% for
    0.01, 47% for 0.005.
  - Analysis script: scripts/analyze_3c.py, written before the run and committed with the protocol.
    In --dry-lorafa mode it reproduces the 3b S3 exactly (+0.0028, 95% CI [-0.0026; +0.0082], 5/10,
    p 0.271) and flags the run INVALID by V3 (|dE| 0.054 to 0.056 in 10/10 seeds).
- **Integrity gates.** Checked before any endpoint is read:
  - V1. G-imp of random_fixed passes in every seed; more than 2 failing seeds invalidate the run.
  - V2. Reallocation integrity at every boundary of every seed: audit present, protected rows > 0,
    heat sum equal before and after (relative tolerance 1e-6), and mean overlap with the arm's own
    selection < 1. Any failure invalidates.
  - V3. |E(random_fixed) - E(exact)| <= 0.01 per seed, with surface E computed over the 3,317,760
    trainable parameters of the 3b vanilla arm. More than 2 seeds outside this invalidate the
    primary as a matched contrast.
  - V4. The manifest commit of each seed equals the protocol commit.
  - V5. A process with a non-zero exit code is relaunched once with the same seed and logged. On a
    second failure the seed is reported missing, and the primary runs on the complete seeds.
- **Secondary endpoints.** Uncorrected, all reported: S1 shape, random_fixed - lrc_exact (E matched
  if V3 passes); S2 random_fixed - frozen_a_control (descriptive); S3 exact - frozen_a_control
  (repeated from 3b); S4 the primary, S1 and S2 on Fgt; S5 mean overlap of the protected rows with
  the arm's own selection (chance = 0.5).
- **Declared deviations and amendments.** This is a deviation from rule 2 of the pre-registration
  (new seed band): the seeds are those of 3b, on purpose. The results of the earlier arms were
  already known, and that of random_fixed was not. Pairing across processes was already the case in
  3b. Pairing with the 3b arms rests on a pre-freeze check that the existing arms give bit-identical
  LoRA weights, heat and mask coverage at deea8d2 and at the implementation commit 929660e.
- **Declared limitations.** One task order, one model (0.5B), 1 epoch, the same seed band as 3b.
  exact's mask is weak (mean 0.91 on B), so the run measures selection in a regime where it has
  little to do, and the result does not generalize to stronger masks. Budget and strength were not
  tuned and will not be.
- **Protocol commit.** 842f54f, goals/protocol_3c_random_fixed.md.
- **Outcome.** NULL AND BOUNDED. Validity verdict OK (V1 0 failures, V2 0 seeds, V3 0 seeds).
  Primary +0.0006 (sd 0.0029), 95% CI [-0.0014; +0.0027], 5/10 > 0, p 4.97e-01, so the CI lies
  within +-0.01 AP (1 pp).

## B.11 RARB: ablation of the importance criterion (BERT-crit)

- **Setting.** BERT-crit: BERT + Split-CLINC150, 2 tasks, the same host as the earlier mechanistic
  diagnostic. Hyperparameters are identical to the existing diagnostic_conditions(): slow_strength =
  3.0, epochs_per_task = 4, max_length = 128, batch_size = 2. plasticity_budget is identical across
  the three arms.
- **Question.** On BERT/Split-CLINC150 with a hard mask and matched capacity, does the magnitude
  criterion produce first-task retention different from random and from functional?
- **Arms.** slowheat_random_hard, slowheat_magnitude_hard (new), slowheat_hard (functional), all
  with hard mask_mode and slow_strength = 3.0. vanilla is a reference arm, always reported, outside
  the family.
- **Seeds.** 10, band 4_000_003, 4_025_011, 4_050_017, 4_075_037, 4_100_043, 4_125_059, 4_150_061,
  4_175_087, 4_200_089, 4_225_097. The seeds of the earlier BERT diagnostics ([2, 9, 10, 28, 30, 32,
  57, 67, 2005, 2012] and [11, 22, 33]) may not be reused.
- **Primary endpoint.** First-task retention (first_task_retention).
- **Test and decision rule.** Family of 2 contrasts, magnitude - random and functional - magnitude;
  functional - random is already known and is left out. Exact two-sided sign test over the 10
  per-seed paired differences, Holm over the 2, alpha 0.05. There is no interim analysis, no early
  stopping and no seed-by-seed inspection before all 10 seeds finish. The declared outcomes are:
  functional > magnitude > random: the gradient adds information; functional ~ magnitude > random:
  magnitude suffices; functional ~ magnitude ~ random: ranking is not the bottleneck; magnitude <
  random: an informative negative finding.
- **Integrity gates.** Two diagnostics are recorded per arm and per stage: the variance of the
  normalized ranking (task_ema.var()), and the top-k protected-set overlap (Jaccard index) of
  magnitude vs random and of magnitude vs functional. Frozen interpretation rule: if the magnitude
  vs random top-k overlap is > 0.8, the magnitude arm is declared degenerate, and any tie between
  the two is reported as inconclusive. Required before the run: a TDD non-regression test showing
  that functional gives a bit-identical task_ema; a test that magnitude ignores the gradient; the
  seeds registered in code; a green test suite; the file committed before the first seed.
- **Secondary endpoints.** FAA, forgetting and last-task accuracy, always reported and never
  promoted.
- **Declared deviations and amendments.** None. The change log has a single entry (creation and
  freeze). The cost is to be measured on 1 seed before the 10 are launched, with the scope
  re-evaluated if the projection exceeds 4 h.
- **Protocol commit.** a3f5117, goals/protocol_importance_criterion_ablation.md (file frozen
  2026-09-28).
- **Outcome.** magnitude - random +12.23 pp 10/10, p_Holm 0.0039 (significant); functional -
  magnitude -0.17 pp 5+/5, p 1.000 (null). The registry's top-k overlap of magnitude vs random is
  0.5748 (degeneracy threshold 0.8).

## B.12 RD: confirmation of the sign inversion by replay selector (CIFAR-replay)

- **Setting.** CIFAR-replay: split_cifar100 only. The runner executes the fixed full matrix of 6
  methods (SWEEP_METHODS) and accepts no filter. Hyperparameters are identical to the exploratory
  sweep: beta = 30, budget = 0.25, and the config_payload recorded in sweep_report.json for
  split_cifar100. Device cuda (1 Pascal GPU) for all 20 seeds, without mixing (see Declared
  deviations).
- **Motivation.** In an exploratory 50-seed sweep (self-declared
  exploratory_not_independent_confirmation), the sign of slowheat_derpp - derpp on FAA inverted with
  the memory selector: loss +0.0035 (44+/6-), representative -0.0050 (7+/43-).
- **Question.** On split_cifar100, does the contrast slowheat_derpp - derpp on final average
  accuracy have OPPOSITE sign under the loss selector and under the representative selector?
- **Hypothesis.** Under loss the per-seed paired difference is positive in most seeds, and under
  representative it is negative in most seeds. Null: at least one of the two contrasts does not
  differ from zero, or both have the same sign.
- **Arms.** Confirmatory contrast slowheat_derpp_hidden_beta_30_budget_0.25 - derpp. The other
  methods of the matrix are computed and reported, but only this contrast is confirmatory.
  Selectors: loss, representative and `first`; `first` is required by the runner as the reference of
  paired_differences_vs_first, and its contrasts are exploratory, outside the family.
- **Seeds.** 20, band 3_000_003, 3_025_011, 3_050_017, 3_075_037, 3_100_043, 3_125_059, 3_150_061,
  3_175_087, 3_200_089, 3_225_097, 3_250_101, 3_275_103, 3_300_119, 3_325_127, 3_350_133, 3_375_149,
  3_400_151, 3_425_163, 3_450_169, 3_475_179.
- **Primary endpoint.** final_average_accuracy, contrast slowheat_derpp - derpp, per selector.
- **Test and decision rule.** Exact two-sided sign test over the 20 per-seed paired differences, one
  per selector, and Holm over the family of 2 (loss, representative), alpha 0.05. Success: p_Holm <
  0.05 for both AND opposite signs of the medians. There is no interim analysis, no early stopping
  and no seed-by-seed inspection before all 20 seeds finish. No outcome authorizes changing the
  dataset, selector or endpoint.
- **Integrity gates.** Mandatory verification before the run: the 20 seeds collide with no band
  already used, including the 50 seeds of the exploratory sweep, which are checked as a set; the
  runner's seed guard-rail accepts them; DERPP_SELECTOR_CONFIRMATORY_SEEDS exists in code and
  matches; the test suite is green; the file is committed before the first seed.
- **Secondary endpoints.** average_forgetting, backward_transfer, wall time and replay bytes, always
  reported and never promoted. Editorial rule: average_forgetting is always reported next to
  final_average_accuracy.
- **Declared deviations and amendments.** Two change-log entries dated 2026-09-28, both before any
  seed and before any analysis:
  - selector `first` added and the confirmatory contrast reworded, because the runner runs the fixed
    6-method matrix; the family stays at 2;
  - device changed to cuda and the cost section rewritten. A CPU attempt was abandoned at 19 min;
    its partial tree is preserved and was not analyzed.
  - Statistical criteria, seeds, contrast and family are unchanged.
- **Declared limitations.** The design does not isolate the method: every ranked learner selects its
  own memory, so the contrast is full algorithm vs full algorithm. One dataset and one pair of
  methods. It does not explain the inversion, and the differences are of the order of thousandths.
- **Protocol commit.** b12ba31 (first committed 2026-09-28),
  goals/protocol_replay_selector_confirmation.md.
- **Outcome.** FAILED. slowheat_derpp - derpp FAA: high-loss selector +0.20 pp (median +0.23),
  14/20, exact sign p 0.115, p_Holm 0.115; representative selector -0.61 pp (median -0.53), 1/20,
  p_Holm 8.0e-5. Median signs are opposite but only one contrast is significant, so the success
  rule is not met. No analysis of the primary had been persisted by the run; these values were
  computed afterwards from the per-seed results with the protocol's rule (R5, R8, R9) and are
  listed as such in Appendix A. The selector-vs-reference contrasts reported in Section 5.5 are
  exploratory (R2).

## B.13 PEN2: penalty pass 2, does MAS survive a matched control?

- **Setting.** Split-MNIST, MLP (256, 128), class-IL, CPU. Plain SGD, without momentum or weight
  decay, base lr 3e-3 (the same as the earlier lambda sweep L3). This host is not one of the named
  settings of Section 4.1.
- **Motivation.** Of the original 6 comparisons (3 penalty methods x 2 hosts), only one with power
  remained after passes 1 (AdamW, no control constructible), L2 (SGD) and L3 (lambda sweep): mas at
  30x, lr_scale = 0.854. This is the first pass of its phase that reads accuracy.
- **Question.** MAS at 30 times its published strength, under plain SGD, reduces the update norm by
  14.6%. Does that reduction buy anything in forgetting beyond what the mere loss of plasticity
  explains?
- **Arms.**
  - vanilla: sequential fine-tuning without a penalty, E 1.0.
  - mas: mas_lambda = 30 (30x Hsu et al.), E 0.8541 (measured in L3).
  - lr_control: no penalty, lr x 0.8541056558955461, E 0.8541 by construction.
  - The scale is a single scalar (the mean of the 10 L3 seeds, read before any accuracy of this
    pass; min 0.8286, max 0.8795, sd 0.0161), not one per seed. It is pinned in
    tests/test_penalty_pass2.py and may not be recomputed.
- **Seeds.** New band 9.500.011+, 12 seeds: 9500011, 9525013, 9550037, 9575041, 9600053, 9625061,
  9650069, 9675077, 9700081, 9725089, 9750097, 9775103. The band is disjoint from 700.001+,
  4.000.003+, 6.000.003+, 7.000.003+, 8.000.011+ and 9.000.011+, which was verified by test.
- **Primary endpoint.** Mean forgetting (average_forgetting), paired difference per seed, contrast
  mas - lr_control.
- **Test and decision rule.** Exact two-sided sign test over the 12 paired differences, threshold p
  < 0.05; the floor at n=12 is 0.00049. The family has 1 comparison and no Holm; the protocol
  declares this explicitly so that nobody applies it later to loosen the test. Predictions declared
  before the run: Q1, declared null on purpose: mas - lr_control on forgetting is not significant
  (p >= 0.05); Q2: lr_control - vanilla reduces forgetting; Q3: mas at 30x has lower final average
  accuracy than vanilla.
- **Integrity gates.** No run-time gates are declared. The pre-run test checks are: lr_scale pinned
  in a test, seed disjointness verified by test, and hyperparameters identical to L3 verified by
  test.
- **Secondary endpoints.** mas - vanilla, lr_control - vanilla, and the final average accuracy of
  the three arms, which must appear in the main table next to forgetting.
- **Declared deviations and amendments.** For execution, the protocol replaces step 3 of section C.1
  of goals/protocol_penalty_reevaluation.md. No change after freezing (single change-log entry,
  frozen 2026-10-05, before any seed of the band).
- **Declared limitations.** The run says nothing about published MAS (lambda 1), EWC, SI or AdamW.
  Matching is exact on average and approximate per seed (the E of mas ranges from 0.829 to 0.879
  across seeds). E was measured in L3, not in this pass. One host, one lambda, one optimizer, one
  lr. 12 seeds detect large effects only: 9 of 12 seeds gives p = 0.146.
- **Protocol commit.** 22461f5.
- **Outcome.** mas - lr control forgetting -0.0400, 12/12, p 0.00049; final acc. .149/.198/.160,
  chance .10. Appendix A gives no status label. Under the declared threshold (p < 0.05) the primary
  contrast is significant, so prediction Q1 is not supported.

## B.14 T3c: selection vs fixed random units of the same shape (MLP-S, MLP-P)

- **Setting.** MLP-S and MLP-P on CPU. Runner scripts/run_t3c_random_mlp.py --setting
  split|permuted. MLP-S: FROZEN_CONFIG of experiments/confirmatory_split_mnist.py, 16 arms in the
  same process; MLP-P: suite_config("permuted_mnist", "mlp", "cpu") of experiments/hard_vs_soft.py.
- **Question.** Does selection beat a fixed random set of units with the same shape (M - R) in MLP-S
  (4 baselines) and MLP-P (hard)?
- **Arms.**
  - MLP-S: for B in {replay, derpp, replay_calibrated, er_ace}, the arms B, method + B, B + uniform
    and B + random_fixed (names *_hidden_beta_30_budget_0.25[_calibrated]).
  - MLP-P: vanilla, hard_freeze, uniform_hard_freeze, random_fixed_hard_freeze.
  - Each random_fixed_* arm mirrors its method arm's MethodSpec field by field. After each
    consolidation, each layer's heat values are sorted and assigned to the units of ONE random order
    per layer, drawn once per run with seed (seed + 500000) * 1009 + layer index (the same offset as
    F16 and 3c). The protected units are order[:k] at every boundary.
  - Each boundary records the per-layer digest of the protected indices, the per-layer count, and
    the overlap with the arm's own selection.
- **Seeds.** New bands, with zero occurrences in the six repositories checked:
  - MLP-S: 19.000.033 + 8.209·k, k = 0..19 (i.e. 19000033 + 8209·k, 20 seeds);
  - MLP-P: 19.500.017 + 8.317·k, k = 0..9 (i.e. 19500017 + 8317·k, 10 seeds).
- **Primary endpoint.** Selection = FAA(M) - FAA(R), paired by seed: final_average_accuracy on the
  test set at the end of the last task.
- **Test and decision rule.** A family of 5 contrasts (MLP-S replay, derpp, replay_calibrated and
  er_ace, plus MLP-P hard): two-sided paired t test, Holm over the 5, alpha 0.05. CONFIRMED for a
  contrast: mean > 0 and Holm p < 0.05. REFUTED: the 95% CI includes 0 or the mean is <= 0. If the
  whole CI lies within +-0.5 pp, the contrast is reported as a bounded null. No joint reading is
  declared; each contrast is a cell of the table and is reported as it comes out, including when
  refuted. Analysis script: scripts/analyze_t3c.py, written before the run. It was validated by
  scripts/validate_t3c_on_f11.py, which reproduces F11 M - U (+0.62 / +1.91 / +1.49 / +0.39 pp,
  17/19/20/15 of 20); the whole pipeline also ran on the smoke data.
- **Integrity gates.** Read BEFORE the primary endpoint; a failure invalidates and is not
  reinterpreted:
  - G1: in every random_fixed arm and seed, one reassignment per consolidation, with each layer's
    digest equal to that of the previous boundary whenever the protected count did not change.
  - G2: mean protected fraction of R equal to that of M at every boundary, tolerance 0.01.
  - G3: no boundary where R's overlap with its own selection is >= 0.999.
  - G4: all arms present in all seeds. No seed is dropped; a failed run is re-run with the same seed
    and the fact is reported.
- **Secondary endpoints.** Descriptive, uncorrected, all reported and none promoted: shape R - U;
  plasticity U - B; M - U (replication of F11/F8 on a new band); M - B; forgetting M - R; sign; mean
  overlap of R with its own selection.
- **Declared deviations and amendments.** not declared.
- **Protocol commit.** ac20479.
- **Outcome.** Run in progress at the time of writing.

## Unresolved

1. **Protocol file paths.** Not recorded in Appendix A or in the protocol texts for F12, F14, F15,
   F16, PM, RC, PEN2 and T3c; only commit hashes are given.
2. **RD commit.** Appendix A and Section 4.3 still carry "[TODO: commit hash]" for RD. This appendix
   uses b12ba31 (first commit of the protocol, 2026-09-28), as supplied with the inputs; Appendix A
   and Section 4.3 should be updated to match.
3. **RD outcome vs protocol.** [RESOLVED 11/11: primary computed with the protocol rule, FAILED; Section 5.5 and Appendix A corrected.] The protocol's single confirmatory contrast is slowheat_derpp - derpp
   per selector (loss, representative), with Holm over a family of 2 and success defined as both
   p_Holm < 0.05 with opposite median signs. Appendix A instead records selector vs no-selection
   contrasts for 4 learners, and Section 4.3 describes RD as "selectors vs the same learner without
   selection, per learner (Holm per family)". The value of the pre-registered contrast is not in
   Appendix A, so the RD outcome is listed as "see Appendix D". The report's
   exploratory_not_independent_confirmation label for cross-learner contrasts would also cover the
   protocol's own confirmatory contrast, which is cross-learner (slowheat_derpp vs derpp).
4. **RD internal cost figures.** Section F gives both 5.3 and 5.9 min per seed (cost only).
5. **PM success criterion.** P6 declares Holm over the family of 2 at alpha 0.05, while P9 and
   section I require p < 0.025 on the primary. The protocol does not say whether that p is raw or
   Holm-adjusted. The Appendix A value (p_Holm 0.0039, 10/10) satisfies either reading.
6. **PEN2 outcome.** Appendix A gives no status label for PEN2, and the final accuracies
   ".149/.198/.160" are listed without arm order. The status written above (significant under p <
   0.05; Q1 not supported) is read off the declared rule, not taken from a label. PEN2 and T3c are
   also absent from the families table in Section 4.3, and the PEN2 host (MLP (256, 128), plain SGD,
   no replay) is not one of the Section 4.1 settings.
7. **RARB host vs Section 4.1.** RARB fixes batch_size = 2 (diagnostic_conditions()), while Section
   4.1 describes BERT-crit as "same BERT host as above", whose BERT row says batch 32. The protocol
   declares the overlap diagnostic as a Jaccard index; Appendix A reports a "top-k overlap" of
   0.5748 (source criterion_degeneracy_overlap.md). These inputs do not show whether the two are the
   same metric.
8. **"All arms of a seed share a process and a device" (Section 4.1).** F13 contradicts this: its
   arms are separate runs, with no guarantee of the same GPU card. 3b also contradicts it (the
   lr_control arms run in separate chained processes), and so does 3c (the new arm runs in its own
   process and is paired with 3b arms run earlier).
9. **Integrity checks claimed in Section 4.2.** Section 4.2 lists integrity checks that run before
   any endpoint. F11, F12, F13, F14 and RC declare no integrity gates, and PEN2 and RD declare only
   pre-run checks.
10. **Fields the protocols do not determine.**
    - F11 does not state soft vs hard or the MLP architecture, so the Section 4.1 TODO for MLP-S
      cannot be resolved from the protocol.
    - RC does not restate its Qwen-LoRA configuration (rank, lr, epochs). The "r = 16" in Section
      4.1 comes from PM.
    - T3c gives its seed bands only as formulas, not as explicit lists.
    - Whether any seed crashed and was re-run in PM and RC (the Section 4.2 TODO) is not recorded in
      these inputs.
11. **Seed counts.** No discrepancy was found between the protocols and Appendix A: F11 20, F12 20
    per pair, F13 10, F14 10, F15 10, F16 10, PM 10, RC 10, 3b 10, 3c 10, RARB 10, RD 20, PEN2 12.
    The sign counts in Appendix A are consistent with these n.

