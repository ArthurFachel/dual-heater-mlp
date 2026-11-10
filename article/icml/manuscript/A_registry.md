# Appendix A: number registry

Every number in the draft, its status and the file it comes from. Paths are on
the Lab (/mnt/B-SSD/fachel/...) unless they start with ~/ (laptop).
"derived" = arithmetic on means printed in the source, done for this draft.

## Confirmatory (protocol frozen before the run)

| claim | value | source |
|---|---|---|
| F11 M - U, FAA, 4 baselines | replay +0.62 [+0.25, +0.98] 17/20; DER++ +1.91 [+1.39, +2.43] 19/20; replay cal. +1.49 [+1.21, +1.78] 20/20; ER-ACE +0.39 [+0.15, +0.64] 15/20; Holm <= 4.3e-3 | ~/Desktop/dual-heat-direcoes/followups/F11_confirm_uniform.txt (copy of dh-uniform-ctrl results/confirm_uniform_control/analysis_preregistered.txt); protocol docs/protocols/confirmatory_uniform_control.md @ e7a2ece |
| F11 means (B / SH / UN FAA) | replay .7586/.7706/.7644; DER++ .8244/.8586/.8395; cal. .8224/.8479/.8329; ER-ACE .7109/.7083/.7044 | same file, descriptive lines |
| U - B from F11 (derived) | +0.58, +1.51, +1.05, -0.65 pp | derived from the means above |
| F12 M - U, T1 retention, 5 pairs | +4.07, +4.40, +3.20, +2.93, +5.20 pp; 19-20/20; Holm <= 1.0e-6 | dh-conc2x2 docs/results/confirmatory_bert_soft30_results.md; protocol @ 77ebece |
| F12 U - V retention (c) | +7.47, +6.97, +5.48, +5.52, +12.10 pp | same file, secondaries |
| F12 FAA SH - UN (c) | +0.84 to +2.40 pp | same file |
| F13 M - U MiniPile loss | -0.00764 nat [-0.00785, -0.00743], 10/10, p 3.5e-14; uniform forgetting +0.07111 nat; -10.7% | ~/Desktop/dual-heat-direcoes/followups/F13_confirm_ncp_ffn_soft30.txt; ncp docs/protocols/confirmatory_ffn_soft30.md @ f8526c8 |
| F14 M - U retention | +5.67 pp [+3.00, +8.33] 9/10, p 9.7e-4; c: U - V +1.60 [+0.29, +2.91] 8/10; M - V +7.27; last-task acq. -0.10 [-1.99, +1.79]; mean ret. hard .0900 | dh-qwen-f14 docs/results/confirmatory_qwen_hard_results.md; protocol @ c495d14 |
| F15 M - R(re-drawn) | +4.10 pp [+0.61, +7.59] 9/10, p 0.026; c: R - U +1.97 [+0.23, +3.71] 5/10; M - U +6.07; mean ret. hard .0793 | dh-qwen-f14 docs/results/confirmatory_qwen_hard_permuted_results.md; protocol @ dab50ed |
| F16 M - R(fixed) | +1.50 pp [-0.07, +3.07] 8/10, p 0.060, REFUTED; c: fixed - re-drawn +0.30 [-1.06, +1.66]; M - re-drawn +1.80 [+0.38, +3.22] 9/10; mean ret. hard .0480 | dh-qwen-f14 docs/results/confirmatory_qwen_hard_random_fixed_results.md; protocol @ cc81913 |
| PM frozen_A - lr control | forgetting -0.1260, FAA +9.92 pp, 10/10, p_Holm 0.0039; c: lr control - vanilla forgetting +0.0616 9/10 p 0.0215; frozen_A - vanilla -0.0644 p 0.109; table of absolutes | dh-qwen-f14 docs/results/plasticity_matched_results.md; protocol @ d9b57fc |
| U - V FAA in Qwen-LoRA (derived) | 0.5275 - 0.5879 = -6.04 pp | derived from PM absolutes |
| RC exact decomposition | frozen_A - vanilla -0.1118 (1+/9-, p_Holm 0.043); exact - frozen_A -0.0092 (5+/5-, p 1.000); FAA +0.69 pp; 611 s vs 336 s | dh-qwen-f14 docs/results/exact_decomposition_results.md; protocol @ f5d77ce |
| R-E original claim | exact - lr control forgetting -0.0921, 9-/1+, p 0.02148; E 0.532 vs 0.850; A = 2,555,904 params | plasticity_matched_results.md, exact_decomposition_results.md |
| R-A / R-B (companion) | magnitude - random +12.23 pp 10/10 p_Holm 0.0039; functional - magnitude -0.17 pp 5+/5 | dh-qwen-f14 docs/results/criterion_ablation_results.md; protocol @ a3f5117 |
| Penalty pass 2 | mas - lr control forgetting -0.0400, 12/12, p 0.00049; final acc. .149/.198/.160, chance .10 | dh-qwen-f14 docs/results/penalty_pass2_results.md; protocol @ 22461f5 |

## Exploratory

| claim | value | source |
|---|---|---|
| BERT hard decomposition | ret. V .0983, R .3433, U .4027, M .4417; M - R +9.83 10/10; U - R +5.93 10/10; M - U +3.90 10/10; FAA M - U +0.65 6/4 | dh-conc2x2 docs/results/uniform_lr_control_exploratory.md (band 13.000.003+) |
| BERT U - V (derived) | +30.44 pp | derived from the means above |
| F3 LM-410M | SH - random -0.00339 nat 6/6; random - uniform -0.00436 nat 6/6 | ~/Desktop/dual-heat-direcoes/followups/F2_F3_ncp_ffn.txt |
| F7 MLP-S | DER++: U - B +1.40 [+0.95, +1.86] 19/20; cal.: +0.84 [+0.49, +1.19] 18/20 | ~/Desktop/dual-heat-direcoes/followups/F7_F8_uniform_baselines_permuted.txt |
| F8 MLP-P | hard - U +13.47 [+11.57, +15.36] 10/10; U - V +3.02 [+1.41, +4.63] 9/10 | same file |
| LoRA rank table | exact/slice/vanilla/rank/leak FAA and E, matched and unmatched | dh-qwen-f14 docs/results/compendium_accuracy_forgetting.md section 12 |
| Penalty E under AdamW, CIFAR-100 | EWC 1.00128, SI 1.01238, MAS 1.01939, 12/12 > 1; MAS cosine min -0.8758 | compendium section 15.1 |
| Penalty E under SGD | 0.955 / 0.976 / 0.993 at lr 3e-3 | goals/roadmap_icml_ijcnn.md item 2.11; docs/results/sgd_plasticity_results.md [TODO: read the table directly] |
| Floor effect | T1 retention 0.0000 in 4 arms at T = 5; FAA 13-14x chance; T = 2 spread 0.354 | docs/results/retention_floor_effect.md |
| Hard vs soft CIFAR-100 MLP | forgetting -11.68 pp, FAA -1.41 pp (.0373 vs .0514), 10/10 | compendium section 6 |
| P1 mask timing | freeze1 -1.31, freeze3 -1.30, warm-up -2.90 pp FAA | ~/Desktop/dual-heat-direcoes/fila.md P1; dh-qwen-f14 docs/results/pilot_mask_timing_results.md |
