# ICML 2027 draft: method + plasticity-matched decomposition

Status: second structure, 2026-11-11 (Venus). Decision of 11/11 with the advisor:
one paper only, ICML 2027. The IJCNN companion is dropped; its results (R-A, R-B,
R-D) are now part of this paper. Not yet reviewed by Fachel or the advisor.

Venue (read on icml.cc 2026-11-11): abstract 14/01/2027 AoE (locks the author
list), full paper 22/01/2027 AoE, Rio de Janeiro. CFP and template for 2027 not
yet published; 2026 rules assumed: 8 pages of body, unlimited references,
impact statement and appendices, one file, double-blind.

Working title: "Importance-Masked Updates for Continual Learning, and How Much of
Their Gain Is Learning Less". Alternative kept: "How Much of Continual Learning
Is Learning Less? A Plasticity-Matched Decomposition".

Double-blind: the LaTeX uses a \method macro with a neutral name; "SlowHeat"
only in the camera-ready. The public GitHub repo must not be referenced.

## Files (read in order)

| file | section | body pages (target) |
|---|---|---|
| 01_abstract_intro.md | Abstract, 1 Introduction | 1.0 |
| 02_method.md | 2 The method | 1.25 |
| 03_controls_decomposition.md | 3 Effective plasticity, controls, decomposition | 0.75 |
| 04_setup.md | 4 Settings, arms, pre-registration | 0.75 |
| 05_results.md | 5.1 method vs matched, 5.2 Table 1, 5.3 signs, 5.4 selection, 5.5 criterion, 5.6 rankings, 5.7 penalties | 2.75 |
| 06_case_study.md | 6 Case study | 0.75 |
| 07_negative_related_limits.md | 7 Negative results, 8 Related work, 9 Limitations | 0.75 |
| A_registry.md | Appendix A: every number, status, source | - |
| APPENDIX_PLAN.md | Appendices B-K: content and sources | - |

Total body target 8.0. Current markdown is over budget in 5 and 6; cuts are
marked in place and move to the appendix.

## Rules used while writing

- Every number comes from a pre-registered analysis output or an exploratory
  result file; the source is in Appendix A. Nothing recomputed by hand except
  sums/differences marked "derived".
- C = confirmatory primary; S = pre-registered secondary; c = descriptive from a
  confirmatory run; E = exploratory.
- Refuted confirmatory results (F16) are reported as refuted.

## Open items (block the final version)

1. Missing cells of Table 1 (plan T3): random arm in MLP-S/MLP-P (CPU);
   random + vanilla arms in LM-410M paired to F13 seeds; F17 (20 seeds, new
   band, NOT 500-519: 500-509 were used by P1).
2. External methods decomposed (plan T4): UPGD in MLP-S; EWC/SI/MAS from the
   existing penalty runs.
3. R-D: protocol commit hash and backbone/width to fill in; decide body
   paragraph vs appendix only.
4. Retention floor in Qwen-FT (4.8 to 9.0%): limitation, or a 5-task
   calibration in the appendix.

## Placeholders

Search for `[CITE` and `[TODO`. No placeholder hides a number.

Change log
- cbeae61: first full draft (decomposition-only framing).
- f88d508: 3b added.
- this commit: one-paper framing; new 02_method; sections renumbered; R-A/R-B/R-D
  folded in (5.5); 3c (C3g) result added (generative selection null, bounded);
  generative row of Table 1 now complete on shared seeds; Limitations fixed
  (SuperNI is in the draft); "five hosts" replaced by four model families,
  seven settings; companion-paper references removed.
