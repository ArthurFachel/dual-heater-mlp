# ICML 2027 draft: plasticity-matched decomposition

Status: first full draft of the body, written 2026-11-10 by Venus from the frozen
analyses. Not yet reviewed by Fachel. Target: ICML 2027 (8 pages + refs; CFP not
published as of 2026-11-10, deadline estimated ~22/01/2027).

Working title: "How Much of Continual Learning Is Learning Less?
A Plasticity-Matched Decomposition"

## Files (read in order)

| file | section | state |
|---|---|---|
| 01_abstract_intro.md | Abstract, 1 Introduction | draft |
| 02_decomposition.md | 2 Effective plasticity and the three-term decomposition | draft |
| 03_setup.md | 3 Hosts, arms and pre-registration | draft |
| 04_results.md | 4 Results (the decomposition table and per-term evidence) | draft |
| 05_case_study.md | 5 How our own confirmed result was wrong | draft |
| 06_negative_related_limits.md | 6 Negative results, 7 Related work, 8 Limitations | draft |
| A_registry.md | Appendix A: every number, its status and its source | draft |

## Rules used while writing

- Every number comes from a pre-registered analysis output or an exploratory
  result file; the source is in Appendix A. Nothing was recomputed by hand except
  the sums/differences marked "derived" in Appendix A.
- C = confirmatory (protocol frozen in a commit before the run, fresh seeds).
  E = exploratory. The tables carry the tag in every cell.
- Refuted confirmatory results (F16) are reported as refuted.

## Open decisions for Fachel (block the final version)

1. Non-overlap rule with the IJCNN companion. R-A/R-B are IJCNN body. This draft
   uses the BERT hard exploratory decomposition (uniform_lr_control, 03/11), which
   reuses the R-A arms plus a uniform arm. Is that a "table line citing the
   companion" or does it cross the rule?
2. Missing confirmatory cells. To make Table 1 confirmatory end to end:
   (a) random arm confirmatory in NCP (F3 is exploratory; ~1 h/run GPU);
   (b) random arm in MLP Split-MNIST (CPU, minutes);
   (c) vanilla arm in NCP (the plasticity term is not measured there);
   (d) Qwen selection with power (F17 proposal, 20 seeds).
3. One external method decomposed (UPGD, ICLR 2024) so the paper is not only
   about our method. Cheap in the MLP.
4. Retention floor in Qwen (4.8 to 9.0% absolute). Report as limitation, or
   re-measure with fewer tasks.

## Placeholders

Search for `[CITE` (missing bibliographic entries) and `[TODO` (content still
owed). No placeholder hides a number.
