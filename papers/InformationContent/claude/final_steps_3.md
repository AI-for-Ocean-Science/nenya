# Final Steps — Phase 3: Manuscript Text

Prompt doc for executing **Phase 3** of `papers/InformationContent/final_steps.md`.
Created 2026-08-02 by Claude Fable 5.

## Before you start

1. Read `papers/InformationContent/claude/Claude.md`,
   `papers/InformationContent/final_steps.md`, `phase0_findings.md`, and
   `phase1_report.md` (if Phase 1 ran — it may have changed the LLC numbers
   used below). Phase 2 must be complete: the text you write references its
   finished figures/tables.
2. Manuscript: `/home/xavier/Projects/Overleaf/Info_content/information_content.tex`.
   Source material: `claude_brainstorming.tex` (literature + DOIs),
   `claude_initial_review.tex`, `Tables/*.tex`.
3. Be critical: where results are ambiguous, say so in the draft with a
   clearly-marked comment for the author rather than papering over it.
   Sections may be drafted in parallel by Claude **Fable** agents, but one
   agent must own final consistency.

## Prompt

Write the missing manuscript text, in this order (Results first, Abstract
last):

1. **PCA Results** (largest gap): write the discussion for `fig_true_pca` and
   `fig_pca_2panel` (~lines 654–666) and the new eigenmodes figure from
   Phase 2 — what the eigenvalue spectra / eigenmodes say about the
   information content of each dataset class (satellite SST vs. model vs.
   reference).
2. **P(k) Results placeholders** (~lines 597–652): fill "$k \approx XX-XX$"
   and "exponents of XX" from `tab_analysis` (betas: MODIS 2.90, VIIRS 3.06,
   LLC 3.80 — use Phase 1 values if rerun); "$\sigma(T) = XX$ K" -> 0.04 K
   (the injected LLC noise, sqrt(0.0016)); the SSH "we find XXX". Resolve the
   editorial notes: "[PMC to discuss a bit further]" (~625), "[Ask Dimitris!]"
   (~642), "[is the LLC SST higher??]" (~652), and the MNIST bracketed note
   (~616) — draft text where possible, otherwise leave a clearly-marked
   %-comment question for the author.
3. **Conclusions** (currently "TEXT", ~line 670).
4. **Introduction** (currently "Blah", line 186): frame the information
   content of remote-sensing imagery, self-supervised/contrastive learning,
   and the motivation for comparing diverse datasets in one latent space;
   pull motivation and citations from `claude_brainstorming.tex`.
5. **Abstract** (currently "TEXT", line 178) — draft LAST, once Results and
   Conclusions are settled. Journal is still open, so keep it ~200 words
   (safe for Copernicus).
6. **Bibliography** (ToDO B, P1): replace the stub `thebibliography`
   (~lines 731–739) with a `.bib` file + `\bibliographystyle{copernicus}`
   (journal still open — bst swap deferred). Resolve every undefined key:
   llc, nenya, viirs, ecco, gallmeier2023, ulmo, pae, nenya_doi,
   field1987/Field1987 (unify capitalization), archer2025, llc_res. Fix the
   `\cite{gallmeiter2023}` typo (~line 364). Import the references + DOIs
   already collected in `claude_brainstorming.tex`.
7. **Macros** (lines 96–98, 126): populate \nocean, \ndataset, \nother with
   real dataset counts (from `info_defs.py` / `tab_datasets`) and \powmnist
   with the measured MNIST power-law exponent, or delete them if unused.
8. **Back matter** (all currently "TEXT"): codeavailability /
   dataavailability / codedataavailability (nenya is open source — cite the
   nenya DOI), authorcontribution, competinginterests, acknowledgements.
   Draft minimally and mark for author review.

**Do NOT touch** (blocked/deferred): the SWOT dataset description (~lines
331–333, waiting on Iury); PR/RankMe integration (on hold); journal-specific
formatting beyond the current Copernicus template.

## Deliverables

- All placeholder sections replaced with draft text; author-decision points
  marked with `% CLAUDE-QUESTION:` comments.
- A real `.bib` file; zero undefined citations on compile (full clean compile
  is Phase 4).
- Tick Phase 3 boxes in `final_steps.md`; update ToDO.txt Sections A/B and
  push Overleaf (`$OVER_TOKEN`).
- Log in `paper_writing.md`; commit and push nenya (`info_content`).
