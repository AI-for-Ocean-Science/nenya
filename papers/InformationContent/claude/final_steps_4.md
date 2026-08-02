# Final Steps — Phase 4: Final Pass

Prompt doc for executing **Phase 4** of `papers/InformationContent/final_steps.md`.
Created 2026-08-02 by Claude Fable 5.

## Before you start

1. Read `papers/InformationContent/claude/Claude.md` and
   `papers/InformationContent/final_steps.md`. Phases 0–3 must be complete
   (Phase 1 may be marked skipped).
2. Work in the Overleaf checkout `/home/xavier/Projects/Overleaf/Info_content`.

## Prompt

Bring the manuscript to a clean, internally-consistent state and hand it back
to the author:

1. **Clean compile** of `information_content.tex` (pdflatex + bibtex cycle,
   or latexmk if available). Fix every error; then eliminate all warnings for
   undefined references, undefined citations, and multiply-defined labels.
   Note (don't chase) purely cosmetic overfull-hbox warnings.
2. **Cross-reference audit**: every figure and table is `\ref`'d at least
   once in the text, and every `\ref`/`\cite` resolves. Every float sits near
   its first mention (per the paper conventions).
3. **Consistency read-through**, checking specifically:
   - SWOT_L2 naming and numbers used everywhere (no SWOT_L3 leftovers);
   - dx values consistent between text and `tab_datasets`;
   - dataset counts match the \nocean/\ndataset/\nother macros;
   - LLC numbers (beta, noise sigma) match the current `tab_analysis` /
     Phase 1 outputs;
   - no leftover placeholders: grep the .tex for "XX", "TEXT", "Blah",
     "TBD", and unmarked bracketed editorial notes.
4. **Author handoff list**: collect every remaining `% CLAUDE-QUESTION:`
   comment and blocked item (SWOT description for Iury, journal choice,
   RankMe hold) into a short section appended to ToDO.txt ("I. FINAL AUTHOR
   ITEMS") so the author sees exactly what needs a human.
5. **Close out**: update ToDO.txt statuses and the Overleaf change log
   (per Claude.md conventions), tick the Phase 4 boxes in `final_steps.md`,
   and push the Overleaf repo (`$OVER_TOKEN`).

## Deliverables

- A PDF that compiles cleanly with zero undefined refs/citations.
- ToDO.txt with an accurate final status picture and the "FINAL AUTHOR
  ITEMS" section; Overleaf pushed.
- Log in `paper_writing.md`; commit and push nenya (`info_content`).
