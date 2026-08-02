# Final Steps — Phase 2: Figures and Tables

Prompt doc for executing **Phase 2** of `papers/InformationContent/final_steps.md`.
Created 2026-08-02 by Claude Fable 5.

## Before you start

1. Read `papers/InformationContent/claude/Claude.md`,
   `papers/InformationContent/final_steps.md`, `phase0_findings.md`, and
   `phase1_report.md` if Phase 1 ran (its regeneration list supersedes any
   stale LLC numbers below).
2. Overleaf checkout: `/home/xavier/Projects/Overleaf/Info_content`
   (figures go in its `Figures/` subdir as PNG; push with `$OVER_TOKEN`).
3. Use the `ocean14` env. Reuse existing figure/table code
   (`Figures/py/figs_nenya_dim.py`, `Tables/py/tables_info.py`) — extend it,
   don't duplicate it. Parallel, independent items may go to Claude **Fable**
   agents.

## Prompt

Complete every figure and table item so that Phase 3 (text) can reference
finished float environments.

Figures:

1. **Eigenmodes figure** (code exists, PNG never generated): run
   `Figures/py/figs_nenya_dim.py` with `flg==53`
   (`fig_eigenmodes_remote_sensing()`; MODIS_SSTa, VIIRS_SSTa, SWOT_L2 from
   `pdict['pca_imgfile']`). Copy the PNG to Overleaf `Figures/`, add a
   `figure*` environment near the PCA results with a real caption, and add
   1–2 sentences of in-text description (Phase 3 expands it).
2. **Learning-curve figure** (decision: include BOTH): add a figure
   environment for `fig_learning_curves.png` (already in Overleaf `Figures/`)
   labelled `fig:learning`, and fix the architecture figure at ~lines 515–520:
   give it its own label (e.g. `fig:architecture`), write a real caption
   (currently bare "Architecture"), and repoint the text reference at
   ~line 538 to the correct labels.
3. **PCA figure captions**: expand `fig_true_pca` ("PCA on actual images")
   and `fig_pca_2panel` ("PCA on latent space") into full captions stating
   dataset, quantity plotted, and takeaway.
4. **Pk_all_datasets caption**: replace the bracketed remote-sensing
   placeholder (~line 584).
5. Evaluate `fig_pca_noise_res.png` (produced by `fig_pca_noise_res()`); if
   it adds value to the noise discussion, include it, otherwise note why not.
6. **Stray "[t]"**: while editing the figure blocks, delete the literal "[t]"
   after `\end{figure*}` at ~lines 659 and 666 (ToDO F, P1).

Tables (regenerate via `Tables/py/tables_info.py`, then copy to Overleaf
`Tables/`):

7. Fix column-count mismatches: `tab_analysis.tex` declares {cccc} but has 3
   columns; `tab_model.tex` declares {ccccccc} but has 5.
8. Add **year** and **geographic coverage** columns to `tab_datasets`
   (MODIS=2021, VIIRS/NOAA-21=2024, LLC days per the text) — update the
   generator script so the table stays reproducible.
9. Reconcile the `tab_model` footnote with its actual columns (it describes a
   "Model" column that does not exist — add the column or trim the footnote).
10. **SWOT_L2 everywhere**: apply the regeneration list from
    `phase0_findings.md` so every SWOT row/curve comes from SWOT_L2.

## Deliverables

- All PNGs and regenerated `.tex` tables in the Overleaf project; manuscript
  compiles with the new/updated floats (a full clean compile is Phase 4).
- Tick Phase 2 boxes in `final_steps.md`; update ToDO.txt Sections C/D/F/H
  and push Overleaf.
- Log in `paper_writing.md`; commit and push nenya (`info_content`) with any
  modified figure/table code.
