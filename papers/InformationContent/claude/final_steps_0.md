# Final Steps — Phase 0: Verification Checks

Prompt doc for executing **Phase 0** of `papers/InformationContent/final_steps.md`.
Created 2026-08-02 by Claude Fable 5.

## Before you start

1. Read `papers/InformationContent/claude/Claude.md` and follow its conventions.
2. Read `papers/InformationContent/final_steps.md` (the master plan) and the
   Overleaf `ToDO.txt` (`/home/xavier/Projects/Overleaf/Info_content/ToDO.txt`).
3. Use the `ocean14` conda environment for all Python. Safe bash commands may
   run without prompting. Use multiple Claude **Fable** agents where parallel
   work helps (the five checks below are independent).

## Prompt

Execute the five Phase 0 verification checks. These are read/verify tasks on
local data and code — no retraining, no Nautilus. Their outcome decides
whether Phase 1 (Nautilus compute) is needed at all.

1. **LLC cutout provenance** (ToDO E, P1, orig). Trace where the LLC SST
   cutouts in the preproc files come from: start at
   `Analysis/py/info_defs.py::grab_paths()` for LLC_SSTa_nonoise /
   LLC_SSTa_noise / LLC_SSHa, then `Analysis/py/extract_llc.py` and the
   wrangler calls it makes. Record the source (LLC4320 run, region set, dates,
   any subsetting) so it can be stated in the dataset table / Methods text.

2. **LLC first-2-months (spin-up) check** (ToDO E, P2, orig). Determine
   whether the LLC preproc files include cutouts from the first 2 months of
   the LLC4320 run. Check the date metadata in the preproc `.h5` files (paths
   from `grab_paths()`, env vars `OS_OGCM` etc.) and/or the extraction date
   lists in `extract_llc.py`. **This is the Phase 1 gate**: if spin-up dates
   are present, Phase 1 must run; if not, document the date range and mark
   the ToDO item done.

3. **P2(k)/P4(k) zero-mean claim** (ToDO E, P2). The manuscript (~line 276 of
   `information_content.tex`) claims the generated Pk2/Pk4 images have zero
   mean and unit std per image. Verify this two ways: read the generation
   code (`Preprocess/py/power_spectrum_images.py`, `build_Pk_images.py`) and
   directly compute per-image mean/std on a sample from the Pk2/Pk4 preproc
   files. Fix the text or flag the code, whichever is wrong.

4. **SWOT_L2 audit** (ToDO H, P1; decision: SWOT_L2 is canonical). Grep
   `information_content.tex`, `Tables/py/tables_info.py`, the three
   `Tables/*.tex`, and `Figures/py/figs_nenya_dim.py` for SWOT_L3 usage.
   List every place a paper figure/table/number is built from SWOT_L3, and
   what must be regenerated from SWOT_L2 (`pca_latents_SWOT_L2.npz` and
   `Pk_SWOT_L2.npz` already exist in `Analysis/pca/`, `Analysis/Pk/`).
   Hand the regeneration list to Phase 2 — do not regenerate here.

5. **dx consistency** (ToDO E, P3). Reconcile the MODIS dx in the text
   ("~XX km", ~line 290) with `tab_datasets` (1.10 km); confirm VIIRS
   (0.75 km) matches. Report the correct value to fill in Phase 3.

## Deliverables

- A findings report written to `papers/InformationContent/claude/phase0_findings.md`
  with one section per check, each ending in a clear verdict
  (OK / needs-fix / triggers-Phase-1), plus the SWOT_L3→L2 regeneration list.
- An explicit **GO / NO-GO decision for Phase 1** at the top of the report.
- Tick the completed Phase 0 boxes in `final_steps.md`; update the matching
  ToDO.txt items ([~] or done) and push the Overleaf repo
  (token: `$OVER_TOKEN` in `~/.bashrc`; never print it).
- Log the work in the Logs section of `paper_writing.md`; commit and push the
  nenya repo (branch `info_content`).
