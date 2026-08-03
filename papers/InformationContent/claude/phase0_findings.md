# Phase 0 Findings — Verification Checks

Executed 2026-08-02 by Claude Fable 5 (three parallel Fable agents + inline
checks), per `final_steps_0.md`. All checks were read-only.

# DECISION: **GO for Phase 1**

All three LLC datasets are contaminated with LLC4320 spin-up data (~17% of
train AND valid cutouts fall in the first 2 months of the run). The LLC
pipeline must be re-run with the spin-up excluded (see Check 2).

Additionally, the SWOT audit found that **every SWOT number/curve currently
in the manuscript is SWOT_L3-sourced** — all SWOT_L2 input products already
exist, so this is Phase 2 regeneration work, not new compute (see Check 4).

---

## Check 1 — LLC cutout provenance: DOCUMENTED

Run: MITgcm **LLC4320**, start date **2011-09-13** (wrangler
`ogcm/llc.py::build_table`, `init_date='2011-09-13'`).

- **LLC_SSTa_nonoise**: 150k/50k train/valid cutouts drawn at random (no date
  cut) from the 2022 Gallmeier/Nenya-LLC parent table
  `LLC_uniform144_r0.5_nonoise.parquet` and its 46 GB extraction file
  `Nenya/PreProc/LLC_uniform144_nonoise_preproc.h5`. Uniform HEALPix
  sampling (r=0.5 deg) over cloud-free ocean; lat −56.9..+57.0; cutouts
  64×64 px = 144 km (2.25 km/pix). No inpainting, no extra preprocessing.
  Temporal: **24 timesteps every 14 days, 2011-09-18 → 2012-08-05**.
- **LLC_SSTa_noise**: exact copy of the nonoise file with per-pixel Gaussian
  noise, **sigma = 0.09 K** (`extract_llc.py` line 67). Same cutouts/dates.
  ⚠ FLAG: ToDO.txt/tab_model cite noise sigma ≈ 0.039–0.04 K; the extraction
  code says 0.09. Resolve in Phase 1/3 (see Flags).
- **LLC_SSHa**: 150k/50k SSHa cutouts via `fronts...generate_from_dbof`
  (DBOF_dev config), lat −78..+57. Temporal: **6 bimonthly timesteps,
  2011-09-30 → 2012-07-31**.
- Preproc files live under `$OS_OGCM/LLC/Info/PreProc/`
  (`train_llc_nonoise.h5`, `train_llc_noise.h5`, `LLC_random_SSHa.h5`, each
  3.28 GB); dates come from companion parquet tables (`Tables/
  train_llc_nonoise.parquet`, `LLC_random_SSHa.parquet` +
  `LLC4320_SSHa_meta.parquet`), aligned to the h5 arrays by `pp_type`/`pp_idx`.

## Check 2 — LLC spin-up contamination: **CONTAMINATED → Phase 1 GO**

Spin-up window: 2011-09-13 to 2011-11-13 (first 2 months).

| Dataset | Date range | Train in spin-up | Valid in spin-up | Verdict |
|---|---|---|---|---|
| LLC_SSTa_nonoise | 2011-09-18 → 2012-08-05 | 25,115 (16.7%) | 8,523 (17.0%) | CONTAMINATED |
| LLC_SSTa_noise | identical to nonoise | 25,115 (16.7%) | 8,523 (17.0%) | CONTAMINATED |
| LLC_SSHa | 2011-09-30 → 2012-07-31 | 25,198 (16.8%) | 8,314 (16.6%) | CONTAMINATED |

Earliest SST cutouts are only 5 days after run start. Note for Phase 1: the
SST timestep 2011-11-13 sits exactly on the 2-month boundary (dropping it
too raises the excluded fraction to ~20.8%) — Phase 1 should state the cut
convention explicitly (recommend: exclude dates < 2011-11-13, i.e. keep the
boundary timestep, or make the author's preferred call).

## Check 3 — Pk2/Pk4 zero-mean/unit-std claim: **CLAIM CORRECT**

The manuscript sentence (information_content.tex line 276) is accurate.
`Preprocess/py/power_spectrum_images.py` normalizes each image individually
(demean + divide by std, lines 74–77; DC amplitude also zeroed at line 54),
`build_Pk_images.py` runs with `normalize=True`, and a 200-image random
sample from each preproc file confirms |mean| < 2e-9, |std−1| < 3e-9
(float32 round-off). No text or code change needed. Incidental: image size
and 150k/50k split in the text also match the files.

## Check 4 — SWOT_L2 audit: **ALL manuscript SWOT content is L3 → regenerate**

Dating evidence: Overleaf figures/tables were committed 2026-01-29; the code
switched to SWOT_L2 later (nenya commits 2026-03-24, 2026-04-06). Verified
numerically (L3: 128 px, dx 0.25, N99=89, beta=1.42±0.11, f_var256=0.801;
L2: 64 px, dx 0.843, N99=132, beta=2.58±0.31, f_var256=0.985).

SWOT_L3-sourced items in the paper:
- `tab_datasets.tex:16`, `tab_model.tex:16`, `tab_analysis.tex:16` (all
  SWOT rows).
- PNGs: `fig_example_images.png`, `Pk_all_datasets.png`, `fig_true_pca.png`,
  `fig_pca_2panel.png` (+ `fig_learning_curves.png`, present but not yet
  included).
- Code still hard-coded to L3: `figs_nenya_dim.py` lines 728–729, 751
  (`fig_example_images`, flg 1) — needs SWOT_L3→SWOT_L2, title 'SWOT L2',
  scale bar `0.843*64 ≈ 54 km`.

### Regeneration list for Phase 2 (all inputs exist unless noted)

| Deliverable | How | Note |
|---|---|---|
| fig_example_images.png | figs_nenya_dim.py flg 1 | edit lines 728/729/751 first |
| fig_learning_curves.png | flg 2 | ready |
| Pk_all_datasets.png | flg 3 | ready |
| fig_true_pca.png | flg 4 | ready |
| fig_pca_2panel.png | flg 5 | ready |
| tab_datasets.tex | tables_info.mktab_datasets() | `__main__` only calls mktab_model — invoke all three |
| tab_model.tex | tables_info.mktab_model() | ready |
| tab_analysis.tex | tables_info.mktab_analysis() | ready |

Expected new SWOT row: npix 64, dx 0.84, "crop 54, jitter 8, flip, rotate",
N99 = 132, beta = 2.58 ± 0.31, f_var256 = 0.985.

Missing-but-not-blocking products: `pca_preproc_SWOT_L2_4096.npz` (only for
the extended true-PCA remote panel; current paper call is one_panel=True)
and `SWOT_L2_eigenimages.npz` (only for fig_eigenimages, not a SWOT paper
deliverable). Recipes are in the audit flags below if ever needed.

## Check 5 — dx consistency: **NEEDS FIXES (text and table)**

- MODIS "XX" (tex line 290): fill with **1.1 km** (matches info_defs and
  tab_datasets 1.10). BUT the same sentence claims the 128×128 px cutout
  spans "≈128×128 km²" — at 1.1 km/pix it spans **≈141×141 km²**. Fix the
  span (or the dx claim) in Phase 3.
- MODIS-2km and VIIRS-2km rows in tab_datasets list dx = 2.00 — that is the
  *default fallback* (`dx = 2.` in info_defs when unset), not a real value.
  Truth: MODIS-2km = 128×1.1/64 = **2.2**, VIIRS-2km = 192×0.75/64 = **2.25**
  (text says ≈2.2 for both). Fix info_defs (set explicit dx for the 2km
  datasets) and regenerate tab_datasets in Phase 2.
- VIIRS (0.75) and LLC (2.25 = 144/64): consistent. OK.

## Flags for later phases

1. **LLC noise sigma conflict**: extraction code adds sigma = 0.09 K;
   ToDO.txt/tab_model reference 0.039–0.04 K. Determine which the trained
   model actually saw (it is 0.09 at the data level) and fix tab_model and
   the "$\sigma(T)=XX$ K" placeholder accordingly (Phase 1 rerun should log
   its sigma explicitly).
2. `figs_nenya_dim.py:208` (`fig_pca_noise_res`, flg 8) does
   `noise_datasets.remove('SWOT_L3')` — will crash now that the list holds
   SWOT_L2. Fix when touched (Phase 2 item 5 evaluates this figure).
3. `claude_rank_metrics.tex` and `claude_brainstorming.tex` embed L3-derived
   numbers (N99=89, beta=1.42). Standalone docs, but update or annotate them
   after the L2 switch so the project is self-consistent.
4. The `\swot` macro text is "SWOT/SSHa" while L2 figure legends will read
   "SWOT/SSHa-L2" — align in Phase 3.
5. L2 beta fit uses only 10 spectral points over the 4–40 pix range (64-px
   images); confirm the fit range when tab_analysis is regenerated.
