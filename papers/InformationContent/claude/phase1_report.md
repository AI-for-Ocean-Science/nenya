# Phase 1 Report — LLC Spin-up Exclusion Rerun

Created: 2026-08-09 (Claude Fable 5)

Phase 0 found that ~17% of the train and valid cutouts in all three LLC
datasets (LLC_SSTa_nonoise, LLC_SSTa_noise, LLC_SSHa) fell within the first
2 months of the LLC4320 run (model spin-up). Phase 1 re-extracted the
cutouts excluding those months, retrained all three models on Nautilus
(4x A10, ~2.2-2.5 days each), re-extracted latents locally, and recomputed
the image-space and latent-space statistics.

## Bottom line

**The spin-up exclusion is a robustness confirmation, not a correction.**
Image-space statistics are unchanged to within uncertainties; the SST latent
dimensionalities move by at most 2 modes. The one visible shift is SSHa
latent space, whose N99 drops 76 -> 70 (N95 50 -> 44) — still well within
run-to-run retraining variability expectations, and in the direction of
*less* information, consistent with removing the spin-up transients.

## Image-space statistics (old = with spin-up, new = without)

| Dataset          | beta (old)  | beta (new)  | f_var256 (old) | f_var256 (new) |
|------------------|-------------|-------------|----------------|----------------|
| LLC_SSTa_nonoise | 3.80 ± 0.06 | 3.80 ± 0.06 | 0.987          | 0.987          |
| LLC_SSTa_noise   | 3.49 ± 0.10 | 3.49 ± 0.10 | 0.958          | 0.958          |
| LLC_SSHa         | 3.21 ± 0.12 | 3.15 ± 0.13 | 0.980          | 0.979          |

beta is the P(k) power-law exponent fit over 4–40 pixel wavelengths;
f_var256 is the variance fraction in the first 256 image-space PCA modes
(from the 4096-component decomposition).

## Latent-space dimensionality (256-dim latents, train+valid = 200k)

| Dataset          | N95 (old) | N95 (new) | N99 (old) | N99 (new) |
|------------------|-----------|-----------|-----------|-----------|
| LLC_SSTa_nonoise | 52        | 54        | 76        | 77        |
| LLC_SSTa_noise   | 46        | 46        | 57        | 57        |
| LLC_SSHa         | 50        | 44        | 76        | 70        |

NXX = number of latent PCA modes needed to reach XX% of the cumulative
explained variance.

## Manuscript impact

- `tab_analysis` regenerated (2026-08-09): the only LLC change is the
  \llcssh row (3.21±0.12/0.980 -> 3.15±0.13/0.979). The regeneration also
  picked up the canonical SWOT_L2 row (\swot 1.42±0.11/0.801 ->
  2.58±0.31/0.985), per Decision 5 (SWOT_L2 canonical) — the old row was
  L3-sourced.
- Any in-text LLC beta / N99 numbers (Phase 3 placeholders) should use the
  NEW values above.
- Still open (flagged Phase 0): the LLC noise file adds sigma = 0.09 K at
  the data level, but the text/tab_model cite ~0.04 K. Needs an author
  decision on which to quote.

## Products (all v1 = with-spinup preserved as *_withspinup)

- Preproc: s3://llc/PreProc/ (v2 same keys; v1 at *_withspinup).
- Models: s3://llc/Nenya/models/LLC_{nonoise,noise,SSHa}/ (v2; v1 backed up
  to s3://llc/Nenya/models_withspinup/). Local: $OS_OGCM/LLC/Info/models/.
- Latents: $OS_OGCM/LLC/Info/latents/LLC_SST_{nonoise,noise}/,
  LLC_SSHa/ (v2). NOTE: the SSHa latents file is named
  train_llc_nonoise_latents.h5 — a pre-existing naming quirk in
  nenya_LLC_SSHa.py (v1 had the same name); contents are SSHa.
- PCA/Pk: Analysis/pca/ and Analysis/Pk/ (v2; *_withspinup preserved).
