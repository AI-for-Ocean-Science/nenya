# Final Steps — Phase 1: Nautilus Compute (CONDITIONAL)

Prompt doc for executing **Phase 1** of `papers/InformationContent/final_steps.md`.
Created 2026-08-02 by Claude Fable 5.

**GATE: Only execute this phase if `phase0_findings.md` says GO** (i.e. the
LLC preproc files include the first ~2 months of LLC4320 spin-up, or another
Phase 0 check found a stale data product). If the decision was NO-GO, mark
this phase skipped in `final_steps.md`, log it, and stop.

## Before you start

1. Read `papers/InformationContent/claude/Claude.md`,
   `papers/InformationContent/final_steps.md`, and
   `papers/InformationContent/claude/phase0_findings.md`.
2. Templates to reuse (do not write new infrastructure from scratch):
   - Training job specs: `papers/InformationContent/Analysis/yaml/nenya_LLC_nonoise_train.yaml`,
     `nenya_LLC_noise_train.yaml`, `nenya_LLC_SSHa_train.yaml` (image
     `profxj/ihop_nvidia:latest`, 4x A10 GPU; clones `wrangler` branch
     `llc_wrangling` and `nenya` branch `info_content`).
   - Operational conventions: `~/Oceanography/python/PAB/nautilus/`
     (esp. `run1k_job.yaml`) — resumable stages, per-stage log lines, fail-fast.
3. Use Claude **Fable** agents to prepare, launch, and monitor jobs where
   possible. Use the `ocean14` env for local Python.

## Prompt

Re-run the LLC pipeline excluding the spin-up period, for each affected
dataset (LLC_SSTa_nonoise, LLC_SSTa_noise, LLC_SSHa):

1. **Re-extract cutouts** with `Analysis/py/extract_llc.py`, dropping all
   dates in the first 2 months of the LLC4320 run (use the exact date range
   documented in `phase0_findings.md`). Rebuild the preproc `.h5` files.
   Keep the old preproc files (rename with a `_withspinup` suffix) — do not
   delete data.
2. **Stage data**: push the new preproc files to Nautilus S3
   (`aws --endpoint http://rook-ceph-rgw-nautiluss3.rook s3 cp ...`, bucket
   layout as in the existing YAMLs).
3. **Retrain on Nautilus**: copy (don't edit in place) the three training
   YAMLs to new versions (e.g. `nenya_LLC_nonoise_train_v2.yaml`), update the
   preproc filenames, `kubectl apply` them, and monitor with
   `kubectl logs -f job/<name>`. Conventions: keep `backoffLimit: 0` for GPU
   jobs; if a job dies, diagnose before re-applying; push checkpoints back to
   S3 as the existing YAMLs do.
4. **Extract latents** with the retrained checkpoints via the per-dataset
   scripts (`Analysis/py/nenya_LLC_*.py`, evaluate mode / `workflow.evaluate`).
5. **Recompute PCA and P(k)** locally with `Analysis/py/calc_pca_pp.py` and
   `Analysis/py/calc_Pk.py`, writing into `Analysis/pca/` and `Analysis/Pk/`
   (again, keep the old `.npz` files under a `_withspinup` suffix).
6. **Record the new beta values** (P(k) slopes) and PCA dimensionalities;
   note every downstream figure/table that must be regenerated in Phase 2,
   and whether the manuscript's LLC numbers (e.g. beta ~3.80, noise
   sigma = 0.04 K) changed.

## Deliverables

- New preproc/latents/pca/Pk products for the LLC datasets, spin-up excluded;
  old products preserved with `_withspinup` names.
- A short report `papers/InformationContent/claude/phase1_report.md`: what
  was rerun, wall-clock per job, old vs. new beta / PCA numbers, and the
  Phase 2 regeneration list.
- Updated ToDO.txt items (Section E) pushed to Overleaf (`$OVER_TOKEN`).
- Log in `paper_writing.md`; commit and push nenya (`info_content`),
  including any new/modified YAMLs.
