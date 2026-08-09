# Final Steps — Information Content Paper

Created: 2026-08-02 (Claude Fable 5)
Last edited: 2026-08-09

Implementation plan to carry the paper from its current state (Methods ~80%,
Results ~20%, front/back matter unwritten) to a submission-ready draft.
Derived from the Overleaf `ToDO.txt` (sections A–H) and the author's
2026-08-02 answers to the Section G open questions.

## Prompts

1. Run Phase 0
2. Run Phase 1
3. Run Phase 2
4. Run Phase 3
5. Run Phase 4

## Decisions now locked in

1. Include BOTH the architecture figure and a separate learning-curve figure.
2. Participation Ratio / RankMe integration is ON HOLD (module
   `Analysis/py/info_rank.py` and `claude_rank_metrics.tex` stay ready).
3. SWOT dataset description WAITS for Iury (blocked; do not draft).
4. Target journal STILL OPEN — keep the Copernicus template; defer the
   bibliography-style decision.
5. **SWOT_L2 is the canonical SWOT dataset** for all figures and tables.

## Phase 0 — Verification checks (local, `ocean14` env; no new compute)

**DONE 2026-08-02 — see `claude/phase0_findings.md`. Verdict: GO for Phase 1**
(all three LLC datasets ~17% spin-up contaminated; also, every SWOT
number/curve in the manuscript is still SWOT_L3-sourced → Phase 2 list).

- [x] **LLC cutout provenance** (ToDO E, P1, orig): LLC4320 (start
      2011-09-13); SST from the 2022 Gallmeier parent table, 24 biweekly
      timesteps 2011-09-18→2012-08-05; SSHa 6 bimonthly timesteps
      2011-09-30→2012-07-31; 64×64 px = 144 km cutouts. Noise file adds
      sigma=0.09 K (conflicts with the 0.04 K in ToDO/tab_model — flagged).
- [x] **LLC first-2-months check** (ToDO E, P2, orig): CONTAMINATED —
      ~16.7% of train and ~17% of valid cutouts in all three LLC datasets
      fall in the first 2 months. **Triggers Phase 1.**
- [x] **P2(k)/P4(k) zero-mean claim** (ToDO E, P2): CLAIM CORRECT — per-image
      demean + unit-std in code and verified numerically on both preproc
      files. No change needed.
- [x] **SWOT_L2 audit** (ToDO H, P1): all manuscript SWOT content (3 table
      rows, 4 included PNGs) is L3-sourced; `fig_example_images()` still
      hard-codes SWOT_L3. Full regeneration list in `phase0_findings.md`;
      all L2 inputs exist (L2: N99=132, beta=2.58±0.31, f_var256=0.985).
- [x] **dx consistency** (ToDO E, P3): MODIS "XX" → 1.1 km, but the claimed
      128 km span is really ≈141 km; MODIS-2km/VIIRS-2km table dx=2.00 is a
      fallback default — true values 2.2 / 2.25 (fix info_defs + regenerate
      tab_datasets in Phase 2).

## Phase 1 — Nautilus compute (CONDITIONAL on Phase 0 findings)

**DONE 2026-08-09 — see `claude/phase1_report.md`. Verdict: robustness
confirmed** (image-space beta/f_var256 unchanged within errors; latent N99:
nonoise 76→77, noise 57→57, SSHa 76→70). tab_analysis regenerated (SSHa +
canonical SWOT_L2 rows); old products preserved as `*_withspinup`.

Only needed if the LLC preproc files include the spin-up months (or if any
SWOT_L2 product turns out stale). Workflow mirrors the existing training jobs
in `Analysis/yaml/` and borrows operational lessons from the PAB repo
(`~/Oceanography/python/PAB/nautilus/`).

Steps (LLC re-run case; applies to LLC_SSTa_nonoise, LLC_SSTa_noise, LLC_SSHa):

- [x] 1. **Re-extract cutouts** excluding the first 2 months — done 2026-08-04;
      v2 preproc on `s3://llc/PreProc/` (same keys), v1 kept as `*_withspinup`.
- [x] 2. **Retrain** on Nautilus — all three v2 jobs Complete (nonoise
      2026-08-05; noise/SSHa 2026-08-07/08 after one preemption + relaunch);
      checkpoints on `s3://llc/Nenya/models/`, v1 backed up to
      `models_withspinup/`. v2 YAMLs: `Analysis/yaml/nenya_LLC_*_train_v2.yaml`.
- [x] 3. **Extract latents** — done locally via
      `Analysis/py/run_latents_local.py` (fork fix for Python 3.14);
      150k train + 50k valid × 256 per dataset, verified.
- [x] 4. **Recompute PCA and P(k)** — image-space Pk + pca_preproc done
      2026-08-05; latent PCAs done 2026-08-05 (nonoise) / 2026-08-09
      (noise, SSHa). N99: nonoise 76→77, noise 57→57, SSHa 76→70.
- [x] 5. **Update `tab_analysis`** — regenerated 2026-08-09 (SSHa row
      3.15±0.13/0.979; SWOT row now canonical SWOT_L2). Remaining
      figure/table regeneration is tracked under Phase 2.

Operational notes (from the PAB Nautilus experience):

- Make each job resumable and log per-stage wall-clock lines
  (`=== STAGE x <timestamp> ===`) so preempted pods resume rather than restart.
- Stop the pipeline when a stage fails instead of letting later stages run on
  partial output.
- Size memory to what workers actually hold; training pods here use
  8 CPU / 24–32 Gi / 4 GPUs per the existing YAMLs.
- Monitor with `kubectl logs -f job/<name>`; keep `backoffLimit` small for
  GPU jobs (existing YAMLs use 0).
- Agent execution: use Claude **Fable** agents to prepare/launch/monitor the
  jobs where possible, per the author's instruction.

## Phase 2 — Figures and tables (local, `ocean14`)

**DONE 2026-08-09 — all figures/tables regenerated SWOT_L2-sourced; see the
work log at the bottom of this file.** Overleaf push `8d59f0f`.

- [x] **Eigenmodes figure** (ToDO C, P2, code done): run
      `Figures/py/figs_nenya_dim.py` with `flg==53`
      (`fig_eigenmodes_remote_sensing()`: MODIS_SSTa, VIIRS_SSTa, SWOT_L2).
      The PNG does not exist yet. Generate it, copy to Overleaf `Figures/`,
      add figure environment + caption + in-text discussion.
      → generated + `fig:eigenmodes` env in new `sec:pca_res` subsection.
- [x] **Learning-curve figure** (ToDO C, P1; decision: BOTH): add a new
      figure environment for `fig_learning_curves.png` (already in Overleaf
      `Figures/`), and fix the architecture figure: relabel (it is currently
      `fig:learning`), write a real caption (now bare "Architecture"), and
      point the text at the right labels.
      → `fig:architecture` (workflow caption + in-text ref) and a real
      `fig:learning` env; PNG regenerated (SWOT_L2 curve).
- [x] **PCA figure captions** (ToDO C, P2): expand `fig_true_pca` and
      `fig_pca_2panel` captions. → full captions, verified against the
      regenerated PNGs.
- [x] **Pk_all_datasets caption** (ToDO C, P3): replace the bracketed
      remote-sensing placeholder (line ~584). → done.
- [x] **Table fixes** (ToDO D):
      - Column-count mismatches: `tab_analysis.tex` ({cccc} vs 3 cols),
        `tab_model.tex` ({ccccccc} vs 5 cols).
      - Add year + geographic-coverage columns to `tab_datasets` via
        `Tables/py/tables_info.py` (MODIS=2021, VIIRS/NOAA-21=2024, LLC days
        per text) and regenerate.
      - Reconcile the `tab_model` footnote with its actual columns.
      → all fixed in the generator + info_defs (also true dx for the 2km
      rows: 2.20/2.25); three tables regenerated. SWOT year/coverage
      (2023-2024, ±78°) awaits text verification when Iury's section lands.
- [x] Optionally include `fig_pca_noise_res.png` if it adds value (ToDO C, P3).
      → INCLUDED as `fig:pca_noise_res` (only figure isolating noise vs
      resolution with controlled pairs); code fixed to drop SWOT from its
      SST panels.

## Phase 3 — Manuscript text (Overleaf `information_content.tex`)

Order matters: Results → Conclusions → Introduction → Abstract (last).

- [ ] **PCA Results text** (largest gap): write the discussion for
      `fig_true_pca` and `fig_pca_2panel` (lines ~654–666).
- [ ] **P(k) Results placeholders** (lines ~597–652): fill "$k \approx XX$",
      exponents (use `tab_analysis` betas: MODIS 2.90, VIIRS 3.06, LLC 3.80 —
      update if Phase 1 reruns change them), the LLC noise sigma
      ($\sigma(T)$ = 0.04 K), the SSH "we find XXX"; resolve the four
      bracketed editorial notes ([PMC...], [Ask Dimitris!], [is the LLC SST
      higher??], MNIST note).
- [ ] **Conclusions**, then **Introduction** (frame: information content of
      remote-sensing imagery; contrastive learning; pull citations from
      `claude_brainstorming.tex`), then **Abstract**.
- [ ] **Bibliography** (ToDO B, P1): replace the stub `thebibliography` with a
      `.bib` file (+ `copernicus.bst` for now, per Decision 4); resolve all
      undefined keys (llc, nenya, viirs, ecco, gallmeier2023, ulmo, pae,
      nenya_doi, field1987 [unify caps], archer2025, llc_res); fix the
      `gallmeiter2023` typo (line ~364); import DOIs from
      `claude_brainstorming.tex`.
- [ ] **Macros**: populate or remove \nocean, \ndataset, \nother ("XXX",
      lines 96–98) and \powmnist ("-XX", line 126).
- [ ] **Back matter** (P3): code/data availability (nenya is open source —
      cite the nenya DOI), author contributions, competing interests,
      acknowledgements.
- [ ] **LaTeX fixes** (ToDO F): remove stray "[t]" after `\end{figure*}`
      (lines ~659, ~666); then a clean compile with zero undefined
      references/citations.

## Phase 4 — Final pass

- [ ] Clean compile of `information_content.tex`; check every figure/table is
      referenced in the text and vice versa.
- [ ] One full read-through for consistency (SWOT_L2 naming, dx values,
      dataset counts vs. macros).
- [ ] Push Overleaf; update `ToDO.txt` statuses and the change log.

## Blocked / deferred (do NOT work these)

- SWOT dataset description — waiting on Iury (ToDO A, [B]).
- PR / RankMe integration — on hold per author (ToDO D/E, [H]).
- Journal choice / bibliography style / abstract length — open (ToDO H, [?]).

## Work log

### 2026-08-09 — Phase 2 executed (Claude Fable 5; two parallel Fable agents)

Prompt 3 ("Run Phase 2") per `claude/final_steps_2.md`. Two Fable agents ran
the compute tracks (figures; tables) while the main session edited the
manuscript. Details:

- **Figures agent**: fixed `figs_nenya_dim.py` — `fig_example_images()`
  SWOT_L3→SWOT_L2 (title 'SWOT L2', 54 km scale bar; also VIIRS_SST→
  VIIRS_SSTa, which would have crashed first) and `fig_pca_noise_res()`
  (defensive SWOT removal; list now carries SWOT_L2). Generated all 7 PNGs
  (flg 53, 1, 2, 3, 4, 5, 8) with no missing inputs and copied them to
  Overleaf `Figures/`. New: `fig_eigenmodes_remote_sensing.png` (2×3: modes
  1–2 × MODIS/VIIRS/SWOT-L2; SST leading modes = orthogonal large-scale
  gradients; SWOT mode 1 = near-uniform mean-like mode, mode 2 = cross-track
  gradient).
- **Tables agent**: `info_defs.py` gained explicit dx for the 2km datasets
  (2.20/2.25) plus `year`/`coverage` entries per dataset; `tables_info.py`
  column declarations fixed (analysis {ccc}, model {ccccc}, datasets
  {ccccccc} with Year+Coverage), 'Model' footnote sentence removed. All
  three tables regenerated + copied to Overleaf. NOTE: SWOT year/coverage
  (2023-2024, ±78°) are inferred, not text-verified — reconcile with Iury's
  SWOT section. tab_analysis betas for the 2km rows shifted slightly
  (3.43→3.44, 3.60→3.62) because the dx fix changes the fit window.
- **Manuscript** (`information_content.tex`): architecture figure relabelled
  `fig:architecture` with a real workflow caption + in-text ref; new
  `fig:learning` environment (learning curves, 8 datasets); Pk caption
  placeholder replaced; full captions for `fig:pca_true` and
  `fig:pca_latent`; stray "[t]" ×2 removed; new Results subsection `\pca`
  (`sec:pca_res`) with brief intro text (Phase 3 expands, marker comment in
  place) hosting `fig:eigenmodes` and — decision: INCLUDE — 
  `fig:pca_noise_res` (isolates noise vs resolution effects with controlled
  pairs). All captions verified against the regenerated PNGs.
- **Deployed**: Overleaf push `8d59f0f` (12 files: tex, 3 tables, 7 PNGs,
  ToDO.txt sections C/D/F/H updated). nenya committed/pushed alongside this
  log.
- **Left for Phase 3/4**: \swot macro reads "SWOT/SSHa" while L2 figure
  legends read "SWOT/SSHa-L2" (align in Phase 3); tab_model footnote still
  defines two notations no row uses (from commented-out generator code);
  LLC noise sigma 0.09 vs 0.04 K conflict still open (author call).


## Logs