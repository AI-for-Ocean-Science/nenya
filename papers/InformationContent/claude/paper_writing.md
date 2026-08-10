# Paper Writing

**IMPORTANT:** At the start of every conversation involving this paper, you MUST read the local `Claude.md` file in this directory (`papers/InformationContent/claude/Claude.md`) and follow its instructions. Do NOT rely solely on the top-level nenya `CLAUDE.md` — the local one contains paper-specific guidance, conventions, and project context.

## Here is how the activity should be organized

### Code

- Place any new analysis code in `papers/InformationContent/Analysis/py/`.
- Place any new figure-generation code in `papers/InformationContent/Figures/py/`.
- Place any new table-generation code in `papers/InformationContent/Tables/py/`.
- Place any Notebooks in the appropriate subdirectory.
- Key data definitions live in `Analysis/py/info_defs.py` — reuse `grab_paths()` for all dataset references.
- Include inline comments in the code to explain the code.

### Data

- Preprocessed HDF5 files are on local disk (paths resolved via environment variables `OS_DATA`, `OS_SST`, `OS_OGCM`, `OS_SSH`).
- Intermediate analysis outputs (PCA, P(k)) go in `Analysis/pca/` and `Analysis/Pk/`.

### LaTeX / Overleaf

- The paper LaTeX source is in this Overleaf-synced directory: Projects/Overleaf/Info_content 
- Figures go in the Overleaf project folder as PNG files in the Figures/ subdirectory.
- Embed figures within the text near their description.
- When adding new files, update the main `.tex` file to include them.
- You may push to git as you work.  The password token is given by $OVER_TOKEN

## Claude Code

- You should be critical of any of my prompts and not simply aim to please.
- You should use Python code exclusively.
- You are allowed to run safe bash commands without prompting me.
- If you need to run Python code, use the `ocean14` conda environment.
- You are welcome to use multiple agents to help you with the task.
- When possible, reuse existing code and modules rather than writing new code.

# Figures

1. Eigenmodes for remote sensing datasets

I wish to add a new method to fig_nenya_dim.py to plot the first 2 eigenmodes for the remote sensing datasets.  

Here is the definition of the method:

- This method should be called fig_eigenmodes_remote_sensing() 
- It should plot the first 2eigenmodes for the remote sensing datasets using the outputs in pdict['pca_imgfile'].  
- The method should be placed in the Figures/py/figs_nenya_dim.py file.  
- It should operate on the 3 primary remote sensing datasets: MODIS_SSTa, VIIRS_SSTa, SWOT_L2.
- It should be a 3x2 grid of images with a colorbar for each image.
- Have it be called using flg==53

# Prompts

## Setup

1. I have started a Claude.md file in the nenya/papers/InformationContent/claude folder.  It will be used to guide the writing of the paper.  Please continue to develop it, drawing upon the instructions in the claude_explore.md file in the /home/xavier/Oceanography/python/fronts/dev/groups folder.  Also build out this file, also drawing from the instructions in the claude_explore.md file.

## Initial review

1. Examine the files in the Overleaf project and generate a report in a separate Latex file that assesses the current state of the paper.  This report should be placed in Overleaf project and name it "claude_initial_review.tex".  Provide a plan before proceeding.
Make sure the claude_initial_review.tex its own stand-alone file that can compiled separtely

2. Push the Overleaf changes to git.

## Brainstorming

1. Examine the files in the Overleaf project and generate a brainstorming document in a separate, stand-alone Latex file that captures the brainstorming session.  This document should be placed in the Overleaf project and name it "claude_brainstorming.tex".  You should access the web as you wish to do your brainstorming. Provide a plan before proceeding or generating any files. 
2. Yes, proceed with that plan and push to git as you go.
3. Do a deeper search of the web for information on the topic of the paper.  Add your findings to the brainstorming document.  Ultrathink.
4. Please find DOIs for the papers that you have found on the web and add them to the brainstorming document.  You are allowed to perform curl commands with bash without prompting me.


## Participation rank and RankMe metric

1. Examine the files in Analysis/ and generate a new module to calculate the participation rank and RankMe metric on the latent space.  This module should be placed in Analysis/py/info_rank.py.  Provide a plan before proceeding and write it to Overleaf as you go.

## Figures

1. Reread this doc and the Figures section above. Proceed to generate code for the first figure.
2. Looks good, but please swap rows and columns in the figure.

## TODO List

1. Please examine all of the files related to this project and generate a TODO list of what needs to be done to complete the project.  Add this TODO list to the file in the Overleaf project named "ToDO.txt".  If you have any questions, please ask me.  Log your work in the Logs section of this file.

2. I have answered the questions in the Q&A section below.  Please update the TODO list to reflect the answers.  Push the Overleaf with git. Then generate an implementation plan named `final_steps.md` in `papers/InformationContent`.  I expect you will need to use Nautilus to run additional tests.  If so, see the `PAB` repo and especially its `Oceanography/python/PAB/nautilus` folder.  Use Fable if you can. Log your work.  

3. Ok, that plan looks good.  Please generate a series of prompt docs in `papers/InformationContent/claude` that will be used to complete the final steps.  Name them `final_steps_<step_number>.md`.  These should be used to complete the final steps.  Log your work.  Use Fable if you can.

### Q&A

Open questions raised while generating the TODO list (2026-06-23); see Section G
of ToDO.txt in the Overleaf project:

1. Learning-curve figure: include BOTH an architecture figure AND a separate
   learning-curve figure, or only one? (fig_learning_curves.png exists but is
   currently unused; the architecture figure is mislabeled as the learning curve.)
   A. Include both
2. Rank metrics (PR / RankMe): integrate into the main paper now? If so, where
   (new Results subsection, table column, or figure)? Module info_rank.py and
   claude_rank_metrics.tex exist but are not referenced in the manuscript.
   A. Let's hold off for now.
3. SWOT dataset description: draft it from the code/opts, or wait for Iury?
   A. Let's wait for Iury.
4. Target journal confirmed as Copernicus GI, or still open? (Affects bib style
   and abstract length.)
   A. Still open
5. SWOT_L2 vs SWOT_L3: which is the canonical SWOT dataset for the paper figures?
   A. Let's use SWOT_L2.

## Logging

Append an entry to the **Logs** section of this file using the format:

```
### <Date> (Short summary of the work)

<Detailed description of the work and what you learned>
```

## Logs

### 2026-06-23 (Generated comprehensive project TODO list)

Examined the full project to assess what remains for completion and wrote a
comprehensive TODO list to ToDO.txt in the Overleaf project (replacing the
sparse 4-item stub, with the original author items retained and flagged "orig").

Files reviewed:
- Overleaf: information_content.tex (main manuscript), claude_initial_review.tex
  (the 2026-03-24 status assessment), claude_brainstorming.tex,
  claude_rank_metrics.tex, and Tables/{tab_datasets,tab_model,tab_analysis}.tex.
- Code: Analysis/py/ (info_defs.py, info_rank.py, extract/PCA/Pk scripts),
  Figures/py/figs_nenya_dim.py, Tables/py/tables_info.py, Preprocess/py/.

Key findings / state of the paper:
- Methods section ~80% complete; Results ~20%; Abstract/Intro/Conclusions/
  back-matter essentially unwritten; bibliography is stub-only.
- LaTeX issues from the initial review are STILL present: stray "[t]" after
  \end{figure*} (lines 659, 666); gallmeiter2023 -> gallmeier2023 typo (line 364);
  column-count mismatches in tab_analysis ({cccc} vs 3 cols) and tab_model
  ({ccccccc} vs 5 cols); architecture figure mislabeled as the learning-curve
  figure while fig_learning_curves.png is never included.
- New work exists but is NOT integrated into the manuscript: the
  Participation Ratio / RankMe analysis (info_rank.py + claude_rank_metrics.tex)
  and the remote-sensing eigenmodes figure (fig_eigenmodes_remote_sensing(),
  flg==53 in figs_nenya_dim.py).

Organized the TODO into sections A-G: Manuscript Text, References, Figures,
Tables, Analysis/new-work, LaTeX fixes, and Open Questions. Logged 5 open
questions for the author in the Q&A section above and in Section G of ToDO.txt.
Did not push to git pending the author's review.

### 2026-08-02 (Updated ToDO.txt with Q&A answers; wrote final_steps.md plan)

Executed task 2 of the TODO List: incorporated the author's answers to the
five Section G open questions into the Overleaf ToDO.txt, pushed the Overleaf
repo to git, and generated the implementation plan
`papers/InformationContent/final_steps.md`.

ToDO.txt changes (committed as ca4bccc and pushed to Overleaf):
- Header note recording the 2026-08-02 decisions; added [B] (blocked) and
  [H] (on hold) status keys.
- Section A: SWOT description marked [B] — wait for Iury, do not draft.
- Section C: architecture/learning-curve item resolved — include BOTH figures.
- Sections D/E: PR/RankMe items marked [H] (author deferred).
- Section G: answers recorded inline under each question.
- New Section H: tasks arising from the answers — [P1] standardize on SWOT_L2
  everywhere (audit tex/tables/figures for SWOT_L3 usage), [P1] add the
  learning-curve figure environment, [?] decide the target journal.

final_steps.md (new): a phased plan to submission-ready draft.
- Phase 0: local verification (LLC cutout provenance, LLC first-2-months
  spin-up check, Pk2/Pk4 zero-mean claim, SWOT_L2 audit, dx consistency).
- Phase 1: CONDITIONAL Nautilus compute — only if Phase 0 shows the LLC
  preproc includes the spin-up months: re-extract, retrain (existing
  Analysis/yaml/ job specs; profxj/ihop_nvidia image, 4x A10), re-extract
  latents, recompute PCA/P(k). Operational conventions borrowed from the PAB
  repo's nautilus folder (resumable stages, per-stage logging, fail-fast);
  Fable agents to drive job prep/launch/monitoring.
- Phase 2: figures/tables (generate the eigenmodes PNG — code exists at
  flg==53 but the PNG was never produced; learning-curve + architecture
  figure fixes; table column-count fixes; year/geo columns).
- Phase 3: manuscript text in order Results -> Conclusions -> Introduction ->
  Abstract, plus a real .bib bibliography.
- Phase 4: clean compile + consistency pass.

Learned along the way: the Overleaf git token lives in ~/.bashrc as
OVER_TOKEN (not "OVERLEAF"); pca_latents_SWOT_L2.npz and Pk_SWOT_L2.npz
already exist, so the SWOT_L2 standardization needs no new compute; the
fig_eigenmodes_remote_sensing PNG is the only missing figure product.

### 2026-08-02 (Generated final_steps prompt docs, phases 0-4)

Executed task 3 of the TODO List: wrote five prompt docs in
papers/InformationContent/claude/, one per phase of final_steps.md, numbered
to match the plan's phases (final_steps_0.md ... final_steps_4.md). Each doc
is a self-contained prompt for a future session: a "Before you start" block
(read Claude.md + the plan + upstream phase reports; ocean14 env; use Fable
agents for parallelizable work), the task list with file paths and line
numbers pulled from ToDO.txt/final_steps.md, an explicit deliverables list,
and the standing close-out steps (tick plan boxes, update ToDO.txt, push
Overleaf with $OVER_TOKEN, log here, push nenya info_content).

- final_steps_0.md — Phase 0 verification. Five independent local checks;
  produces claude/phase0_findings.md with a GO/NO-GO decision that gates
  Phase 1.
- final_steps_1.md — Phase 1 Nautilus compute, CONDITIONAL on the Phase 0
  gate. LLC re-extract (spin-up excluded) -> retrain via copies of the
  existing Analysis/yaml job specs -> latents -> PCA/P(k); preserves old
  products under _withspinup names; produces claude/phase1_report.md with
  old-vs-new betas and a Phase 2 regeneration list.
- final_steps_2.md — Phase 2 figures/tables. Eigenmodes PNG (flg==53),
  learning-curve + architecture figure fixes, captions, table column-count
  and year/geo fixes, SWOT_L2 regeneration list, stray "[t]" removal.
- final_steps_3.md — Phase 3 manuscript text in order Results -> Conclusions
  -> Introduction -> Abstract, plus .bib bibliography, macros, back matter;
  author-decision points marked as % CLAUDE-QUESTION: comments; explicit
  do-not-touch list (Iury's SWOT text, RankMe, journal formatting).
- final_steps_4.md — Phase 4 final pass. Clean compile, cross-ref audit,
  consistency greps (XX/TEXT/Blah/SWOT_L3 leftovers), and a "FINAL AUTHOR
  ITEMS" section appended to ToDO.txt collecting everything needing a human.

Design choices: docs are numbered 0-4 (not 1-5) so names match the plan's
phase numbers; phase reports (phase0_findings.md, phase1_report.md) are the
hand-off artifacts between phases so each doc can run in a fresh session
without rereading the whole project.

### 2026-08-02 (Executed Phase 0 verification checks -- GO for Phase 1)

Ran final_steps_0.md: three parallel Fable agents (LLC provenance+spin-up,
Pk zero-mean, SWOT audit) plus an inline dx check. Full report with evidence
in claude/phase0_findings.md; final_steps.md Phase 0 boxes ticked; ToDO.txt
Sections E/H updated and pushed to Overleaf.

Verdicts:
1. LLC provenance DOCUMENTED: LLC4320 (start 2011-09-13); SST cutouts from
   the 2022 Gallmeier parent table, 24 biweekly timesteps 2011-09-18 to
   2012-08-05; SSHa 6 bimonthly timesteps 2011-09-30 to 2012-07-31; 64x64 px
   = 144 km cutouts; noise file adds sigma=0.09 K at the data level.
2. LLC spin-up: CONTAMINATED -- ~16.7% of train and ~17% of valid cutouts in
   ALL THREE LLC datasets fall within 2 months of run start (earliest SST
   cutouts just 5 days after start). ==> PHASE 1 IS A GO (Nautilus rerun).
3. Pk2/Pk4 zero-mean claim: CORRECT in code (per-image demean + unit std)
   and verified numerically (|mean|<2e-9, |std-1|<3e-9 on 200-image samples).
4. SWOT audit: every SWOT number/curve in the manuscript is L3-sourced
   (Overleaf floats predate the code's L2 switch): 3 table rows + 4 PNGs;
   fig_example_images() still hard-codes SWOT_L3 in figs_nenya_dim.py
   (lines 728/729/751). All L2 inputs already exist (L2: npix 64, dx 0.843,
   N99=132, beta=2.58+/-0.31, f_var256=0.985 vs L3's 128/0.25/89/1.42/0.801).
   Regeneration list handed to Phase 2 in phase0_findings.md.
5. dx: MODIS XX -> 1.1 km but the "128x128 km^2" span is really ~141 km;
   MODIS-2km/VIIRS-2km table dx=2.00 is the info_defs fallback default --
   true values 2.2 / 2.25 km/pix (fix info_defs, regenerate tab_datasets).

Notable flags: LLC noise sigma conflict (0.09 K in extraction code vs
~0.04 K cited in ToDO/tab_model -- resolve during the Phase 1 rerun);
fig_pca_noise_res() will crash (removes 'SWOT_L3' from a list that now
holds SWOT_L2); claude_rank_metrics.tex and claude_brainstorming.tex embed
L3-era numbers; SST timestep 2011-11-13 sits exactly on the 2-month cut
boundary -- Phase 1 must state its cut convention.

### 2026-08-02 (Phase 1 IN PROGRESS -- re-extraction done, Nautilus training launched)

Executing final_steps_1.md. Cut convention (per author): exclude all cutouts
with dates in the first 2 months of the LLC4320 run, i.e. datetime <
2011-11-13; the 2011-11-13 boundary timestep is RETAINED.

DONE in this session (commit a86d61b on info_content):
1. Code: extract_utils.prep_for_training gained min_date; extract_llc.py
   gained LLC_SPINUP_END='2011-11-13', ex_nonoise(min_date, seed) and
   ex_ssh(min_date, seed) (the SSHa date-cut path samples the DBOF table
   directly and reuses fronts create_hdf5_cutouts). Seed 12345 used.
   NOTE: wrangler/fronts/nenya are NOT pip-installed locally -- run with
   PYTHONPATH=/mnt/tank/Oceanography/python/{wrangler,fronts,nenya} under
   conda env ocean14.
2. Fixed latent bugs: nenya_LLC_nonoise.py / nenya_LLC_noise.py used stale
   dataset keys LLC_SST_* (grab_paths now wants LLC_SSTa_*), and the noise
   script resumed from load_epoch=49 (v1 restart logic) -- now fresh.
3. Re-extraction complete + validated: 150k/50k per dataset, zero cutouts
   before 2011-11-13 (SST 2011-11-13 -> 2012-08-05, 20 timesteps; SSHa
   2011-11-30 -> 2012-07-31, 5 timesteps), noise sigma verified 0.090.
   Old products preserved locally AND on S3 as *_withspinup; old model
   checkpoints backed up to s3://llc/Nenya/models_withspinup/.
4. New preproc uploaded to s3://llc/PreProc/ (same keys as v1). Local S3
   access: default aws profile + --endpoint-url https://s3-west.nrp-nautilus.io.
5. Launched 3 Nautilus jobs (namespace sea-meets-the-stars):
   xavier-nenya-llc-{nonoise,noise,ssha}-train-v2 from the new v2 YAMLs
   (branches: nenya info_content, wrangler llc_wrangling; 4x A10 each;
   backoffLimit 0 -- a preempted/failed job must be re-applied by hand).
   At session pause: nonoise Running, noise/ssha Pending (GPU scheduling).

UPDATE 2026-08-05 (second session pause):
- nonoise training COMPLETED on Nautilus (~55h); v2 checkpoints pushed to
  s3://llc/Nenya/models/LLC_nonoise/ (2026-08-05 05:17). last.pth + opts +
  learning_curve downloaded to $OS_OGCM/LLC/Info/models/LLC_nonoise/ (v1
  moved to models/LLC_nonoise_withspinup; v1 latents dir moved to
  latents/LLC_SST_nonoise_withspinup).
- noise + ssha jobs FAILED once (BackoffLimitExceeded, pods gone, likely
  preemption); RELAUNCHED 2026-08-05 ~15:40, currently Pending/Running.
- nonoise latent extraction attempted locally (CPU; run from Analysis/ as
  `python py/nenya_LLC_nonoise.py evaluate` with the PYTHONPATH trio under
  ocean14): FAILED with a multiprocessing serialization error under
  `conda run` (log: scratchpad latents_nonoise_v2.log copy in
  /tmp/claude-1000/.../tasks/bfg0i6lz3.output). NOT yet debugged -- likely
  the DataLoader num_workers=8 + conda-run interaction; try running with
  `conda activate ocean14` directly or num_workers=0, or extract on the
  cluster instead.

UPDATE 2026-08-05 evening (third session pause):
- Latents bug DIAGNOSED + FIXED: Python 3.14 defaults multiprocessing to
  forkserver, which pickles the DataLoader dataset; HDF5RGBDataset holds an
  open h5py handle -> 'h5py objects cannot be pickled'. Fix: force fork via
  new Analysis/py/run_latents_local.py (committed e6aba6a). Usage from
  Analysis/: PYTHONPATH=py:<wrangler>:<fronts>:<nenya> conda run -n ocean14
  python -u py/run_latents_local.py nenya_LLC_nonoise evaluate
  ALSO add /mnt/tank/Oceanography/python/remote_sensing to PYTHONPATH for
  calc_Pk (nenya.pk imports remote_sensing).
- nonoise latents extraction RUNNING (nohup, survives exit; ~2h on CPU;
  log ~/.claude/jobs/5f7b75e3/tmp/latents_nonoise_v2.log; done when it
  prints 'Latents saved to').
- noise + ssha v2 training RUNNING on Nautilus (relaunched 2026-08-05
  ~15:40; noise reached epoch 1 by 16:20; expect ~2 days).
- Image-space products for the 3 no-spinup LLC datasets DONE (Pk npz/png,
  pca_preproc 256 + 4096; v1 files preserved as *_withspinup).
- EARLY RESULT: spin-up exclusion does NOT move the image-space numbers.
  beta (4-40 pix fit): nonoise 3.80+/-0.06 -> 3.80+/-0.06; noise 3.49 ->
  3.49; SSHa 3.21+/-0.12 -> 3.15+/-0.13. f_var256 (from 4096 PCA): 0.987 ->
  0.987, 0.958 -> 0.958, 0.980 -> 0.979. tab_analysis image-space columns
  are effectively unchanged; any paper-level impact must come via the
  retrained latents (N99 etc.), still pending.

UPDATE 2026-08-05 night (fourth session):
- nonoise v2 latents DONE (200k x 256, latents/LLC_SST_nonoise/.../
  train_llc_nonoise_latents.h5) and latent PCA recomputed
  (pca/pca_latents_LLC_SSTa_nonoise.npz; run pca_latents() from Analysis/,
  NOT Analysis/pca/ -- pca_file is 'pca/...' relative).
- KEY RESULT: N99 = 76 (v1 withspinup) -> 77 (v2 no-spinup); N95 = 52 -> 54.
  Together with the unchanged image-space stats, the spin-up exclusion is a
  robustness CONFIRMATION -- the paper's LLC numbers barely move.
- Housekeeping: a duplicate nonoise extraction accidentally launched this
  session was killed (the session-3 nohup run won the race); the session-3
  version of run_latents_local.py was overwritten by an equivalent one with
  CLI 'run_latents_local.py nonoise|noise|SSHa' (commit eac7914) -- use THAT
  interface going forward.

REMAINING for Phase 1 (resume here):
- When noise/ssha jobs complete: pull last.pth+opts+learning_curve from
  s3://llc/Nenya/models/{LLC_noise,LLC_SSHa}/ to $OS_OGCM/LLC/Info/models/
  (back up v1 dirs to *_withspinup first, same for latents dirs), run
  run_latents_local.py for nenya_LLC_noise / nenya_LLC_SSHa, then their
  latent PCAs.
- Write claude/phase1_report.md (include the beta/f_var256 table above +
  N99 old-vs-new); update ToDO.txt Section E; push Overleaf; log; push nenya.
- Verify the 3 jobs completed (kubectl -n sea-meets-the-stars get jobs |
  grep v2); on completion each pushes checkpoints to s3://llc/Nenya/models.
- Pull new checkpoints locally (under $OS_OGCM/LLC/Info/models/...),
  run 'evaluate' in nenya_LLC_{nonoise,noise,SSHa}.py to extract latents.
- Rename old pca/Pk npz to *_withspinup, rerun calc_pca_pp.py + calc_Pk.py
  for the LLC datasets, compare old-vs-new beta/N99.
- Write claude/phase1_report.md; update ToDO.txt (Section E), push
  Overleaf; log here; push nenya.
UPDATE 2026-08-09 (fifth session): PHASE 1 COMPLETE.
- noise + ssha v2 training COMPLETED on Nautilus (Complete 1/1; ~2d8h and
  ~2d12h); checkpoints on S3 2026-08-07 23:43 / 2026-08-08 04:01.
- v1 model + latents dirs backed up to *_withspinup; v2 last.pth + opts +
  learning_curve pulled to $OS_OGCM/LLC/Info/models/{LLC_noise,LLC_SSHa}.
- Latents extracted locally for both (run_latents_local.py noise SSHa,
  ~2h each on CPU; train 150k + valid 50k x 256 each). NOTE: conda run
  buffers stdout -- the log stays empty until the process exits; not a
  stall. The SSHa latents file is named train_llc_nonoise_latents.h5 (same
  as v1) -- pre-existing quirk in nenya_LLC_SSHa.py, contents are SSHa.
- Latent PCAs recomputed (dataset keys are LLC_SSTa_noise / LLC_SSHa in
  info_defs -- NOT LLC_SST_noise). Results: N99 noise 57->57 (N95 46->46),
  SSHa 76->70 (N95 50->44); with nonoise 76->77 the spin-up exclusion is a
  robustness confirmation; SSHa is the only mover (fewer modes, consistent
  with removing spin-up transients).
- tab_analysis regenerated via Tables/py/tables_info.py mktab_analysis():
  \llcssh 3.21+/-0.12/0.980 -> 3.15+/-0.13/0.979; \swot row now the
  canonical SWOT_L2 (2.58+/-0.31/0.985, was L3 1.42/0.801). Copied to repo
  Tables/ + Overleaf; Overleaf ToDO.txt Section E spin-up item marked DONE;
  Overleaf pushed (d307708).
- Wrote claude/phase1_report.md (full old-vs-new tables + product paths).
- Phase 1 DONE. Next: Phase 2 (figures/tables) -- note tab_analysis {cccc}
  vs 3-col mismatch and tab_model fixes are still Phase 2 items; in-text
  LLC/SWOT numbers get fixed in Phase 3 with the new values.

UPDATE 2026-08-09 (sixth session): PHASE 2 COMPLETE (figures + tables).
- Executed via two parallel Fable agents (figures track, tables track) +
  main-session manuscript edits, per final_steps_2.md.
- All 7 paper figures regenerated SWOT_L2-sourced (figs_nenya_dim.py fixes:
  fig_example_images L3->L2 + VIIRS_SSTa rename; fig_pca_noise_res SWOT
  removal fix). NEW figures in the manuscript: fig:eigenmodes
  (fig_eigenmodes_remote_sensing.png, first generation), fig:learning
  (fig_learning_curves.png finally included), fig:pca_noise_res (INCLUDED
  after evaluation). Architecture fig relabelled fig:architecture with a
  real caption.
- All 3 tables regenerated: column declarations fixed, tab_datasets gained
  Year+Coverage columns, 2km dx fallback fixed (2.20/2.25; shifts those
  betas to 3.44/3.62), tab_model footnote 'Model' sentence dropped.
  CAVEAT: SWOT year/coverage inferred (2023-2024, +-78 deg), verify against
  Iury's SWOT section when it lands.
- Manuscript: new \pca Results subsection (sec:pca_res) with intro text and
  a Phase-3 expansion marker; full captions everywhere; stray [t] x2 gone;
  Pk caption placeholder resolved.
- ToDO.txt C/D/F/H updated; Overleaf pushed (8d59f0f). Work log added at
  the bottom of final_steps.md (author request: log there).
- Phase 2 DONE. Next: Phase 3 (Results text -> Conclusions -> Intro ->
  Abstract; bibliography; macros; back matter).

UPDATE 2026-08-10 (seventh session): PHASE 3 COMPLETE (manuscript text).
- Executed via two parallel Fable agents (references.bib: 61 CrossRef-
  verified entries, nenya Zenodo DOI 10.5281/zenodo.21730571 found;
  Introduction: 7 paragraphs/25 citations from a shared key manifest) with
  the main session writing P(k)+PCA Results, Conclusions, Abstract, macros,
  back matter, and owning consistency.
- KEY DATA FIX: injected LLC noise measured from the v2 preproc files =
  0.090 K (NOT the 0.04 K in the text nor tab_model's 0.039 from the stale
  ulmo json). Text, tables_info.py, and tab_model updated; CLAUDE-QUESTION
  left for the author. Also: N99 76-vs-77 was a threshold-convention
  difference; the text quotes tab_model.
- All placeholders gone: Abstract/Intro/Conclusions written; P(k) section
  numbers filled; four editorial notes resolved (PMC + Dimitris questions
  now % CLAUDE-QUESTION comments); MODIS dx=1.1 km + 141 km^2 span fix;
  \powmnist=-1.4; unused count macros removed; \swot -> "SWOT/SSHa-L2";
  back matter drafted (codedata availability cites the Zenodo DOI).
- Bibliography: thebibliography stub -> copernicus.bst + references.bib.
  COMPILES CLEAN: zero errors, zero undefined refs/citations (pdflatex x3 +
  bibtex). \runningauthor{Prochaska} was required by copernicus.cls.
- Overleaf pushed (4a1acd0); ToDO.txt Sections A + B all [x] (SWOT
  description still [B]locked on Iury), Section E noise flag RESOLVED.
- Phase 3 DONE. Next: Phase 4 final pass (read-through, cross-ref audit,
  ToDO Section I "FINAL AUTHOR ITEMS" from the CLAUDE-QUESTION comments).

UPDATE 2026-08-10 (eighth session): PHASE 4 COMPLETE -- FINAL PASS DONE.
- Fable agent consistency read-through (~30 findings) + main-session
  compile/cross-ref audits. All findings fixed (see final_steps.md work
  log): number fixes (dx 2.25, N99 range, k-lambda), hedged overclaims,
  macro misuse (\llcsst vs \llcsstn), skimage (not sklearn), grammar,
  caption fixes, table order swap, \llcsstn -> "+noise" convention,
  fig_example_images + fig_learning_curves regenerated with proper labels,
  tab_model footnote rebuilt.
- Author answered all 10 CLAUDE-QUESTIONs inline on Overleaf mid-phase;
  rebased over their commit and acted on 3 (priority claim softened, SWOT
  eigenimage future-work sentence added, Dryad archiving statement).
- Compile: zero errors/warnings/undefined/overfull (settled chain).
- ToDO.txt: Section I "FINAL AUTHOR ITEMS" added + updated with answers.
- Overleaf pushed (6eb4934 rebased + 5e3a630). ALL FIVE PHASES COMPLETE;
  draft is submission-ready pending Section I items (Iury SWOT section,
  PMC/Dimitris inputs, journal choice, Dryad DOI, dates/affiliations).
