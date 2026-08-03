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