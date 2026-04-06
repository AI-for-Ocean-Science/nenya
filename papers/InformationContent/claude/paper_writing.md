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

- The paper LaTeX source is in this Overleaf-synced directory: /home/xavier/Projects/overleaf/Info_content 
- Figures go in the Overleaf project folder as PNG files in the Figures/ subdirectory.
- Embed figures within the text near their description.
- When adding new files, update the main `.tex` file to include them.
- You may push to git as you work.

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