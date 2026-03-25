# Information Content Paper — Claude Guide

## Paper Overview

This paper investigates the information content of self-supervised (contrastive) learning latent representations applied to ocean remote sensing data. Using the Nenya framework (SimCLR with ResNet50 backbones), we train on multiple datasets (satellite SST, SSH, simulated fields, and reference datasets like MNIST/ImageNet) and analyze the latent space via PCA decomposition and power spectra P(k).

Reference: Prochaska et al. 2023 (IEEE TGRS), https://ui.adsabs.harvard.edu/abs/2023ITGRS..6100272P/abstract

## Communication Style
- Be direct and concise. No sycophantic preamble.
- When uncertain, say so explicitly.
- Be critical of prompts; do not simply aim to please.

## Sounds
- Use standard sound to announce task completion.
- Use a different sound when user input is needed.

## Code Conventions
- Use Python exclusively. Conda environment: `ocean14`.
- Separate analysis scripts (`Analysis/py/`) from figure scripts (`Figures/py/`) so figures can be regenerated without rerunning analysis.
- Jupyter notebooks for exploratory analysis; `.py` scripts for reproducible runs.
- When possible, reuse existing code and modules rather than writing new code.
- Key module: `Analysis/py/info_defs.py` defines all datasets and paths via `grab_paths()`.
- No centralized dependency management — each project is self-contained.

## Directory Structure

```
papers/InformationContent/
├── Analysis/
│   ├── opts/          # JSON config files for each dataset
│   ├── py/            # Analysis scripts (extraction, PCA, P(k))
│   ├── pca/           # PCA output files (.npz)
│   ├── Pk/            # Power spectrum output files
│   └── yaml/          # Training YAML configs (for Nautilus)
├── Figures/
│   ├── py/            # Figure generation scripts (figs_nenya_dim.py)
│   └── *.png          # Output figures
├── Preprocess/
│   └── py/            # Preprocessing scripts
├── Tables/
│   └── py/            # Table generation scripts (tables_info.py)
└── claude/            # This directory — paper writing coordination
```

## Datasets

Defined in `Analysis/py/info_defs.py`:
- **Ocean SST**: MODIS_SSTa, MODIS_SSTa_2km, VIIRS_SSTa, VIIRS_SSTa_2km, VIIRS_SSTa_sub
- **Ocean model (LLC)**: LLC_SSTa_nonoise, LLC_SSTa_noise, LLC_SSHa
- **Satellite SSH**: SWOT_L3
- **Reference/Natural**: WNoise, Pk2, Pk4, MNIST, ImageNet

## LaTeX / Overleaf Conventions
- Overleaf-synced.
- Figures go in `Figures/` with informative filenames.
- 1-inch margins. Author: `JXP \& Claude [model version]`.
- Documents show creation date and last-edit date at the top.
- Embed figures within the text near their description.
- Each project with an Overleaf document maintains a chronological change log.

## Bash Commands
- Safe bash commands may be run without prompting.
- Multiple agents may be used to parallelize work.

## Key Analysis Outputs
- PCA decomposition of latent vectors: `Analysis/pca/pca_latents_*.npz`
- Power spectra: `Analysis/Pk/Pk_*.npz`
- Eigenimages: `{dataset_path}/eigen/{dataset}_eigenimages.npz`
- Tables: `Tables/tab_datasets.tex`, `Tables/tab_model.tex`, `Tables/tab_analysis.tex`
- Figures: `Figures/fig_learning_curves.png`, `Figures/fig_pca_2panel.png`, `Figures/fig_pca_noise_res.png`, `Figures/fig_true_pca.png`, `Figures/Pk_all_datasets.png`

## Update History

### Overleaf Document
- Overleaf project: "Claude history"
- Keep a log of changes:
  - Start a new section for each 24h period. Draw horizontal lines between them. All entries chronological.
  - Write the date and a description of both the requests and changes implemented. Separate them clearly.
  - Write the number of total code lines of the project (if applicable).
- The document should be easily readable two ways: understanding what has been done as a function of time, and/or understanding what has been done for a specific project as a function of time.
- You may push to git as you work.  The access token is in my .bashrc profile with the name OVERLEAF.
