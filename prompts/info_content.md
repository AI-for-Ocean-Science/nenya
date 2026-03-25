# Prompts related to the Information Content paper

# Figures

## Add a new method to nenya/papers/InformationContent/Figures/py/figs_nenya_dim.py to plot the PCA outputs on the latent space that has 2 panels, similar to the fig_pca_2panel method.  In this new method, call it fig_pca_noise_res(), one panel will examine the effects of noise and the other will examine the effects of resolution.  For now in the left panel, just use SST datasets and compare LLC_SST_nonoise and LLC_SST_noise as well as the Remote Sensing measures.  For resolution, compare VIIRS_SSTa_2km and VIIRS_SSTa and the LLC_SSTa_noise.

# SWOT

## Preproc file

### Plan

Modify swot.py in nenya/papers/InformationContent/Preprocess/py to create the preproc file for the SWOT L2 dataset.  Use a similar approach to main_L3, but for the SWOT L2 dataset.  Use SWOT_L2 in info_defs.grab_paths() to get the correct paths.  Provide a plan before you start writing the code.

### More planning 

Unlike the main_L3() method, also create a parquet table in Info/Tables/SWOT_L2_54km.parquet that contains the metadata for the SWOT L2 dataset.  This table should contain at least the following columns:

- pp_file: the path to the preproc file
- pp_type: the type of preproc file
- pp_idx: the index of the preproc file
- lon: the latitude of the center of the cutout, provided in longitude_avg
- lat: the latitude of the center of the cutout, provided in latitude_avg
- datetime: the timestamp of the cutout, provided in time

Update your plan

### Proceed

Proceed to generate the code.

### Modifications

The code should draw a total of 200,000 random cutouts from the SWOT L2 dataset, before splitting into train and valid sets.  

The table needs to be updated to record the indices of the cutouts in the preproc file.