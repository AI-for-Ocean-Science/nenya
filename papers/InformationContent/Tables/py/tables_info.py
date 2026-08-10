"""  Module for Tables for the SSL paper """
# Imports
import os, sys
import json

import numpy as np
import h5py

# Local
sys.path.append(os.path.abspath("../Analysis/py"))
import info_defs
import calc_Pk

from IPython import embed


def mktab_datasets(outfile='tab_datasets.tex', sub=False, local=True):
    """Generate LaTeX table describing the datasets.

    This table focuses on data characteristics: name, type, source,
    spatial resolution, and pixel size.
    """

    if sub:
        outfile = outfile.replace('.tex', '_sub.tex')

    # Open
    tbfil = open(outfile, 'w')

    # Header
    tbfil.write('\\begin{table*}\n')
    tbfil.write('\\centering\n')
    tbfil.write('\\caption{Datasets\\label{tab:datasets}}\n')
    tbfil.write('\\begin{tabular}{ccccccc}\n')
    tbfil.write('\\hline \n')
    tbfil.write('Name & Type & Source & Year & Coverage & \\npix & \\dx \\\\ \n')
    tbfil.write('\\hline \n')

    # Loop me
    for dataset in info_defs.all_datasets:
        pdict = info_defs.grab_paths(dataset)

        # Name (convert _ to \_)
        #cdataset = dataset.replace('_', '\\_')
        slin = f'{pdict['macro']}'

        # Type (SST, SSH, etc.)
        if 'SST' in dataset:
            slin += ' & SST'
        elif 'SSHa' in dataset or 'SWOT' in dataset:
            slin += ' & SSH'
        elif dataset == 'WNoise':
            slin += ' & Noise'
        elif dataset in ['Pk2', 'Pk4']:
            slin += ' & Power-law'
        elif dataset == 'MNIST':
            slin += ' & Digits'
        elif dataset == 'ImageNet':
            slin += ' & Natural'
        else:
            slin += ' &'

        # Sensor/Source
        if 'SST' in dataset or 'SSH' in dataset:
            slin += f' & {dataset.split("_")[0]}'
        elif 'SWOT' in dataset:
            slin += ' & SWOT'
        elif dataset in ['MNIST', 'ImageNet']:
            slin += f' & {dataset}'
        else:
            slin += ' & Generated'

        # Year
        if 'year' in pdict:
            slin += f' & {pdict["year"]}'
        else:
            slin += ' & ...'

        # Geographic coverage
        if 'coverage' in pdict:
            slin += f' & {pdict["coverage"]}'
        else:
            slin += ' & ...'

        # Npix
        preproc_file = pdict['preproc_file']
        preproc = h5py.File(preproc_file, 'r')
        slin += f' & {preproc["train"].shape[1]}'

        # km/pix
        if 'dx' in pdict and dataset not in info_defs.natural_datasets:
            slin += f' & {pdict["dx"]:0.2f}'
        else:
            slin += ' & ...'

        tbfil.write(slin)
        tbfil.write(' \\\\ \n')

    # End
    tbfil.write('\\hline \n')
    tbfil.write('\\end{tabular} \n')
    tbfil.write('\\\\ \n')
    # Table notes
    tbfil.write('\\begin{minipage}{0.9\\textwidth}\n')
    tbfil.write('\\small\n')
    tbfil.write('\\textbf{Column descriptions:} ')
    tbfil.write('\\textit{Name}: dataset identifier (see text for definitions); ')
    tbfil.write('\\textit{Type}: data type (SST = sea surface temperature, SSH = sea surface height, ')
    tbfil.write('Noise = white noise, Power-law = synthetic power-law fields, Digits = handwritten digits, ')
    tbfil.write('Natural = natural images); ')
    tbfil.write('\\textit{Source}: instrument or data source; ')
    tbfil.write('\\textit{Year}: time period of the observations or model run, ')
    tbfil.write('with "..." indicating not applicable; ')
    tbfil.write('\\textit{Coverage}: geographic (latitude) coverage, ')
    tbfil.write('with "..." indicating not applicable; ')
    tbfil.write('\\npix\\ = number of pixels per side of the square cutouts; ')
    tbfil.write('\\dx\\ = spatial resolution in km per pixel with "..." indicating non-physical data.\n')
    tbfil.write('\\end{minipage}\n')
    tbfil.write('\\end{table*} \n')

    tbfil.close()

    print('Wrote {:s}'.format(outfile))


def mktab_model(outfile='tab_model.tex', sub=False, local=True):
    """Generate LaTeX table describing the contrastive learning model and its outputs.

    This table has separate columns for pre-processing steps and augmentations,
    as well as model architecture and output characteristics.
    """

    # Path to ulmo preproc options
    ulmo_preproc_path = '/home/xavier/Oceanography/python/ulmo/ulmo/preproc/options'

    # Mapping from dataset prefixes to ulmo preproc JSON files
    ulmo_preproc_files = {
        'MODIS': os.path.join(ulmo_preproc_path, 'preproc_standard.json'),
        'VIIRS': os.path.join(ulmo_preproc_path, 'preproc_viirs_std.json'),
        'LLC_SSTa_nonoise': os.path.join(ulmo_preproc_path, 'preproc_llc_144_nonoise.json'),
        'LLC_SSTa_noise': os.path.join(ulmo_preproc_path, 'preproc_llc_144.json'),
        'LLC_SSHa': os.path.join(ulmo_preproc_path, 'preproc_llc_std.json'),
    }

    if sub:
        outfile = outfile.replace('.tex', '_sub.tex')

    # Open
    tbfil = open(outfile, 'w')

    # Header
    tbfil.write('\\begin{table*}\n')
    tbfil.write('\\centering\n')
    tbfil.write('\\caption{Contrastive Learning Model Configuration\\label{tab:model}}\n')
    tbfil.write('\\begin{tabular}{ccccc}\n')
    tbfil.write('\\hline \n')
    tbfil.write('Name & Pre-processing & Augmentations & \\nfeature & $N_{99}$ \\\\ \n')
    tbfil.write('\\hline \n')

    # Loop me
    for dataset in info_defs.all_datasets:
        pdict = info_defs.grab_paths(dataset)

        # Name (convert _ to \_)
        cdataset = dataset.replace('_', '\\_')
        #slin = f'{cdataset}'
        slin = pdict['macro']

        # Load opts file for processing info
        opts_file = os.path.join('../Analysis', pdict['opts_file'])
        preproc_str = '...'
        augment_str = '...'
        model_str = '...'
        ndim_str = '...'

        # Determine which ulmo preproc file to use based on dataset name
        ulmo_preproc_file = None
        if dataset in ulmo_preproc_files:
            ulmo_preproc_file = ulmo_preproc_files[dataset]
        elif dataset.startswith('MODIS'):
            ulmo_preproc_file = ulmo_preproc_files['MODIS']
        elif dataset.startswith('VIIRS'):
            ulmo_preproc_file = ulmo_preproc_files['VIIRS']

        # Load ulmo preproc JSON to get pre-processing steps
        if ulmo_preproc_file and os.path.exists(ulmo_preproc_file):
            try:
                with open(ulmo_preproc_file, 'r') as f:
                    ulmo_opts = json.load(f)

                preproc_parts = []

                # Field size (original cutout size)
                #if 'field_size' in ulmo_opts:
                #    preproc_parts.append(f'{ulmo_opts["field_size"]}x{ulmo_opts["field_size"]}')

                # Downscale
                if '2km' in dataset and ulmo_opts.get('downscale', False) and 'dscale_size' in ulmo_opts:
                    dscale = ulmo_opts['dscale_size']
                    preproc_parts.append(f'downsample {dscale[0]}x{dscale[1]}')

                # Median filter
                if ulmo_opts.get('median', False) and 'med_size' in ulmo_opts:
                    med = ulmo_opts['med_size']
                    preproc_parts.append(f'median {med[0]}x{med[1]}')

                # Clear threshold
                #if 'clear_threshold' in ulmo_opts:
                #    preproc_parts.append(f'{ulmo_opts["clear_threshold"]}\\% clear')

                # Noise (for LLC simulations)
                if 'noise' in ulmo_opts and ulmo_opts['noise'] > 0:
                    noise_val = ulmo_opts['noise']
                    # The legacy ulmo json holds sigma=0.039, but the dataset was
                    # built by extract_llc.py add_noise(noise=0.09); verified
                    # numerically against the preproc h5 files (2026-08-10).
                    if dataset == 'LLC_SSTa_noise':
                        noise_val = 0.09
                    preproc_parts.append(f'noise $\\sigma$={noise_val:.3f}')

                if preproc_parts:
                    preproc_str = ', '.join(preproc_parts)
            except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
                pass

        if os.path.exists(opts_file):
            try:
                with open(opts_file, 'r') as f:
                    opts = json.load(f)

                # Augmentations (random transformations during training)
                augment_parts = []

                # Random crop (the base crop dimension)
                if 'random_cropjitter' in opts and opts['random_cropjitter']:
                    crop_dim, _ = opts['random_cropjitter']
                    augment_parts.append(f'crop {crop_dim}')

                # Jitter is an augmentation
                if 'random_cropjitter' in opts and opts['random_cropjitter']:
                    _, jitter = opts['random_cropjitter']
                    if jitter > 0:
                        augment_parts.append(f'jitter {jitter}')

                # Flip
                if opts.get('flip', False):
                    augment_parts.append('flip')

                # Rotate
                if opts.get('rotate', False):
                    augment_parts.append('rotate')

                # Gaussian noise
                if 'gauss_noise' in opts and opts['gauss_noise'] > 0:
                    augment_parts.append(f'noise {opts["gauss_noise"]}')

                if augment_parts:
                    augment_str = ', '.join(augment_parts)

                # Model architecture
                #ssl_method = opts.get('ssl_method', 'SimCLR')
                #ssl_model = opts.get('ssl_model', 'resnet50')
                #model_str = f'{ssl_method}/{ssl_model}'

                # Number of dimensions (feat_dim)
                if 'feat_dim' in opts:
                    ndim_str = f'{opts["feat_dim"]}'

            except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
                pass

        slin += f' & {preproc_str}'
        slin += f' & {augment_str}'
        #slin += f' & {model_str}'
        slin += f' & {ndim_str}'

        # N_99: number of latent vectors to explain 99% of variance
        pca_file = os.path.join('../Analysis', pdict['pca_file'])
        n99_str = '...'
        if os.path.exists(pca_file):
            try:
                d = np.load(pca_file)
                cumsum = 1 - np.cumsum(d['explained_variance'])
                # Find index where cumulative variance reaches 99%
                n99 = np.argmin(np.abs((1 - cumsum) - 0.99)) + 1  # +1 for 1-indexed
                n99_str = f'{n99}'
            except Exception as e:
                pass
        slin += f' & {n99_str}'

        tbfil.write(slin)
        tbfil.write(' \\\\ \n')

    # End
    tbfil.write('\\hline \n')
    tbfil.write('\\end{tabular} \n')
    tbfil.write('\\\\ \n')
    # Table note describing columns
    tbfil.write('\\begin{minipage}{0.9\\textwidth}\n')
    tbfil.write('\\small\n')
    tbfil.write('\\textbf{Column descriptions:} ')
    tbfil.write('\\textit{Pre-processing}: transformations applied to satellite/model data before training ')
    tbfil.write('($N$x$N$ = original cutout size in pixels; ')
    tbfil.write('downsample $M$x$M$ = spatial downsampling factor; ')
    tbfil.write('median $M$x$N$ = median filter kernel size; ')
    tbfil.write('$X$\\% clear = minimum cloud-free threshold; ')
    tbfil.write('noise $\\sigma$ = synthetic noise added to simulations); ')
    tbfil.write('\\textit{Augmentations}: random transformations applied during contrastive learning ')
    tbfil.write('(jitter $M$ = random spatial shift up to $M$ pixels; ')
    tbfil.write('flip = random horizontal/vertical flip; ')
    tbfil.write('rotate = random 90$^\\circ$ rotation; ')
    tbfil.write('noise $\\sigma$ = additive Gaussian noise with standard deviation $\\sigma$); ')
    tbfil.write('$N_{\\rm f}$ = latent space dimensionality; ')
    tbfil.write('$N_{99}$ = PCA components needed for 99\\% variance.\n')
    tbfil.write('\\end{minipage}\n')
    tbfil.write('\\end{table*} \n')

    tbfil.close()

    print('Wrote {:s}'.format(outfile))


def mktab_analysis(outfile='tab_analysis.tex', sub=False, local=True,
                   pix_min: int = 4, pix_max: int = 40):
    """Generate LaTeX table describing P(k) and PCA outputs for each dataset.

    This table includes power-law exponents from P(k) spectra and PCA
    variance statistics.

    Parameters
    ----------
    outfile : str
        Output filename for the LaTeX table
    sub : bool
        If True, append '_sub' to the output filename
    local : bool
        Unused parameter (kept for consistency with other methods)
    pix_min : int
        Minimum wavelength in pixels for power-law fit range
    pix_max : int
        Maximum wavelength in pixels for power-law fit range
    """

    if sub:
        outfile = outfile.replace('.tex', '_sub.tex')

    # Open
    tbfil = open(outfile, 'w')

    # Header
    tbfil.write('\\begin{table*}\n')
    tbfil.write('\\centering\n')
    tbfil.write('\\caption{Power Spectrum and PCA Analysis\\label{tab:analysis}}\n')
    tbfil.write('\\begin{tabular}{ccc}\n')
    tbfil.write('\\hline \n')
    tbfil.write('Name & $\\beta$ & $f_{\\rm var,256}$ \\\\ \n')
    tbfil.write('\\hline \n')

    # Loop me
    for dataset in info_defs.all_datasets:
        pdict = info_defs.grab_paths(dataset)

        # Name (convert _ to \_)
        cdataset = dataset.replace('_', '\\_')
        #slin = f'{cdataset}'
        slin = pdict['macro']

        # Power-law exponent from P(k)
        pk_result = calc_Pk.fit_powerlaw(dataset, pix_min=pix_min, pix_max=pix_max)
        if pk_result is not None:
            slin += f' & ${pk_result["exponent"]:.2f} \\pm {pk_result["exponent_err"]:.2f}$'
            #slin += f' & {pk_result["r_squared"]:.3f}'
        else:
            slin += ' & ... '#& ...'
            #slin += ' & ... & ...'

        # PCA statistics - variance explained by 256 eigenvectors
        pca_file = os.path.join('../Analysis', pdict['pca_imgfile'])
        var256_str = '...'

        if os.path.exists(pca_file):
            try:
                d = np.load(pca_file)
                explained_var = d['explained_variance_ratio']

                # Sum variance explained by first 256 components
                if len(explained_var) >= 256:
                    var256 = np.sum(explained_var[:256])
                    var256_str = f'{var256:.3f}'
                else:
                    # If fewer than 256 components, sum all available
                    var256 = np.sum(explained_var)
                    var256_str = f'{var256:.3f}'

            except Exception as e:
                print(f"Error loading PCA file for {dataset}: {e}")
                pass
        else:
            print(f"PCA file not found for {dataset}: {pca_file}")

        slin += f' & {var256_str}'

        tbfil.write(slin)
        tbfil.write(' \\\\ \n')

    # End
    tbfil.write('\\hline \n')
    tbfil.write('\\end{tabular} \n')
    tbfil.write('\\\\ \n')
    # Table note describing columns
    tbfil.write('\\begin{minipage}{0.9\\textwidth}\n')
    tbfil.write('\\small\n')
    tbfil.write('\\textbf{Column descriptions:} ')
    tbfil.write(f'$\\beta$ = power-law exponent from $P(k) \\propto \\lambda^\\beta$ fit over {pix_min}--{pix_max} pixel wavelengths; ')
    #tbfil.write('$R^2$ = coefficient of determination for power-law fit; ')
    tbfil.write('$f_{{\\rm var,256}}$ = fraction of variance explained by the first 256 PCA components.\n')
    tbfil.write('\\end{minipage}\n')
    tbfil.write('\\end{table*} \n')

    tbfil.close()

    print('Wrote {:s}'.format(outfile))


# Command line execution
if __name__ == '__main__':

    #mktab_datasets()
    #mktab_analysis()
    mktab_model()