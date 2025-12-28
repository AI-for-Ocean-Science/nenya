"""  Module for Tables for the SSL paper """
# Imports
import os, sys
import json

import h5py

# Local
sys.path.append(os.path.abspath("../Analysis/py"))
import info_defs

from IPython import embed

def mktab_datasets(outfile='tab_datasets.tex', sub=False, local=True):

    if sub:
        outfile=outfile.replace('.tex', '_sub.tex')

    # Open
    tbfil = open(outfile, 'w')

    # Header
    #tbfil.write('\\clearpage\n')
    tbfil.write('\\begin{table*}\n')
    tbfil.write('\\centering\n')
    tbfil.write('\\caption{Datasets\\label{tab:datasets}}\n')
    tbfil.write('\\begin{tabular}{ccccccccccc}\n')
    tbfil.write('\\hline \n')
    tbfil.write('Name & Type & Source & \\npix & km pix$^{-1}$ & Processing \\\\ \n')
    #tbfil.write('(deg) & (deg) & & (K) \n')
    tbfil.write('\\\\ \n')
    tbfil.write('\\hline \n')

    # Loop me
    for dataset in info_defs.all_datasets:
        pdict = info_defs.grab_paths(dataset)

        # Name (convert _ to \_)
        cdataset = dataset.replace('_', '\\_')
        slin = f'{cdataset}'

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
        if 'SST' in dataset:
            slin += f' & {dataset.split("_")[0]}'
        elif 'SWOT' in dataset:
            slin += ' & SWOT'
        else:
            slin += ' &'

        # Npix
        preproc_file = pdict['preproc_file']
        preproc = h5py.File(preproc_file, 'r')
        slin += f'& {preproc["train"].shape[1]}'

        # km/pix
        if 'dx' in pdict:
            slin += f'& {pdict["dx"]:0.2f}'
        else:
            slin += f'& ...'

        # Processing - load from JSON opts file
        opts_file = os.path.join('../Analysis', pdict['opts_file'])
        processing_str = '& ...'
        if os.path.exists(opts_file):
            try:
                with open(opts_file, 'r') as f:
                    opts = json.load(f)

                # Build processing string from JSON parameters
                proc_parts = []

                # Random crop and jitter
                if 'random_cropjitter' in opts and opts['random_cropjitter']:
                    crop_dim, jitter = opts['random_cropjitter']
                    proc_parts.append(f'crop {crop_dim}')
                    if jitter > 0:
                        proc_parts.append(f'jit {jitter}')

                # Flip
                if opts.get('flip', False):
                    proc_parts.append('flip')

                # Rotate
                if opts.get('rotate', False):
                    proc_parts.append('rot')

                # Gaussian noise
                if 'gauss_noise' in opts and opts['gauss_noise'] > 0:
                    proc_parts.append(f'noise {opts["gauss_noise"]}')

                # Demean
                if opts.get('demean', False):
                    proc_parts.append('demean')

                if proc_parts:
                    processing_str = f'& {", ".join(proc_parts)}'
            except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
                # If there's an error reading the file, keep the default '...'
                pass

        slin += processing_str

        tbfil.write(slin)
        tbfil.write('\\\\ \n')

    # End
    tbfil.write('\\hline \n')
    tbfil.write('\\end{tabular} \n')
    tbfil.write('\\\\ \n')
    # Table note describing processing steps
    tbfil.write('\\begin{minipage}{0.9\\textwidth}\n')
    tbfil.write('\\small\n')
    tbfil.write('\\textbf{Processing abbreviations:} ')
    tbfil.write('crop $N$ = random crop to $N \\times N$ pixels; ')
    tbfil.write('jit $M$ = random spatial jitter up to $M$ pixels; ')
    tbfil.write('flip = random horizontal/vertical flip; ')
    tbfil.write('rot = random 90$^\\circ$ rotation; ')
    tbfil.write('noise $\\sigma$ = additive Gaussian noise with standard deviation $\\sigma$; ')
    tbfil.write('demean = subtract mean value from each cutout.\n')
    tbfil.write('\\end{minipage}\n')
    tbfil.write('\\end{table*} \n')

    tbfil.close()

    print('Wrote {:s}'.format(outfile))



# Command line execution
if __name__ == '__main__':

    mktab_datasets()