"""  Module for Tables for the SSL paper """
# Imports
import os, sys

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
    tbfil.write('\\begin{tabular}{cccccccccc}\n')
    tbfil.write('\\hline \n')
    tbfil.write('Name & Source & \\npix & km pix$^{-1}$ & Processing \\\\ \n')
    #tbfil.write('(deg) & (deg) & & (K) \n') 
    tbfil.write('\\\\ \n')
    tbfil.write('\\hline \n')

    # Loop me 
    for dataset in info_defs.all_datasets:
        pdict = info_defs.grab_paths(dataset)

        # Name (convert _ to \_)
        cdataset = dataset.replace('_', '\\_')
        slin = f'{cdataset}'

        # Sensor
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

        # Processing
        
        tbfil.write(slin)
        tbfil.write('\\\\ \n')

    # End
    tbfil.write('\\hline \n')
    tbfil.write('\\end{tabular} \n')
    #tbfil.write('\\end{minipage} \n')
    tbfil.write('\\\\ \n')
    #tbfil.write('Notes: The \\DT\\ value listed here is measured from the inner $40 \\times 40$\,pixel$^2$ region of the cutout. \\\\ \n')
    #tbfil.write('LL is the log-likelihood metric calculated from the \\ulmo\\ algorithm. \\\\ \n')
    #tbfil.write('$U_{0,\\rm all}, U_{1,\\rm all}$ are the UMAP values for the UMAP analysis on the full dataset. \\\\ \n')
    #tbfil.write('$U_0, U_1$ are the UMAP values for the UMAP analysis in the \\DT\\ bin for this cutout. \\\\ \n')
    #tbfil.write('{$^b$}Assumes $\\nu=1$GHz, $n_e = 4 \\times 10^{-3} \\cm{-3}$, $z_{\\rm DLA} = 1$, $z_{\\rm source} = 2$.\\\\ \n')
    tbfil.write('\\end{table*} \n')

    tbfil.close()

    print('Wrote {:s}'.format(outfile))



# Command line execution
if __name__ == '__main__':

    mktab_datasets()