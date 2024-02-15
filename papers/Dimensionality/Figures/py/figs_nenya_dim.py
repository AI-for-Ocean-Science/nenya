""" Figures for Paper I on IHOP """

# imports
import os
import sys
from importlib import resources

import numpy as np

import torch

from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec

import seaborn as sns

from oceancolor.hydrolight import loisel23
from oceancolor.utils import plotting 

mpl.rcParams['font.family'] = 'stixgeneral'

# Local
#sys.path.append(os.path.abspath("../Analysis/py"))
#import reconstruct

from IPython import embed


def fig_pca(outfile:str='fig_pca_variance.png'): 

    # Load PCA
    d = np.load('../Analysis/pca_latents_MODIS_SST.npz')

    # Fit?
    xs = np.arange(len(d['explained_variance'])) + 1
    exponent = -0.5
    ys = d['explained_variance'][10] * (xs/xs[10])**(exponent) 

    fig = plt.figure(figsize=(8,6))
    gs = gridspec.GridSpec(1,1)


    ax = plt.subplot(gs[0])
    ax.plot(xs, d['explained_variance'], 'ob',
            label='Explained Variance')
    ax.plot(xs, ys, '--', color='g', label=f'Power law: {exponent}')
    # Label
    ax.set_ylabel('Variance explained per mode')
    ax.set_xlabel('Number of PCA components')
    #
    #ax.set_xlim(0,10.)
    ax.legend()
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Minor ticks
    ax.minorticks_on()
    # Horizontal line at 0
    #ax.axhline(0., color='k', ls='--')

    loc = 'lower left'
    ax.legend(fontsize=17, loc=loc)

    # Turn on grid
    ax.grid(True, which='both', ls='--', lw=0.5)

    plotting.set_fontsize(ax, 20)

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    # PCA variaince
    if flg & (2**19):
        fig_pca()

    # Example spectra
    if flg & (2**20):
        fig_emulator_rmse('L23', 3, [512, 512, 512, 256])
        #fig_emulator_rmse(['L23_NMF', 'L23_PCA'], [3, 3])

    # L23 IHOP performance vs. perc error
    if flg & (2**21):
        fig_mcmc_fit()#use_quick=True)

    # L23 IHOP performance vs. perc error
    if flg & (2**22):
        #fig_rmse_vs_sig()
        fig_rmse_vs_sig(decomp='nmf')


# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0

        flg += 2 ** 19  # PCA veriance
        #flg += 2 ** 20  # RMSE of emulators
        #flg += 2 ** 21  # Single MCMC fit (example)
        #flg += 2 ** 22  # RMSE of L23 fits

        #flg += 2 ** 2  # 4 -- Indiv
        #flg += 2 ** 3  # 8 -- Coeff
        #flg += 2 ** 4  # 16 -- Fit CDOM
        #flg += 2 ** 5  # 32 -- Explained variance
        
    else:
        flg = sys.argv[1]

    main(flg)