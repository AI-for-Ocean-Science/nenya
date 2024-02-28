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

    # Load PCAs
    ds = []
    datasets = ['MODIS_SST', 'VIIRS_SST']
    #datasets = ['MODIS_SST']
    for dataset in datasets:
        d = np.load(f'../Analysis/pca_latents_{dataset}.npz')
        ds.append(d)

    # Fit
    d = ds[0]
    xs = np.arange(len(d['explained_variance'])) + 1
    exponent = -0.5
    ys = d['explained_variance'][10] * (xs/xs[10])**(exponent) 

    fig = plt.figure(figsize=(8,6))
    gs = gridspec.GridSpec(1,1)


    ax = plt.subplot(gs[0])
    for ss, d in enumerate(ds):
        ax.plot(xs, d['explained_variance'], 'o', label=datasets[ss])
        #ax.plot(xs, d['explained_variance'], 'o', label='Explained Variance')
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

    #loc = 'upper right' if ss == 1 else 'upper left'
    ax.legend(fontsize=15)#, loc=loc)

    # Turn on grid
    ax.grid(True, which='both', ls='--', lw=0.5)

    plotting.set_fontsize(ax, 18)

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

# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0

        flg += 2 ** 19  # PCA veriance
        
    else:
        flg = sys.argv[1]

    main(flg)