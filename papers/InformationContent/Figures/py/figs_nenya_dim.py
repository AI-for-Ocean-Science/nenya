""" Figures for Paper I on IHOP """

# imports
import os
import sys
from importlib import resources

import numpy as np
import h5py

import torch
from sklearn.metrics.pairwise import cosine_similarity

from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as patheffects

import seaborn as sns

from remote_sensing.plotting import utils as rsp_utils

from wrangler import s3_io

from nenya import plotting as nenya_plotting
from nenya import params 
from nenya import io as nenya_io
from nenya import analysis
from nenya import pca as nenya_pca

mpl.rcParams['font.family'] = 'stixgeneral'

# Local
sys.path.append(os.path.abspath("../Analysis/py"))
import info_defs

from IPython import embed

# Color eict
cdict = {}
cdict['MODIS'] = '#1f77b4'  # Blue
cdict['VIIRS'] = '#ff7f0e'  # Orange
cdict['LLC'] = '#2ca02c'  # Green
cdict['LLC_SSHa'] = '#9467bd'  # Purple
# Red
cdict['SWOT_L3'] = '#d62728'  # Red
# Black
cdict['ImageNet'] = '#000000'  # Black
# Silver
cdict['WNoise'] = '#C0C0C0'
# Brown
cdict['Pk2'] = '#8c564b'  # Brown
# Tan
cdict['Pk4'] = '#D2B48C'
# Gray
cdict['MNIST'] = '#7f7f7f'  # Gray


def grab_clr(dataset:str):
    if 'SST' in dataset:
        clr = cdict[dataset.split('_')[0]]
    else:
        clr = cdict[dataset]
    return clr

def grab_ls(dataset:str):
    if 'sub' in dataset:
        ls = '--' 
    elif '_noise' in dataset:
        ls = '--' 
    elif '2km' in dataset:
        ls = ':' 
    else:
        ls = '-'

    return ls

def fig_pca_2panel(outfile:str='fig_pca_2panel.png',
                   cumulative:bool=False,
                   show_cum_point:float=None,
                   xmnx:tuple=None,
                   exponent:float=-0.5):
    """
    Generate a 2-panel PCA variance explained plot.

    Panel 1 (left): Natural Images - MNIST + ImageNet
    Panel 2 (right): Remote Sensing - MODIS_SST + ImageNet

    Args:
        outfile (str): The output file path for the saved plot.
        cumulative (bool): If True, plot the cumulative variance explained.
        show_cum_point (float): If provided, marks the point where cumulative
            variance reaches this value.
        xmnx (tuple): Sets xlim of the x-axis if provided.
        exponent (float): The exponent for the power-law fit line.
    """
    # Define datasets for each panel
    natural_datasets_panel = info_defs.natural_datasets
    remote_datasets_panel = info_defs.primary_remote_datasets
    remote_datasets_panel += ['ImageNet']

    # Cumulative filename adjustment
    if cumulative:
        if 'variance' in outfile:
            outfile = outfile.replace('variance', 'cumulative')

    # Create figure with 2 panels
    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(1, 2)

    panels = [
        (natural_datasets_panel, 'Natural Images'),
        (remote_datasets_panel, 'Remote Sensing')
    ]

    for panel_idx, (datasets, title) in enumerate(panels):
        ax = plt.subplot(gs[panel_idx])

        # Load and plot each dataset
        for ss, dataset in enumerate(datasets):
            pdict = info_defs.grab_paths(dataset)
            clr = grab_clr(dataset)
            ls = grab_ls(dataset)

            pca_file = f'../Analysis/{pdict["pca_file"]}'
            print(f"Loading PCA file: {pca_file}")
            d = np.load(pca_file)

            # Calculate y values
            cumsum = 1 - np.cumsum(d['explained_variance'])
            if cumulative:
                yvals = cumsum
            else:
                yvals = d['explained_variance']

            xs = np.arange(d['explained_variance'].size) + 1
            ax.plot(xs, yvals, label=dataset.replace('_', '/'),
                    color=clr, ls=ls, lw=2)

            # Add cumulative point marker
            if show_cum_point is not None:
                imin = np.argmin(np.abs((1 - cumsum) - show_cum_point))
                ax.plot(imin + 1, yvals[imin], 'x', color=clr, markersize=10)

        # Add power-law reference line
        xs_ref = np.arange(d['explained_variance'].size) + 1
        ys = d['explained_variance'][10] * (xs_ref / xs_ref[10])**(exponent)
        ax.plot(xs_ref, ys, '--', color='gray', label=f'Power law: {exponent}')

        # Labels and formatting
        ax.set_title(title, fontsize=18)
        if cumulative:
            ax.set_ylabel('Cumulative Variance explained per mode')
        else:
            ax.set_ylabel('Variance explained per mode')
        ax.set_xlabel('Number of PCA components (Latent Space)')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.minorticks_on()
        ax.legend(fontsize=13, loc='lower left')
        ax.grid(True, which='both', ls='--', lw=0.5)

        if xmnx is not None:
            ax.set_xlim(xmnx)

        rsp_utils.set_fontsize(ax, 16)

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


def fig_pca(outfile:str='fig_pca_variance.png',
            datasets:list=None, cumulative:bool=False,
            frac_remain:bool=False,
            show_cum_point:float=None,
            xmnx:tuple=None,
            exponent:float=-0.5):
    """
    Generate and save a PCA variance explained plot.
    This function creates a plot to visualize the variance explained by PCA components
    for a given set of datasets. It supports cumulative variance, fractional remaining
    variance, and power-law fitting.
    Args:
        outfile (str): The output file path for the saved plot. Defaults to 'fig_pca_variance.png'.
        datasets (list): A list of dataset names to include in the plot. If None, a default
            list of datasets is used. Defaults to None.
        cumulative (bool): If True, plot the cumulative variance explained. Defaults to False.
        frac_remain (bool): If True, plot the fractional remaining variance. Defaults to False.
        show_cum_point (float): If provided, marks the point on the plot where the cumulative
            variance reaches this value. Defaults to None.
        xmnx (tuple): Sets xlim of the x-axis if provided. Defaults to None.
        exponent (float): The exponent for the power-law fit line. Defaults to -0.5.
    Returns:
        None: The function saves the plot to the specified output file.
    Notes:
        - The function expects PCA data files to be located in the '../Analysis/pca/' directory
            with filenames formatted as 'pca_latents_<dataset>.npz'.
        - The datasets are color-coded, and different line styles are used to distinguish
            between dataset types.
        - The plot is saved in log-log scale with grid lines enabled.
    """
    # Cumulative?
    if cumulative:
        if 'variance' in outfile:
            outfile = outfile.replace('variance', 'cumulative')

    # Load PCAs
    if datasets is None:
        datasets = info_defs.all_datasets
    clrs = []
    ds = []
    for dataset in datasets:
        #if dataset in ['Pk2', 'Pk4']:
        #    continue
        pdict = info_defs.grab_paths(dataset)
        clr = grab_clr(dataset)
        
        pca_file = f'../Analysis/{pdict['pca_file']}'
        print(f"Loading PCA file: {pca_file}")
        d = np.load(pca_file)
        ds.append(d)
        #
        clrs.append(clr)

    #embed(header='PCA Variance Explained 89')

    # 
    fig = plt.figure(figsize=(8,6))
    gs = gridspec.GridSpec(1,1)

    ax = plt.subplot(gs[0])
    for ss, d in enumerate(ds):
        ls = grab_ls(datasets[ss])
        # Cumulative?
        cumsum = 1-np.cumsum(d['explained_variance'])
        if cumulative:
            yvals = cumsum
        elif frac_remain:
            cumsum = 1-np.cumsum(d['explained_variance'])
            yvals = d['explained_variance'] / cumsum
        else:
            yvals = d['explained_variance']
        ax.plot(np.arange(d['explained_variance'].size)+1, 
                yvals,  label=datasets[ss].replace('_','/'),
                color=clrs[ss], ls=ls)
        # Add cum point
        if show_cum_point is not None:
            imin = np.argmin(np.abs((1-cumsum) - show_cum_point))
            ax.plot(imin+1, yvals[imin], 'x', color=clrs[ss])

            
        if ss == 0:
            xs = np.arange(d['explained_variance'].size)+1

    ys = d['explained_variance'][10] * (xs/xs[10])**(exponent) 
    ax.plot(xs, ys, '--', color='gray', label=f'Power law: {exponent}')
    # Label
    if cumulative:
        ax.set_ylabel('Cumulative Variance explained per mode')
    else:
        ax.set_ylabel('Variance explained per mode')
    ax.set_xlabel('Number of PCA components (Latent Space)')
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
    ax.legend(fontsize=15, loc='lower left')

    # Turn on grid
    ax.grid(True, which='both', ls='--', lw=0.5)

    # xlim?
    if xmnx is not None:
        ax.set_xlim(xmnx)

    rsp_utils.set_fontsize(ax, 18)

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


def fig_true_pca(outfile:str='fig_true_pca.png',
            datasets:list=None, cumulative:bool=False,
            frac_remain:bool=False,
            show_cum_point:float=None,
            xmnx:tuple=None,
            exponent:float=-0.5):
    """
    Generate and save a 2-panel PCA variance explained plot for true (image-space) PCA.

    Panel 1 (left): Natural Images
    Panel 2 (right): Primary Remote Sensing datasets

    Args:
        outfile (str): The output file path for the saved plot. Defaults to 'fig_true_pca.png'.
        datasets (list): Ignored - uses natural_datasets and primary_remote_datasets.
        cumulative (bool): If True, plot the cumulative variance explained. Defaults to False.
        frac_remain (bool): If True, plot the fractional remaining variance. Defaults to False.
        show_cum_point (float): If provided, marks the point on the plot where the cumulative
            variance reaches this value. Defaults to None.
        xmnx (tuple): Sets xlim of the x-axis if provided. Defaults to None.
        exponent (float): The exponent for the power-law fit line. Defaults to -0.5.
    Returns:
        None: The function saves the plot to the specified output file.
    Notes:
        - The function expects PCA data files to be located in the '../Analysis/pca/' directory
            with filenames formatted as 'pca_preproc_<dataset>.npz'.
        - The datasets are color-coded, and different line styles are used to distinguish
            between dataset types.
        - The plot is saved in log-log scale with grid lines enabled.
    """
    # Cumulative filename adjustment
    if cumulative:
        if 'true_pca' in outfile:
            outfile = outfile.replace('true_pca', 'true_pca_cumulative')

    # Define datasets for each panel
    natural_datasets_panel = info_defs.natural_datasets
    remote_datasets_panel = info_defs.primary_remote_datasets
    remote_datasets_panel += ['ImageNet']

    # Create figure with 2 panels
    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(1, 2)

    panels = [
        (natural_datasets_panel, 'Natural Images'),
        (remote_datasets_panel, 'Remote Sensing')
    ]

    for panel_idx, (panel_datasets, title) in enumerate(panels):
        ax = plt.subplot(gs[panel_idx])

        xs = None  # Will be set from first valid dataset

        # Load and plot each dataset
        for ss, dataset in enumerate(panel_datasets):
            clr = grab_clr(dataset)
            ls = grab_ls(dataset)

            pca_file = f'../Analysis/pca/pca_preproc_{dataset}.npz'
            print(f"Loading PCA file: {pca_file}")
            try:
                d = np.load(pca_file)
            except:
                print(f"PCA file for {dataset} not found, skipping -- {pca_file}")
                continue

            # Calculate y values
            cumsum = 1 - np.cumsum(d['explained_variance_ratio'])
            if cumulative:
                yvals = cumsum
            elif frac_remain:
                yvals = d['explained_variance'] / cumsum
            else:
                yvals = d['explained_variance_ratio']

            xs_curr = np.arange(d['explained_variance_ratio'].size) + 1
            if xs is None:
                xs = xs_curr

            ax.plot(xs_curr, yvals, label=dataset.replace('_', '/'),
                    color=clr, ls=ls, lw=2)

            # Add cumulative point marker
            if show_cum_point is not None:
                imin = np.argmin(np.abs((1 - cumsum) - show_cum_point))
                ax.plot(imin + 1, yvals[imin], 'x', color=clr, markersize=10)

        # Labels and formatting
        ax.set_title(title, fontsize=18)
        if cumulative:
            ax.set_ylabel('Cumulative Variance explained per mode')
        else:
            ax.set_ylabel('Variance explained per mode')
        ax.set_xlabel('Number of True PCA components')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.minorticks_on()
        ax.legend(fontsize=13, loc='lower left')
        ax.grid(True, which='both', ls='--', lw=0.5)

        if xmnx is not None:
            ax.set_xlim(xmnx)

        rsp_utils.set_fontsize(ax, 16)

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


def fig_learning_curves(outfile:str='fig_learning_curves.png',
                        show_train:bool=False):
    """Plot the learning curves for SWOT, VIIRS, MODIS and MNIST datasets."""
    
    # Define the datasets
    #datasets = ['VIIRS_SST', 'MODIS_SST', 'MNIST', 'SWOT_L3', 
    #            'WNoise', 'ImageNet']
    datasets = info_defs.all_datasets
    
    # Create a figure
    fig = plt.figure(figsize=(12, 10))
    plt.clf()
    ax = plt.gca()

    for ss, dataset in enumerate(datasets):
        print(f'Processing dataset: {dataset}')
        clr = grab_clr(dataset)
        ls = grab_ls(dataset)
        #path = dataset_path(dataset)
        pdict = info_defs.grab_paths(dataset)
        opt = params.Params('../Analysis/'+pdict['opts_file'])
        params.option_preprocess(opt)
        #embed(header=f"Learning curves for {dataset}")
        losses_train, losses_valid = nenya_io.losses_filenames(opt)

        valid_file = os.path.join(pdict['path'], losses_valid)
        train_file = os.path.join(pdict['path'], losses_train)
        with s3_io.open(valid_file, 'rb') as f:
            valid_hf = h5py.File(f, 'r')
            loss_valid = valid_hf['loss_valid'][:]
        with s3_io.open(train_file, 'rb') as f:
            train_hf = h5py.File(f, 'r')
            loss_train = train_hf['loss_train'][:]

        # Plot
        if ss == 0:
            lbl0 = f'{dataset} validation'
            lbl1 = f'{dataset} training'
        else:
            lbl0 = f'{dataset}'
            lbl1 = None
        ax.plot(np.arange(loss_valid.size)+1, loss_valid, label=lbl0, lw=3, color=clr, ls=ls)
        if show_train:
            ax.plot(np.arange(loss_train.size)+1, loss_train, label=lbl1, lw=3, color=clr, ls='--')

        
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_yscale('log')
    ax.grid()

    ax.legend(fontsize=15, loc='upper right')

    rsp_utils.set_fontsize(ax, 24.)
    
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def fig_learning_curve(dataset:str='SWOT'):
    outfile=f'fig_{dataset}_learning_curve.png'
    # Load the learning curve files
    path = dataset_path(dataset)
    valid_file = os.path.join(path, 'learning_curve',
                              'SimCLR_resnet50_lr_0.05_decay_0.0001_bsz_64_temp_0.07_trial_5_cosine_warm_losses_valid.h5')
    train_file = os.path.join(path, 'learning_curve',
                              'SimCLR_resnet50_lr_0.05_decay_0.0001_bsz_64_temp_0.07_trial_5_cosine_warm_losses_train.h5')
    # Plot the learning curve
    nenya_plotting.learn_curve(valid_file, train_file, 
                                   outfile=outfile, ylog=True)

def fig_swot_umap(outfile:str='fig_swot_umap_gallery.png'):
    swot_path = os.getenv('SWOT_PNGs')
    tbl_file = os.path.join(swot_path,'Pass_006.parquet')
    img_file = os.path.join(swot_path,'Pass_006.h5')
    nenya_plotting.umap_gallery(tbl_file, img_file, 
                                dxv=1.0, dyv=1.0,
                                #scl_inset=(0.9,0.9),
                                in_vmnx=(-0.5, 1.5),
                                cbar_lbl='SWOT/SSR',
                                outfile=outfile,
                                debug=True,
                                cmap="Greys",)

def fig_eigenimages(dataset:str, cmap:str, Nimages:int=9, 
                    outroot:str='fig_eigenmodes'):

    outfile = f'{outroot}_{dataset}.png'

    # Load eigenimages
    pdict = info_defs.grab_paths(dataset)
    d = np.load(pdict['eigen_file'])


    fig = plt.figure(figsize=(6,6))
    gs = gridspec.GridSpec(3,3)

    for ss in range(Nimages):
        img = d['eigen_images'][ss][0,...]

        ax = plt.subplot(gs[ss]) 
        _ = sns.heatmap(np.flipud(img), xticklabels=[], 
                     #vmin=vmnx[0], vmax=vmnx[1], 
                     ax=ax,
                     yticklabels=[], cmap=cmap, cbar=False) 
                     #cbar_kws={'label': clbl})# 'fontsize': 20})
        # Title
        title = f'Eigenmode {ss+1} sim={d["similarities"][ss]:.2f}'
        ax.set_title(title, fontsize=12)

    #rsp_utils.set_fontsize(ax, 18)

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def fig_eigenmatches(dataset:str, cmap:str, Nmodes:int=9, 
                     partition:str='train', 
                     last_ones:bool=False,
                     outroot:str='fig_eigenmatches'):

    outfile = f'{outroot}_{dataset}.png'

    # Load the PCA model
    pdict = info_defs.grab_paths(dataset)
    pca_file = '../Analysis/'+pdict['pca_file']

    # Open the preproc file
    preproc_file = pdict['preproc_file']
    preproc = h5py.File(preproc_file, 'r')

    if last_ones:
        d = np.load(pca_file)
        cumsum = 1-np.cumsum(d['explained_variance'])
        imin = np.argmin(np.abs((1-cumsum) - 0.99))
        modes=imin+np.arange(-Nmodes,0)
    else:
        modes=np.arange(Nmodes)
    image_idx, similarities = nenya_pca.find_eigenmatches(
        pca_file, pdict['latents_file'],
        modes=modes)


    fig = plt.figure(figsize=(6,6))
    gs = gridspec.GridSpec(3,3)

    for ss,mode in enumerate(modes):
        # Grab the image
        img = preproc[partition][image_idx[0, ss]]
        if img.ndim == 3:
            img = img[0,...]

        ax = plt.subplot(gs[ss]) 
        _ = sns.heatmap(np.flipud(img), xticklabels=[], 
                     #vmin=vmnx[0], vmax=vmnx[1], 
                     ax=ax,
                     yticklabels=[], cmap=cmap, cbar=False) 
                     #cbar_kws={'label': clbl})# 'fontsize': 20})
        # Title
        title = f'Ematch: mode={mode+1} sim={similarities[0, ss]:.2f}'
        ax.set_title(title, fontsize=10)

    #rsp_utils.set_fontsize(ax, 18)

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def fig_example_images(outfile:str='fig_example_images.png',
                       idx:int=0, fourth:str='Pk4'):
    """
    Generate a 4-panel figure showing example images from VIIRS_SST, SWOT_L3,
    ImageNet, and LLC_SST datasets.

    Args:
        outfile (str): The output file path for the saved plot.
        idx (int): Index of the image to show from each dataset.
    """
    datasets = ['VIIRS_SST', 'SWOT_L3', 'ImageNet']
    titles = ['VIIRS SST', 'SWOT L3', 'ImageNet']
    cmaps = ['jet', 'RdBu_r', 'gray']#, 'jet']  # gray for ImageNet if single-channel
    cbar_labels = ['SSTa (K)', 'SSHa (m)', 'Intensity']#, 'SSTa (K)']
    if fourth == 'LLC':
        datasets += ['LLC_SST_nonoise']
        titles += ['LLC_SST']
        cmaps += ['jet']
        cbar_labels += ['SSTa (K)']
    elif fourth == 'Pk4':
        datasets += ['Pk4']
        titles += [r'$P(k) \propto k^{-4}$']
        cmaps += ['Greens']
        cbar_labels += ['Intensity']
    else:
        raise IOError(f"Bad fourth: {fourth}")

    # Physical scales (km) - from info_defs
    # VIIRS: 0.75 km/pixel, 64 pixels -> 48 km
    # SWOT: 0.25 km/pixel, 64 pixels -> 16 km
    # LLC: 144/64 km/pixel, 64 pixels -> 144 km
    scales = {
        'VIIRS_SST': 0.75 * 192,  # 48 km
        'SWOT_L3': 0.25 * 128,    # 16 km
        'LLC_SST_nonoise': (144./64) * 64,  # 144 km
        'ImageNet': None,
        'Pk4': None
    }

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    for ax, dataset, title, cmap, cbar_lbl in zip(axes.flatten(), datasets, titles, cmaps, cbar_labels):
        pdict = info_defs.grab_paths(dataset)
        preproc_file = pdict['preproc_file']

        with h5py.File(preproc_file, 'r') as f:
            img = f['train'][idx]

        # Handle different image formats
        if img.ndim == 3:
            if img.shape[0] == 1:
                # Single channel, squeeze
                img = img[0, ...]
            elif img.shape[0] == 3:
                # RGB image (C, H, W) -> (H, W, C)
                img = np.transpose(img, (1, 2, 0))
                # Normalize to [0, 1] for display
                img = (img - img.min()) / (img.max() - img.min())

        # Plot
        if img.ndim == 2:
            orig = None if dataset in ['ImageNet', 'Pk4'] else 'lower'
            im = ax.imshow(img, cmap=cmap, origin=orig)
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cbar.set_label(cbar_lbl, fontsize=16)
            cbar.ax.tick_params(labelsize=12)
        else:
            embed(header='637 of figs')

        ax.set_title(title, fontsize=16)
        ax.set_xticks([])
        ax.set_yticks([])

        # Add physical scale bar for SST and SWOT
        scale_km = scales[dataset]
        if scale_km is not None:
            npix = img.shape[0]
            # Add scale bar (10 km or appropriate fraction)
            bar_km = 10 if scale_km < 50 else (50 if scale_km < 150 else 50)
            bar_pix = bar_km / (scale_km / npix)
            # Position scale bar
            x0, y0 = 5, 5
            ax.plot([x0, x0 + bar_pix], [y0, y0], 'k-', lw=3)
            ax.text(x0 + bar_pix/2, y0 + 3, f'{bar_km} km',
                    ha='center', va='bottom', fontsize=14,
                    color='black', fontweight='bold')

    plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    print(f"Saved: {outfile}")


def fig_Pk():

    # go
    plt.figure(figsize=(12,6))
    # two panels: Natural (left), Remote Sensing (right)
    gs = gridspec.GridSpec(1,2)
    ax_natural = plt.subplot(gs[0])
    ax_remote = plt.subplot(gs[1])

    for dataset in info_defs.all_datasets:
        # Skip 2km and sub datasets for remote sensing
        if dataset not in info_defs.natural_datasets:
            if '2km' in dataset or 'sub' in dataset:
                continue

        pdict = info_defs.grab_paths(dataset)
        pk_file = os.path.join('../Analysis', pdict['Pk_file'])
        if not os.path.exists(pk_file):
            print(f"Pk file for {dataset} not found, skipping -- {pk_file}")
            continue
        # Load
        data = np.load(pk_file)
        k = data['wavenumber']
        power = data['power']
        wavelength = data['wavelength']

        if 'sub' in dataset:
            ls = '--'
        elif '_noise' in dataset:
            ls = '--'
        elif '2km' in dataset:
            ls = ':'
        else:
            ls = '-'
        clr = grab_clr(dataset)

        # Pixel units
        dx = pdict['dx'] if 'dx' in pdict else 1.0

        # Skip the highest wavenumber (first element, since k is typically sorted high to low
        # or last element if sorted low to high) - skip last point
        skip_i = -1
        skip_j = 1
        k = k[skip_j:skip_i]
        power = power[skip_j:skip_i]
        wavelength = wavelength[skip_j:skip_i]

        print(f'{dataset}: k={k[0]}, wave={wavelength[-1]*0.70}')

        # Scale SSH
        if 'SSH' in dataset or 'SWOT' in dataset:
            power /= 2.  #  Geostrophy gives Deta/Dt ~ 2cm/K

        # Plot to appropriate panel
        if dataset in info_defs.natural_datasets:
            ax_natural.loglog(k, power*k, label=dataset,
                        color=clr, ls=ls)
        else:
            ax_remote.loglog(k, power*k, label=dataset,
                      color=clr, ls=ls)

    # Add power-law reference curves on the left panel (Natural Images)
    # Use wavelength range from the natural panel
    wv_ref = np.logspace(0.5, 2, 50)  # wavelengths in pixels
    k_ref = 1.0 / wv_ref
    # Normalize to a reference point
    norm_idx = len(wv_ref) // 2
    norm_val = 1e-3  # arbitrary normalization for visibility

    # k^-2 power law: P(k) ~ k^-2, so k*P(k) ~ k^-1 ~ wavelength^1
    pk2_power = norm_val * (wv_ref / wv_ref[norm_idx])**1
    ax_natural.loglog(k_ref, pk2_power, ':', color=cdict['Pk2'],
                      label=r'$k^{-2}$', lw=2)

    # k^-4 power law: P(k) ~ k^-4, so k*P(k) ~ k^-3 ~ wavelength^3
    pk4_power = norm_val * (wv_ref / wv_ref[norm_idx])**3
    ax_natural.loglog(k_ref, pk4_power, ':', color=cdict['Pk4'],
                      label=r'$k^{-4}$', lw=2)

    # Labels and formatting
    ax_natural.set_title('Natural Images', fontsize=16)
    ax_natural.legend(fontsize=12, loc='lower left')
    ax_natural.set_xlabel('Wavenumber (cycles/pixels)')
    ax_natural.set_ylabel(r'Power Spectrum per log bin: $k \, P(k)$')
    ax_natural.grid(True, which='both', ls='--', lw=0.5)

    ax_remote.set_title('Remote Sensing', fontsize=16)
    ax_remote.legend(fontsize=12, loc='lower left')
    ax_remote.set_xlabel('Wavenumber (cycles/km)')
    ax_remote.set_ylabel(r'Power Spectrum per log bin: $k \, P(k)$')
    ax_remote.grid(True, which='both', ls='--', lw=0.5)

    # Invert x-axes so wavenumber increases left to right (wavelength decreases)
    #ax_natural.invert_xaxis()
    #ax_remote.invert_xaxis()

    # Add wavelength on the top axis (swapped from wavenumber)
    ax_top = ax_remote.secondary_xaxis('top', functions=(lambda x: 1/x, lambda x: 1/x))
    ax_top.set_xlabel('Wavelength (km)')

    #ax_top2 = ax_natural.secondary_xaxis('top', functions=(lambda x: 1e3/x, lambda x: 1e3/x))
    #ax_top2.set_xlabel('Size (pixels)')

    for ax in [ax_natural, ax_remote, ax_top]:#, ax_top2]:
        rsp_utils.set_fontsize(ax, 18)


    plt.tight_layout()
    plt.savefig('Pk_all_datasets.png', dpi=300)
    plt.close()
    print(f'Wrote: Pk_all_datasets.png')


def fig_multi_eigenmatches(
    dataset:str, cmap:str, modes:list=np.arange(9),
    partition:str='train', nimages:int=9,
    outroot:str='fig_multi_eigenmatches'):


    # Load the PCA model
    pdict = info_defs.grab_paths(dataset)

    # Open the preproc file
    preproc_file = pdict['preproc_file']
    preproc = h5py.File(preproc_file, 'r')

    image_idx, similarities = nenya_pca.find_eigenmatches(
        '../Analysis/'+pdict['pca_file'], pdict['latents_file'],
        modes=modes, nimages=nimages)

    for tt, mode in enumerate(modes):

        outfile = f'{outroot}_{dataset}_mode{mode+1}.png'
        fig = plt.figure(figsize=(6,6))
        gs = gridspec.GridSpec(3,3)

        for ss in range(nimages):
            # Grab the image
            img = preproc[partition][image_idx[ss, tt]]
            if img.ndim == 3:
                img = img[0,...]

            ax = plt.subplot(gs[ss]) 
            _ = sns.heatmap(np.flipud(img), xticklabels=[], 
                        #vmin=vmnx[0], vmax=vmnx[1], 
                        ax=ax,
                        yticklabels=[], cmap=cmap, cbar=False) 
                        #cbar_kws={'label': clbl})# 'fontsize': 20})
            # Title
            title = f'sim={similarities[ss, tt]:.2f}'
            ax.set_title(title, fontsize=12)

        #rsp_utils.set_fontsize(ax, 18)
        # Plot title
        fig.suptitle(f'Eigematches mode={mode+1} for {dataset}', fontsize=16)

        plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
        plt.savefig(outfile, dpi=300)
        print(f"Saved: {outfile}")

def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    # SWOT learning curve
    if flg == 30:
        #fig_learning_curve('SWOT')
        #fig_learning_curve('VIIRS')
        #fig_learning_curve('MODIS')
        fig_learning_curve('MNIST')

    if flg == 31:
        fig_multi_eigenmatches('MODIS_SST', 'jet')

    # SWOT UMAP gallery
    if flg == 40:
        fig_swot_umap()


    # GRHSST talk
    if flg == 50:
        # Just MODIS
        fig_pca(outfile='fig_pca_MODIS.png',
            datasets=['MODIS_SST'],
            show_cum_point=0.99)
        #fig_pca(outfile='fig_pca_MODIS.png',
        #    datasets=['MODIS_SST', 'MODIS_SST_2km'],
        #    show_cum_point=0.99)
        #fig_pca(outfile='fig_pca_MMVV.png',
        #    datasets=['MODIS_SST', 'MODIS_SST_2km', 
        #              'VIIRS_SST', 'VIIRS_SST_2km'],
        #fig_pca(outfile='fig_pca_MODIS.png',
        #    datasets=['MODIS_SST', 'MODIS_SST_2km'],
        #    show_cum_point=0.99)
        #fig_pca(outfile='fig_pca_MMV.png',
        #    datasets=['MODIS_SST', 'MODIS_SST_2km', 'VIIRS_SST_2km'],
        #    show_cum_point=0.99)
        #fig_pca(outfile='fig_pca_MODIS_cumul.png',
        #    datasets=['MODIS_SST', 'MODIS_SST_2km'],
        #    cumulative=True)
        #fig_pca(outfile='fig_pca_MODIS_frac.png',
        #    datasets=['MODIS_SST', 'MODIS_SST_2km'],
        #    frac_remain=True)
        # MNIST + MODIS
        #fig_pca(outfile='fig_pca_MM.png',
        #    datasets=['MODIS_SST', 'MNIST'],
        #    show_cum_point=0.99)
        # MNIST + MODIS + ImageNet
        #fig_pca(outfile='fig_pca_MMI.png',
        #    datasets=['MODIS_SST', 'MNIST',
        #              'ImageNet'],
        #    show_cum_point=0.99)
        # MODIS + VIIRS
        #fig_pca(outfile='fig_pca_MV.png',
        #    datasets=['MODIS_SST', 'VIIRS'])
        # MODIS + LLC (noise)
        fig_pca(outfile='fig_pca_MLnoise.png',
            datasets=['MODIS_SST', 
                      'LLC_SST_nonoise', 'LLC_SST_noise'],
            show_cum_point=0.99)
        # MODIS + VIIRS + LLC + SWOT
        #fig_pca(outfile='fig_pca_MVLS.png',
        #    datasets=['MODIS_SST', 'VIIRS',
        #              'LLC_SST', 'SWOT_L3'])
    

    # Team brainstorming
    if flg == 60:
        # 103
        fig_multi_eigenmatches('MODIS_SST', 'jet', modes=[103-1])

    # Paper figures

    # Learning curves
    if flg == 1:
        fig_learning_curves()
        #fig_learning_curves(outfile='fig_learning_curves.pdf')

    # PCA variance on latent space
    if flg == 2:
        #fig_pca(show_cum_point=0.99, outfile='fig_pca_variance_zoomin.png',
        #        xmnx=(30, 300))

        # Natural
        if False:
            fig_pca(show_cum_point=0.99, 
                datasets=info_defs.natural_datasets,
                outfile='fig_pca_natural.png')
        # All
        #fig_pca(show_cum_point=0.99)
        fig_pca_2panel(show_cum_point=0.99)

        #fig_pca(outfile='fig_pca_noise.png',
        #    datasets=['MODIS_SST', 'MODIS_SST_2km', 'LLC_SST_nonoise', 'LLC_SST_noise'],
        #    show_cum_point=0.99)

    # Eigenmodes 
    if flg == 3:
        #fig_eigenimages('MNIST', 'Greys')
        fig_eigenimages('MODIS_SST', 'jet')

    # Closest matched images
    if flg == 4:
        fig_eigenmatches('MODIS_SST', 'jet')
        #fig_eigenmatches('MODIS_SST', 'jet', last_ones=True,
        #                 outroot='fig_last_eigenmatches')

    # P(k)
    if flg == 5:
        fig_Pk()

    # PCA variance
    if flg == 6:
        fig_true_pca(show_cum_point=0.99)

    # Example images
    if flg == 7:
        fig_example_images()



# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
    else:
        flg = sys.argv[1]

    main(flg)