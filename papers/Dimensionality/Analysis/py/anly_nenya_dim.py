""" Modules for analysis of dimensionality of Nenya data."""
import os
import h5py

import numpy as np

from sklearn import decomposition

def pca_latents(dataset:str):

    # Load
    if dataset == 'MODIS_SST':
        filename = 'MODIS_R2019_2004_95clear_128x128_latents_std.h5'
        outfile='pca_latents_MODIS_SST.npz'
        path = os.path.join(os.getenv('OS_SST'), 'MODIS_L2',
                        'Nenya', 'latents/MODIS_R2019_v4_REDO',
                        'SimCLR_resnet50_lr_0.05_decay_0.0001_bsz_256_temp_0.07_trial_5_cosine_warm')
    elif dataset == 'VIIRS_SST':
        filename = 'VIIRS_2013_98clear_192x192_latents_viirs_std_train.h5'
        outfile='pca_latents_VIIRS_SST.npz'
        path = os.path.join(os.getenv('OS_SST'), 'VIIRS',
                        'Nenya', 'latents', 'VIIRS_v1') 
    else:
        raise IOError("Bad dataset: {}".format(dataset))

    fpath = os.path.join(path, filename)
    print(f'Loading: {fpath}')
    f = h5py.File(fpath, 'r')
    latents = f['valid'][:]
    f.close()

    # PCA
    print("Fitting PCA")
    pca_fit = decomposition.PCA().fit(latents)

    # Save
    coeff = pca_fit.transform(latents)
    #
    outputs = dict(Y=coeff,
                M=pca_fit.components_,
                mean=pca_fit.mean_,
                explained_variance=pca_fit.explained_variance_ratio_)
    # Save
    print(f"Saving: {outfile}")
    np.savez(outfile, **outputs)

# Command line execution
if __name__ == '__main__':
    # PCA MODIS SST
    pca_latents('MODIS_SST')

    #  VIIRS SST
    #pca_latents('VIIRS_SST')