
import os
import numpy as np

from nenya import pk as nenya_pk

import info_defs

def calc_one(dataset:str, clobber:bool=False):

    pdict = info_defs.grab_paths(dataset)
    if os.path.exists(pdict['Pk_file']) and not clobber:
        print(f'Already analyzed {dataset}. Set clobber=True to overwrite')
        return

    # Load
    cutouts = nenya_pk.load_images(pdict['preproc_file'], partition='train')

    if dataset in ['VIIRS_SST']:
        batch_size = 16
    else:
        batch_size = None
    
    # Compute spectrum (parallel version - recommended)
    k, power, wavelength = nenya_pk.orig_compute_ensemble_spectrum_parallel(
        cutouts=cutouts,
        dx=pdict['dx'],         # 2 km resolution
        detrend=False,   # Remove linear trends
        window=True,    # Apply Hanning window
        n_workers=None,  # Uses cpu_count - 1
        #batch_size=batch_size,
    )
    
    # Plot results
    nenya_pk.plot_spectrum(k, power, wavelength, show=False, 
                           outfile=pdict['Pk_plot'],
                           title=dataset)
    
    # Save results
    np.savez(pdict['Pk_file'], wavenumber=k, power=power, wavelength=wavelength)

if __name__ == "__main__":
    datasets = ['MODIS_SST', 'MODIS_SST_2km',
        'VIIRS_SST', 'VIIRS_SST_2km', 'VIIRS_SST_sub', 
        'LLC_SST_nonoise', 'SWOT_L3', 
        'WNoise', 'MNIST',
        'ImageNet']

    # Loop me
    for dataset in datasets:
        print(f"Working on {dataset}")
        # Analyze
        calc_one(dataset) 


    