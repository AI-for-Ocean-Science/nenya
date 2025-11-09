
import os
import numpy as np
    
import matplotlib.pyplot as plt

from nenya import pk as nenya_pk

import info_defs

#datasets = ['MODIS_SST', 'MODIS_SST_2km',
#        'VIIRS_SST', 'VIIRS_SST_2km', 'VIIRS_SST_sub', 
#        'LLC_SST_nonoise', 'SWOT_L3', 
#        'WNoise', 'MNIST', 'ImageNet']
datasets = info_defs.all_datasets

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


def calc_all():
    # Loop me
    for dataset in datasets:
        print(f"Working on {dataset}")
        # Analyze
        calc_one(dataset) 

def plot_em_all_in_one():

    plt.figure(figsize=(8,6))
    ax = plt.gca()

    for dataset in datasets:
        pdict = info_defs.grab_paths(dataset)
        if not os.path.exists(pdict['Pk_file']):
            print(f"Pk file for {dataset} not found, skipping")
            continue
        # Load
        data = np.load(pdict['Pk_file'])
        k = data['wavenumber']
        power = data['power']
        wavelength = data['wavelength']

        plt.loglog(wavelength, power, label=dataset)

    plt.xlabel('Wavelength (km)')
    plt.ylabel('Power')
    plt.title('Power Spectra for Various Datasets')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig('Pk_all_datasets.png', dpi=300)
    plt.close()

    
if __name__ == "__main__":
    calc_all()
    #plot_em_all_in_one()