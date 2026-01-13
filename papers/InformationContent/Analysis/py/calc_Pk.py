
import os
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt

from nenya import pk as nenya_pk

import info_defs

from IPython import embed

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
    print("Saved combined power spectrum plot as 'Pk_all_datasets.png'")


def fit_powerlaw(dataset: str, pix_min: int = 4, pix_max: int = 40):
    """Fit a power-law to the P(k) spectrum over a specified pixel range.

    Parameters
    ----------
    dataset : str
        Name of the dataset
    pix_min : int
        Minimum wavelength in pixels for the fit range
    pix_max : int
        Maximum wavelength in pixels for the fit range

    Returns
    -------
    dict
        Dictionary containing:
        - 'exponent': power-law exponent (slope in log-log space)
        - 'exponent_err': standard error of the exponent
        - 'intercept': y-intercept in log-log space
        - 'r_squared': R^2 value of the fit
        - 'wavelength_range': (min, max) wavelength in km used for fit
    """
    pdict = info_defs.grab_paths(dataset)

    dfile = f"../Analysis/{pdict['Pk_file']}"
    if not os.path.exists(dfile):
        print(f"Pk file for {dataset} not found")
        #embed(header='114 of calc')
        return None

    # Load P(k) data
    data = np.load(dfile)
    k = data['wavenumber']
    power = data['power']
    wavelength = data['wavelength']

    # Convert pixel range to wavelength range using dx
    dx = pdict['dx']  # km per pixel
    wl_min = pix_min * dx
    wl_max = pix_max * dx

    # Select data in the specified wavelength range
    mask = (wavelength >= wl_min) & (wavelength <= wl_max)

    if np.sum(mask) < 3:
        print(f"Not enough points in range for {dataset}")
        return None

    wl_fit = wavelength[mask]
    power_fit = power[mask]

    # Fit in log-log space: log(P) = slope * log(wl) + intercept
    # P(k) ~ k^n means P(wl) ~ wl^(-n) since k = 1/wl
    log_wl = np.log10(wl_fit)
    log_power = np.log10(power_fit)

    slope, intercept, r_value, p_value, std_err = stats.linregress(log_wl, log_power)

    return {
        'exponent': slope,
        'exponent_err': std_err,
        'intercept': intercept,
        'r_squared': r_value**2,
        'wavelength_range': (wl_min, wl_max),
        'n_points': np.sum(mask),
    }


def fit_all_powerlaws(pix_min: int = 4, pix_max: int = 40):
    """Fit power-laws to all datasets.

    Returns
    -------
    dict
        Dictionary mapping dataset names to their fit results
    """
    results = {}
    for dataset in datasets:
        result = fit_powerlaw(dataset, pix_min, pix_max)
        if result is not None:
            results[dataset] = result
            print(f"{dataset}: exponent = {result['exponent']:.2f} ± {result['exponent_err']:.2f}, "
                  f"R² = {result['r_squared']:.3f}")
    return results

    
if __name__ == "__main__":
    #calc_all()
    plot_em_all_in_one()