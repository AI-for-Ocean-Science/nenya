import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy import signal
from pathlib import Path
from joblib import Parallel, delayed
from multiprocessing import cpu_count

from remote_sensing.plotting.utils import set_fontsize


def load_images(pp_file, partition='train'):
    """
    Load images from HDF5 file.

    Args:
        pp_file (str): Path to preprocessed data file of cutouts
        partition (str): 'train' or 'test' partition 

    Returns:
        images (3D array): cutouts (n_images, ny, nx)
    """
    print(f'Loading cutouts from {pp_file} on partition={partition}')
    with h5py.File(pp_file, 'r') as preproc:
        return preproc[partition][:]


def preprocess_image(image, demean:bool=True, detrend:bool=False, window=True):
    """
    Preprocess image before FFT to reduce spectral leakage.
    
    Parameters:
    -----------
    cutout : 2D array
        cutout
    demean : bool
        Remove mean?
    detrend : bool
        Remove linear trends
    window : bool
        Apply 2D Hanning window
    
    Returns:
    --------
    processed : 2D array
        Preprocessed image
    """
    processed = image.copy()
    
    # Remove mean
    if demean:
        processed = processed - np.nanmean(processed)
    
    # Detrend (remove linear trends in x and y)
    if detrend:
        processed = signal.detrend(processed, axis=0, type='linear')
        processed = signal.detrend(processed, axis=1, type='linear')
    
    # Apply 2D Hanning window to reduce edge effects
    if window:
        window_x = np.hanning(processed.shape[1])
        window_y = np.hanning(processed.shape[0])
        window_2d = np.outer(window_y, window_x)
        processed = processed * window_2d
    
    return processed


def compute_2d_spectrum(image):
    """
    Compute 2D power spectrum of a single image.
    
    Parameters:
    -----------
    image : 2D array
        Preprocessed image
    
    Returns:
    --------
    power_spectrum : 2D array
        Power spectral density
    """
    ny, nx = image.shape
    
    # Compute 2D FFT
    fft_2d = np.fft.fft2(image)
    
    # Shift zero frequency to center
    fft_2d_shifted = np.fft.fftshift(fft_2d)
    
    # Compute power spectrum (normalize by number of points)
    power = np.abs(fft_2d_shifted)**2 / (nx * ny)
    
    return power


def get_wavenumber_grids(ny, nx, dx):
    """
    Create wavenumber grids in physical units.
    
    Parameters:
    -----------
    ny, nx : int
        Grid dimensions
    dx : float
        Spatial resolution in km
    
    Returns:
    --------
    kx, ky : 2D arrays
        Wavenumber grids in cycles/km
    k_radial : 2D array
        Radial wavenumber magnitude
    """
    # Frequency arrays (cycles per grid point)
    freq_x = np.fft.fftshift(np.fft.fftfreq(nx))
    freq_y = np.fft.fftshift(np.fft.fftfreq(ny))
    
    # Convert to physical wavenumbers (cycles/km)
    kx_1d = freq_x / dx
    ky_1d = freq_y / dx
    
    # Create 2D grids
    kx, ky = np.meshgrid(kx_1d, ky_1d)
    
    # Radial wavenumber
    k_radial = np.sqrt(kx**2 + ky**2)
    
    return kx, ky, k_radial


def radial_average(power_2d, k_radial, nbins=None):
    """
    Compute radially-averaged 1D power spectrum.
    
    Parameters:
    -----------
    power_2d : 2D array
        2D power spectrum
    k_radial : 2D array
        Radial wavenumber grid
    nbins : int
        Number of radial bins (default: nx//2)
    
    Returns:
    --------
    k_bins : 1D array
        Radial wavenumber bin centers (cycles/km)
    power_1d : 1D array
        Radially-averaged power spectrum
    """
    nx = power_2d.shape[1]
    if nbins is None:
        nbins = nx // 2
    
    # Flatten arrays
    k_flat = k_radial.flatten()
    power_flat = power_2d.flatten()
    
    # Create bins
    k_max = np.max(k_flat)
    k_bins = np.linspace(0, k_max, nbins + 1)
    k_centers = 0.5 * (k_bins[1:] + k_bins[:-1])
    
    # Compute average in each bin
    power_1d = np.zeros(nbins)
    counts = np.zeros(nbins)
    
    for i in range(nbins):
        mask = (k_flat >= k_bins[i]) & (k_flat < k_bins[i+1])
        if np.sum(mask) > 0:
            power_1d[i] = np.mean(power_flat[mask])
            counts[i] = np.sum(mask)
    
    # Only return bins with data
    valid = counts > 0
    
    return k_centers[valid], power_1d[valid]


def _process_single_image(image, detrend, window):
    """
    Helper function to process a single image (for parallelization).
    
    Parameters:
    -----------
    image : 2D array
        Input image
    detrend : bool
        Apply detrending
    window : bool
        Apply windowing
    
    Returns:
    --------
    power_2d : 2D array
        2D power spectrum
    """
    image_proc = preprocess_image(image, detrend=detrend, window=window)
    power_2d = compute_2d_spectrum(image_proc)
    return power_2d



def compute_ensemble_spectrum_parallel(cutouts, dx=2.0, detrend=True, 
                                      window=True, 
                                      n_workers=None, batch_size=None):
    """
    Compute ensemble-averaged power spectrum from all images using parallelization.
    
    Parameters:
    -----------
    cutouts : 3D array
        Stack of cutouts (n_images, ny, nx)
    dx : float
        Spatial resolution in km
    detrend : bool
        Apply detrending to each image
    window : bool
        Apply windowing to each image
    n_workers : int
        Number of parallel workers (default: cpu_count - 1)
    batch_size : int
        Number of images to process per batch (default: None = all at once)
        Use this to limit RAM usage by processing in smaller chunks
    
    Returns:
    --------
    k : 1D array
        Wavenumber (cycles/km)
    power : 1D array
        Radially-averaged power spectrum
    wavelength : 1D array
        Wavelength (km)
    power_2d_avg : 2D array (optional)
        Averaged 2D power spectrum
    k_radial : 2D array (optional)
        Radial wavenumber grid
    """
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)
    
    n_images = cutouts.shape[0]
    ny, nx = cutouts[0].shape
    
    print(f"Processing {n_images} images using {n_workers} workers...")
    
    # Get wavenumber grids
    _, _, k_radial = get_wavenumber_grids(ny, nx, dx)
    power_2d_sum = np.zeros((ny, nx))
    
    # If no batch_size specified, process all at once
    if batch_size is None:
        batch_size = n_images
    
    # Process in batches with parallelization within each batch
    n_batches = int(np.ceil(n_images / batch_size))
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_images)
        
        print(f"Batch {batch_idx+1}/{n_batches} (images {start_idx}-{end_idx})...")
        
        # Process this batch in parallel
        power_2d_batch = Parallel(n_jobs=n_workers, backend='loky', verbose=5)(
            delayed(_process_single_image)(cutouts[i], detrend, window)
            for i in range(start_idx, end_idx)
        )
        
        # Accumulate results from this batch
        power_2d_sum += np.sum(power_2d_batch, axis=0)
        
        # Free memory
        del power_2d_batch
    
    # Average
    power_2d_avg = power_2d_sum / n_images
    
    # Radial average
    k, power_1d = radial_average(power_2d_avg, k_radial)
    
    # Compute wavelength
    wavelength = 1.0 / k
    
    print("Spectral analysis complete!")
    
    return k, power_1d, wavelength


def orig_compute_ensemble_spectrum_parallel(cutouts, dx=2.0, detrend=True, 
                                      window=True, save_2d=False, 
                                      n_workers=None):
    """
    Compute ensemble-averaged power spectrum from all images using parallelization.
    
    Parameters:
    -----------
    cutouts : 3D array
        Stack of cutouts (n_images, ny, nx)
    dx : float
        Spatial resolution in km
    detrend : bool
        Apply detrending to each image
    window : bool
        Apply windowing to each image
    save_2d : bool
        Also return 2D averaged spectrum
    n_workers : int
        Number of parallel workers (default: cpu_count - 1)
    
    Returns:
    --------
    k : 1D array
        Wavenumber (cycles/km)
    power : 1D array
        Radially-averaged power spectrum
    wavelength : 1D array
        Wavelength (km)
    power_2d_avg : 2D array (optional)
        Averaged 2D power spectrum
    k_radial : 2D array (optional)
        Radial wavenumber grid
    """
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)
    
    n_images = cutouts.shape[0]
    ny, nx = cutouts[0].shape
    
    print(f"Processing {n_images} images using {n_workers} workers...")
    
    # Get wavenumber grids
    _, _, k_radial = get_wavenumber_grids(ny, nx, dx)
    
    # Process all images in parallel
    power_2d_list = Parallel(n_jobs=n_workers, backend='loky', verbose=10)(
        delayed(_process_single_image)(cutouts[i], detrend, window)
        for i in range(n_images)
    )
    
    # Sum and average all results
    power_2d_sum = np.sum(power_2d_list, axis=0)
    power_2d_avg = power_2d_sum / n_images
    
    # Radial average
    k, power_1d = radial_average(power_2d_avg, k_radial)
    
    # Compute wavelength
    wavelength = 1.0 / k
    
    print("Spectral analysis complete!")
    
    if save_2d:
        return k, power_1d, wavelength, power_2d_avg, k_radial
    else:
        return k, power_1d, wavelength


def compute_ensemble_spectrum_batched(cutouts, dx=2.0, detrend=True, 
                                     window=True, save_2d=False, 
                                     batch_size=1000):
    """
    Compute ensemble-averaged power spectrum from all images using batched processing.
    (Original non-parallel version for memory-constrained situations)
    
    Parameters:
    -----------
    cutouts : 3D array
        Stack of cutouts (n_images, ny, nx)
    dx : float
        Spatial resolution in km
    detrend : bool
        Apply detrending to each image
    window : bool
        Apply windowing to each image
    save_2d : bool
        Also return 2D averaged spectrum
    batch_size : int
        Number of images to process at once
    
    Returns:
    --------
    k : 1D array
        Wavenumber (cycles/km)
    power : 1D array
        Radially-averaged power spectrum
    wavelength : 1D array
        Wavelength (km)
    power_2d_avg : 2D array (optional)
        Averaged 2D power spectrum
    k_radial : 2D array (optional)
        Radial wavenumber grid
    """
    n_images = cutouts.shape[0]
    print(f"Processing {n_images} images in batches of {batch_size}...")
    
    # Initialize from first image
    ny, nx = cutouts[0].shape
    _, _, k_radial = get_wavenumber_grids(ny, nx, dx)
    power_2d_sum = np.zeros((ny, nx))
    
    # Process in batches
    n_batches = int(np.ceil(n_images / batch_size))
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_images)
        
        print(f"Batch {batch_idx+1}/{n_batches} (images {start_idx}-{end_idx})...")
        
        for i in range(start_idx, end_idx):
            # Load image
            image = cutouts[i]
            
            # Preprocess
            image_proc = preprocess_image(image, detrend=detrend, window=window)
            
            # Compute spectrum
            power_2d = compute_2d_spectrum(image_proc)
            
            # Accumulate
            power_2d_sum += power_2d
    
    # Average
    power_2d_avg = power_2d_sum / n_images
    
    # Radial average
    k, power_1d = radial_average(power_2d_avg, k_radial)
    
    # Compute wavelength
    wavelength = 1.0 / k
    
    print("Spectral analysis complete!")
    
    if save_2d:
        return k, power_1d, wavelength, power_2d_avg, k_radial
    else:
        return k, power_1d, wavelength


def plot_spectrum(k, power, wavelength, ax=None, plot_wavelength=True, 
                 reference_slopes=True, show:bool=False,
                 title:str=None, outfile:str=None):
    """
    Plot the power spectrum.
    
    Parameters:
    -----------
    k : 1D array
        Wavenumber (cycles/km)
    power : 1D array
        Power spectrum
    wavelength : 1D array
        Wavelength (km)
    ax : matplotlib axis
        Axis to plot on (creates new figure if None)
    plot_wavelength : bool
        Show wavelength on top x-axis
    reference_slopes : bool
        Plot reference slopes (k^-2, k^-3, k^-5/3)
    
    Returns:
    --------
    ax : matplotlib axis
        The axis object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot spectrum
    ax.loglog(k, power, 'k-', linewidth=2, label='Power Spectrum')
    
    # Add reference slopes
    if reference_slopes:
        k_ref = np.linspace(k.min(), k.max(), 100)
        
        # k^-2 (mesoscale)
        power_ref_2 = power[len(k)//3] * (k_ref / k[len(k)//3])**(-2)
        ax.loglog(k_ref, power_ref_2, 'b--', alpha=0.5, label='$k^{-2}$')
        
        # k^-5/3 (inertial subrange)
        power_ref_53 = power[2*len(k)//3] * (k_ref / k[2*len(k)//3])**(-5/3)
        ax.loglog(k_ref, power_ref_53, 'r--', alpha=0.5, label='$k^{-5/3}$')
        
        # k^-3 (enstrophy cascade)
        power_ref_3 = power[2*len(k)//3] * (k_ref / k[2*len(k)//3])**(-3)
        ax.loglog(k_ref, power_ref_3, 'g--', alpha=0.5, label='$k^{-3}$')
    
    ax.set_xlabel('Wavenumber (cycles/km)')
    ax.set_ylabel('Power Spectral Density (km$^2$)') 
    if title is None:
        title = 'Fluctuation Power Spectrum'
    
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=22)
    
    # Add wavelength axis on top
    if plot_wavelength:
        ax2 = ax.twiny()
        ax2.set_xscale('log')
        ax2.set_xlim(ax.get_xlim())
        
        # Set wavelength ticks
        wavelength_ticks = [128, 64, 32, 16, 8, 4]
        k_ticks = [1.0/w for w in wavelength_ticks]
        ax2.set_xticks(k_ticks)
        ax2.set_xticklabels([f'{w}' for w in wavelength_ticks])
        ax2.set_xlabel('Wavelength (km)')
    
    # Font size
    for iax in [ax, ax2]:
        set_fontsize(iax, 20)

    plt.title(title,  fontsize=30)
    plt.tight_layout()
    if outfile is not None:
        plt.savefig(outfile, dpi=300, bbox_inches='tight')
        print(f'Wrote: {outfile}')
    if show:
        plt.show()
    return ax

def plot_from_file(pk_file:str, outfile:str=None, show:bool=True,
                   title:str=None):
    """Quick plot from saved spectrum file."""
    data = np.load(pk_file)
    k = data['wavenumber']
    power = data['power']
    wavelength = data['wavelength']

    fig, ax = plt.subplots(figsize=(10, 7))
    plot_spectrum(k, power, wavelength, ax=ax, show=show, title=title,
                  outfile=outfile)



def test():    
    # Load your data
    pp_file = '/home/xavier/Projects/Oceanography/data/Natural/White_Noise/Info/PreProc/wnoise_64x64_processed.h5'
    cutouts = load_images(pp_file, partition='train')
    
    # Compute spectrum (parallel version - recommended)
    k, power, wavelength = compute_ensemble_spectrum_parallel(
        cutouts=cutouts,
        dx=2.0,         # 2 km resolution
        detrend=True,   # Remove linear trends
        window=True,    # Apply Hanning window
        n_workers=None  # Uses cpu_count - 1
    )
    
    # Alternative: batched version for memory-constrained situations
    # k, power, wavelength = compute_ensemble_spectrum_batched(
    #     cutouts=cutouts,
    #     dx=2.0,
    #     detrend=True,
    #     window=True,
    #     batch_size=1000
    # )
    
    # Plot results
    fig, ax = plt.subplots(figsize=(10, 7))
    plot_spectrum(k, power, wavelength, ax=ax, show=True)
    
    # Save results
    np.savez('wnoise_spectrum_results.npz',
             wavenumber=k,
             power=power,
             wavelength=wavelength)
    
    print(f"\nSpectral range:")
    print(f"  Wavelengths: {wavelength[-1]:.2f} km to {wavelength[0]:.2f} km")
    print(f"  Wavenumbers: {k[0]:.6f} to {k[-1]:.6f} cycles/km")

if __name__ == "__main__":
    #test()

    plot_from_file('wnoise_spectrum_results.npz', 
                   show=False, title='White Noise',
                   outfile='wnoise_spectrum.png')