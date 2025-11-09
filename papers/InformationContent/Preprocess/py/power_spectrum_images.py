"""
Generate synthetic images with power spectrum P(k) = k^-n with random phase.

This module creates synthetic 64x64 images that follow a specified power law
spectrum with random phases. The images are saved as train/valid datasets in
a single HDF5 file.
"""

import os
import sys
import numpy as np
import h5py
from typing import Tuple, Optional
import argparse

# If using with nenya framework, uncomment these lines:
# sys.path.append(os.path.abspath("../Analysis/py"))
# import info_defs


def generate_power_law_image(
    size: int, 
    n: float, 
    rng: np.random.Generator,
    normalize: bool = True
) -> np.ndarray:
    """
    Generate a single image with power spectrum P(k) = k^-n.
    
    Args:
        size: Image size (will create size x size image)
        n: Power law exponent
        rng: Random number generator
        normalize: Whether to normalize the output image
        
    Returns:
        2D numpy array of the generated image
    """
    # Create frequency grids
    freq = np.fft.fftfreq(size)
    fx, fy = np.meshgrid(freq, freq, indexing='ij')
    
    # Calculate radial frequency (k = sqrt(kx^2 + ky^2))
    k = np.sqrt(fx**2 + fy**2)
    
    # Avoid division by zero at DC component
    k[0, 0] = 1.0
    
    # Generate power spectrum P(k) = k^-n
    # Note: We use -n/2 for amplitude since power = amplitude^2
    amplitude = k ** (-n / 2)
    
    # Set DC component to zero (mean-zero image)
    amplitude[0, 0] = 0.0
    
    # Generate random phases
    phases = rng.uniform(0, 2 * np.pi, size=(size, size))
    
    # Construct complex Fourier coefficients
    fourier = amplitude * np.exp(1j * phases)
    
    # Ensure Hermitian symmetry for real output
    # For a real signal, F(k) = F*(-k)
    for i in range(size):
        for j in range(size//2 + 1, size):
            # Mirror indices
            mi = (-i) % size
            mj = (-j) % size
            fourier[i, j] = np.conj(fourier[mi, mj])
    
    # Inverse FFT to get the image
    image = np.fft.ifft2(fourier).real
    
    if normalize:
        # Normalize to zero mean and unit variance
        image = image - np.mean(image)
        image = image / np.std(image)
    
    return image


def generate_power_law_dataset(
    ntrain: int = 150000,
    nvalid: int = 50000,
    size: int = 64,
    n: float = 2.0,
    seed: int = 1234,
    normalize: bool = True,
    batch_size: int = 1000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate complete dataset of power law images.
    
    Args:
        ntrain: Number of training images
        nvalid: Number of validation images
        size: Image size (size x size)
        n: Power law exponent (P(k) = k^-n)
        seed: Random seed
        normalize: Whether to normalize images
        batch_size: Process images in batches to save memory
        
    Returns:
        Tuple of (train_images, valid_images)
    """
    # Initialize random generator
    rng = np.random.default_rng(seed)
    
    # Total number of images
    nimg = ntrain + nvalid
    
    print(f"Generating {nimg} images with power spectrum P(k) = k^-{n}")
    print(f"Image size: {size}x{size}")
    print(f"Training: {ntrain}, Validation: {nvalid}")
    
    # Initialize arrays
    all_images = np.zeros((nimg, size, size), dtype=np.float32)
    
    # Generate images in batches
    for i in range(0, nimg, batch_size):
        batch_end = min(i + batch_size, nimg)
        batch_size_curr = batch_end - i
        
        if i % 10000 == 0:
            print(f"  Generated {i}/{nimg} images...")
        
        for j in range(batch_size_curr):
            all_images[i + j] = generate_power_law_image(size, n, rng, normalize)
    
    print(f"  Generated {nimg}/{nimg} images.")
    
    # Split into train and validation
    train_images = all_images[:ntrain]
    valid_images = all_images[ntrain:ntrain+nvalid]
    
    return train_images, valid_images


def save_to_hdf5(
    train_images: np.ndarray,
    valid_images: np.ndarray,
    output_file: str,
    n: float,
    dataset_name: str = None,
    compression: str = 'gzip'
) -> None:
    """
    Save images to HDF5 file.
    
    Args:
        train_images: Training images array
        valid_images: Validation images array
        output_file: Output HDF5 file path
        n: Power law exponent used
        dataset_name: Optional dataset name for metadata
        compression: Compression algorithm ('gzip', 'lzf', or None)
    """
    print(f"\nSaving to: {output_file}")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with h5py.File(output_file, 'w') as f:
        # Save datasets
        f.create_dataset('train', data=train_images, dtype=np.float32, compression=compression)
        f.create_dataset('valid', data=valid_images, dtype=np.float32, compression=compression)
        
        # Add metadata
        f.attrs['power_law_exponent'] = n
        f.attrs['dataset'] = dataset_name or f'PowerLaw_n{n}'
        f.attrs['n_train'] = train_images.shape[0]
        f.attrs['n_valid'] = valid_images.shape[0]
        f.attrs['image_shape'] = train_images.shape[1:]
        f.attrs['normalized'] = True
        
    print(f"Dataset saved successfully!")
    print(f"  Training images: {train_images.shape}")
    print(f"  Validation images: {valid_images.shape}")


def verify_power_spectrum(
    image: np.ndarray,
    n_expected: float,
    plot: bool = False
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Verify that an image has the expected power spectrum.
    
    Args:
        image: Input image
        n_expected: Expected power law exponent
        plot: Whether to plot the spectrum (requires matplotlib)
        
    Returns:
        Tuple of (k_values, power_spectrum, fitted_n)
    """
    # Compute 2D FFT
    fft = np.fft.fft2(image)
    power_2d = np.abs(fft) ** 2
    
    # Get frequency grid
    size = image.shape[0]
    freq = np.fft.fftfreq(size)
    fx, fy = np.meshgrid(freq, freq, indexing='ij')
    k = np.sqrt(fx**2 + fy**2)
    
    # Radial binning
    k_bins = np.linspace(0, 0.5, size//2)
    k_centers = (k_bins[:-1] + k_bins[1:]) / 2
    power_radial = np.zeros(len(k_centers))
    
    for i, (k_min, k_max) in enumerate(zip(k_bins[:-1], k_bins[1:])):
        mask = (k >= k_min) & (k < k_max)
        if np.sum(mask) > 0:
            power_radial[i] = np.mean(power_2d[mask])
    
    # Fit power law (excluding DC and very high frequencies)
    valid = (k_centers > 0.02) & (k_centers < 0.4) & (power_radial > 0)
    if np.sum(valid) > 2:
        log_k = np.log10(k_centers[valid])
        log_power = np.log10(power_radial[valid])
        fitted_n = -np.polyfit(log_k, log_power, 1)[0]
    else:
        fitted_n = np.nan
    
    if plot:
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Show image
            axes[0].imshow(image, cmap='viridis')
            axes[0].set_title(f'Generated Image (n={n_expected:.1f})')
            axes[0].set_xlabel('x')
            axes[0].set_ylabel('y')
            
            # Show power spectrum
            axes[1].loglog(k_centers[valid], power_radial[valid], 'b.-', 
                          label=f'Measured (fitted n={fitted_n:.2f})')
            axes[1].loglog(k_centers[valid], k_centers[valid]**(-n_expected) * power_radial[valid][0], 
                          'r--', label=f'Expected (n={n_expected:.1f})')
            axes[1].set_xlabel('Wavenumber k')
            axes[1].set_ylabel('Power P(k)')
            axes[1].set_title('Power Spectrum')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("Matplotlib not available for plotting")
    
    return k_centers, power_radial, fitted_n


def main(
    ntrain: int = 150000,
    nvalid: int = 50000,
    n: float = 2.0,
    seed: int = 1234,
    size: int = 64,
    output_file: str = None,
    verify: bool = False
):
    """
    Main function to generate power law image dataset.
    
    Args:
        ntrain: Number of training images
        nvalid: Number of validation images
        n: Power law exponent (P(k) = k^-n)
        seed: Random seed
        size: Image size
        output_file: Output HDF5 file path
        verify: Whether to verify the power spectrum of sample images
    """
    # Default output file name
    if output_file is None:
        output_file = f'power_law_n{n}_64x64_processed.h5'
    
    # Generate dataset
    train_images, valid_images = generate_power_law_dataset(
        ntrain=ntrain,
        nvalid=nvalid,
        size=size,
        n=n,
        seed=seed,
        normalize=True
    )
    
    # Optional: Verify power spectrum of a few samples
    if verify:
        print("\nVerifying power spectrum of sample images...")
        rng = np.random.default_rng(seed + 1000)
        for i in range(3):
            idx = rng.integers(0, ntrain)
            k, power, fitted_n = verify_power_spectrum(
                train_images[idx], 
                n, 
                plot=(i == 0)  # Only plot first one
            )
            print(f"  Sample {i+1}: Expected n={n:.2f}, Fitted n={fitted_n:.2f}")
    
    # Save to HDF5
    save_to_hdf5(
        train_images,
        valid_images,
        output_file,
        n,
        dataset_name=f'PowerLaw_n{n}'
    )
    
    # If using nenya framework, uncomment to save with info_defs paths:
    # dataset = f'PowerLaw_n{n}'
    # odict = info_defs.grab_paths(dataset)
    # save_to_hdf5(train_images, valid_images, odict['preproc_file'], n, dataset)
    
    return train_images, valid_images


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate synthetic images with power law spectrum P(k) = k^-n'
    )
    parser.add_argument('--n', type=float, default=2.0,
                       help='Power law exponent (default: 2.0)')
    parser.add_argument('--ntrain', type=int, default=150000,
                       help='Number of training images (default: 150000)')
    parser.add_argument('--nvalid', type=int, default=50000,
                       help='Number of validation images (default: 50000)')
    parser.add_argument('--size', type=int, default=64,
                       help='Image size (default: 64)')
    parser.add_argument('--seed', type=int, default=1234,
                       help='Random seed (default: 1234)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output HDF5 file path')
    parser.add_argument('--verify', action='store_true',
                       help='Verify power spectrum of sample images')
    
    args = parser.parse_args()
    
    # Generate the dataset
    main(
        ntrain=args.ntrain,
        nvalid=args.nvalid,
        n=args.n,
        seed=args.seed,
        size=args.size,
        output_file=args.output,
        verify=args.verify
    )

def run(option:int)

    if option == 2:
        # n=-2
        print("Example 1: Generating dataset with n=2.0 (similar to natural images)")
        print("-" * 60)

        main(
            ntrain=150000,
            nvalid=50000,
            n=2.0,
            seed=1234,
            size=64,
            output_file='power_law_n2_64x64.h5',
            verify=True  # This will verify the power spectrum
        )

        print("\n" + "="*60 + "\n")

    if option == 2:
        # n=-4
        print("Example 2: Generating dataset with n=3.0 (smoother images)")
        print("-" * 60)

        main(
            ntrain=150000,
            nvalid=50000,
            n=4.0,
            seed=1234,
            size=64,
            output_file='power_law_n4_64x64.h5',
            verify=True  # Skip verification for faster generation
        )

        print("\n" + "="*60 + "\n")

        # Example 3: Load and inspect the generated dataset
        print("Example 3: Inspecting the generated datasets")
        print("-" * 60)

    if option == 0:

        for n in [2.0, 4.0]:
            filename = f'power_law_n{n}_64x64.h5'
            print(f"\nInspecting: {filename}")
            
            with h5py.File(filename, 'r') as f:
                # Check metadata
                print("  Metadata:")
                for key in f.attrs.keys():
                    print(f"    {key}: {f.attrs[key]}")
                
                # Check data shapes
                train_data = f['train']
                valid_data = f['valid']
                print(f"  Training data shape: {train_data.shape}")
                print(f"  Validation data shape: {valid_data.shape}")
                
                # Load a few samples for statistics
                sample_train = train_data[:100]
                print(f"  Sample statistics (first 100 training images):")
                print(f"    Mean: {np.mean(sample_train):.6f}")
                print(f"    Std: {np.std(sample_train):.6f}")
                print(f"    Min: {np.min(sample_train):.3f}")
                print(f"    Max: {np.max(sample_train):.3f}")

        print("\nDone! The datasets are ready for use.")
        print("Files created:")
        print("  - power_law_n2.0_64x64.h5")
        print("  - power_law_n4.0_64x64.h5")

if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
    else:
        flg = sys.argv[1]

    run(flg)
