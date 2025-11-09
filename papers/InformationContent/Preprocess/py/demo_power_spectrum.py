#!/usr/bin/env python3
"""
Demonstration script for generating and analyzing power law spectrum images.

This script shows how to:
1. Generate datasets with different power law exponents
2. Verify the power spectra
3. Visualize sample images
4. Compare different power law behaviors
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from power_spectrum_images import (
    generate_power_law_dataset, 
    save_to_hdf5,
    verify_power_spectrum,
    generate_power_law_image
)


def visualize_power_law_comparison(n_values=[1.0, 2.0, 3.0, 4.0], size=64, seed=1234):
    """
    Generate and visualize images with different power law exponents.
    """
    fig, axes = plt.subplots(2, len(n_values), figsize=(15, 7))
    rng = np.random.default_rng(seed)
    
    for i, n in enumerate(n_values):
        # Generate a single image
        image = generate_power_law_image(size, n, rng, normalize=True)
        
        # Display image
        im = axes[0, i].imshow(image, cmap='viridis')
        axes[0, i].set_title(f'n = {n}')
        axes[0, i].set_xlabel('x')
        axes[0, i].set_ylabel('y')
        axes[0, i].axis('off')
        plt.colorbar(im, ax=axes[0, i], fraction=0.046)
        
        # Compute and display power spectrum
        k, power, fitted_n = verify_power_spectrum(image, n, plot=False)
        
        # Plot power spectrum
        valid = (k > 0.02) & (k < 0.4) & (power > 0)
        axes[1, i].loglog(k[valid], power[valid], 'b.-', alpha=0.7, label='Measured')
        axes[1, i].loglog(k[valid], k[valid]**(-n) * power[valid][0], 
                         'r--', label=f'k^{-n}')
        axes[1, i].set_xlabel('Wavenumber k')
        if i == 0:
            axes[1, i].set_ylabel('Power P(k)')
        axes[1, i].set_title(f'Fitted n = {fitted_n:.2f}')
        axes[1, i].legend(fontsize=8)
        axes[1, i].grid(True, alpha=0.3)
    
    plt.suptitle('Power Law Images with Different Exponents', fontsize=14)
    plt.tight_layout()
    plt.savefig('power_law_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Visualization saved as 'power_law_comparison.png'")


def batch_generate_datasets(n_values=[1.5, 2.0, 2.5, 3.0], 
                           ntrain=150000, 
                           nvalid=50000,
                           size=64,
                           seed=1234):
    """
    Generate multiple datasets with different power law exponents.
    """
    print("="*60)
    print("BATCH GENERATION OF POWER LAW DATASETS")
    print("="*60)
    
    for n in n_values:
        print(f"\n--- Generating dataset with n = {n} ---")
        
        # Generate dataset
        train_images, valid_images = generate_power_law_dataset(
            ntrain=ntrain,
            nvalid=nvalid,
            size=size,
            n=n,
            seed=seed,
            normalize=True
        )
        
        # Save to HDF5
        output_file = f'power_law_n{n}_64x64_processed.h5'
        save_to_hdf5(
            train_images,
            valid_images,
            output_file,
            n,
            dataset_name=f'PowerLaw_n{n}'
        )
        
        # Verify a few samples
        print(f"\nVerifying power spectrum for n={n}...")
        rng = np.random.default_rng(seed + int(n*1000))
        fitted_values = []
        for i in range(5):
            idx = rng.integers(0, ntrain)
            _, _, fitted_n = verify_power_spectrum(train_images[idx], n, plot=False)
            fitted_values.append(fitted_n)
        
        mean_fitted = np.mean(fitted_values)
        std_fitted = np.std(fitted_values)
        print(f"  Expected n: {n:.2f}")
        print(f"  Fitted n: {mean_fitted:.2f} ± {std_fitted:.3f} (mean ± std from 5 samples)")
        
        # Compute basic statistics
        print(f"\nDataset statistics for n={n}:")
        print(f"  Training set:")
        print(f"    Shape: {train_images.shape}")
        print(f"    Mean: {np.mean(train_images):.6f}")
        print(f"    Std: {np.std(train_images):.6f}")
        print(f"    Min: {np.min(train_images):.3f}")
        print(f"    Max: {np.max(train_images):.3f}")
        print(f"  Validation set:")
        print(f"    Shape: {valid_images.shape}")
        print(f"    Mean: {np.mean(valid_images):.6f}")
        print(f"    Std: {np.std(valid_images):.6f}")
    
    print("\n" + "="*60)
    print("BATCH GENERATION COMPLETE")
    print("="*60)


def analyze_dataset_from_file(hdf5_file):
    """
    Load and analyze a previously generated dataset.
    """
    print(f"\nAnalyzing dataset from: {hdf5_file}")
    
    with h5py.File(hdf5_file, 'r') as f:
        # Load metadata
        print("\nMetadata:")
        for key, value in f.attrs.items():
            print(f"  {key}: {value}")
        
        # Load images
        train_images = f['train'][:]
        valid_images = f['valid'][:]
        n = f.attrs.get('power_law_exponent', 2.0)
        
        print(f"\nDataset shapes:")
        print(f"  Training: {train_images.shape}")
        print(f"  Validation: {valid_images.shape}")
        
        # Analyze a few random samples
        print(f"\nVerifying power spectrum (n={n})...")
        rng = np.random.default_rng(42)
        fitted_values = []
        
        for i in range(10):
            idx = rng.integers(0, len(train_images))
            _, _, fitted_n = verify_power_spectrum(train_images[idx], n, plot=False)
            fitted_values.append(fitted_n)
        
        fitted_values = np.array(fitted_values)
        print(f"  Expected n: {n:.2f}")
        print(f"  Fitted n: {np.mean(fitted_values):.2f} ± {np.std(fitted_values):.3f}")
        print(f"  Min/Max fitted: {np.min(fitted_values):.2f} / {np.max(fitted_values):.2f}")
        
        # Display some sample images
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        for i in range(5):
            # Training samples
            axes[0, i].imshow(train_images[i], cmap='viridis')
            axes[0, i].set_title(f'Train {i}')
            axes[0, i].axis('off')
            
            # Validation samples
            axes[1, i].imshow(valid_images[i], cmap='viridis')
            axes[1, i].set_title(f'Valid {i}')
            axes[1, i].axis('off')
        
        plt.suptitle(f'Sample Images from Dataset (n={n})', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'samples_n{n}.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"Sample visualization saved as 'samples_n{n}.png'")


def quick_test():
    """
    Quick test with small dataset to verify everything works.
    """
    print("\n" + "="*60)
    print("QUICK TEST - Small Dataset")
    print("="*60)
    
    # Generate small test dataset
    n = 2.5
    train_images, valid_images = generate_power_law_dataset(
        ntrain=1000,
        nvalid=500,
        size=64,
        n=n,
        seed=42,
        normalize=True
    )
    
    # Save test dataset
    output_file = 'test_power_law.h5'
    save_to_hdf5(
        train_images,
        valid_images,
        output_file,
        n,
        dataset_name='TestPowerLaw'
    )
    
    print(f"\nTest dataset saved to: {output_file}")
    
    # Verify and visualize
    print("\nVerifying power spectrum...")
    k, power, fitted_n = verify_power_spectrum(train_images[0], n, plot=True)
    print(f"  Expected n: {n:.2f}")
    print(f"  Fitted n: {fitted_n:.2f}")
    
    print("\nQuick test complete!")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Demonstrate power law image generation'
    )
    parser.add_argument('--mode', type=str, default='test',
                       choices=['test', 'visualize', 'batch', 'analyze'],
                       help='Mode: test, visualize, batch, or analyze')
    parser.add_argument('--file', type=str, default=None,
                       help='HDF5 file to analyze (for analyze mode)')
    parser.add_argument('--n', type=float, nargs='+', default=[1.5, 2.0, 2.5, 3.0],
                       help='Power law exponents for batch mode')
    
    args = parser.parse_args()
    
    if args.mode == 'test':
        quick_test()
    
    elif args.mode == 'visualize':
        visualize_power_law_comparison(n_values=args.n)
    
    elif args.mode == 'batch':
        # For full dataset generation (this will take a while)
        batch_generate_datasets(
            n_values=args.n,
            ntrain=150000,
            nvalid=50000
        )
    
    elif args.mode == 'analyze':
        if args.file is None:
            print("Please provide a file to analyze with --file")
        else:
            analyze_dataset_from_file(args.file)
    
    print("\nDone!")
