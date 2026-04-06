"""
PCA Analysis on PreProcessed Images

Performs PCA analysis directly on preprocessed images (64x64 or nx x ny),
reducing to 256 dimensions. Images are flattened before PCA.

To run: python py/calc_preproc_pca.py
"""

import os
import sys
import numpy as np
import h5py
from sklearn.decomposition import PCA

# Add parent directory to path for imports
sys.path.append(os.path.abspath("../Analysis/py"))
import info_defs


def load_preproc_images(
    preproc_file: str,
    key: str = 'train',
    max_samples: int = 150000
) -> np.ndarray:
    """
    Load preprocessed images from HDF5 file and flatten them.
    
    Args:
        preproc_file: Path to HDF5 file containing preprocessed images
        key: HDF5 key ('train', 'valid', etc.)
        max_samples: Maximum number of samples to load
    
    Returns:
        Flattened image array (n_samples, nx*ny) or (n_samples, nx*ny*channels)
    """
    print(f"Loading preprocessed images from: {preproc_file}")
    print(f"Key: {key}")
    
    with h5py.File(preproc_file, 'r') as f:
        if key not in f:
            available_keys = list(f.keys())
            raise ValueError(f"Key '{key}' not found. Available keys: {available_keys}")
        
        # Get the full dataset shape
        full_shape = f[key].shape
        print(f"Full dataset shape: {full_shape}")
        
        # Determine how many samples to load
        n_samples = min(max_samples, full_shape[0]) if max_samples is not None else full_shape[0]
        print(f"Loading {n_samples} samples...")
        
        # Load the images
        images = f[key][:n_samples]
    
    print(f"Loaded images shape: {images.shape}")
    
    # Flatten the images
    # If images are (n_samples, nx, ny) -> flatten to (n_samples, nx*ny)
    # If images are (n_samples, channels, nx, ny) -> flatten to (n_samples, channels*nx*ny)
    n_samples = images.shape[0]
    flattened = images.reshape(n_samples, -1)
    
    print(f"Flattened shape: {flattened.shape}")
    print(f"Features per sample: {flattened.shape[1]}")
    
    return flattened


def fit_pca_on_images(
    images_flat: np.ndarray,
    n_components: int = 256
) -> dict:
    """
    Fit PCA on flattened images.
    
    Args:
        images_flat: Flattened images (n_samples, n_features)
        n_components: Number of PCA components to keep
    
    Returns:
        Dictionary containing PCA results
    """
    print(f"\nFitting PCA...")
    print(f"  Input shape: {images_flat.shape}")
    print(f"  Target components: {n_components}")
    
    # Fit PCA
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(images_flat)
    
    print(f"  Transformed shape: {transformed.shape}")
    print(f"  Total explained variance: {pca.explained_variance_ratio_.sum():.6f}")
    
    # Package results
    results = {
        'Y': transformed,  # PCA-transformed data (n_samples, 256)
        'M': pca.components_,  # Principal components (256, n_features)
        'mean': pca.mean_,  # Mean of training data
        'explained_variance': pca.explained_variance_,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'singular_values': pca.singular_values_,
        'n_components': n_components,
        'n_samples': images_flat.shape[0],
        'n_features': images_flat.shape[1]
    }
    
    return results


def analyze_variance(results: dict):
    """
    Print variance analysis.
    
    Args:
        results: Dictionary containing PCA results
    """
    print("\n" + "="*70)
    print("VARIANCE ANALYSIS")
    print("="*70)
    
    var_ratio = results['explained_variance_ratio']
    cumsum = np.cumsum(var_ratio)
    
    print(f"\nTop 10 components:")
    for i in range(min(10, len(var_ratio))):
        print(f"  Component {i+1:3d}: {var_ratio[i]:.6f} (cumulative: {cumsum[i]:.6f})")
    
    # Check various thresholds
    for threshold in [0.90, 0.95, 0.99]:
        n_comp = np.argmax(cumsum >= threshold) + 1
        print(f"\nComponents for {threshold*100:.0f}% variance: {n_comp}")
    
    print(f"\nTotal variance with {results['n_components']} components: {cumsum[-1]:.6f}")
    print("="*70 + "\n")


def save_results(results: dict, output_file: str):
    """
    Save PCA results to .npz file.
    
    Args:
        results: Dictionary containing PCA results
        output_file: Path to output file
    """
    # Create output directory if needed
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    print(f"Saving results to: {output_file}")
    np.savez_compressed(output_file, **results)
    
    file_size = os.path.getsize(output_file) / 1024 / 1024
    print(f"Saved! File size: {file_size:.2f} MB\n")


def pca_preproc_dataset(
    dataset: str,
    key: str = 'train',
    max_samples: int = 150000,
    n_components: int = 256,
    clobber: bool = False
):
    """
    Perform PCA on a preprocessed dataset.
    
    Args:
        dataset: Dataset name (e.g., 'MNIST', 'MODIS_SST', etc.)
        key: HDF5 key to use ('train', 'valid', etc.)
        max_samples: Maximum number of samples (150000 by default)
        n_components: Number of PCA components (256 by default)
    """
    print("\n" + "="*70)
    print(f"PCA ANALYSIS: {dataset}")
    print("="*70 + "\n")
    
    # Get paths using info_defs
    pdict = info_defs.grab_paths(dataset)
    preproc_file = pdict['preproc_file']
    
    # Output file
    output_file = os.path.join('pca', f'pca_preproc_{dataset}.npz')
    if n_components != 256:
        output_file = os.path.join('pca', f'pca_preproc_{dataset}_{n_components}.npz')
    if os.path.exists(output_file) and not clobber:
        print(f"Output file already exists: {output_file}")
        print("Use clobber=True to overwrite.\n")
        return
    
    print(f"Dataset: {dataset}")
    print(f"Input file: {preproc_file}")
    print(f"Output file: {output_file}")
    print(f"Key: {key}")
    print(f"Max samples: {max_samples}")
    print(f"PCA components: {n_components}\n")
    
    # Check if input file exists
    if not os.path.exists(preproc_file):
        raise FileNotFoundError(f"Preprocessed file not found: {preproc_file}")
    
    # Load and flatten images
    images_flat = load_preproc_images(preproc_file, key=key, max_samples=max_samples)
    
    # Fit PCA
    results = fit_pca_on_images(images_flat, n_components=n_components)
    
    # Analyze variance
    analyze_variance(results)
    
    # Save results
    save_results(results, output_file)
    
    print("="*70)
    print(f"COMPLETED: {dataset}")
    print("="*70 + "\n")


# Main execution
if __name__ == '__main__':

    # Example: Process MNIST
    #pca_preproc_dataset('MNIST', key='train', max_samples=150000, n_components=256)

    # Standard 256
    if False:
        for dataset in info_defs.all_datasets:
            pca_preproc_dataset(dataset, key='train', max_samples=150000, n_components=256)

    # Extend to 4096
    if True:
        for dataset in info_defs.all_datasets:
            ncomp = 28**2 if dataset == 'MNIST' else 4096
            pca_preproc_dataset(dataset, key='train', max_samples=150000, n_components=ncomp)
        
    
    
    # Example: Process MODIS SST
    # pca_preproc_dataset('MODIS_SST', key='train', max_samples=150000, n_components=256)
    
    # Example: Process VIIRS SST
    # pca_preproc_dataset('VIIRS_SST', key='train', max_samples=150000, n_components=256)
    
    # Example: Process SWOT L3
    # pca_preproc_dataset('SWOT_L3', key='train', max_samples=150000, n_components=256)
    
    # Example: Process LLC SST (no noise)
    # pca_preproc_dataset('LLC_SST_nonoise', key='train', max_samples=150000, n_components=256)
    
    # Example: Process ImageNet
    # pca_preproc_dataset('ImageNet', key='train', max_samples=150000, n_components=256)
    
    # Example: Process White Noise
    # pca_preproc_dataset('WNoise', key='train', max_samples=150000, n_components=256)
