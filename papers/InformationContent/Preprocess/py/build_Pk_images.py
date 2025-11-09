#!/usr/bin/env python3
"""
Simple example of generating power law spectrum images.
"""

import numpy as np
import h5py
from power_spectrum_images import main

def run(option:int):

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

    if option == 4:
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
        flg = -1
    else:
        flg = int(sys.argv[1])

    run(flg)