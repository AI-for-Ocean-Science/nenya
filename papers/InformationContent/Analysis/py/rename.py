#!/usr/bin/env python
"""
Simple script to rename files containing 'SST' to 'SSTa' in Analysis/Pk and pca/ directories.
"""

import os
from pathlib import Path


def rename_sst_files(directory):
    """
    Rename all files in directory that contain 'SST' to 'SSTa'.

    Args:
        directory: Path to the directory to process
    """
    directory = Path(directory)

    if not directory.exists():
        print(f"Directory not found: {directory}")
        return

    renamed_count = 0

    for file_path in directory.iterdir():
        if file_path.is_file() and 'SST' in file_path.name:
            new_name = file_path.name.replace('SST', 'SSTa')
            new_path = file_path.parent / new_name

            print(f"Renaming: {file_path.name} -> {new_name}")
            file_path.rename(new_path)
            renamed_count += 1

    print(f"Renamed {renamed_count} files in {directory}")


def main():
    # Get the base directory (Analysis/)
    script_dir = Path(__file__).parent
    analysis_dir = script_dir.parent

    # Define directories to process
    pk_dir = analysis_dir / "Pk"
    pca_dir = analysis_dir / "pca"

    print("Starting file renaming...\n")

    # Rename files in Pk directory
    print("Processing Pk directory:")
    rename_sst_files(pk_dir)

    print("\nProcessing pca directory:")
    rename_sst_files(pca_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
