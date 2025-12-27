# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Nenya is a contrastive learning framework for remote sensing data (sea surface temperature from VIIRS/MODIS satellites). It implements SimCLR/SupCon self-supervised learning using ResNet backbones to learn meaningful latent representations of ocean SST images.

Reference paper: https://ui.adsabs.harvard.edu/abs/2023ITGRS..6100272P/abstract

## Installation

```bash
pip install -e .
```

Requires Python >3.11.0. Key dependencies: PyTorch, torchvision, scikit-learn, h5py, xarray.

## Commands

### Training
```python
from nenya.train import main as train_main
train_main("path/to/opts.json", debug=False, load_epoch=None)
```

Or via workflow:
```python
from nenya import workflow
workflow.train("path/to/opts.json", load_epoch=None, debug=False)
```

### Latent Extraction
```python
from nenya import workflow
workflow.evaluate(
    opts_file="path/to/opts.json",
    preproc_file="path/to/preproc.h5",
    latents_file="output_latents.h5",
    local_model_path=None,
    use_gpu=False
)
```

### PCA on Latents
```python
from nenya import pca
pca.fit_latents("latents.h5", "pca_output.npz", key="train")
```

### Run Tests
```bash
cd nenya/tests
python test_sst.py
```
Tests require downloading data from S3 (see comments in test_sst.py).

### Build Docs
```bash
cd docs
make html
```

## Architecture

### Core Modules

- **train.py** / **train_util.py**: Training loop and data loading. `train.main()` is the entry point. Uses `NenyaDataset` for HDF5 data and `TwoCropTransform` for contrastive augmentations.

- **models/resnet_big.py**: ResNet backbone (`SupConResNet` class) with projection head. Supports resnet18/34/50/101. The encoder outputs features that are L2-normalized after the projection head.

- **losses.py**: `SupConLoss` implements the contrastive loss (SimCLR when no labels provided, SupCon with labels).

- **latents_extraction.py**: `evaluate()` extracts latent vectors from trained models. `HDF5RGBDataset` handles data loading for inference.

- **pca.py**: PCA analysis of latent space. `fit_latents()` fits PCA, `generate_eigenmode_with_regularization()` reconstructs images from eigenmodes.

- **params.py**: `Params` class loads JSON config. `option_preprocess()` sets up model paths and warmup schedules.

- **workflow.py**: High-level API wrapping train/evaluate/eigenmode workflows.

### Data Flow

1. **Input**: HDF5 files with SST cutouts (train/valid partitions)
2. **Augmentations**: RandomFlip, RandomRotate, JitterCrop, GaussianNoise, Demean (configured in opts JSON)
3. **Training**: SimCLR contrastive learning with SupConResNet
4. **Output**: Model checkpoints (`.pth`), latent vectors (`.h5`), PCA decomposition (`.npz`)

### Configuration

Training is configured via JSON files (see `runs/viirs/opts_nenya_viirs_v1.json`):
- `ssl_method`: "SimCLR" or "SupCon"
- `ssl_model`: "resnet18", "resnet34", "resnet50", "resnet101"
- `feat_dim`: latent space dimension
- `batch_size_train`/`batch_size_valid`: batch sizes
- `nchannels`: 1 (grayscale) or 3 (RGB-replicated)
- Augmentation flags: `flip`, `rotate`, `demean`, `gauss_noise`, `random_cropjitter`

### Key Conventions

- Preprocessing files use `.h5` extension with `train`/`valid` keys
- Latent files have same structure but store feature vectors
- Models saved as `ckpt_epoch_N.pth` or `last.pth`
- S3 paths prefixed with `s3://` are handled via wrangler library
