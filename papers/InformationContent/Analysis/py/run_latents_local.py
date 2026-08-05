""" Run a nenya_<dataset>.py task locally (CPU box, ocean14 env).

Works around two local quirks:
- Python >=3.14 defaults multiprocessing to 'forkserver', which pickles the
  DataLoader dataset; HDF5RGBDataset holds an open h5py handle and h5py
  objects cannot be pickled.  Forcing 'fork' restores the pre-3.14 behavior
  (and matches the cluster image, where extraction works as-is).
- Run from the Analysis/ directory so the relative opts/ paths resolve, e.g.:
    cd papers/InformationContent/Analysis
    PYTHONPATH=py:<wrangler>:<fronts>:<nenya> python -u py/run_latents_local.py nenya_LLC_nonoise evaluate
"""
import sys
import importlib

import torch.multiprocessing as mp

if __name__ == '__main__':
    module_name = sys.argv[1]            # e.g. nenya_LLC_nonoise
    task = sys.argv[2] if len(sys.argv) > 2 else 'evaluate'

    # Force fork BEFORE any DataLoader is created (see docstring)
    mp.set_start_method('fork', force=True)

    import torch
    torch.set_num_threads(32)

    module = importlib.import_module(module_name)
    module.main(task)
