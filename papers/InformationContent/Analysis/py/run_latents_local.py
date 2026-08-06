""" Local (CPU) latent-extraction runner for the Information Content paper.

Python >= 3.14 defaults multiprocessing to the 'forkserver' start method on
Linux, which breaks nenya.latents_extraction: its HDF5RGBDataset holds an
open h5py file handle ('h5f'), and h5py objects cannot be pickled when the
DataLoader spawns workers.  The historical behaviour (and what the Nautilus
pods with older Pythons use) is 'fork', where the handle is inherited and
never pickled -- so force 'fork' before torch creates any workers.

Run from the Analysis/ directory (the opts paths are relative to it):

    python py/run_latents_local.py nonoise
    python py/run_latents_local.py noise SSHa
"""
import sys
import multiprocessing as mp

# Map the command-line names to the per-dataset run modules in this directory
MODULES = {
    'nonoise': 'nenya_LLC_nonoise',
    'noise': 'nenya_LLC_noise',
    'SSHa': 'nenya_LLC_SSHa',
}

if __name__ == '__main__':
    # Must be set before torch builds any DataLoader workers (see docstring)
    mp.set_start_method('fork', force=True)

    targets = sys.argv[1:]
    if not targets or any(t not in MODULES for t in targets):
        raise SystemExit(f"Usage: python py/run_latents_local.py [{'|'.join(MODULES)}] ...")

    for t in targets:
        print(f"===== Extracting latents: {t} =====")
        mod = __import__(MODULES[t])
        mod.main('evaluate')
        print(f"===== Done: {t} =====")
