""" Nenya Analayis of LLC -- 
"""

import os
from typing import IO
import numpy as np

import h5py
import numpy as np
import argparse

import pandas
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from matplotlib import pyplot as plt
import seaborn as sns

from nenya import train as nenya_train
from nenya import latents_extraction

from IPython import embed

def build_training_set(ntrain:int=300000, nvalid:int=100000,
                       seed:int=76543, debug:bool=False):
    """
    Builds the training set for the LLC model.

    Parameters:
    - ntrain (int): Number of training samples to include in the training set. Default is 300000.
    - nvalid (int): Number of validation samples to include in the training set. Default is 100000.
    - seed (int): Seed value for random number generation. Default is 76543.
    - debug (bool): Flag to enable debug mode. If True, a smaller number of samples will be used for training and validation. Default is False.
    """
    if debug:
        ntrain = 1000
        nvalid = 1000

    # Init seed
    np.random.seed(seed)
    
    #s3://llc/PreProc/LLC_uniform144_nonoise_preproc.h5
    train_file= os.path.join(os.getenv('OS_OGCM'),
                             'LLC', 'Gallmeier2023', 'PreProc', 
                             'LLC_uniform144_nonoise_preproc.h5')
    #s3://llc/mae/PreProc/Enki_LLC_valid_nonoise_preproc.h5
    valid_file= os.path.join(os.getenv('OS_OGCM'),
                             'LLC', 'Enki', 'PreProc', 
                             'Enki_LLC_valid_nonoise_preproc.h5')

    # Load

    print(f"Loding: {train_file}")
    with h5py.File(train_file, 'r') as f:
        train = f['valid'][:]
        train_meta = f['valid_metadata'][:]
        train_clms = f['valid_metadata'].attrs['columns']

    print(f"Loding: {valid_file}")
    with h5py.File(valid_file, 'r') as f:
        valid = f['valid'][:]
        valid_meta = f['valid_metadata'][:]
        valid_clms = f['valid_metadata'].attrs['columns']

    # Grab randmoly from each
    train_idx = np.random.choice(np.arange(train.shape[0]), ntrain, replace=False)
    valid_idx = np.random.choice(np.arange(valid.shape[0]), nvalid, replace=False)

    if debug:
        embed(header='60 of nenya_llc')
    # Select
    train = train[train_idx]
    valid = valid[valid_idx]
    train_meta = train_meta[train_idx]
    valid_meta = valid_meta[valid_idx]

    # Write to disk
    with h5py.File('LLC_nenya_training.h5', 'w') as f:
        f.create_dataset('train', data=train)
        f.create_dataset('valid', data=valid)
        # Meta
        dset = f.create_dataset('valid_metadata', data=valid_meta.astype('S'))
        dset.attrs['columns'] = valid_clms
        dset = f.create_dataset('train_metadata', data=train_meta.astype('S'))
        dset.attrs['columns'] = train_clms
    

def train(opt_path:str, debug:bool=False, save_file:str=None):
    """Train the model

    Args:
        opt_path (str): Path + filename of options file
        debug (bool, optional): 
        save_file (str, optional): 
    """
    # Do it
    nenya_train.main(opt_path, debug=debug, save_file=save_file)
        
def evaluate(opt_path, debug=False, clobber=False):
    """
    This function is used to obtain the latents of the trained model
    for all of VIIRS

    Args:
        opt_path: (str) option file path.
        model_name: (str) model name 
        clobber: (bool, optional)
            If true, over-write any existing file
    """
    pp_files = ['s3://llc/Nenya/PreProc/LLC_nenya_training.h5']
    # Evaluate
    latents_extraction.main(opt_path, pp_files, clobber=clobber, debug=debug)
    

def parse_option():
    """
    This is a function used to parse the arguments in the training.
    
    Returns:
        args: (dict) dictionary of the arguments.
    """
    parser = argparse.ArgumentParser("argument for training.")
    parser.add_argument("func_flag", type=str, 
                        help="function to execute: train,evaluate,umap,umap_ndim3,sub2010,collect")
    parser.add_argument("--opt_path", type=str, 
                        default='opts_ssl_modis_v4.json',
                        help="Path to options file")
    parser.add_argument("--model", type=str, 
                        default='2010', help="Short name of the model used [2010,CF]")
    parser.add_argument('--debug', default=False, action='store_true',
                        help='Debug?')
    parser.add_argument('--local', default=False, action='store_true',
                        help='Local?')
    parser.add_argument('--clobber', default=False, action='store_true',
                        help='Clobber existing files')
    parser.add_argument('--redo', default=False, action='store_true',
                        help='Redo?')
    parser.add_argument("--outfile", type=str, 
                        help="Path to output file")
    parser.add_argument("--umap_file", type=str, 
                        help="Path to UMAP pickle file for analysis")
    parser.add_argument("--table_file", type=str, 
                        help="Path to Table file")
    parser.add_argument("--ncpu", type=int, help="Number of CPUs")
    parser.add_argument("--years", type=str, help="Years to analyze")
    parser.add_argument("--cf", type=float, 
                        help="Clear fraction (e.g. 96)")
    args = parser.parse_args()
    
    return args

        
if __name__ == "__main__":
    # get the argument of training.
    args = parse_option()

    # run the 'main_train()' function.
    if args.func_flag == 'dataset':
        build_training_set(debug=args.debug)
        print("Dataset Ends.")
        # python -u nenya_llc.py dataset --debug
        # python -u nenya_llc.py dataset 

    
    # Train python -u nenya_llc.py evaluate --opt_path opts_llc_v1.json 
        # python -u nenya_llc.py evaluate --opt_path opts_llc_v1.json 