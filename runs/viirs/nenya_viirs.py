""" Nenya Analayis of VIIRS -- 
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

from ulmo import io as ulmo_io

from nenya import train as nenya_train
from nenya import latents_extraction
from nenya import io as nenya_io
from nenya.train_util import option_preprocess

from IPython import embed


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
    # Parse the model
    opt = nenya_io.Params(opt_path)
    option_preprocess(opt)

    # Prep
    model_base, existing_files = latents_extraction.prep(opt)

    # Data afiles
    pp_files = ['s3://viirs/PreProc/VIIRS_2013_98clear_192x192_preproc_viirs_std_train.h5']

    for ifile in pp_files:
        print(f"Working on {ifile}")
        data_file = os.path.basename(ifile)

        # Setup
        latents_file = data_file.replace('_preproc', '_latents')
        if latents_file in existing_files and not clobber:
            print(f"Not clobbering {latents_file} in s3")
            continue

        s3_file = os.path.join(opt.s3_outdir, opt.latents_folder, latents_file) 

        # Download
        if not os.path.isfile(data_file):
            ulmo_io.download_file_from_s3(data_file, ifile)
        else:
            print(f"Data file already downloaded: {data_file}")

        # Extract
        latent_dict = latents_extraction.model_latents_extract(
            opt, data_file, model_base, debug=debug)
        # Save
        latents_hf = h5py.File(latents_file, 'w')
        for partition in latent_dict.keys():
            latents_hf.create_dataset(partition, data=latent_dict[partition])
        latents_hf.close()

        # Push to s3
        print("Uploading to s3..")
        ulmo_io.upload_file_to_s3(latents_file, s3_file)

        # Remove data file
        if not debug:
            os.remove(data_file)
            print(f'{data_file} removed')


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
    
    # Train the model
    if args.func_flag == 'train':
        print("Training Starts.")
        train(args.opt_path, debug=args.debug)
        print("Training Ends.")
        # python -u nenya_viirs.py train --opt_path opts_viirs_v1.json 

    # Evaluate
    if args.func_flag == 'evaluate':
        print("Evaluation Starts.")
        evaluate(args.opt_path, debug=args.debug)
        print("Evaluation Ends.")
        # python -u nenya_viirs.py evaluate --opt_path opts_viirs_v1.json 

    # python ssl_modis_v4.py --func_flag extract_new --ncpu 20 --local --years 2020 --debug
    if args.func_flag == 'extract_new':
        ncpu = args.ncpu if args.ncpu is not None else 10
        years = [int(item) for item in args.years.split(',')] if args.years is not None else [2020,2021]
        extract_modis(debug=args.debug, n_cores=ncpu, local=args.local, years=years)

    # python ssl_modis_v4.py --func_flag revert_mask --debug
    if args.func_flag == 'revert_mask':
        revert_mask(debug=args.debug)

    # python ssl_modis_v4.py --func_flag preproc --debug
    if args.func_flag == 'preproc':
        modis_20s_preproc(debug=args.debug)

    # python ssl_modis_v4.py --func_flag ulmo_evaluate --debug
    #  This comes before the slurp and cut
    if args.func_flag == 'ulmo_evaluate':
        modis_ulmo_evaluate(debug=args.debug)

    # python ssl_modis_v4.py --func_flag slurp_tables --debug
    if args.func_flag == 'slurp_tables':
        slurp_tables(debug=args.debug)

    # python ssl_modis_v4.py --func_flag cut_96 --debug
    #if args.func_flag == 'cut_96':
    #    cut_96(debug=args.debug)

    # python ssl_modis_v4.py --func_flag ssl_evaluate --debug
    if args.func_flag == 'ssl_evaluate':
        main_ssl_evaluate(args.opt_path, debug=args.debug)
        
    # python ssl_modis_v4.py --func_flag DT40 --debug --local
    if args.func_flag == 'DT40':
        calc_dt40(args.opt_path, debug=args.debug, local=args.local,
                  redo=args.redo)

    # python ssl_modis_v4.py --func_flag umap --debug --local
    if args.func_flag == 'umap':
        ssl_v4_umap(args.opt_path, debug=args.debug, local=args.local)

    # Repeat UMAP analysis by DT using alpha instead
    # python ssl_modis_v4.py --func_flag alpha --debug --local
    if args.func_flag == 'alpha':
        ssl_v4_umap(args.opt_path, metric='alpha', debug=args.debug, local=args.local)

