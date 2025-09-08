""" Module to extract a training set for the LLC """

import os
import numpy as np

import extract_utils

from wrangler.ogcm import llc
from wrangler.tables import io as tbl_io
from wrangler.preproc import io as pp_io
from wrangler.extract import ogcm as ex_ogcm
from wrangler.datasets.loader import load_dataset

from IPython import embed

import info_defs

pdict = info_defs.grab_paths('LLC_SST')

local_llc_path = os.path.join(os.getenv('OS_OGCM'), 'LLC')
local_tables_path = os.path.join(local_llc_path, 'Info', 'Tables')
local_orig_preproc_path = os.path.join(local_llc_path, 'Nenya', 'PreProc')
local_preproc_path = os.path.join(local_llc_path, 'Info', 'PreProc')
tables_path = os.path.join(local_llc_path, 'Tables')

def ex_nonoise():
    # Open the LLC Uniform file
    poptions=None
    extract_utils.prep_for_training(os.path.join(tables_path, 'LLC_uniform144_r0.5_nonoise.parquet'),
                    os.path.join(local_orig_preproc_path, 'LLC_uniform144_nonoise_preproc.h5'),
                    os.path.join(local_preproc_path, 'train_llc_nonoise.h5'), 
                    os.path.join(local_tables_path, 'train_llc_nonoise.parquet'), 
                    inpaint=False, poptions=poptions,
                    use_ppidx=True, 
                    n_train=150000, n_valid=50000,
                    orig_key='valid')

def ex_noise():
    # Add noise to the original file
    map_fn = extract_utils.partial(extract_utils.add_noise, noise=0.09)
    extract_utils.modify(
        os.path.join(local_preproc_path, 'train_llc_nonoise.h5'),
        os.path.join(local_preproc_path, 'train_llc_noise.h5'),
        map_fn, n_cores=15)

# Extract SSH data
def ex_ssh(n_train:int=170000, n_valid:int=60000):
    """ Extract SSH data from LLC and prepare for training """

    # Instantiate the AIOS_DataSet
    aios_ds = load_dataset('LLC4320_SSH')

    tbl_file = os.path.join(local_tables_path, 'LLC_uniform_SSH.parquet')
    out_file = os.path.join(local_preproc_path, 'LLC_uniform_SSH.h5')

    if not os.path.exists(tbl_file):
        # Generate a table
        llc_table = llc.build_table(debug=False, resol=0.5, minmax_lat=(-100., 57.))

        # Grab random rows
        idx_tv = np.random.choice(llc_table.index, n_train+n_valid, replace=False)
        llc_table = llc_table.loc[idx_tv].copy()
        llc_table.reset_index(inplace=True, drop=True)

        tv_idx = np.ones(n_train+n_valid, dtype=int)
        tv_idx[n_train:] = 0
        llc_table['pp_type'] = tv_idx

        # Write
        tbl_io.write_main_table(llc_table, tbl_file)
    else:
        print(f'Loading existing table: {tbl_file}')
        # Load
        llc_table = tbl_io.load_main_table(tbl_file)

    # Load options
    pp_dict = pp_io.load_options('preproc_llc_ssh_nonoise.json')

    # Run me
    llc_table = ex_ogcm.extract_llc(
        llc_table, aios_ds, pp_dict, out_file, n_cores=15, debug=True)

    # Write new table (there is some loss during extraction)
    tbl_io.write_main_table(llc_table, tbl_file)


# Command line execution
if __name__ == '__main__':
    #ex_nonoise()
    #ex_noise()
    ex_ssh()