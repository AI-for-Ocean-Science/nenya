""" Module to extract a training set for the LLC """

import os
import numpy as np
from importlib.resources import files

import h5py
import extract_utils

from wrangler.ogcm import llc
from wrangler.tables import io as tbl_io
from wrangler.preproc import io as pp_io
from wrangler.extract import ex_ogcm
from wrangler import utils as wr_utils
from wrangler.datasets.loader import load_dataset

from fronts import io as fronts_io
from fronts.dbof import io as dbof_io
from fronts.train import datasets

from IPython import embed

import info_defs

pdict = info_defs.grab_paths('LLC_SST')

# Hard code as needed

if 'OS_OGCM' not in os.environ.keys():
    os.environ['OS_OGCM'] = '/orcd/data/abodner/002/abigail/swot/nenya_data'

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
def ex_ssh():
    """ Extract SSH data from LLC and prepare for training """

    dbof_dev_json_file = os.path.join(files('fronts.runs.dbof.dev'), 'llc4320_dbof_dev.json')

    dbof_config = {
        "name": "LLC4320_SSH",
        "description": "A small test set for Jake to try out the DBOF model training",
        "DBOF": "DBOF_dev",
        "dataset": "LLC4320",
        "sampling": {
            "type": "random", 
        },
        "inputs": ["SSH"],
        "ntest": 0,
        "ntrain": 150000,
        "nvalid": 50000,
        "targets": []
    }

    # Generate the individual train, valid files
    meta_tbl = datasets.generate_from_dbof(
            dbof_dev_json_file, 
            dbof_config,
            path_outdir=local_preproc_path,
            skip_test=True, clobber=True)

    # Fuss about
    dbof_dict = fronts_io.loadjson(dbof_dev_json_file)
    dbof_table = dbof_io.load_main_table(dbof_dict)

    # Grab the entries in meta_tbl using UID
    idx = wr_utils.match_ids(meta_tbl.UID.values, dbof_table.UID.values, require_in_match=True)
    llc_table = dbof_table.iloc[idx].copy()

    # Load up h5 files
    train_file = os.path.join(local_preproc_path, f"{dbof_config['name']}_train.h5")
    valid_file = os.path.join(local_preproc_path, f"{dbof_config['name']}_valid.h5")

    # Concatenate
    with h5py.File(train_file, 'r') as f:
        train_ssh = f['inputs'][:,0,:,:]
    with h5py.File(valid_file, 'r') as f:
        valid_ssh = f['inputs'][:,0,:,:]

    tbl_file = os.path.join(local_tables_path, 'LLC_random_SSH.parquet')
    out_file = os.path.join(local_preproc_path, 'LLC_random_SSH.h5')

    # Write
    tbl_io.write_main_table(llc_table, tbl_file)
    with h5py.File(out_file, 'w') as f:
        f.create_dataset('train', data=train_ssh)
        f.create_dataset('valid', data=valid_ssh)
        f.attrs['n_train'] = dbof_config['ntrain']
        f.attrs['n_valid'] = dbof_config['nvalid']
        f.attrs['image_shape'] = train_ssh.shape[1:]

# Command line execution
if __name__ == '__main__':
    #ex_nonoise()
    #ex_noise()
    ex_ssh()