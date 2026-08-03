""" Module to extract a training set for the LLC """

import os
import numpy as np
import pandas
from importlib.resources import files

import h5py
import extract_utils

from wrangler import defs as wr_defs

from wrangler.ogcm import llc
from wrangler.tables import io as tbl_io
from wrangler.preproc import io as pp_io
from wrangler.extract import ex_ogcm
from wrangler import utils as wr_utils
from wrangler.datasets.loader import load_dataset

from fronts import io as fronts_io
from fronts.dbof import io as dbof_io
from fronts.train import datasets
from fronts.train import cutouts as t_cutouts

from IPython import embed

import info_defs

#pdict = info_defs.grab_paths('LLC_SST')

# Hard code as needed

if 'OS_OGCM' not in os.environ.keys():
    os.environ['OS_OGCM'] = '/orcd/data/abodner/002/abigail/swot/nenya_data'

# End of the LLC4320 spin-up period.  The run starts 2011-09-13 and the
# first 2 months are excluded from all training sets (final_steps.md,
# Phase 1).  The 2011-11-13 SST timestep sits exactly on the 2-month
# boundary and is retained.
LLC_SPINUP_END = '2011-11-13'

local_llc_path = os.path.join(os.getenv('OS_OGCM'), 'LLC')
local_tables_path = os.path.join(local_llc_path, 'Info', 'Tables')
local_orig_preproc_path = os.path.join(local_llc_path, 'Nenya', 'PreProc')
local_gall_preproc_path = os.path.join(local_llc_path, 'Gallmeier', 'PreProc')
local_preproc_path = os.path.join(local_llc_path, 'Info', 'PreProc')
tables_path = os.path.join(local_llc_path, 'Tables')
gall_tables_path = os.path.join(local_llc_path, 'Gallmeier', 'Tables')

def ex_nonoise(min_date:str=None, seed:int=None):
    """ Extract the nonoise SST training set from the LLC uniform table

    Args:
        min_date (str, optional): exclude cutouts earlier than this date
            (e.g. LLC_SPINUP_END to drop the model spin-up period).
        seed (int, optional): RNG seed for the random cutout selection
            (reproducibility of the train/valid draw).
    """
    if seed is not None:
        np.random.seed(seed)
    # Open the LLC Uniform file
    poptions=None
    extract_utils.prep_for_training(os.path.join(tables_path, 'LLC_uniform144_r0.5_nonoise.parquet'),
                    os.path.join(local_orig_preproc_path, 'LLC_uniform144_nonoise_preproc.h5'),
                    os.path.join(local_preproc_path, 'train_llc_nonoise.h5'),
                    os.path.join(local_tables_path, 'train_llc_nonoise.parquet'),
                    inpaint=False, poptions=poptions,
                    use_ppidx=True,
                    min_date=min_date,
                    n_train=150000, n_valid=50000,
                    orig_key='valid')

def ex_viirs_match():
    pdict = info_defs.grab_paths('LLC_SSTa_VIIRS')
    # Open the LLC VIIRS file
    poptions=None
    extract_utils.prep_for_training(os.path.join(gall_tables_path, 'llc_viirs_match.parquet'),
                    os.path.join(local_gall_preproc_path, 'LLC_VIIRS144_preproc.h5'),
                    pdict['preproc_file'],
                    os.path.join(local_tables_path, 'LLC4320_SSTa_VIIRS.parquet'), 
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
def ex_ssh(min_date:str=None, seed:int=None):
    """ Extract SSH data from LLC and prepare for training

    Args:
        min_date (str, optional): exclude DBOF timesteps earlier than this
            date (e.g. LLC_SPINUP_END to drop the model spin-up period).
            When set, the train/valid sampling is done here (mirroring the
            'random' branch of fronts.train.tables.dbof_gen_tvt, which has
            no date-cut option) and the cutout files are built directly
            with fronts.train.cutouts.create_hdf5_cutouts.
        seed (int, optional): RNG seed for the random cutout selection.
    """

    dbof_dev_json_file = os.path.join(files('fronts.runs.dbof.dev'), 'llc4320_dbof_dev.json')

    dbof_config = {
        "name": "LLC4320_SSHa",
        "description": "LLC SSH for the Information Content paper",
        "DBOF": "DBOF_dev",
        "dataset": "LLC4320",
        "sampling": {
            "type": "random",
        },
        "inputs": ["SSHa"],
        "ntest": 0,
        "ntrain": 150000,
        "nvalid": 50000,
        "targets": []
    }

    if seed is not None:
        np.random.seed(seed)

    if min_date is None:
        # Original path: let fronts do the sampling + cutout generation
        meta_tbl = datasets.generate_from_dbof(
                dbof_dev_json_file,
                dbof_config,
                path_outdir=local_preproc_path,
                skip_test=True, clobber=True)
    else:
        # Date-cut path: sample the DBOF table ourselves, then reuse the
        # fronts cutout machinery.  Restrict to rows with the SSHa field.
        dbof_dict = fronts_io.loadjson(dbof_dev_json_file)
        super_tbl = dbof_io.load_main_table(dbof_dict)
        super_tbl = super_tbl[super_tbl['SSHa']].copy()

        # Exclude the spin-up timesteps
        super_tbl = super_tbl[super_tbl['datetime'] >= min_date].copy()
        print(f"Cut the DBOF table to {len(super_tbl)} rows with datetime >= {min_date}")

        # Random train/valid split (as in dbof_gen_tvt 'random' sampling)
        ntrain, nvalid = dbof_config['ntrain'], dbof_config['nvalid']
        ridx = np.random.choice(super_tbl.index.values, ntrain+nvalid, replace=False)
        train_tbl = super_tbl.loc[ridx[:ntrain]].copy()
        valid_tbl = super_tbl.loc[ridx[ntrain:]].copy()

        # Build the cutout files and the meta table (as generate_from_dbof does)
        all_tables = []
        for tbl, dtype in zip([train_tbl, valid_tbl], ['train', 'valid']):
            outfile = os.path.join(local_preproc_path,
                                   f"{dbof_config['name']}_{dtype}.h5")
            print(f"Generating {dtype} set with {len(tbl)} entries to {outfile}")
            t_cutouts.create_hdf5_cutouts(
                dbof_dev_json_file, dbof_config, tbl, outfile, clobber=True)
            tbl['pp_type'] = wr_defs.tbl_dmodel['pp_type'][dtype]  # train=1, valid=0
            all_tables.append(tbl)
        meta_tbl = pandas.concat(all_tables, ignore_index=True)
        meta_file = os.path.join(local_preproc_path,
                                 f"{dbof_config['name']}_meta.parquet")
        meta_tbl.to_parquet(meta_file)
        print(f"Wrote {meta_file}")

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

    tbl_file = os.path.join(local_tables_path, 'LLC_random_SSHa.parquet')
    out_file = os.path.join(local_preproc_path, 'LLC_random_SSHa.h5')

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
    # Uniform -- re-extraction excluding the LLC4320 spin-up period
    #   (Phase 1 of final_steps.md, 2026-08-02).  Seeded for reproducibility.
    ex_nonoise(min_date=LLC_SPINUP_END, seed=12345)
    ex_noise()
    ex_ssh(min_date=LLC_SPINUP_END, seed=12345)

    # VIIRS
    #ex_viirs_match()