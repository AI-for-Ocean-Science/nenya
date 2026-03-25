import os, sys
import numpy as np
import h5py
import pandas
import xarray as xr

# Local
sys.path.append(os.path.abspath("../Analysis/py"))
import info_defs

def main_L3(ntrain=150000, nvalid=50000):
    # Load up the existing file
    h5_file = os.path.join(os.getenv('OS_SSH'), 'swot_prototype_train.h5')

    print("Loading images from:", h5_file)
    with h5py.File(h5_file, 'r') as f:
        all_images = [f['valid'][:], f['train'][:]]
    # Combine the two sets of images
    all_images = np.concatenate(all_images, axis=0)[:,0,...]

    opts_file, path, preproc_file, latents_file = info_defs.grab_paths('SWOT_L3')
    # Split by 150000 and 50000

    print("Loaded")
    with h5py.File(preproc_file, 'w') as f:
        f.create_dataset('train', data=all_images[:ntrain])
        f.create_dataset('valid', data=all_images[ntrain:ntrain+nvalid])
        
        # Add metadata
        f.attrs['dataset'] = 'SWOT_L3'
        f.attrs['n_train'] = ntrain
        f.attrs['n_valid'] = nvalid
        f.attrs['image_shape'] = all_images.shape[1:]
    print(f"SWOT_L3 preprocessed and saved to: {preproc_file}")

def main_L2(ntrain=150000, nvalid=50000, debug=False):
    if debug:
        ntrain = 750
        nvalid = 250

    # Load the NetCDF source file
    nc_file = os.path.join(os.getenv('OS_SSH'), 'SWOT_v2',
                           'ssha_unfiltered_64x64_54km.nc')
    print("Loading SWOT L2 data from:", nc_file)
    ds = xr.open_dataset(nc_file)

    if debug:
        ds = ds.isel(cutout=slice(0, 1000))

    ssha = ds.ssha_unfiltered.values  # (cutout, y, x)
    lon = ds.longitude_avg.values
    lat = ds.latitude_avg.values
    time = ds.time.values

    # Filter out cutouts with any NaN
    good = ~np.any(np.isnan(ssha.reshape(ssha.shape[0], -1)), axis=1)
    good_indices = np.where(good)[0]  # original NetCDF cutout indices
    ssha = ssha[good]
    lon = lon[good]
    lat = lat[good]
    time = time[good]
    print(f"Kept {ssha.shape[0]} cutouts out of {good.size} (removed {good.size - ssha.shape[0]} with NaN)")

    # Randomly draw ntrain+nvalid cutouts
    total = ntrain + nvalid
    rng = np.random.default_rng(12345)
    idx = rng.choice(ssha.shape[0], size=total, replace=False)
    cutout_idx = good_indices[idx]  # map back to original NetCDF indices
    ssha = ssha[idx]
    lon = lon[idx]
    lat = lat[idx]
    time = time[idx]
    print(f"Randomly selected {total} cutouts")

    # Paths
    out_dict = info_defs.grab_paths('SWOT_L2')
    preproc_file = out_dict['preproc_file']
    path = out_dict['path']

    # Create output directories
    os.makedirs(os.path.dirname(preproc_file), exist_ok=True)
    tbl_dir = os.path.join(path, 'Tables')
    os.makedirs(tbl_dir, exist_ok=True)

    # Write preproc HDF5
    print("Writing preproc file:", preproc_file)
    with h5py.File(preproc_file, 'w') as f:
        f.create_dataset('train', data=ssha[:ntrain])
        f.create_dataset('valid', data=ssha[ntrain:ntrain+nvalid])

        f.attrs['dataset'] = 'SWOT_L2'
        f.attrs['n_train'] = ntrain
        f.attrs['n_valid'] = nvalid
        f.attrs['image_shape'] = ssha.shape[1:]
    print(f"SWOT_L2 preprocessed and saved to: {preproc_file}")

    # Build parquet metadata table
    pp_type = np.array(['train'] * ntrain + ['valid'] * nvalid)
    pp_idx = np.concatenate([np.arange(ntrain), np.arange(nvalid)])

    df = pandas.DataFrame({
        'pp_file': preproc_file,
        'pp_type': pp_type,
        'pp_idx': pp_idx,
        'cutout_idx': cutout_idx,
        'lon': lon,
        'lat': lat,
        'datetime': time,
    })

    tbl_file = os.path.join(tbl_dir, 'SWOT_L2_54km.parquet')
    df.to_parquet(tbl_file, index=False)
    print(f"Metadata table saved to: {tbl_file}")


def grabbing_iury_data():

    # Install pixi 
    #   curl -fsSL https://pixi.sh/install.sh | bash
    #   This failed on profx, so I grabbed the binary


    # Grab the swot_patterns Repo

    # pixi install
    # pixi shell

    import xarray as xr
    import matplotlib.pyplot as plt
    import numpy as np
    from glob import glob
    import random
    import icechunk

    from IPython import embed

    import os
    os.environ["AWS_PROFILE"] = "swot-user"

    storage = icechunk.s3_storage(
        bucket="iuryt-shared",
        prefix=f"icechunk/ocean/swot_cutouts/64x64_resampled_54km",
        region="us-west-2",
    )

    repo = icechunk.Repository.open(storage)
    session = repo.readonly_session("main")

    ds = xr.open_zarr(session.store, consolidated=False)

    #embed(header='65 of swot.py')
    ssha_unfiltered = ds.ssha_unfiltered

    # Grab 256 cutouts of ssha_unfiltered
    #cutouts = ssha_unfiltered.isel(cutout=slice(0, 200000))

    # Write to disk as netcdf
    print("Writing to disk...")
    #outfile = os.path.join(os.getenv('OS_SSH'), 'SWOT_v2', 
    #    'ssha_unfiltered_64x64_54km_0-200000.nc')
    #cutouts.to_netcdf(outfile)
    outfile = os.path.join(os.getenv('OS_SSH'), 'SWOT_v2', 
        'ssha_unfiltered_64x64_54km.nc')
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    ssha_unfiltered.to_netcdf(outfile)
    print(f"Wrote {outfile}")

if __name__ == '__main__':
    #main_L3()

    # Iury / L2
    #grabbing_iury_data()

    # L2
    main_L2()#debug=True)
