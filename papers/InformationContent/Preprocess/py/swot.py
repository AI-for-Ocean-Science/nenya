import os, sys
import numpy as np
import h5py

# Local
sys.path.append(os.path.abspath("../Analysis/py"))
import info_defs

def main(ntrain=150000, nvalid=50000):
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
    outfile = os.path.join(os.getenv('OS_SSH'), 'SWOT_v2', 
        'ssha_unfiltered_64x64_54km_0-200000.nc')
    #cutouts.to_netcdf(outfile)
    outfile = os.path.join(os.getenv('OS_SSH'), 'SWOT_v2', 
        'ssha_unfiltered_64x64_54km.nc')
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    ssha_unfiltered.to_netcdf(outfile)
    print(f"Wrote {outfile}")

if __name__ == '__main__':
    #main()

    # Iury
    grabbing_iury_data()