""" Extract latents from a model """


import time
import os
import h5py
import numpy as np
import pandas as pd
from tqdm.auto import trange

import torch
import tqdm

from ulmo.utils import id_collate
from ulmo import io as ulmo_io

from nenya.train_util import option_preprocess
from nenya.train_util import set_model
from nenya import io as nenya_io

from IPython import embed


class HDF5RGBDataset(torch.utils.data.Dataset):
    """Represents an abstract HDF5 dataset.
    
    Parameters:
        file_path: Path to the HDF5 file.
        dataset_names: List of dataset names to gather. 
        allowe_indices (np.ndarray): Set of images that can be grabbed
        
    Objects will be returned in this order.
    """
    def __init__(self, file_path, partition, allowed_indices=None):
        super().__init__()
        self.file_path = file_path
        self.partition = partition
        self.meta_dset = partition + '_metadata'
        # s3 is too risky and slow here
        self.h5f = h5py.File(file_path, 'r')
        # Indices -- allows us to work on a subset of the images by indices
        self.allowed_indices = allowed_indices
        if self.allowed_indices is None:
            self.allowed_indices = np.arange(self.h5f[self.partition].shape[0])

    def __len__(self):
        return self.allowed_indices.size
        #return self.h5f[self.partition].shape[0]
    
    def __getitem__(self, index):
        # Grab it
        data = self.h5f[self.partition][self.allowed_indices[index]]
        # Resize
        data = np.resize(data, (1, data.shape[-1], data.shape[-1]))
        data = np.repeat(data, 3, axis=0)
        # Metadata
        metadata = None
        return data, metadata


def main(opt_path:str, pp_files:list, clobber:bool=False, debug:bool=False):
    # Parse the model
    opt = nenya_io.Params(opt_path)
    option_preprocess(opt)

    # Prep
    model_base, existing_files = prep(opt)

    # Data afiles
    #pp_files = ['s3://viirs/PreProc/VIIRS_2013_98clear_192x192_preproc_viirs_std_train.h5']

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
        latent_dict = model_latents_extract(
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

def build_loader(data_file, dataset, batch_size=1, num_workers=1,
                 allowed_indices=None):
    # Generate dataset
    """
    This function is used to create the data loader for the latents
    creating (evaluation) process.

    Args: 
        data_file: (str) path of data file.
        dataset: (str) key of the used data in data_file.
        batch_size: (int) batch size of the evalution process.
        num_workers: (int) number of workers used in loading data.
    
    Returns:
        dset: (HDF5RGBDataset) HDF5 dataset of data_file.
        loader: (torch.utils.data.Dataloader) Dataloader created 
            using data_file.
    """
    dset = HDF5RGBDataset(data_file, partition=dataset, allowed_indices=allowed_indices)

    # Generate DataLoader
    loader = torch.utils.data.DataLoader(
        dset, batch_size=batch_size, shuffle=False, 
        collate_fn=id_collate,
        drop_last=False, num_workers=num_workers)
    
    return dset, loader

def calc_latent(model, image_tensor, using_gpu):
    """
    This is a function to calculate the latents.
    Args:
        model: (SupConResNet) model class used for latents.
        image_tensor: (torch.tensor) image tensor of the data set.
        using_gpu: (bool) flag for cude usage.
    """
    model.eval()
    if using_gpu:
        latents_tensor = model(image_tensor.cuda())
        latents_numpy = latents_tensor.cpu().numpy()
    else:
        latents_tensor = model(image_tensor)
        latents_numpy = latents_tensor.numpy()
    return latents_numpy


def prep(opt):
    """
    Prepare the environment for latent extraction.

    Args:
        opt (object): An object containing the options for the extraction.

    Returns:
        tuple: A tuple containing the model base name and a list of existing latent files.
    """
    # Grab the model from s3
    model_file = os.path.join(opt.s3_outdir,
        opt.model_folder, 'last.pth')
    model_base = os.path.basename(model_file)
    if not os.path.isfile(model_base):
        print(f"Grabbing model: {model_file}")
        ulmo_io.download_file_from_s3(model_base, model_file)
    else:
        print(f"Model was already downloaded: {model_base}")

    # Grab existing for clobber
    latents_path = os.path.join(opt.s3_outdir, opt.latents_folder)
    latent_files = ulmo_io.list_of_bucket_files(latents_path)
    existing_files = [os.path.basename(ifile) for ifile in latent_files]

    # Return
    return model_base, existing_files

def model_latents_extract(opt, data_file, 
                          model_path, 
                          remove_module=True, loader=None,
                          partitions=('train', 'valid'),
                          allowed_indices=None,
                          debug:bool=False):
    """
    This function is used to obtain the latents of input data.
    And write them to disk
    
    Args:
        opt: (Parameters) parameters used to create the model.
        data_file: (str) path of data_file.
        model_path: (string) path of the saved model file.
        save_path: (string or None) path for saving the latents.
        partitions: (list) list of keys in the h5py file [e.g. 'train', 'valid'].
        loader: (torch.utils.data.DataLoader, optional) Use this DataLoader, if provided
        allowed_indices: (np.ndarray) Set of images that can be grabbed

    Returns:
        latent_dict: (dict) dictionary of latents for each partition.
    """
    using_gpu = torch.cuda.is_available()
    model, _ = set_model(opt, cuda_use=using_gpu)
    if not using_gpu:
        model_dict = torch.load(model_path, map_location=torch.device('cpu'))
    else:
        model_dict = torch.load(model_path)

    if remove_module:
        new_dict = {}
        for key in model_dict['model'].keys():
            new_dict[key.replace('module.','')] = model_dict['model'][key]
        model.load_state_dict(new_dict)
    else:
        model.load_state_dict(model_dict['model'])
    print("Model loaded")

    # Create Data Loader for evaluation
    batch_size_eval, num_workers_eval = opt.batch_size_valid, opt.num_workers


    # Loop on partitions
    latent_dict = {}
    for partition in partitions:
        # parition exists?
        with h5py.File(data_file, 'r') as f:
            if partition in f.keys():
                print(f"Working on: {partition}")
            else:
                print(f"Partition {partition} not found in {data_file}")
                continue 
        # Data
        _, loader = build_loader(data_file, partition, 
                                    batch_size_eval, num_workers_eval,
                                    allowed_indices=allowed_indices)
        # Debug?
        if debug:
            total = 2
        else:
            total = len(loader)

        print("Beginning to evaluate")
        model.eval()
        with torch.no_grad():
            latents_numpy = [calc_latent(
                model, data[0], using_gpu) for data in tqdm.tqdm(
                    loader, total=total, unit='batch', 
                    desc='Computing latents')]

        latent_dict[partition] = np.concatenate(latents_numpy)

    #return np.concatenate(latents_numpy)
    return latent_dict
    
