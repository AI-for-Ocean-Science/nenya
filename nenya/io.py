""" I/O routines for SSL analysis """
import os
from pkg_resources import resource_filename

import json

class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    This module comes from:
    https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/vision/utils.py

    """

    def __init__(self, json_path):
        self.nenya_data = None
        self.data_folder = None
        self.lr_decay_epochs = None
        self.model_name = None
        self.cosine = None
        self.warmup_from = None
        self.warmup_to = None
        self.warmup_epochs = None
        self.model_folder = None
        self.latents_folder = None
        self.cuda_use = None
        self.valid_freq = None
        self.save_freq = None

        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
            
    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__

def load_opt(nenya_model:str):
    """ Load the SSL model options

    Args:
        nenya_model (str): name of the model
            e.g. 'LLC', 'CF', 'v4', 'v5'

    Raises:
        IOError: _description_

    Returns:
        tuple: SSL options, model file (str)
    """
    # Prep
    ssl_model_file = None
    if nenya_model == 'LLC' or nenya_model == 'LLC_local':
        ssl_model_file = 's3://llc/SSL/LLC_MODIS_2012_model/SimCLR_LLC_MODIS_2012_resnet50_lr_0.05_decay_0.0001_bsz_64_temp_0.07_trial_0_cosine_warm/last.pth'
        opt_path = os.path.join(resource_filename('ulmo', 'runs'),
                                'Nenya', 'LLC', 'experiments', 
                                'llc_modis_2012', 'opts.json')
    elif nenya_model == 'CF': 
        opt_path= os.path.join(resource_filename('ulmo', 'runs'),
            'Nenya', 'MODIS', 'v2', 'experiments',
            'modis_model_v2', 'opts_cloud_free.json')
    elif nenya_model == 'v4':  
        opt_path= os.path.join(resource_filename('ulmo', 'runs'),
            'Nenya', 'MODIS', 'v4', 'opts_ssl_modis_v4.json')
    elif nenya_model == 'v5': 
        opt_path= os.path.join(resource_filename('ulmo', 'runs'),
            'Nenya', 'MODIS', 'v5', 'opts_nenya_modis_v5.json')
        ssl_model_file = os.path.join(os.getenv('OS_SST'),
                                  'MODIS_L2', 'Nenya', 'models', 
                                  'v4_last.pth')  # Only the UMAP was retrained (for now)
    else:
        raise IOError("Bad model!!")

    opt = option_preprocess(ulmo_io.Params(opt_path))

    if ssl_model_file is None:
        ssl_model_file = os.path.join(opt.s3_outdir, 
                                  opt.model_folder, 'last.pth')

    # Return
    return opt, ssl_model_file
    