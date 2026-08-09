import os
from IPython import embed

all_datasets = ['MODIS_SSTa', 
                'MODIS_SSTa_2km',
                'VIIRS_SSTa', 
                'VIIRS_SSTa_2km', 
                'VIIRS_SSTa_sub', 
                #'LLC_SSTa_VIIRS', 
                'LLC_SSTa_nonoise', 
                'LLC_SSTa_noise', 
                'LLC_SSHa', 
                'SWOT_L2', 
                #'SWOT_SSR', 
                'WNoise',
                'Pk2',
                'Pk4',
                'MNIST',
                'ImageNet',
                ]

primary_remote_datasets = ['MODIS_SSTa', 
                'VIIRS_SSTa', 
                'SWOT_L2'] 

all_sst_datasets = ['MODIS_SSTa', 
                'MODIS_SSTa_2km',
                'VIIRS_SSTa', 
                'VIIRS_SSTa_2km', 
                'VIIRS_SSTa_sub', 
                'LLC_SSTa_nonoise', 
                'LLC_SSTa_noise'] 

natural_datasets = ['WNoise',
                    'Pk2',
                    'Pk4',
                    'MNIST',
                    'ImageNet',
                    ]

def grab_paths(dataset:str):

    out_dict = {}
    out_dict['path'] = None
    out_dict['preproc_file'] = None
    out_dict['latents_file'] = None
    out_dict['model_file'] = None

    if dataset == 'MNIST':
        if 'OS_DATA' in os.environ:
            path = os.path.join(os.getenv('OS_DATA'), 'Natural', 'MNIST', 'Info')
            out_dict['path'] = path
            out_dict['preproc_file'] = os.path.join(path, 'PreProc', 'mnist_resampled.h5')
            out_dict['latents_file'] = os.path.join(path, 'latents', 
                                'SimCLR_resnet50_lr_0.05_decay_0.0001_bsz_64_temp_0.07_trial_5_cosine_warm',
                                'mnist_resampled_latents.h5')
        out_dict['opts_file'] = 'opts_nenya_mnist.json'
        out_dict['pca_file'] = 'pca_latents_MNIST.npz'
        out_dict['macro'] = '\\mnist'
        out_dict['label'] = 'MNIST'
    elif dataset == 'ImageNet':
        if 'OS_DATA' in os.environ:
            path = os.path.join(os.getenv('OS_DATA'), 'Natural', 'ImageNet', 'Info')
            out_dict['path'] = path
            out_dict['preproc_file'] = os.path.join(path, 'PreProc', 'imagenet_processed.h5')
            out_dict['latents_file'] = os.path.join(path, 'latents', 'imagenet',
                                'SimCLR_resnet50_lr_0.05_decay_0.0001_bsz_64_temp_0.07_trial_5_cosine_warm',
                                'imagenet_latents.h5')
        out_dict['opts_file'] = 'opts_nenya_imagenet.json'
        out_dict['pca_file'] = 'pca_latents_ImageNet.npz'
        out_dict['macro'] = '\\inet'
        out_dict['label'] = 'ImageNet'
    elif dataset == 'WNoise':
        if 'OS_DATA' in os.environ:
            path = os.path.join(os.getenv('OS_DATA'), 'Natural', 'White_Noise', 'Info')
            out_dict['path'] = path
            out_dict['preproc_file'] = os.path.join(path, 'PreProc', 'wnoise_64x64_processed.h5')
            out_dict['latents_file'] = os.path.join(path, 'latents', 'wnoise',
                                'SimCLR_resnet50_lr_0.05_decay_0.0001_bsz_64_temp_0.07_trial_5_cosine_warm',
                                'wnoise_latents.h5')
            out_dict['latents_file_extended'] = out_dict['latents_file'].replace('wnoise_latents.h5', 'wnoise_latents_extended.h5')
        out_dict['opts_file'] = 'opts_nenya_wnoise.json'
        out_dict['opts_file_extended'] = 'opts_nenya_wnoise_extended.json'
        out_dict['pca_file'] = 'pca_latents_WNoise.npz'
        out_dict['macro'] = '\\wnoise'
        out_dict['label'] = 'WhiteNoise'
        # Extended training 
    elif dataset == 'Pk2':
        if 'OS_DATA' in os.environ:
            path = os.path.join(os.getenv('OS_DATA'), 'Natural', 'Pk', 'Info')
            out_dict['path'] = path
            out_dict['preproc_file'] = os.path.join(path, 'PreProc', 'power_law_n2_64x64.h5')
            out_dict['latents_file'] = os.path.join(path, 'latents', 'Pk2',
                                'SimCLR_resnet50_lr_0.05_decay_0.0001_bsz_64_temp_0.07_trial_5_cosine_warm',
                                'pk2_latents.h5')
        out_dict['opts_file'] = 'opts_nenya_pk2.json'
        out_dict['pca_file'] = 'pca_latents_Pk2.npz'
        out_dict['macro'] = '\\pktwo'
        out_dict['label'] = r'$P_2(k)$'
    elif dataset == 'Pk4':
        if 'OS_DATA' in os.environ:
            path = os.path.join(os.getenv('OS_DATA'), 'Natural', 'Pk', 'Info')
            out_dict['path'] = path
            out_dict['preproc_file'] = os.path.join(path, 'PreProc', 'power_law_n4_64x64.h5')
            out_dict['latents_file'] = os.path.join(path, 'latents', 'Pk4',
                                'SimCLR_resnet50_lr_0.05_decay_0.0001_bsz_64_temp_0.07_trial_5_cosine_warm',
                                'pk2_latents.h5')
        out_dict['opts_file'] = 'opts_nenya_pk4.json'
        out_dict['pca_file'] = 'pca_latents_Pk4.npz'
        out_dict['macro'] = '\\pkfour'
        out_dict['label'] = r'$P_4(k)$'
    elif dataset == 'orig_MODIS_SST_2km':
        if 'OS_SST' in os.environ:
            path = os.path.join(os.getenv('OS_SST'), 'MODIS_L2', 'Nenya')
            out_dict['path'] = path
            out_dict['preproc_file'] = os.path.join(path, 'PreProc', 'MODIS_R2019_2004_95clear_128x128_preproc_std.h5')
            out_dict['latents_file'] = os.path.join(path,
                        'latents/MODIS_R2019_v4_REDO',
                        'SimCLR_resnet50_lr_0.05_decay_0.0001_bsz_256_temp_0.07_trial_5_cosine_warm', 
                        'MODIS_R2019_2004_95clear_128x128_latents_std.h5')
        out_dict['opts_file'] = None
        # 200,000
        out_dict['pca_file'] = 'pca_latents_MODIS_SST_2km_sub.npz'
    elif dataset == 'MODIS_SSTa':
        if 'OS_SST' in os.environ:
            path = os.path.join(os.getenv('OS_SST'), 'MODIS_L2', 'Info')
            out_dict['path'] = path
            out_dict['preproc_file'] = os.path.join(path, 'PreProc', 'train_MODIS_2021_128x128.h5')
            out_dict['latents_file'] = os.path.join(path, 'latents', 'MODIS_2021',
                                'SimCLR_resnet50_lr_0.05_decay_0.0001_bsz_64_temp_0.07_trial_5_cosine_warm',
                                'train_MODIS_2021_128x128_latents.h5')
        out_dict['opts_file'] = 'opts_nenya_modis.json'
        out_dict['pca_file'] = 'pca_latents_MODIS_SSTa.npz'
        out_dict['dx'] = 1.1
        out_dict['year'] = '2021'
        out_dict['coverage'] = 'Global'
        out_dict['macro'] = '\\modis'
        out_dict['label'] = 'MODIS/SSTa'
    elif dataset == 'MODIS_SSTa_2km':
        if 'OS_SST' in os.environ:
            path = os.path.join(os.getenv('OS_SST'), 'MODIS_L2', 'Info')
            out_dict['path'] = path
            out_dict['preproc_file'] = os.path.join(path, 'PreProc', 'train_MODIS_2021_64x64.h5')
            out_dict['latents_file'] = os.path.join(path, 'latents', 'MODIS_2021_2km',
                                'SimCLR_resnet50_lr_0.05_decay_0.0001_bsz_64_temp_0.07_trial_5_cosine_warm',
                                'train_MODIS_2021_64x64_latents.h5')
        out_dict['opts_file'] = 'opts_nenya_modis_2km.json'
        out_dict['pca_file'] = 'pca_latents_MODIS_SSTa_2km.npz'
        out_dict['dx'] = 128.*1.1/64  # 2.2 km/pix (128 native pixels at 1.1 km resampled to 64)
        out_dict['year'] = '2021'
        out_dict['coverage'] = 'Global'
        out_dict['macro'] = '\\modistwo'
        out_dict['label'] = 'MODIS/SSTa-2km'
    elif dataset == 'VIIRS_SSTa':
        if 'OS_SST' in os.environ:
            path = os.path.join(os.getenv('OS_SST'), 'VIIRS', 'Info')
            out_dict['path'] = path
            out_dict['preproc_file'] = os.path.join(path, 'PreProc', 'train_VIIRS_N21_2024.h5')
            out_dict['latents_file'] = os.path.join(path, 'latents', 'VIIRS_SST',
                                'SimCLR_resnet50_lr_0.05_decay_0.0001_bsz_64_temp_0.07_trial_5_cosine_warm', 
                                'train_VIIRS_N21_2024_latents.h5')
        out_dict['opts_file'] = 'opts_nenya_viirs.json'
        out_dict['pca_file'] = 'pca_latents_VIIRS_SSTa.npz'
        out_dict['dx'] = 0.75
        out_dict['year'] = '2024'
        out_dict['coverage'] = 'Global'
        out_dict['macro'] = '\\viirs'
        out_dict['label'] = 'VIIRS/SSTa'
    elif dataset == 'VIIRS_SSTa_2km':
        if 'OS_SST' in os.environ:
            path = os.path.join(os.getenv('OS_SST'), 'VIIRS', 'Info')
            out_dict['path'] = path
            out_dict['preproc_file'] = os.path.join(path, 'PreProc', 'train_VIIRS_N21_2024_2km.h5')
            out_dict['latents_file'] = os.path.join(path, 'latents', 'VIIRS_2km',
                                'SimCLR_resnet50_lr_0.05_decay_0.0001_bsz_64_temp_0.07_trial_5_cosine_warm', 
                                'train_VIIRS_N21_2024_2km_latents.h5')
        out_dict['opts_file'] = 'opts_nenya_viirs_2km.json'
        out_dict['pca_file'] = 'pca_latents_VIIRS_SSTa_2km.npz'
        out_dict['dx'] = 192.*0.75/64  # 2.25 km/pix (192 native pixels at 0.75 km resampled to 64)
        out_dict['year'] = '2024'
        out_dict['coverage'] = 'Global'
        out_dict['macro'] = '\\viirstwo'
        out_dict['label'] = 'VIIRS/SSTa-2km'
    elif dataset == 'VIIRS_SSTa_sub':  # Native resolution but 64x64 pixels
        if 'OS_SST' in os.environ:
            path = os.path.join(os.getenv('OS_SST'), 'VIIRS', 'Info')
            out_dict['path'] = path
            out_dict['preproc_file'] = os.path.join(path, 'PreProc', 'train_VIIRS_N21_2024_sub.h5')
            out_dict['latents_file'] = os.path.join(path, 'latents', 'VIIRS_sub',
                                'SimCLR_resnet50_lr_0.05_decay_0.0001_bsz_64_temp_0.07_trial_5_cosine_warm', 
                                'train_VIIRS_N21_2024_sub_latents.h5')
        out_dict['opts_file'] = 'opts_nenya_viirs_sub.json'
        out_dict['pca_file'] = 'pca_latents_VIIRS_SSTa_sub.npz'
        out_dict['dx'] = 0.75
        out_dict['year'] = '2024'
        out_dict['coverage'] = 'Global'
        out_dict['macro'] = '\\viirssub'
        out_dict['label'] = 'VIIRS/SSTa-sub'
    elif dataset == 'orig_VIIRS_SST_2km':
            out_dict['pca_file'] = 'pca_latents_orig_VIIRS_SST_2km.npz'
    elif 'LLC_SSTa' in dataset:
        if 'OS_OGCM' in os.environ:
            path = os.path.join(os.getenv('OS_OGCM'), 'LLC', 'Info')
            out_dict['path'] = path
            if 'VIIRS' in dataset:
                out_dict['preproc_file'] = os.path.join(path, 'PreProc', 'LLC4320_SSTa_VIIRS.h5')
                out_dict['latents_file'] = os.path.join(path, 'latents', 'LLC_SSTa_VIIRS',
                                'SimCLR_resnet50_lr_0.05_decay_0.0001_bsz_64_temp_0.07_trial_5_cosine_warm',
                                'train_llc_nonoise_latents.h5')
            elif 'nonoise' in dataset:
                out_dict['preproc_file'] = os.path.join(path, 'PreProc', 'train_llc_nonoise.h5')
                out_dict['latents_file'] = os.path.join(path, 'latents', 'LLC_SST_nonoise',
                                'SimCLR_resnet50_lr_0.05_decay_0.0001_bsz_64_temp_0.07_trial_5_cosine_warm',
                                'train_llc_nonoise_latents.h5')
            else:
                out_dict['preproc_file'] = os.path.join(path, 'PreProc', 'train_llc_noise.h5')
                out_dict['latents_file'] = os.path.join(path, 'latents', 'LLC_SST_noise',
                                'SimCLR_resnet50_lr_0.05_decay_0.0001_bsz_64_temp_0.07_trial_5_cosine_warm',
                                'train_llc_noise_latents.h5')
        if 'VIIRS' in dataset:
            out_dict['opts_file'] = 'opts_nenya_llc.json'
            out_dict['macro'] = '\\llcsstv'
            out_dict['pca_file'] = 'pca_latents_LLC_SSTa_VIIRS.npz'
            out_dict['label'] = 'LLC/SSTa+VIIRS'
        elif 'nonoise' in dataset:
            out_dict['opts_file'] = 'opts_nenya_llc.json'
            out_dict['macro'] = '\\llcsst'
            out_dict['pca_file'] = 'pca_latents_LLC_SSTa_nonoise.npz'
            out_dict['label'] = 'LLC/SSTa'
        else:
            out_dict['opts_file'] = 'opts_nenya_llc_noise.json'
            out_dict['macro'] = '\\llcsstn'
            out_dict['pca_file'] = 'pca_latents_LLC_SSTa_noise.npz'
            out_dict['label'] = 'LLC/SSTa+noise'
        out_dict['dx'] = 144./64
        out_dict['year'] = '2011-2012'  # LLC4320 run period
        out_dict['coverage'] = '$\\pm 57^\\circ$'
    elif 'LLC_SSHa' in dataset:
        if 'OS_OGCM' in os.environ:
            path = os.path.join(os.getenv('OS_OGCM'), 'LLC', 'Info')
            out_dict['path'] = path
            out_dict['preproc_file'] = os.path.join(path, 'PreProc', 'LLC_random_SSHa.h5')
            out_dict['latents_file'] = os.path.join(path, 'latents', 'LLC_SSHa',
                                'SimCLR_resnet50_lr_0.05_decay_0.0001_bsz_64_temp_0.07_trial_5_cosine_warm',
                                'train_llc_nonoise_latents.h5')
        out_dict['opts_file'] = 'opts_nenya_llc_ssha.json'
        out_dict['pca_file'] = 'pca_latents_LLC_SSHa.npz'
        out_dict['dx'] = 144./64
        out_dict['year'] = '2011-2012'  # LLC4320 run period
        out_dict['coverage'] = '$-78^\\circ$ to $+57^\\circ$'
        out_dict['macro'] = '\\llcssh'
        out_dict['label'] = 'LLC/SSHa'
    elif dataset == 'SWOT_L3':
        if 'OS_SSH' in os.environ:
            path = os.path.join(os.getenv('OS_SSH'), 'SWOT_L3', 'Info')
            out_dict['path'] = path
            out_dict['preproc_file'] = os.path.join(path, 'PreProc', 'SWOT_L3_250m_preproc.h5')
            out_dict['latents_file'] = os.path.join(path, 'latents', 
                                'SimCLR_resnet50_lr_0.05_decay_0.0001_bsz_64_temp_0.07_trial_5_cosine_warm', 
                                'SWOT_L3_250m_latents.h5')
        out_dict['opts_file'] = 'opts_nenya_swot_l3.json'
        out_dict['pca_file'] = 'pca_latents_SWOT_L3.npz'
        out_dict['dx'] = 0.25
        out_dict['macro'] = '\\swotl3'
        out_dict['label'] = 'SWOT/SSHa_L3'
    elif dataset == 'SWOT_L2':
        if 'OS_SSH' in os.environ:
            path = os.path.join(os.getenv('OS_SSH'), 'SWOT_v2', 'Info')
            out_dict['path'] = path
            out_dict['preproc_file'] = os.path.join(path, 'PreProc', 'SWOT_L2_54km_preproc.h5')
            out_dict['latents_file'] = os.path.join(path, 'latents', 
                                'SimCLR_resnet50_lr_0.05_decay_0.0001_bsz_64_temp_0.07_trial_5_cosine_warm', 
                                'SWOT_L2_54km_latents.h5')
        out_dict['opts_file'] = 'opts_nenya_swot_l2.json'
        out_dict['pca_file'] = 'pca_latents_SWOT_L2.npz'
        out_dict['dx'] = 0.843  # km
        out_dict['year'] = '2023-2024'  # SWOT science orbit
        out_dict['coverage'] = '$\\pm 78^\\circ$'
        out_dict['macro'] = '\\swot'
        out_dict['label'] = 'SWOT/SSHa-L2'
    else:
        raise ValueError(f"Dataset {dataset} not supported for Nenya.")

    # Add pca/ to pca_file
    out_dict['pca_file'] = os.path.join('pca', out_dict['pca_file'])
    out_dict['opts_file'] = os.path.join('opts', out_dict['opts_file'])
    if 'opts_file_extended' in out_dict.keys():
        out_dict['opts_file_extended'] = os.path.join('opts', out_dict['opts_file_extended'])

    # Image PCA file
    out_dict['pca_imgfile'] = out_dict['pca_file'].replace('latents', 'preproc')

    # Auto-generate Pk
    out_dict['Pk_file'] = os.path.join('Pk', f'Pk_{dataset}.npz')
    out_dict['Pk_plot'] = os.path.join('Pk', f'Pk_{dataset}.png')

    # Eigen
    if out_dict['path'] is not None:
        out_dict['eigen_file'] = os.path.join(out_dict['path'], 'eigen',
            f'{dataset}_eigenimages.npz')

    # dx
    if 'dx' not in out_dict.keys():
        out_dict['dx'] = 2.

    # Return
    return out_dict