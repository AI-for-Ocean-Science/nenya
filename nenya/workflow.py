""" utilities for nenya analysis
"""
import os
import numpy as np
import h5py
from sklearn.metrics.pairwise import cosine_similarity


from wrangler.plotting import cutout

from nenya.train import main as train_main
from nenya import latents_extraction
from nenya import analysis
from nenya import pca
from nenya import plotting
from nenya import params 
from nenya import io as nenya_io

from IPython import embed

def evaluate(opts_file:str, preproc_file:str, latents_file:str=None,
             local_model_path:str=None, use_gpu:bool=False, clobber:bool=False,
             base_model_name:str='last.pth', debug:bool=False):
    """
    Evaluate the latents extraction process using the specified options and preprocessing files.

    Args:
        opts_file (str): Path to the options file containing configuration settings.
        preproc_file (str): Path to the preprocessing file required for evaluation.
        local_model_path (str, optional): Path to the local model file. Defaults to None.
        latents_file (str, optional): Path to the file where latents will be stored. Defaults to None.
        clobber (bool, optional): If True, overwrite existing latents file. Defaults to False.
        use_gpu (bool, optional): Flag indicating whether to use GPU for evaluation. Defaults to False.

    Returns:
        None: This function does not return a value. It performs evaluation and may modify files or output logs.
    """
    latents_extraction.evaluate(opts_file,
                preproc_file,
                local_model_path=local_model_path,
                latents_file=latents_file,
                use_gpu=use_gpu,
                debug=debug, clobber=clobber,
                base_model_name=base_model_name)

def chk_latents(dataset:str, latents_file:str, preproc_file:str,
                query_idx:int, partition:str='train', top_N:int=5):

    # Grab the latents
    with h5py.File(latents_file, 'r') as f:
        latents = f[partition][:]
        print(f"Latents shape: {latents.shape}")


    # Closest
    closest_idx, similarities = analysis.find_closest_latents(latents, query_idx)
    indices = [query_idx]+closest_idx[:top_N].tolist()

    # Grab the images
    with h5py.File(preproc_file, 'r') as f:
        images = [f[partition][idx] for idx in [query_idx]+closest_idx[:top_N].tolist()]
        print(f"Grabbed {len(images)} images for plotting including the query.")

    # Plot
    #embed(header='53 of nenya')
    plotting.closest_latents(images, indices, similarities,
                          output_png=f'nenya_{dataset}_{partition}_chk_latents_{query_idx}.png')


def train(opts_file:str, load_epoch:int=None, debug:bool=False):
    """
    Train the model using the specified options file.

    Args:
        opts_file (str): Path to the options file containing training configurations.
        load_epoch (int, optional): Epoch number to load for resuming training. Defaults to None.
        debug (bool, optional): Flag to enable debug mode. Defaults to False.

    Returns:
        None
    """
    # Train the model
    train_main(opts_file, debug=debug, load_epoch=load_epoch)

def find_eigenmatches(pca_file:str, latents_file:str, nmodes:int,
                     partition:str='train', from_mode:int=0):
    """
    Find the closest latent vectors to the eigenmodes of a PCA model.

    This function loads a PCA model and a set of latent vectors, and for a specified
    number of modes, it computes the most similar latent vector to each eigenmode
    using cosine similarity.

    Args:
        pca_file (str): Path to the PCA model file (in `.npz` format).
        latents_file (str): Path to the file containing latent vectors (in `.h5` format).
        nmodes (int): Number of eigenmodes to process.
        partition (str, optional): Dataset partition to use from the latent file 
            (e.g., 'train', 'test'). Defaults to 'train'.
        from_mode (int, optional): Starting mode index for processing eigenmodes. 
            Defaults to 0.

    Returns:
        Tuple[List[int], List[float]]: A tuple containing:
            - indices (List[int]): Indices of the closest latent vectors for each eigenmode.
            - sims (List[float]): Cosine similarity scores of the closest latent vectors.
    """

    # Load the PCA model
    d = np.load(pca_file)

    # Grab the latents
    with h5py.File(latents_file, 'r') as f:
        latents = f[partition][:]

    indices, sims = [], []
    for tt in range(nmodes):
        # Increment
        ss = from_mode + tt
        #
        eigenmode = d['M'][ss, :]
        # Closest
        query_vector = eigenmode.reshape(1, -1)
        similarities = cosine_similarity(query_vector, latents)[0]
        # Sort
        sorted_indices = np.argsort(-similarities)
        similarities = similarities[sorted_indices]
        # Save
        indices.append(sorted_indices[0])
        sims.append(similarities[0])

    # Return
    return indices, sims


def find_eigenmodes(opt_path:str, pca_file:str, image_shape:tuple,
                    output_file:str, Neigenmodes:int=10, use_gpu:bool=False,
                    clamp_value:float=None, local_model_path:str=None,
                    base_model_name:str='last.pth', num_iterations:int=1000,
                    tv_weight:float=0.0, show:bool=False, debug:bool=False,
                    use_eigenmatch_start:bool=False, latents_file:str=None,
                    preproc_file:str=None, partition:str='train'):
    """
    Generate and save eigenmodes using a pre-trained model and PCA data.

    Args:
        opt_path (str): Path to the configuration file for the model.
        pca_file (str): Path to the PCA file containing eigenmodes.
        image_shape (tuple): Shape of the output images (height, width).
        output_file (str): Path to save the generated eigenmodes and similarities.
        Neigenmodes (int, optional): Number of eigenmodes to generate. Defaults to 10.
        use_gpu (bool, optional): Whether to use GPU for computation. Defaults to False.
        clamp_value (float, optional): Value to clamp the generated images. Defaults to None.
        local_model_path (str, optional): Path to the local model directory. Defaults to None.
        base_model_name (str, optional): Name of the base model file. Defaults to 'last.pth'.
        num_iterations (int, optional): Number of iterations for eigenmode generation. Defaults to 1000.
        tv_weight (float, optional): Total variation regularization weight. Defaults to 0.0.
        show (bool, optional): Whether to display the generated images. Defaults to False.
        debug (bool, optional): Whether to enable debug mode. Defaults to False.
        use_eigenmatch_start (bool, optional): If True, use the closest eigenmatch image
            as the starting point for optimization instead of random noise. Requires
            latents_file and preproc_file to be provided. Defaults to False.
        latents_file (str, optional): Path to the latents file. Required if
            use_eigenmatch_start is True. Defaults to None.
        preproc_file (str, optional): Path to the preprocessing file containing images.
            Required if use_eigenmatch_start is True. Defaults to None.
        partition (str, optional): Dataset partition to use for eigenmatches
            (e.g., 'train', 'valid'). Defaults to 'train'.

    Returns:
        None: The function saves the generated eigenmodes and similarities to the specified output file.
    Notes:
        - The function loads a pre-trained model and PCA data to generate eigenmodes.
        - Eigenmodes are generated with optional total variation regularization and clamping.
        - If `show` is True, the generated images are displayed during the process.
        - If `debug` is True, debugging information is displayed, and the process is interactive.
        - The generated eigenmodes and their similarities are saved in `.npz` format.
        - If `use_eigenmatch_start` is True, the optimization starts from the image whose
          latent vector is closest to each eigenmode, which may improve convergence.
    """
    # Load model
    opt = params.Params(opt_path)
    params.option_preprocess(opt)

    if use_gpu is None:
        use_gpu = torch.cuda.is_available()

    # Model name and opt
    opt, model_name = nenya_io.load_model_name(
        opt_path, local_model_path=local_model_path,
        base_model_name=base_model_name)

    # Load model
    model, _ = nenya_io.load_model(model_name, opt, use_gpu,
                               remove_module=True,
                               weights_only=False)

    # Load the PCA model
    d = np.load(pca_file)

    # Find eigenmatches if requested
    start_images = None
    eigenmatch_indices = None
    eigenmatch_sims = None
    if use_eigenmatch_start:
        if latents_file is None or preproc_file is None:
            raise ValueError("latents_file and preproc_file must be provided when use_eigenmatch_start=True")
        # Find the closest latent vectors to each eigenmode
        modes = list(range(Neigenmodes))
        eigenmatch_indices, eigenmatch_sims = pca.find_eigenmatches(
            pca_file, latents_file, modes, nimages=1, partition=partition)
        # eigenmatch_indices shape is (1, Neigenmodes), get the first match for each mode
        eigenmatch_indices = eigenmatch_indices[0, :]  # shape: (Neigenmodes,)
        eigenmatch_sims = eigenmatch_sims[0, :]  # shape: (Neigenmodes,)
        # Load the corresponding images
        with h5py.File(preproc_file, 'r') as f:
            start_images = [f[partition][idx] for idx in eigenmatch_indices]
        print(f"Using eigenmatch starting images with cosine similarities: {eigenmatch_sims}")

    # Run it
    eigen_images = []
    similarities = []
    for ss in range(Neigenmodes):
        print(f"Working on eigenmode: {ss}")
        # Grab the eigenmode
        eigenmode = d['M'][ss, :]
        # Get starting image if available
        start_image = start_images[ss] if start_images is not None else None
        # Generate the eigenmode with regularization
        img, cosi = pca.generate_eigenmode_with_regularization(model, eigenmode, image_shape,
            tv_weight=tv_weight, clamp_value=clamp_value, num_iterations=num_iterations,
            start_image=start_image)
        eigen_images.append(img)
        similarities.append(cosi)
        # Show the image?
        if show:
            ax = cutout.show_image(img[0], show=True)
            ax.set_title=f'Eigenmode {ss+1}'
        if debug:
            embed(header=f"Generated eigenmode {ss+1}/{Neigenmodes} with shape {img.shape}")

    # Save the eigenmodes
    if not debug:
        print(f"Saving eigenmodes to: {output_file}")
        np.savez(output_file, 
                 eigen_images=np.array(eigen_images), 
                 similarities=np.array(similarities), 
                 eigenmodes=d['M'][:Neigenmodes, :],
                 modes=np.arange(Neigenmodes))


def return_task():
    import sys

    if len(sys.argv) == 1:
        print("Usage: python nenya_ImageNet.py <task>")
        print("Tasks: train, evaluate, chk_latents")
    elif len(sys.argv) > 2:
        print("Too many arguments. Only one task is allowed.")
        sys.exit(1)
    elif len(sys.argv) == 2:
        task = sys.argv[1].lower()
        if task not in ['train', 'evaluate', 'chk_latents', 'eigenimages',
            'train_extended', 'evaluate_extended']:
            print(f"Unknown task: {task}. Use 'train', 'evaluate', 'eigenimages', or 'chk_latents'.")
            sys.exit(1)
        print(f"Running task: {task}")
        task = sys.argv[1].lower()

    return task