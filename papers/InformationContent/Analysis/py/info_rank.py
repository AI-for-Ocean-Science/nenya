"""Compute participation ratio and RankMe for latent and image PCA."""

import os
import numpy as np
import info_defs


def participation_ratio(ev: np.ndarray) -> float:
    """Participation ratio from explained variance ratios.

    PR = (sum ev_i)^2 / sum(ev_i^2) = 1 / sum(ev_i^2)
    when ev sums to 1.

    Parameters
    ----------
    ev : np.ndarray
        Explained variance ratios (must sum to ~1).

    Returns
    -------
    float
        Participation ratio (1 to N_f).
    """
    return 1.0 / np.sum(ev**2)


def rankme(ev: np.ndarray, eps: float = 1e-12) -> float:
    """RankMe metric from explained variance ratios.

    Converts eigenvalues (proportional to sigma^2) to singular values
    (sigma), normalizes, then computes exp(Shannon entropy).

    Parameters
    ----------
    ev : np.ndarray
        Explained variance ratios (must sum to ~1).
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    float
        RankMe (1 to N_f).
    """
    sigma = np.sqrt(np.maximum(ev, 0.0))
    p = sigma / (sigma.sum() + eps)
    # Remove zeros for log stability
    mask = p > eps
    H = -np.sum(p[mask] * np.log(p[mask]))
    return float(np.exp(H))


def rankme_from_sv(sv: np.ndarray, eps: float = 1e-12) -> float:
    """RankMe directly from singular values.

    Parameters
    ----------
    sv : np.ndarray
        Singular values.
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    float
        RankMe (1 to len(sv)).
    """
    sv = np.maximum(sv, 0.0)
    p = sv / (sv.sum() + eps)
    mask = p > eps
    H = -np.sum(p[mask] * np.log(p[mask]))
    return float(np.exp(H))


def compute_metrics(pca_file: str) -> dict:
    """Compute PR and RankMe from a PCA .npz file.

    Handles two formats:
    - Latent PCA: 'explained_variance' sums to 1 (variance ratios)
    - Image PCA: 'explained_variance_ratio' for ratios,
      'explained_variance' for raw eigenvalues,
      'singular_values' for direct SVD output

    Parameters
    ----------
    pca_file : str
        Path to PCA output file.

    Returns
    -------
    dict with keys 'PR', 'RankMe', 'N_f', 'N_99'
    """
    d = np.load(pca_file)

    # Get explained variance ratios (summing to ~1)
    if 'explained_variance_ratio' in d:
        ev = d['explained_variance_ratio']
    else:
        ev = d['explained_variance']
        # Normalize if not already
        if ev.sum() > 1.01:
            ev = ev / ev.sum()

    # N_99: components for 99% variance
    cumvar = np.cumsum(ev)
    n99 = int(np.searchsorted(cumvar, 0.99) + 1)

    # For RankMe, use singular values directly if available
    if 'singular_values' in d:
        sv = d['singular_values']
        rm = rankme_from_sv(sv)
    else:
        rm = rankme(ev)

    return {
        'PR': participation_ratio(ev),
        'RankMe': rm,
        'N_f': len(ev),
        'N_99': n99,
    }


def compute_all(output_file: str = None):
    """Compute metrics for all datasets, both latent and image PCA.

    Parameters
    ----------
    output_file : str, optional
        If provided, save results to this .npz file.

    Returns
    -------
    dict
        Nested dict: results[dataset]['latent'] and results[dataset]['image']
    """
    results = {}

    for dataset in info_defs.all_datasets:
        pdict = info_defs.grab_paths(dataset)
        results[dataset] = {}

        # Latent PCA
        latent_file = os.path.join('pca', pdict['pca_file'].split('/')[-1])
        if os.path.exists(latent_file):
            results[dataset]['latent'] = compute_metrics(latent_file)
        else:
            print(f"  Missing latent PCA: {latent_file}")

        # Image PCA
        image_file = os.path.join('pca', pdict['pca_imgfile'].split('/')[-1])
        if os.path.exists(image_file):
            results[dataset]['image'] = compute_metrics(image_file)
        else:
            print(f"  Missing image PCA: {image_file}")

    # Print table
    print_table(results)

    # Save
    if output_file is not None:
        save_results(results, output_file)

    return results


def print_table(results: dict):
    """Print a formatted comparison table."""
    header = (f"{'Dataset':<22s} "
              f"{'N_f':>4s} {'N_99':>5s} "
              f"{'PR_lat':>7s} {'RM_lat':>7s} "
              f"{'PR_img':>7s} {'RM_img':>7s}")
    print(header)
    print("-" * len(header))

    for dataset in info_defs.all_datasets:
        if dataset not in results:
            continue
        r = results[dataset]
        lat = r.get('latent', {})
        img = r.get('image', {})

        nf = lat.get('N_f', 0)
        n99 = lat.get('N_99', 0)
        pr_l = lat.get('PR', 0)
        rm_l = lat.get('RankMe', 0)
        pr_i = img.get('PR', 0)
        rm_i = img.get('RankMe', 0)

        print(f"{dataset:<22s} "
              f"{nf:>4d} {n99:>5d} "
              f"{pr_l:>7.1f} {rm_l:>7.1f} "
              f"{pr_i:>7.1f} {rm_i:>7.1f}")


def save_results(results: dict, output_file: str):
    """Save results to .npz file."""
    datasets = []
    nf_arr, n99_arr = [], []
    pr_lat, rm_lat = [], []
    pr_img, rm_img = [], []

    for dataset in info_defs.all_datasets:
        if dataset not in results:
            continue
        r = results[dataset]
        lat = r.get('latent', {})
        img = r.get('image', {})

        datasets.append(dataset)
        nf_arr.append(lat.get('N_f', 0))
        n99_arr.append(lat.get('N_99', 0))
        pr_lat.append(lat.get('PR', 0.0))
        rm_lat.append(lat.get('RankMe', 0.0))
        pr_img.append(img.get('PR', 0.0))
        rm_img.append(img.get('RankMe', 0.0))

    np.savez(output_file,
             datasets=np.array(datasets),
             N_f=np.array(nf_arr),
             N_99=np.array(n99_arr),
             PR_latent=np.array(pr_lat),
             RankMe_latent=np.array(rm_lat),
             PR_image=np.array(pr_img),
             RankMe_image=np.array(rm_img))
    print(f"\nSaved to {output_file}")


if __name__ == '__main__':
    compute_all(output_file='pca/rank_metrics.npz')
