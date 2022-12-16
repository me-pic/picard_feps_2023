import os
import pickle
import numpy as np
from numpy.linalg import norm
from nilearn.image import math_img
from nilearn import datasets
from nilearn.maskers import NiftiMasker
from scipy.stats import permutation_test
import pandas as pd
import pickle
import nibabel as nib
from argparse import ArgumentParser

def cosine_similarity(A,B):
    """
    Compute the cosine similarity between two vectors
    """
    return (np.dot(A,B.T)/(norm(A)*norm(B)))


def similarity_across_networks(path_signature, path_feps, path_mask, labels=None, metric=None, permutation=False, n_permutation=10000):
    """
    Compute spatial similarity metrics between two signatures defined in path_signature and path_feps
    
    Parameters
    ----------
    path_signature: string
        signature path (path to nii file) on which to calculate the similarity with the signature defined in the path_feps
    path_feps: string
        feps path (path to nii file) on which to calculate the similarity with the signature defined in the path_signature
    path_mask: string
        path to the masks used to extract the signature and the feps signal
    labels: list (None)
        list containing the name of the networks/regions contained in the mask from path_mask
    metric: string (None)
        metric to use to compute the spatial similarity between the signature and the feps. If the metric is not defined, 
        both the cosine similarity and the pearson product-moment correlation will be computed
        'cosine': compute the cosine similarity between the signature and the feps
        'pearson': compute the pearson product-moment correlation between the signature and the feps
    permutation: bool (False)
        if True, compute permutations in order to assess the significativity of the similarity metric; otherwise no permutations are computed
    n_permutation: int (10000)
        number of iterations to do if the permutation tests are conducted
    
    Returns
    -------
    similarity: list 
        list containing the similarity metric(s) for each region/network contained in the mask
    res_dict: dictionnary 
        dictionnary containing the results of the permutation. If permutation == False, return an empty dictionnary
    
    """
    similarity = []
    res = []
    statistic = []
    pval = []
    null_dist = []
    res_dict = {}

    for idx, label in enumerate(labels):
        #Define the masker based on mask in path_mask
        masker = NiftiMasker(path_mask[idx])
        feps = masker.fit_transform(path_feps)
        masker = NiftiMasker(path_mask[idx])
        signature = masker.fit_transform(path_signature)
        if metric=='cosine':
            #compute cosine similarity
            similarity.append((label, cosine_similarity(feps, signature)[0][0]))
        elif metric=='pearson':
            #Compute pearson product-moment correlation
            similarity.append(np.corrcoef(feps, signature)[0][1])
        else:
            #Compute both cosine similarity and pearson product-moment correlation
            cos_sim.append((label, cosine_similarity(feps, signature), np.corrcoef(feps, signature)[0][1]))

        if permutation:
            res_ = permutation_test((signature, feps), 
                                    cosine_similarity, 
                                    permutation_type='independent', 
                                    n_resamples=n_permutation, 
                                    alternative='two-sided', 
                                    random_state=42, axis=1
                                   )
            res.append(res_)
    
            for network in res:
                statistic.append(network.statistic)
                pval.append(network.pvalue)
                null_dist.append(network.null_distribution)

            res_dict = {'statistic': statistic,'pval': pval,'null_dist': null_dist}
    
    return similarity, res_dict
    


def compute_cosine_cortical_networks(path_signature, path_feps, path_output, permutation=False, n_permutation=10000):

    atlas_yeo_2011 = datasets.fetch_atlas_yeo_2011()
    atlas_yeo = atlas_yeo_2011.thick_7
    labels_cortical  = ['Visual', 'Somatosensory', 'Dorsal Attention', 'Ventral Attention', 'Limbic', 'Frontoparietal', 'Default']

    separated_regions = []
    for i in range(1,8):
        separated_regions.append(math_img(f"img == {i}", img=atlas_yeo))

    similarity_cortical, res_cortical = similarity_across_networks(path_signature=path_signature, path_feps=path_feps, path_mask=separated_regions, labels=labels_cortical, cosine=True, permutation=permutation, n_permutation=n_permutation)
    with open(os.path.join(path_output, f"cosine_similarity_cortical_networks_feps_{path_signature.split('/')[-1].split('.')[0]}.pickle"), 'wb') as output_file:
        pickle.dump([similarity_cortical, res_cortical], output_file)
    output_file.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--path_signature', type=str, default=None)
    parser.add_argument('--path_feps', type=str, default=None)
    parser.add_argument('--path_output', type=str, default=None)
    args = parser.parse_args()

    compute_cosine_cortical_networks(args.path_signature, args.path_feps, args.path_output)
