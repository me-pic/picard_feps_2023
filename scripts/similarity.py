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
    return (np.dot(A,B.T)/(norm(A)*norm(B)))


def similarity_across_networks(path_signature, path_feps, path_mask=None, labels=None, cosine=True, permutation=False, n_permutation=5000):
    similarity = []
    res = []
    statistic = []
    pval = []
    null_dist = []

    for idx, label in enumerate(labels):
        #Define the masker based on mask in path_mask
        masker = NiftiMasker(path_mask[idx])
        feps = masker.fit_transform(path_feps)
        masker = NiftiMasker(path_mask[idx])
        signature = masker.fit_transform(path_signature)
        if cosine:
            #compute cosine similarity
            similarity.append((label, cosine_similarity(feps, signature)[0][0]))
        else:
            #Compute Pearson product-moment correlation
            similarity.append(np.corrcoef(feps, signature)[0][1])

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

    compute_cosine_cortical_networks(args.path_signature, args.path_feps, args.path_output, permutation=False, n_permutation=5000)