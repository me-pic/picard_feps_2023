import os
import pickle
import numpy as np
from numpy.linalg import norm
from nilearn.image import math_img
from nilearn import datasets
from nilearn.maskers import NiftiMasker
from scipy.stats import permutation_test
import pickle
from argparse import ArgumentParser

def cosine_similarity(A,B):
    """
    Compute the cosine similarity between two vectors
    """
    return (np.dot(A,B.T)/(norm(A)*norm(B)))


def similarity(path_signature, path_feps, gr_mask, metric=None):
    """
    Compute spatial similarity metrics between two signatures defined in path_signature and path_feps
    
    Parameters
    ----------
    path_signature: string
        signature path (path to nii file) on which to calculate the similarity with the signature defined in the path_feps
    path_feps: string
        feps path (path to nii file) on which to calculate the similarity with the signature defined in the path_signature
    metric: string (None)
        metric to use to compute the spatial similarity between the signature and the feps. If the metric is not defined, 
        both the cosine similarity and the pearson product-moment correlation will be computed
        'cosine': compute the cosine similarity between the signature and the feps
        'pearson': compute the pearson product-moment correlation between the signature and the feps
    
    Returns
    -------
    similarity: list
        list containing the similarity metric(s)
    """
    similarity=[]

    #Extract the signature and the feps signal
    masker = NiftiMasker(gr_mask)
    feps = masker.fit_transform(path_feps)
    masker = NiftiMasker(gr_mask)
    signature = masker.fit_transform(path_signature)
    
    if metric=='cosine':
        #compute cosine similarity
        similarity = cosine_similarity(feps, signature)[0][0]
    elif metric=='pearson':
        #Compute pearson product-moment correlation
        similarity = np.corrcoef(feps, signature)[0][1]
    else:
        #Compute both cosine similarity and pearson product-moment correlation
        similarity = (cosine_similarity(feps, signature), np.corrcoef(feps, signature)[0][1])
    
    return similarity


def similarity_across_networks(path_signature, path_feps, path_mask, labels=None, metric=None, permutation=False, n_permutation=10000):
    """
    Compute spatial similarity metrics between two signatures defined in path_signature and path_feps across different networks/regions in path_mask
    
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
    perm_out: list
        list containing the results of the permutation for each network/region. If permutation == False, return an empty list
    
    """
    similarity = []
    perm_out = []

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
            similarity.append((label, np.corrcoef(feps, signature)[0][1]))
        else:
            #Compute both cosine similarity and pearson product-moment correlation
            similarity.append((label, cosine_similarity(feps, signature), np.corrcoef(feps, signature)[0][1]))

        if permutation:
            res_ = permutation_test((signature, feps), 
                                    cosine_similarity, 
                                    permutation_type='independent', 
                                    n_resamples=n_permutation, 
                                    alternative='two-sided', 
                                    random_state=42, axis=1
                                   )
            perm_out.append({'statistic': res_.statistic,'pval': res_.pvalue,'null_dist': res_.null_distribution})
    
    return similarity, perm_out



#Arguments to pass to the script
parser = ArgumentParser()
parser.add_argument('--path_signature', type=str, default=None)
parser.add_argument('--path_feps', type=str, default=None)
parser.add_argument('--path_output', type=str, default=None)
args = parser.parse_args()

#./similarity.py --path_signature '/Users/mepicard/Documents/master_analysis/picard_feps_2022_v1/data/brain_signatures/nonnoc_v11_4_137subjmap_weighted_mean.nii' --path_feps '/Users/mepicard/Documents/master_analysis/picard_feps_2022_v1/data/brain_signatures/z_bootstrap_lasso_standard_True_sample_5000_None.nii' --path_output '/Users/mepicard/Documents/master_analysis/picard_feps_2022_outputs'

#Define the parameters
metric=None #'cosine'
permutation=True 
n_permutation=10000
gr_mask='../masks/masker.nii.gz'

#Compute the spatial similarity between signatures defined in path_signature and path_feps
similarity_signatures = similarity(args.path_signature, args.path_feps, gr_mask, metric=None)
#Save the output
with open(os.path.join(args.path_output, f"spatial_similarity_{args.path_feps.split('/')[-1].split('.')[0]}_{args.path_signature.split('/')[-1].split('.')[0]}.pickle"), 'wb') as output_file:
    pickle.dump(similarity_signatures, output_file)
output_file.close()

#Load the altas
atlas_yeo_2011 = datasets.fetch_atlas_yeo_2011()
atlas_yeo = atlas_yeo_2011.thick_7
labels_cortical  = ['Visual', 'Somatosensory', 'Dorsal Attention', 'Ventral Attention', 'Limbic', 'Frontoparietal', 'Default']

#Separated the atlas networks according to their index
separated_regions = []
for i in range(1,8):
    separated_regions.append(math_img(f"img == {i}", img=atlas_yeo))

#Compute the spatial similarity across the cortical networks
similarity_cortical, perm_cortical = similarity_across_networks(path_signature=args.path_signature, path_feps=args.path_feps, path_mask=separated_regions, labels=labels_cortical, metric=metric, permutation=permutation, n_permutation=n_permutation)
#Save the output
with open(os.path.join(args.path_output, f"spatial_similarity_cortical_networks_{args.path_feps.split('/')[-1].split('.')[0]}_{args.path_signature.split('/')[-1].split('.')[0]}.pickle"), 'wb') as output_file:
    pickle.dump([similarity_cortical, perm_cortical], output_file)
output_file.close()