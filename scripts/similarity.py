import os
import pickle
import numpy as np
from numpy.linalg import norm
from neuromaps import nulls
from neuromaps.stats import compare_images
from nilearn.image import math_img
from nilearn.masking import unmask
from nilearn import datasets
from nilearn.maskers import NiftiMasker
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
    similarity: int or tuple
        similarity metric. If cosine similarity and person correlation are computed, return a tuple
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


def similarity_across_networks(path_signature, path_feps, path_mask, labels=None, metric=None):
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
    
    Returns
    -------
    similarity: list 
        list containing the similarity metric(s) for each region/network contained in the mask
    perm_out: list
        list containing the results of the permutation for each network/region. If permutation == False, return an empty list
    """
    similarity = []

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
    
    return similarity


def similarity_nulls(path_signature, apply_gm = True, n_perm=1000):
    """
    Parameters
    ----------
    path_signature: string
        signature path (path to nii file) on which to calculate the null distribution
    apply_gm: bool
        if true, apply gray matter mask on 'path_signature'. Else data in 
        'path_signature' will be used to compute null distribution
    n_perm: int
        number of permutation to compute
    
    Returns
    -------
    nulls_sign: array
        generated null distribution
    
    References
    ----------
    Burt, J. B., Helmer, M., Shinn, M., Anticevic, A., & Murray, J. D. (2020). 
        Generative modeling of brain maps with spatial autocorrelation. 
        NeuroImage, 220, 117038. https://doi.org/10.1016/j.neuroimage.2020.117038
    
    See also
    --------
    https://netneurolab.github.io/neuromaps/user_guide/nulls.html
    """
    #Apply gray matter mask on signature according to Neuromaps doc
    if apply_gm:
        mask_gm = NiftiMasker(mask_img=datasets.load_mni152_gm_mask())
        masked_data = mask_gm.fit_transform(os.path.join(path_signature))
        data = unmask(masked_data, mask_gm.mask_img_)
    else:
        data = path_signature
    
    #Compute null models
    nulls_sign = nulls.burt2020(data, 
                                atlas='MNI152', 
                                density='3mm', 
                                n_perm=n_perm, 
                                n_proc=-1, 
                                seed=1234)

    return nulls_sign

def similarity_nulls_significance(nulls, x, y, apply_gm=True, metric=None):
    """
    Parameters
    ----------
    nulls: array
        generated null distribution
    x: string
        signature path (path to nii file) on which the null distribution was computed
    y: string
        signature path (path to nii file) on which to compute the similarity with x
    metric: string
        type of similarity metric to compare x and y
    
    Returns
    -------
    similarity_value:
        similarity of the comparison between x and y 
    pval:
        pvalue of 'similarity_value'
    distr: array
        null distribution of similarity metrics

    See also
    --------
    https://netneurolab.github.io/neuromaps/generated/neuromaps.stats.compare_images.html#neuromaps.stats.compare_images
    """
    if apply_gm:
        mask_gm = NiftiMasker(mask_img=datasets.load_mni152_gm_mask())
        masked_data = mask_gm.fit_transform(os.path.join(y))
        y = unmask(masked_data, mask_gm.mask_img_)

    if metric is None:
        similarity_value, pval, distr = compare_images(x, y, metric=cosine_similarity, nulls=nulls, return_nulls=True)
    else:
        similarity_value, pval, distr = compare_images(x, y, metric=metric, nulls=nulls, return_nulls=True)

    return similarity_value, pval, distr

if __name__ == "__main__":
    #Arguments to pass to the script
    parser = ArgumentParser()
    parser.add_argument('--path_signature', type=str, default=None)
    parser.add_argument('--path_feps', type=str, default=None)
    parser.add_argument('--path_output', type=str, default=None)
    args = parser.parse_args()

    #Define the parameters
    metric='cosine'
    permutation=False
    n_permutation=5000
    gr_mask='../masks/masker.nii.gz'

    #Compute the spatial similarity between signatures defined in path_signature and path_feps
    nulls_distr = similarity_nulls(args.path_feps)
    similarity_value, pval, distr = similarity_nulls_significance(nulls_distr, args.path_feps, args.path_signature)
    #Save the output
    with open(os.path.join(args.path_output, f"spatial_similarity_{args.path_feps.split('/')[-1].split('.')[0]}_{args.path_signature.split('/')[-1].split('.')[0]}.pickle"), 'wb') as output_file:
        pickle.dump([similarity_value, pval, distr], output_file)
    output_file.close()

    #Load the altas
    atlas_yeo_2011 = datasets.fetch_atlas_yeo_2011()
    atlas_yeo = atlas_yeo_2011.thick_7
    labels_cortical  = ['Visual', 'Somatosensory', 'Dorsal Attention', 'Ventral Attention', 'Limbic', 'Frontoparietal', 'Default']

    #Separated the atlas networks according to their index
    separated_regions = []
    for i in range(1,len(labels_cortical)+1):
        separated_regions.append(math_img(f"img == {i}", img=atlas_yeo))

    #Compute the spatial similarity across the cortical networks
    similarity_cortical = similarity_across_networks(path_signature=args.path_signature, path_feps=args.path_feps, path_mask=separated_regions, labels=labels_cortical, metric=metric, permutation=permutation, n_permutation=n_permutation)
    #Save the output
    with open(os.path.join(args.path_output, f"spatial_similarity_cortical_networks_{args.path_feps.split('/')[-1].split('.')[0]}_{args.path_signature.split('/')[-1].split('.')[0]}.pickle"), 'wb') as output_file:
        pickle.dump(similarity_cortical, output_file)
    output_file.close()
