#!/usr/bin/env python

# ./main.py --path "data_FEPS.json" --seed 42

import json
import pickle
import prepping_data
import building_model
import numpy as np
import nibabel as nib
from nilearn.masking import unmask
from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_argument("--path", type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, choices=["whole-brain","M1","without M1"], default="whole-brain")
    args = parser.parse_args()

    #Loading the dataset
    data = open(args.path, "r")
    data_feps = json.loads(data.read())

    #Predicted variable
    y = np.array(data_feps["target"])
    
    #Group variable: how the data is grouped (by subjects)
    gr = np.array(data_feps["group"])

    #Convert fmri files to Nifti-like objects
    array_feps = prepping_data.hdr_to_Nifti(data_feps["data"])
    #Extract signal from gray matter
    if args.model == "whole-brain" :
        for i, element in enumerate(model_voxel):
            (masker.inverse_transform(element)).to_filename(f"coefs_whole_brain_{i}.nii.gz")
        
        model_to_averaged = model_voxel.copy()
        model_averaged = sum(model_to_averaged)/len(model_to_averaged)
        (masker.inverse_transform(model_averaged)).to_filename("coefs_whole_brain_ave.nii.gz")

    else :
        array_model_voxel = []
        if args.model == "M1" :
            unmask_model = unmask(model_voxel, mask_M1)
        if args.model == "without M1": 
            unmask_model = unmask(model_voxel, mask_NoM1)

        for element in unmask_model:
            array_model_voxel.append(np.array(element.dataobj))

        model_ave = sum(array_model_voxel)/len(array_model_voxel)
        model_to_nifti = nib.nifti1.Nifti1Image(model_ave, affine = array_feps[0].affine)
        model_to_nifti.to_filename(f"coefs_{args.model}_ave.nii.gz")
    
    
    for i in range(len(X_train)):
        filename = f"train_test_{i}.npz"
        np.savez(filename, X_train=X_train[i],y_train=y_train[i],X_test=X_test[i],y_test=y_test[i],y_pred=y_pred[i])

    #Saving the model
    filename_model = f"lasso_models_{args.model}.pickle" 
    pickle_out = open(filename_model,"wb")
    pickle.dump(model, pickle_out)
    pickle_out.close()

    #Compute permutation tests
    score, perm_scores, pvalue = building_model.compute_permutation(X, y, gr, random_sedd=args.seed)
    perm_dict = {'score': score, 'perm_scores': perm_scores.tolist(), 'pvalue': pvalue}
    filename_perm = f"permutation_output_{args.model}_{args.seed}.json"
    with open(filename_perm, 'w') as fp:
        json.dump(perm_dict, fp)

    #compute bootstrap tests
    resampling_coef = building_model.boostrap_test(X, y, gr, random_seed=args.seed)
    filename_bootstrap = f"bootstrap_models_{args.model}_{args.seed}.pickle"
    pickle_out = open(filename_bootstrap,"wb")
    pickle.dump(resampling_coef, pickle_out)
    pickle_out.close()
    

if __name__ == "__main__":
    main()
