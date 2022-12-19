#!/usr/bin/env python

import os
import json
import pickle
import prepping_data
import building_model
import pandas as pd
import numpy as np
import nibabel as nib
from nilearn.masking import unmask
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from argparse import ArgumentParser

def main(path_dataset, path_fmri, path_output, seed, mask, reg, confound, run_regression, run_permutations, run_bootstrap):
    """
    path_dataset: string
        specifies the path to json file containing the dataset
    path_fmri: string
        specifies the path containing the fmri data (not in BIDS format)
    path_output: string
        specifies the path to output the results of the regression analysis
    seed: int
        specifies the integer to initialize a pseudorandom number generator. The default value is 42
    mask: string
        specifies the mask to use to extract the signal. This argument can take the path to a nii file containing a mask. The default value is 'whole-brain', meaning that the signal from the whole-brain will be used
    reg: string
        specifies the regression algorithm to use in the analysis. The default value is 'lasso', meaning that a LASSO regression will be performed
    confound: string
        specifies the path to the counfounds file if needed. The default value is None, meaning that no confounds will be taken into account for the signal extraction
    run_regression: boolean
        if True, the regression analysis is computed
    run_permutations: boolean
        if True, the permutation tests are computed
    run_bootstrap: boolean
        if True, the bootstrap tests are computed
    """
    ########################################################################################
    #Loading the datasets
    ########################################################################################
    ##Open json file containing the dataset
    open_json = open(path_dataset, "r")
    data = json.loads(open_json.read())
    ##Predicted variable
    y = np.array(data["target"])
    ##Group variable: how the data is grouped (by subjects)
    gr = np.array(data["group"])
    ##Close json file
    open_json.close()
    ##Load confound file if specified
    if confound == None:
        confound = None
    else:
        confound = pd.read_csv(confound)

    ########################################################################################
    #Extract fmri signal
    ########################################################################################
    ##Convert fmri files to Nifti-like objects
    array_feps = prepping_data.hdr_to_Nifti(data["data"], path_fmri)
    if mask == "whole-brain":
        masker, extract_X = prepping_data.extract_signal(array_feps, mask="template", standardize = True, confound=confound)
    else:
        masker = nib.load(mask)
        extract_X = prepping_data.extract_signal_from_mask(array_feps, masker, affine=False)
    #Standardize the signal
    stand_X = StandardScaler().fit_transform(extract_X.T)
    X = stand_X.T
    
    ########################################################################################
    #Train and test the model
    ########################################################################################
    #Define the algorithm
    if reg == "lasso":
        algo = Lasso()
    elif reg == "ridge":
        algo = Ridge()
    elif reg == "svr":
        algo = SVR(kernel="linear")
    elif reg == 'linear':
        algo = LinearRegression()
    
    #Set model parameters
    splits=10
    split_procedure='GSS'
    test_size=0.3
    if mask == 'whole-brain':
            mask_name = mask
    else:
        mask_name = os.path.basename(mask).split('.')[0]

    ########################################################################################
    #Regression
    ########################################################################################
    if run_regression:
        X_train, y_train, X_test, y_test, y_pred, model, model_voxel, df_metrics = building_model.train_test_model(X, y, gr, splits=splits, test_size=test_size,reg=algo,random_seed=42, standard=True)

        #Save the outputs
        ##Save readme file with analysis info
        filename_txt = os.path.join(path_output, "readme.txt")
        with open(filename_txt, 'w') as f:
            f.write(f"cv procedure = {split_procedure} \nnumber of folds = {splits} \ntest size = {test_size} \nalgorithm = {reg} \nrandom seed = {seed} \nconfounds = {confound}")

        ##Save the performance metrics
        df_metrics.to_csv(os.path.join(path_output, f'dataframe_metrics_{mask_name}.csv'))

        ##Save model's coefficients
        if mask == "whole-brain" :
            for i, element in enumerate(model_voxel):
                (masker.inverse_transform(element)).to_filename(f"coefs_whole_brain_{i}.nii.gz")
            model_to_averaged = model_voxel.copy()
            model_averaged = sum(model_to_averaged)/len(model_to_averaged)
            (masker.inverse_transform(model_averaged)).to_filename("coefs_whole_brain_ave.nii.gz")
        else :
            array_model_voxel = []
            if mask == "M1" :
                unmask_model = unmask(model_voxel, mask_M1)
            if mask == "without M1": 
                unmask_model = unmask(model_voxel, mask_NoM1)

            for element in unmask_model:
                array_model_voxel.append(np.array(element.dataobj))

            model_ave = sum(array_model_voxel)/len(array_model_voxel)
            model_to_nifti = nib.nifti1.Nifti1Image(model_ave, affine = array_feps[0].affine)
            model_to_nifti.to_filename(f"coefs_{mask_name}_ave.nii.gz")
        
        #Predict on the left out dataset
        print("Test accuray: ", building_model.predict_on_test(X_train=X[:len(y_0)], y_train=y_0, X_test=X[len(y_0):], y_test=y_1, reg=reg))
        
        for i in range(len(X_train)):
            filename = f"train_test_{i}.npz"
            np.savez(filename, X_train=X_train[i],y_train=y_train[i],X_test=X_test[i],y_test=y_test[i],y_pred=y_pred[i])

        ##Saving the model
        filename_model = f"lasso_models_{model}.pickle" 
        pickle_out = open(filename_model,"wb")
        pickle.dump(model, pickle_out)
        pickle_out.close()

    ########################################################################################
    #Compute permutation tests
    ########################################################################################
    if run_permutations:
        score, perm_scores, pvalue = building_model.compute_permutation(X, y, gr, reg=algo, random_seed=seed)
        perm_dict = {'score': score, 'perm_scores': perm_scores.tolist(), 'pvalue': pvalue}
        filename_perm = f"permutation_output_{mask_name}_{seed}.json"
        with open(filename_perm, 'w') as fp:
            json.dump(perm_dict, fp)

    ########################################################################################
    #Compute bootstrap tests
    ########################################################################################
    if run_bootstrap:
        True

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path_dataset", type=str, default=None)
    parser.add_argument("--path_fmri", type=str, default=None)
    parser.add_argument("--path_output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="whole-brain")
    parser.add_argument("--reg", type=str, choices=['lasso','ridge','svr','linear'], default='lasso')
    parser.add_argument('--confound', type=str, default=None)
    parser.add_argument('--run_regression', action='store_true') 
    parser.add_argument('--run_permutations', action='store_true')
    parser.add_argument('--run_bootstrap', action='store_true')
    args = parser.parse_args()

    main(args.path_dataset, args.path_fmri, args.path_output, args.seed, args.model, args.reg, args.confound, )
